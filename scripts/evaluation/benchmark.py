#!/usr/bin/env python
"""Inference benchmark: float32 latency, throughput and graphics memory, in PyTorch and in ONNX Runtime.

    bash scripts/run_benchmark.sh              every architecture, both runtimes, one pass per seed
    python scripts/evaluation/benchmark.py --gpu 0 --onnx

**What is measured.** Each architecture at its representative rank, with the adapters merged into the base
weights, batch size 1, at its own deployment input size, in PyTorch and through ONNX Runtime with the CUDA
execution provider. The forward pass only: reading, resizing, standardising and the handling of the output are all
outside the timed region, and the inputs are already on the graphics card. Accuracy is never computed inside a
timed loop.

**Why the rank does not matter.** Merging folds the low-rank update into the original weights and deletes the
adapter modules, so the deployed network has the shape and the cost of the original one at every rank. The
benchmark times one rank per architecture and, unless asked not to, checks that the parameter count after merging
is the same at every rank, so the claim is shown rather than asserted. Only the **trainable** count depends on the
rank, and that is tabulated for every rank without any timing.

**Repetition.** The whole benchmark is repeated once per pass seed, each pass in its own process, with the order
of the architectures rotated between passes. A pass seed fixes which validation images are used as inputs. Every
ONNX model is timed in several fresh sessions within a pass, because the convolution algorithms are chosen when a
session is created; the timed runs of those sessions are pooled inside the pass. Latencies of different passes are
never pooled: each percentile is taken inside one pass, and the summary is the mean and the sample standard
deviation over the passes, with the per-pass values reported beside it.

**Fairness.**
  * float32 means float32 in both runtimes. TensorFloat-32 is turned off in PyTorch, and NVIDIA_TF32_OVERRIDE=0
    is set before the libraries load so that ONNX Runtime does not quietly use it either.
  * ONNX Runtime inputs and outputs are bound to the graphics card, so it is not charged for host transfers that
    the PyTorch path, whose inputs are already on the card, does not pay.
  * The determinism switches used during training are off: what is measured is the library default.
  * The exported graph is compared with the PyTorch network on the same inputs, outside the timed loop, so a
    speed-up is a speed-up of the same function. The comparison is reported with the timings.
  * Memory is read the same way for both runtimes: device-level used memory after the timed loop, minus a
    baseline taken before the model reached the card, with the CUDA libraries already loaded so that the first
    model of a process does not carry them. It is a high-water footprint, not a continuously sampled peak, and it
    includes the weights, the activations, the workspace and the device-resident inputs.

**Multi-head attention and the exporter.** PyTorch computes attention with fused kernels that the ONNX exporter
cannot translate. For the duration of the export only, the parameters of each multi-head attention module are
marked as requiring gradients and gradients are enabled, which makes PyTorch take its ordinary path instead of the
fused one, and scaled dot-product attention and tensor unflattening are replaced by equivalent explicit
operations. The patches are removed afterwards, and the PyTorch timings always use the stock kernels.

Outputs, in results/consolidated/benchmark/:

    benchmark.json              everything: protocol, environment, per-pass values, parameters
    benchmark.csv               one row per architecture and runtime, mean and standard deviation over passes
    benchmark_per_pass.csv      one row per architecture, runtime and pass
    benchmark_parameters.csv    one row per architecture and rank
    benchmark_summary.md        a readable table
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# TensorFloat-32 changes what float32 means, and the libraries read this when they load, so it is set before torch
# is imported rather than after.
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from qpicell.cli import add_common, config  # noqa: E402
from qpicell.io import write_json  # noqa: E402
from qpicell.models import ARCHITECTURES, build  # noqa: E402
from qpicell.models.lora import count_parameters, merge, parameter_report  # noqa: E402
from qpicell.training.runs import Run, conditions as all_conditions, training_dir  # noqa: E402

PROTOCOL_VERSION = 5
PRECISION = "float32"
BACKENDS = ("pytorch", "onnx")
N_DEVICE_INPUTS = 16          # distinct real validation images kept on the card and cycled through
ORT_SESSIONS_PER_PASS = 3     # fresh ONNX Runtime sessions timed per architecture and pass
METRICS = ("latency_p50_ms", "latency_p99_ms", "images_per_second", "memory_mib")


# ------------------------------------------------------------------------------- device and environment

def sync(dev) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def device_used_mib(dev) -> Optional[float]:
    if dev.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(dev)
    return (total - free) / 1024 ** 2


def release(dev) -> None:
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(dev)


def apply_tf32_policy() -> Dict[str, Any]:
    """float32 has to mean float32 in both runtimes, or the comparison is between different arithmetic.

    PyTorch uses TensorFloat-32 for convolutions by default while ONNX Runtime uses it for convolutions and
    matrix products, so without this the two runtimes would not be computing the same thing.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    state = {"cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
             "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
             "float32_matmul_precision": torch.get_float32_matmul_precision(),
             "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
             "mode": "strict float32, TensorFloat-32 off"}
    state["strict_effective"] = os.environ.get("NVIDIA_TF32_OVERRIDE") == "0"
    if not state["strict_effective"]:
        print("[benchmark] WARNING: NVIDIA_TF32_OVERRIDE=0 was not set before the libraries loaded, so ONNX "
              "Runtime may still use TensorFloat-32. Start through scripts/run_benchmark.sh.", flush=True)
    return state


def gpu_state(index: int) -> Optional[Dict[str, float]]:
    """Clock, temperature, power, utilisation and used memory, so an odd timing can be explained afterwards."""
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(index),
                              "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu,"
                              "memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return None
        v = [float(t) for t in out.stdout.strip().split(",")]
        return {"sm_clock_mhz": v[0], "sm_clock_max_mhz": v[1], "temperature_c": v[2], "power_w": v[3],
                "utilization_pct": v[4], "memory_used_mib": v[5]}
    except Exception:                                                   # noqa: BLE001
        return None


def host_details() -> Dict[str, Any]:
    out: Dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform(),
                           "torch": torch.__version__, "cpu_logical_cores": os.cpu_count()}
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            names = [l.split(":", 1)[1].strip() for l in fh if l.lower().startswith("model name")]
        out["cpu_model"] = names[0] if names else None
    except OSError:
        out["cpu_model"] = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            out["ram_gib"] = round(int(fh.readline().split()[1]) / 1024 ** 2, 1)
    except (OSError, ValueError):
        out["ram_gib"] = None
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=15).stdout.strip().splitlines()
        if q:
            f = [x.strip() for x in q[0].split(",")]
            out.update({"gpu_name": f[0], "gpu_driver": f[1], "gpu_memory_total_mib": float(f[2]),
                        "gpu_max_sm_clock_mhz": float(f[3]), "gpu_power_limit_w": float(f[4])})
    except Exception:                                                   # noqa: BLE001
        pass
    try:
        import onnxruntime as ort
        out["onnxruntime"] = ort.__version__
        out["onnxruntime_providers"] = list(ort.get_available_providers())
    except ImportError:
        out["onnxruntime"] = None
    if torch.cuda.is_available():
        out["cuda"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version()
    return out


def warm_libraries(dev, scratch: Path, with_onnx: bool) -> None:
    """Load the CUDA libraries before any memory baseline is taken, so the first model does not carry them."""
    if dev.type != "cuda":
        return
    conv = torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.ReLU(),
                               torch.nn.Conv2d(8, 4, 1)).to(dev).eval()
    x = torch.randn(1, 1, 64, 64, device=dev)
    with torch.inference_mode():
        for _ in range(3):
            conv(x)
        torch.softmax(torch.matmul(x.flatten(2), x.flatten(2).transpose(1, 2)), dim=-1)
        F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    if with_onnx:
        scratch.mkdir(parents=True, exist_ok=True)
        path = scratch / "_warm.onnx"
        try:
            export_onnx(conv.cpu(), x.cpu(), path, 17)
            sess = make_session(path, dev)
            sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})
            del sess
        except Exception:                                               # noqa: BLE001
            pass
        finally:
            path.unlink(missing_ok=True)
    del conv, x
    release(dev)


# ---------------------------------------------------------------------------------------------- the model

def representative_ranks(cfg: Dict) -> Dict[str, Optional[int]]:
    """The rank the selection rule chose for each architecture, read from the selection output.

    Nothing is hard-coded here: if the selection has not been run, the caller is told to run it or name a rank.
    """
    path = Path(cfg["paths_abs"]["results"]) / "consolidated" / "selection" / "rank_selection.csv"
    if not path.exists():
        return {}
    out: Dict[str, Optional[int]] = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            c = str(r.get("selected_condition", ""))
            out[r["architecture"]] = None if c == "fullft" else int(c.lstrip("r"))
    return out


def checkpoint_for(cfg: Dict, arch: str, rank: Optional[int], arm: str, seed: int, tag: str) -> Path:
    cond = "fullft" if rank is None else f"r{rank}"
    return training_dir(cfg, Run(arch, cond, arm, int(seed))) / "checkpoints" / f"{tag}.pt"


def load_merged(cfg: Dict, arch: str, rank: Optional[int], ckpt: Path) -> tuple:
    """The deployed network: trained weights loaded, adapters folded in and deleted, evaluation mode."""
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found: the benchmark times a trained checkpoint")
    net = build(cfg, arch, rank, pretrained=False)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(state["model_state"] if "model_state" in state else state)
    info: Dict[str, Any] = {"checkpoint": str(ckpt), "weights": "trained",
                            "checkpoint_epoch": state.get("epoch") if isinstance(state, dict) else None,
                            "checkpoint_sha256_16": hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16]}
    if rank is not None:
        merge(net)
    net = net.eval().float()
    for p in net.parameters():
        p.requires_grad_(False)
    info["parameters_after_merge"] = count_parameters(net)["total"]
    return net, info


def device_inputs(cfg: Dict, size: int, seed: int, n: int) -> List[torch.Tensor]:
    """Distinct real validation images, standardised exactly as the evaluation standardises them."""
    from qpicell.data.dataset import PhaseDataset
    ds = PhaseDataset(cfg, "val", size, cfg["data"]["eval_rotations"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[: min(n, len(ds))]
    return [ds[int(i)]["input"][None].float() for i in idx]


# ------------------------------------------------------------------- ONNX export, patched only while exporting

def _unflatten(self, dim, sizes):
    shape = list(self.shape)
    dim = dim + len(shape) if dim < 0 else dim
    return self.reshape(shape[:dim] + list(sizes) + shape[dim + 1:])


def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **_):
    """Scaled dot-product attention written out. The scale is split between the two factors, as the original
    attention implementations do, which gives the same result."""
    scale = 1.0 / (query.size(-1) ** 0.5) if scale is None else scale
    root = scale ** 0.5
    scores = torch.matmul(query * root, key.transpose(-2, -1) * root)
    if attn_mask is not None:
        scores = (scores.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool
                  else scores + attn_mask.to(scores.dtype))
    return torch.matmul(torch.softmax(scores, dim=-1).to(value.dtype), value)


@contextlib.contextmanager
def onnx_export_patches():
    original = (torch.Tensor.unflatten, F.scaled_dot_product_attention)
    torch.Tensor.unflatten = _unflatten
    F.scaled_dot_product_attention = _sdpa
    try:
        yield
    finally:
        torch.Tensor.unflatten, F.scaled_dot_product_attention = original


def export_onnx(model, example, path: Path, opset: int) -> None:
    """Export the merged network.

    A multi-head attention module takes an inference fast path when gradients are off, and the exporter has no
    translation for the operator that path emits. The fast path is skipped when gradients are enabled and a
    parameter requires them, which gives the ordinary, exportable graph of the same function. Only the export is
    affected; the timings always use the stock kernels.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    m, x = model.eval().float(), example.float()
    extra = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    attention = [mod for mod in m.modules() if isinstance(mod, torch.nn.MultiheadAttention)]
    for mod in attention:
        for prm in mod.parameters():
            prm.requires_grad_(True)
    grad = torch.enable_grad() if attention else torch.no_grad()
    try:
        with onnx_export_patches(), torch.inference_mode(False), grad:
            torch.onnx.export(m, x, str(path), export_params=True, opset_version=opset, do_constant_folding=True,
                              input_names=["input"], output_names=["output"], **extra)
    finally:
        for mod in attention:
            for prm in mod.parameters():
                prm.requires_grad_(False)


def make_session(path: Path, dev):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    if dev.type == "cuda":
        # the pool grows by what is asked for rather than by powers of two, so the reading reflects the model
        providers = [("CUDAExecutionProvider", {"device_id": dev.index or 0,
                                                "arena_extend_strategy": "kSameAsRequested"}),
                     "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
    if dev.type == "cuda" and "CUDAExecutionProvider" not in sess.get_providers():
        raise RuntimeError(f"ONNX Runtime did not enable the CUDA provider; it offered {sess.get_providers()}")
    return sess


def compare_outputs(ref: torch.Tensor, got: np.ndarray) -> Dict[str, float]:
    r = ref.detach().float().cpu().numpy()
    g = np.asarray(got, dtype=np.float32)
    out = {"max_abs_logit_difference": float(np.max(np.abs(r - g))),
           "mean_abs_logit_difference": float(np.mean(np.abs(r - g)))}
    if r.ndim == 4 and r.shape[1] > 1:
        out["class_map_agreement_pct"] = float(100.0 * np.mean(r.argmax(1) == g.argmax(1)))
    return out


# ------------------------------------------------------------------------------------------------- timing

def latency_stats(ms: Sequence[float]) -> Dict[str, Any]:
    a = np.asarray(ms, dtype=float)
    p50 = float(np.percentile(a, 50))
    return {"latency_p50_ms": p50, "latency_p99_ms": float(np.percentile(a, 99)),
            "latency_mean_ms": float(a.mean()), "latency_sd_ms": float(a.std(ddof=1)) if a.size > 1 else None,
            "latency_min_ms": float(a.min()), "latency_max_ms": float(a.max()),
            "images_per_second": 1e3 / p50, "n_timed": int(a.size)}


def time_loop(fn, inputs: List, warmup: int, timed: int, dev) -> Dict[str, Any]:
    n = len(inputs)
    for i in range(warmup):
        fn(inputs[i % n])
    sync(dev)
    t: List[float] = []
    for i in range(timed):
        sync(dev)
        t0 = time.perf_counter()
        fn(inputs[i % n])
        sync(dev)
        t.append((time.perf_counter() - t0) * 1e3)
    return {"latencies_ms": [round(float(v), 4) for v in t], **latency_stats(t)}


def pool_sessions(stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The timed runs of the fresh sessions of one pass, pooled."""
    if len(stats) == 1:
        return stats[0]
    a = np.concatenate([np.asarray(s["latencies_ms"], dtype=float) for s in stats])
    return {"latencies_ms": [round(float(v), 4) for v in a], **latency_stats(a)}


def benchmark_one(cfg: Dict, arch: str, backend: str, dev, scratch: Path, seed: int, rank: Optional[int],
                  arm: str, tag: str, warmup: int, timed: int, opset: int, n_check: int) -> Dict[str, Any]:
    """One architecture in one runtime, for one pass."""
    size = int(cfg["models"][arch]["image_size"])
    ckpt = checkpoint_for(cfg, arch, rank, arm, cfg["training"]["seeds"][0], tag)
    net, info = load_merged(cfg, arch, rank, ckpt)
    xs_cpu = device_inputs(cfg, size, seed, N_DEVICE_INPUTS)
    row: Dict[str, Any] = {"architecture": arch, "architecture_display": cfg["models"][arch]["display"],
                           "backend": backend, "precision": PRECISION, "pass_seed": seed,
                           "lora_rank": rank, "lora_merged": rank is not None, "input_size": size,
                           "batch_size": 1, "n_distinct_inputs": len(xs_cpu), "device": str(dev), **info}
    release(dev)

    if backend == "pytorch":
        base = device_used_mib(dev)
        net = net.to(dev).eval()
        xs = [x.to(dev) for x in xs_cpu]
        row["gpu_before"] = gpu_state(dev.index or 0) if dev.type == "cuda" else None
        with torch.inference_mode():
            stats = time_loop(net, xs, warmup, timed, dev)
        used = device_used_mib(dev)
        row["memory_mib"] = (used - base) if (used is not None and base is not None) else None
        row["gpu_after"] = gpu_state(dev.index or 0) if dev.type == "cuda" else None
        row.update(stats)
        del net, xs

    elif backend == "onnx":
        import onnxruntime as ort
        net = net.to(dev).eval()
        xs = [x.to(dev) for x in xs_cpu]
        with torch.inference_mode():
            outs = [net(x) for x in xs[:n_check]]
        refs = [(o[0] if isinstance(o, (tuple, list)) else o).detach().float().cpu() for o in outs]
        path = scratch / f"{arch}_{tag}_pass{seed}.onnx"
        export_onnx(net, xs[0], path, opset)
        row["onnx_file_mib"] = round(path.stat().st_size / 1024 ** 2, 2)
        np_in = [x.detach().cpu().contiguous().numpy() for x in xs]
        del net, xs, outs
        release(dev)

        where = "cuda" if dev.type == "cuda" else "cpu"
        dev_id = dev.index or 0
        sess_stats: List[Dict[str, Any]] = []
        sess_mem: List[Optional[float]] = []
        checks: List[Dict[str, float]] = []
        for si in range(ORT_SESSIONS_PER_PASS):
            base = device_used_mib(dev)
            sess = make_session(path, dev)                     # fresh: the algorithms are chosen again
            in_name, out_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name
            binds = []
            for arr in np_in:
                ov = ort.OrtValue.ortvalue_from_numpy(arr, where, dev_id)
                b = sess.io_binding()
                b.bind_ortvalue_input(in_name, ov)
                b.bind_output(out_name, where, dev_id)
                binds.append((b, ov))

            def run(t, _s=sess):
                _s.run_with_iobinding(t[0])
                t[0].synchronize_outputs()

            if si == 0:                                        # outside the timed loop
                for (b, _), ref in zip(binds[:n_check], refs):
                    run((b, None))
                    checks.append(compare_outputs(ref, b.copy_outputs_to_cpu()[0]))
                row["gpu_before"] = gpu_state(dev_id) if dev.type == "cuda" else None
                row["onnx_providers"] = list(sess.get_providers())
            sess_stats.append(time_loop(run, binds, warmup, timed, dev))
            used = device_used_mib(dev)
            sess_mem.append((used - base) if (used is not None and base is not None) else None)
            if si == ORT_SESSIONS_PER_PASS - 1:
                row["gpu_after"] = gpu_state(dev_id) if dev.type == "cuda" else None
            binds = None                                       # the bindings keep the session and its pool alive
            del sess, run
            release(dev)

        mem = [m for m in sess_mem if m is not None]
        row["memory_mib"] = float(max(mem)) if mem else None
        row["memory_mib_by_session"] = sess_mem
        row["n_sessions"] = ORT_SESSIONS_PER_PASS
        row["session_p50_ms"] = [s["latency_p50_ms"] for s in sess_stats]
        row["class_map_agreement_pct"] = min(c["class_map_agreement_pct"] for c in checks) if checks else None
        row["max_abs_logit_difference"] = max(c["max_abs_logit_difference"] for c in checks) if checks else None
        row["n_images_checked"] = len(checks)
        row.update(pool_sessions(sess_stats))
        path.unlink(missing_ok=True)
    else:
        raise ValueError(f"unknown runtime {backend!r}")
    release(dev)
    return row


# ------------------------------------------------------------------------------------------------ one pass

def run_pass(job: tuple) -> Dict[str, Any]:
    (cfg, gpu, cpu_only, seed, index, scratch, ranks, arm, tag, warmup, timed, opset, with_onnx, n_check) = job
    dev = torch.device("cpu") if cpu_only else torch.device(f"cuda:{gpu}")
    tf32 = apply_tf32_policy()
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = False
    archs = [a for a in ARCHITECTURES if a in ranks]
    k = index % max(len(archs), 1)
    archs = archs[k:] + archs[:k]                              # rotate the order between passes
    backends = BACKENDS if with_onnx else ("pytorch",)
    start = gpu_state(gpu) if dev.type == "cuda" else None
    warm_libraries(dev, Path(scratch), with_onnx)
    rows = []
    for arch in archs:
        for backend in backends:
            name = cfg["models"][arch]["display"].replace("_", "-")
            print(f"[benchmark] pass {seed}  {name:<16} {backend}", flush=True)
            r = benchmark_one(cfg, arch, backend, dev, Path(scratch), seed, ranks[arch], arm, tag,
                              warmup, timed, opset, n_check)
            rows.append(r)
            extra = ""
            if backend == "onnx" and r.get("class_map_agreement_pct") is not None:
                extra = (f"; class map agrees on {r['class_map_agreement_pct']:.4f} % of pixels, largest "
                         f"difference {r['max_abs_logit_difference']:.3g}")
            mem = "-" if r.get("memory_mib") is None else f"{r['memory_mib']:.0f} MiB"
            print(f"[benchmark]           p50 {r['latency_p50_ms']:7.3f} ms  p99 {r['latency_p99_ms']:7.3f} ms  "
                  f"{r['images_per_second']:7.1f} images/s  {mem}{extra}", flush=True)
    return {"pass_seed": seed, "pass_index": index, "architecture_order": archs, "gpu_at_start": start,
            "rows": rows, "tf32": tf32,
            "gpu_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "processor"}


# ----------------------------------------------------------------------------------------------- aggregation

PER_PASS_KEYS = ("architecture", "architecture_display", "backend", "precision", "pass_seed", "lora_rank",
                 "input_size", "batch_size", "checkpoint_sha256_16", "latency_p50_ms", "latency_p99_ms",
                 "images_per_second", "memory_mib", "n_timed", "n_sessions", "session_p50_ms",
                 "memory_mib_by_session", "class_map_agreement_pct", "max_abs_logit_difference")


def mean_sd(values: List[Optional[float]]) -> tuple:
    v = [float(x) for x in values if x is not None]
    if not v:
        return None, None
    a = np.asarray(v, float)
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else None)


def aggregate(passes: List[Dict]) -> Dict[str, Any]:
    """Per-pass rows and the summary over passes. Latencies of different passes are never pooled."""
    per_pass, groups = [], {}
    for p in passes:
        for r in p["rows"]:
            per_pass.append({k: r[k] for k in PER_PASS_KEYS if k in r})
            groups.setdefault((r["architecture"], r["backend"]), []).append(r)
    order = list(dict.fromkeys(r["architecture"] for p in passes for r in p["rows"]))
    rows = []
    for key in sorted(groups, key=lambda k: (order.index(k[0]), BACKENDS.index(k[1]))):
        arch, backend = key
        g = groups[key]
        out: Dict[str, Any] = {"architecture": arch, "architecture_display": g[0]["architecture_display"],
                               "backend": backend, "precision": PRECISION, "input_size": g[0]["input_size"],
                               "batch_size": 1, "lora_rank": g[0]["lora_rank"],
                               "n_passes": len(g), "pass_seeds": [r["pass_seed"] for r in g]}
        for m in METRICS:
            mu, sd = mean_sd([r.get(m) for r in g])
            out[m], out[m + "_sd"] = mu, sd
            out[m + "_per_pass"] = [r.get(m) for r in g]
        if backend == "onnx":
            a = [r.get("class_map_agreement_pct") for r in g if r.get("class_map_agreement_pct") is not None]
            d = [r.get("max_abs_logit_difference") for r in g if r.get("max_abs_logit_difference") is not None]
            out["class_map_agreement_pct"] = min(a) if a else None
            out["max_abs_logit_difference"] = max(d) if d else None
        rows.append(out)
    return {"rows": rows, "per_pass": per_pass}


def parameters_by_rank(cfg: Dict) -> List[Dict[str, Any]]:
    """Trainable parameters for every architecture and condition. Nothing is timed here."""
    rows = []
    for arch in ARCHITECTURES:
        for cond in all_conditions(cfg):
            rank = None if cond == "fullft" else int(cond.lstrip("r"))
            net = build(cfg, arch, rank, pretrained=False)
            rep = parameter_report(net)
            before = count_parameters(net)
            if rank is not None:
                merge(net)
            rows.append({"architecture": arch, "architecture_display": cfg["models"][arch]["display"],
                         "condition": cond, "lora_rank": rank,
                         "parameters_total": before["total"], "parameters_trainable": before["trainable"],
                         "parameters_trainable_pct": 100.0 * before["trainable"] / max(before["total"], 1),
                         "adapter_parameters": rep["lora_parameters"],
                         "trainable_without_adapters": rep["trainable_non_lora"],
                         "adapted_layers": rep["lora_layers"],
                         "parameters_after_merge": count_parameters(net)["total"]})
            del net
    gc.collect()
    return rows


def rank_independence(params: List[Dict], archs: Sequence[str]) -> List[Dict[str, Any]]:
    """Merged parameter counts across the ranks: identical counts are what makes one rank enough to time."""
    out = []
    for a in archs:
        counts = {r["condition"]: r["parameters_after_merge"] for r in params
                  if r["architecture"] == a and r["lora_rank"] is not None}
        out.append({"architecture": a, "conditions": ", ".join(counts),
                    "parameters_after_merge": sorted(set(counts.values())),
                    "identical_at_every_rank": len(set(counts.values())) == 1})
    return out


def write_csv(path: Path, keys: Sequence[str], rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(keys), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


SUMMARY_CSV = ("architecture_display", "backend", "precision", "input_size", "batch_size", "lora_rank",
               "n_passes", "latency_p50_ms", "latency_p50_ms_sd", "latency_p99_ms", "latency_p99_ms_sd",
               "images_per_second", "images_per_second_sd", "memory_mib", "memory_mib_sd",
               "class_map_agreement_pct", "max_abs_logit_difference")

PARAMETER_KEYS = ("architecture_display", "condition", "lora_rank", "parameters_total", "parameters_trainable",
                  "parameters_trainable_pct", "adapter_parameters", "trainable_without_adapters",
                  "adapted_layers", "parameters_after_merge")


def summary_md(agg: Dict, params: List[Dict], independence: List[Dict], protocol: Dict, host: Dict,
               budget_ms: float) -> str:
    def q(v, p=2):
        return "-" if v is None else f"{v:,.{p}f}"

    def pm(r, k, p=2):
        return "" if r.get(k + "_sd") is None else f" ± {r[k + '_sd']:,.{p}f}"

    L = ["# Inference cost", "",
         f"{host.get('gpu_name', 'unknown device')}, torch {host.get('torch')}, "
         f"ONNX Runtime {host.get('onnxruntime') or 'not used'}. Batch size 1, {protocol['precision']}, "
         f"TensorFloat-32 off.",
         f"{protocol['warmup_iterations']} warm-up and {protocol['timed_iterations']} timed passes per runtime "
         f"per pass, over {protocol['n_distinct_inputs']} distinct validation images already on the device; "
         f"{protocol['n_passes']} passes (seeds {', '.join(str(s) for s in protocol['pass_seeds'])}), each in "
         f"its own process with the architecture order rotated."]
    if protocol["ort_sessions_per_pass"]:
        L.append(f"ONNX Runtime is timed in {protocol['ort_sessions_per_pass']} fresh sessions per pass, pooled "
                 f"within the pass, with inputs and outputs bound to the device.")
    L += ["", "| architecture | runtime | input | rank | median (ms) | 99th (ms) | images/s | memory (MiB) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in agg["rows"]:
        L.append(f"| {r['architecture_display'].replace('_', '-')} | "
                 f"{'PyTorch' if r['backend'] == 'pytorch' else 'ONNX Runtime'} | "
                 f"{r['input_size']} x {r['input_size']} | "
                 f"{'full fine-tuning' if r['lora_rank'] is None else 'r = ' + str(r['lora_rank'])} | "
                 f"{q(r['latency_p50_ms'])}{pm(r, 'latency_p50_ms')} | "
                 f"{q(r['latency_p99_ms'])}{pm(r, 'latency_p99_ms')} | "
                 f"{q(r['images_per_second'], 1)}{pm(r, 'images_per_second', 1)} | "
                 f"{q(r['memory_mib'], 0)}{pm(r, 'memory_mib', 0)} |")
    L += ["", "Mean ± sample standard deviation over the passes. Latencies of different passes are not pooled: "
              "each percentile is taken inside one pass, and the throughput is 1000 divided by that pass's "
              "median. Memory is device-level used memory after the timed loop minus a baseline taken before "
              "the model reached the device; it is a high-water footprint, not a continuously sampled peak."]
    worst = [r["latency_p99_ms"] for r in agg["rows"] if r.get("latency_p99_ms") is not None]
    if worst:
        L += ["", f"Slowest 99th percentile of any architecture and runtime: {max(worst):.2f} ms, against a "
                  f"{budget_ms:.1f} ms budget at {1000.0 / budget_ms:.0f} frames per second."]
    sizes = {r["input_size"] for r in agg["rows"]}
    if len(sizes) > 1:
        L.append(f"The architectures are **not** compared at one input size: {', '.join(str(s) + ' x ' + str(s) for s in sorted(sizes))}. "
                 "A latency comparison between them has to say so.")
    onnx = [r for r in agg["rows"] if r["backend"] == "onnx" and r.get("class_map_agreement_pct") is not None]
    if onnx:
        L += ["", "## Agreement between the exported graph and PyTorch", "",
              "| architecture | class map agrees | largest output difference |", "|---|---|---|"]
        for r in onnx:
            L.append(f"| {r['architecture_display'].replace('_', '-')} | "
                     f"{r['class_map_agreement_pct']:.4f} % | {r['max_abs_logit_difference']:.3g} |")
        L += ["", "Measured outside the timed loop on the same inputs, so a speed-up is a speed-up of the same "
                  "function. The worst value over all passes is reported."]
    if independence:
        L += ["", "## The cost does not depend on the rank", "",
              "| architecture | ranks merged | parameters after merging | identical |", "|---|---|---|---|"]
        for r in independence:
            L.append(f"| {r['architecture'].replace('_', '-')} | {r['conditions']} | "
                     f"{', '.join(f'{v:,}' for v in r['parameters_after_merge'])} | "
                     f"{'yes' if r['identical_at_every_rank'] else 'NO'} |")
        L += ["", "Merging folds the low-rank update into the original weights and deletes the adapters, so the "
                  "deployed network is the same at every rank and one rank is enough to time."]
    L += ["", "## Trainable parameters per rank", "",
          "| architecture | condition | total | trainable | trainable (%) | adapter | adapted layers |",
          "|---|---|---|---|---|---|---|"]
    for r in params:
        L.append(f"| {r['architecture_display'].replace('_', '-')} | "
                 f"{'full fine-tuning' if r['lora_rank'] is None else 'r = ' + str(r['lora_rank'])} | "
                 f"{r['parameters_total']:,} | {r['parameters_trainable']:,} | "
                 f"{r['parameters_trainable_pct']:.2f} | {r['adapter_parameters']:,} | {r['adapted_layers']} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="time on the processor instead (exploratory)")
    ap.add_argument("--warmup", type=int, default=50, help="passes discarded before timing (default: 50)")
    ap.add_argument("--timed", type=int, default=500, help="timed passes per runtime per pass (default: 500)")
    ap.add_argument("--passes", default=None,
                    help="comma-separated pass seeds; these choose the input images and are not training seeds "
                         "(default: the seeds of the configuration)")
    ap.add_argument("--loss", default="dice_cipp", help="which trained run's weights to load")
    ap.add_argument("--checkpoint", default="last", help="which checkpoint to time (default: last)")
    ap.add_argument("--rank", type=int, default=None,
                    help="time this rank for every architecture instead of the representative rank")
    ap.add_argument("--onnx", action="store_true", help="also export and time through ONNX Runtime")
    ap.add_argument("--onnx-opset", type=int, default=17)
    ap.add_argument("--check-images", type=int, default=4,
                    help="images the exported graph is compared on, outside the timed loop (default: 4)")
    ap.add_argument("--frame-rate", type=float, default=30.0)
    ap.add_argument("--no-rank-check", action="store_true",
                    help="skip the demonstration that the merged network is the same at every rank")
    ap.add_argument("--parameters-only", action="store_true",
                    help="do not time anything: recount the parameters of every rank and rewrite "
                         "benchmark_parameters.csv and benchmark_summary.md from the timings already in "
                         "benchmark.json (processor only, no graphics card needed)")
    add_common(ap)
    a = ap.parse_args()
    cfg = config(a)

    ranks = {arch: a.rank for arch in ARCHITECTURES} if a.rank is not None else representative_ranks(cfg)
    if not ranks:
        sys.exit("[benchmark] results/consolidated/selection/rank_selection.csv is missing: run "
                 "scripts/evaluation/objective_and_rank_selection.py first, or name a rank with --rank")
    seeds = [int(s) for s in a.passes.split(",")] if a.passes else list(cfg["training"]["seeds"])
    out = Path(cfg["paths_abs"]["results"]) / "consolidated" / "benchmark"
    scratch = out / "onnx"
    budget_ms = 1000.0 / a.frame_rate

    if a.parameters_only:
        src = out / "benchmark.json"
        if not src.exists():
            sys.exit(f"[benchmark] {src} not found: --parameters-only reuses the timings of an earlier run")
        prev = json.loads(src.read_text(encoding="utf-8"))
        print("[benchmark] --parameters-only: nothing is timed; the timings are read from benchmark.json",
              flush=True)
        params = parameters_by_rank(cfg)
        independence = [] if a.no_rank_check else rank_independence(params, list(ranks))
        agg = {"rows": prev["rows"], "per_pass": prev["per_pass"]}
        prev["parameters_by_rank"] = params
        prev["rank_independence"] = independence
        write_json(src, prev)
        write_csv(out / "benchmark_parameters.csv", PARAMETER_KEYS, params)
        (out / "benchmark_summary.md").write_text(
            summary_md(agg, params, independence, prev["protocol"], prev["host"], budget_ms), encoding="utf-8")
        for r in params:
            print(f"[benchmark] {r['architecture_display']:<16} {r['condition']:<8} "
                  f"adapter {r['adapter_parameters']:>9,}  over {r['adapted_layers']:>3} layers", flush=True)
        print(f"\n[benchmark] rewritten: {out / 'benchmark_parameters.csv'}", flush=True)
        print(f"[benchmark] rewritten: {out / 'benchmark_summary.md'}", flush=True)
        return

    print("=" * 78, flush=True)
    print(f"[benchmark] protocol {PROTOCOL_VERSION}, {PRECISION}, batch size 1", flush=True)
    print("[benchmark] ranks timed: " + ", ".join(
        f"{cfg['models'][k]['display'].replace('_', '-')} "
        f"{'full fine-tuning' if v is None else 'r = ' + str(v)}" for k, v in ranks.items()), flush=True)
    print(f"[benchmark] {len(seeds)} passes (seeds {', '.join(str(s) for s in seeds)}), each in its own "
          f"process; {a.warmup} warm-up and {a.timed} timed passes", flush=True)
    print(f"[benchmark] runtimes: PyTorch{' and ONNX Runtime' if a.onnx else ''}", flush=True)
    print("=" * 78 + "\n", flush=True)

    jobs = [(cfg, a.gpu, bool(a.cpu), int(s), i, str(scratch), ranks, a.loss, a.checkpoint,
             a.warmup, a.timed, a.onnx_opset, bool(a.onnx), a.check_images) for i, s in enumerate(seeds)]
    ctx = mp.get_context("spawn")
    passes = []
    for job in jobs:                                   # one process per pass, run one after the other
        with ctx.Pool(1) as pool:
            passes.append(pool.apply(run_pass, (job,)))

    print("\n[benchmark] parameter counts for every rank (nothing is timed here)", flush=True)
    params = parameters_by_rank(cfg)
    independence = [] if a.no_rank_check else rank_independence(params, list(ranks))
    agg = aggregate(passes)
    host = host_details()
    host["gpu_name"] = passes[0].get("gpu_name")
    protocol = {"version": PROTOCOL_VERSION, "precision": PRECISION,
                "backends": list(BACKENDS) if a.onnx else ["pytorch"],
                "batch_size": 1, "warmup_iterations": a.warmup, "timed_iterations": a.timed,
                "n_passes": len(seeds), "pass_seeds": seeds, "n_distinct_inputs": N_DEVICE_INPUTS,
                "ort_sessions_per_pass": ORT_SESSIONS_PER_PASS if a.onnx else 0,
                "onnx_io_binding": bool(a.onnx), "onnx_opset": a.onnx_opset,
                "ort_arena_extend_strategy": "kSameAsRequested",
                "checkpoint": a.checkpoint, "loss_of_loaded_weights": a.loss,
                "ranks_timed": dict(ranks),
                "scope": "model forward pass only, at each architecture's deployment input size, with the "
                         "inputs already on the device",
                "accuracy_in_timed_loop": False,
                "aggregation": "per-pass values; the summary is the mean and the sample standard deviation over "
                               "the passes; latencies of different passes are not pooled",
                "memory": "device-level used memory after the timed loop minus a baseline taken before the "
                          "model reached the device, with the CUDA libraries already loaded; for ONNX Runtime "
                          "the largest value over the fresh sessions of the pass",
                "tf32": passes[0].get("tf32"),
                "architecture_order_per_pass": {str(p["pass_seed"]): p["architecture_order"] for p in passes}}

    write_json(out / "benchmark.json",
               {"protocol": protocol, "host": host, "rows": agg["rows"], "per_pass": agg["per_pass"],
                "parameters_by_rank": params, "rank_independence": independence,
                "gpu_at_start_of_pass": {str(p["pass_seed"]): p.get("gpu_at_start") for p in passes}})
    write_csv(out / "benchmark.csv", SUMMARY_CSV, agg["rows"])
    write_csv(out / "benchmark_per_pass.csv", PER_PASS_KEYS, agg["per_pass"])
    write_csv(out / "benchmark_parameters.csv", PARAMETER_KEYS, params)
    (out / "benchmark_summary.md").write_text(
        summary_md(agg, params, independence, protocol, host, budget_ms), encoding="utf-8")

    print(f"\n[benchmark] benchmark.csv: {len(agg['rows'])} rows", flush=True)
    print(f"[benchmark] benchmark_per_pass.csv: {len(agg['per_pass'])} rows", flush=True)
    print(f"[benchmark] benchmark_parameters.csv: {len(params)} rows", flush=True)
    for r in independence:
        if not r["identical_at_every_rank"]:
            print(f"[benchmark] WARNING: {r['architecture']} does not have the same merged parameter count at "
                  f"every rank: {r['parameters_after_merge']}", flush=True)
    print(f"[benchmark] {out}", flush=True)


if __name__ == "__main__":
    main()
