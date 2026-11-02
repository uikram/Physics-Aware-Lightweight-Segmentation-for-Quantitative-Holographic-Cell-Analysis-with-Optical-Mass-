"""Inference benchmark: FP32 latency, throughput and GPU memory of PyTorch and ONNX Runtime. Protocol v5 (D-26).

What is measured
    Six combinations: EdgeSAM, MobileSAM and MobileNet-UNet, each in PyTorch FP32 and in ONNX Runtime FP32 (CUDA execution
    provider). Trained reference checkpoint (best.pt of the selected rank, LoRA merged), batch size 1, each model at its own
    deployment input size (EdgeSAM and MobileNet-UNet 256 px, MobileSAM 1024 px), forward pass only (no pre- or post-processing,
    no data loading). ``warmup_iterations`` unrecorded passes, then ``timed_iterations`` synchronised passes on 16 distinct real
    validation phase maps that stay on the GPU. Accuracy is never computed inside a timed loop.

Metrics (per model and runtime)
    P50 (ms) and P99 (ms): percentiles of the timed per-inference latencies.
    FPS = 1000 / P50.
    Peak VRAM (MB): GPU memory used at the device after the timed loop minus the used memory before the model was placed on the
    GPU. The PyTorch caching allocator and the ONNX Runtime arena do not give memory back while the loop runs, so this is the
    high-water footprint of both runtimes measured the same way (weights, activations, workspace, the 16 device-resident
    inputs, library kernels loaded on first use; the CUDA context is excluded). For ONNX Runtime it is the largest value over
    the fresh sessions of the seed.

Repetition and aggregation
    The benchmark is repeated for each seed (42, 43, 44), every seed in its own process, with the architecture order rotated
    between seeds. A seed fixes the validation maps used as inputs. Every ONNX Runtime model is timed in ORT_SESSIONS_PER_PASS
    fresh sessions on one exported file (the time of a session depends on the convolution algorithms cuDNN picks when the
    session is created); inside one seed the timed runs of those sessions are pooled, so a seed has ``timed_iterations`` runs
    (PyTorch) or ``ORT_SESSIONS_PER_PASS * timed_iterations`` runs (ONNX Runtime). P50, P99, FPS and Peak VRAM are computed
    per seed. Latency samples of different seeds are never pooled: the summary is the mean and the sample SD (n - 1) of the
    per-seed values, and the per-seed values are reported next to it.

Fairness and checks
    * Strict FP32 in both runtimes (TF32 off in PyTorch; NVIDIA_TF32_OVERRIDE=0 for ONNX Runtime), recorded in the output.
    * The training-time determinism switches are off: the benchmark measures the library defaults.
    * ONNX Runtime inputs and outputs are bound to the GPU (IO binding), so ORT is not charged for host-device copies that the
      PyTorch path (inputs already on the GPU) does not pay.
    * Each ONNX model is compared with its PyTorch counterpart on the same inputs (argmax agreement, maximum logit difference),
      outside the timed loop, so a speed-up is a speed-up of the same function. The full correctness and quality gate is in
      qpi/equivalence.py.
    * A failing ONNX export or session stops the run; no combination is skipped silently.
"""
from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import inspect
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from qpi.experiments import build_grid, resolve, run_config, run_dir
from qpi.utils.env import _visible_physical_ids, collect_environment
from qpi.utils.io import write_json
from qpi.utils.seed import seed_everything

PROTOCOL_VERSION = 5
PRECISION = "fp32"
BACKENDS = ("pytorch", "onnx")
DEFAULT_SEEDS = (42, 43, 44)
N_DEVICE_INPUTS = 16                  # distinct real inputs kept on the GPU and cycled through
ORT_SESSIONS_PER_PASS = 3             # fresh ONNX Runtime sessions timed per model and seed
METRICS = ("latency_p50_ms", "latency_p99_ms", "fps", "peak_vram_mb")


def reference_rank(cfg: Dict) -> int:
    """LoRA rank of the checkpoint that is timed and checked (the selected rank). Merged LoRA has the same cost for every rank."""
    return int(cfg["benchmark"].get("reference_rank") or cfg["lora"]["selected_rank"])


def benchmark_architectures(cfg: Dict) -> List[str]:
    return [a for a in cfg["architectures"] if not cfg.get("smoke") or a in cfg["smoke"]["architectures"]]


def load_reference_weights(cfg: Dict, spec, model) -> Dict[str, Any]:
    """Load best.pt into ``model`` (LoRA not yet merged) and return what was loaded. A missing checkpoint stops the run."""
    ckpt = run_dir(cfg, spec.run_id) / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found: the benchmark times the trained reference checkpoint")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model_state"], strict=True)
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16]
    return {"weights": "trained", "checkpoint": f"{spec.run_id}/checkpoints/best.pt", "checkpoint_epoch": state.get("epoch"),
            "checkpoint_sha256_16": digest}


def latency_stats(latencies_ms: Any) -> Dict[str, Any]:
    """P50, P99 and FPS (= 1000 / P50) of a set of per-inference latencies."""
    a = np.asarray(latencies_ms, dtype=float)
    p50 = float(np.percentile(a, 50))
    return {"latency_p50_ms": p50, "latency_p99_ms": float(np.percentile(a, 99)), "fps": 1e3 / p50, "n_timed": int(a.size)}


def _combine_stats(sts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One stats dict from the timed loops of several sessions of the same seed: the timed runs are pooled."""
    if len(sts) == 1:
        return sts[0]
    a = np.concatenate([np.asarray(s["latencies_ms"], dtype=float) for s in sts])
    return {"latencies_ms": [round(float(v), 4) for v in a], **latency_stats(a)}


# --------------------------------------------------------------------------------------------------
# ONNX export patches (applied only while exporting, so PyTorch timings use stock kernels)
# --------------------------------------------------------------------------------------------------
def _unflatten(self, dim, sizes):
    shape = list(self.shape)
    dim = dim + len(shape) if dim < 0 else dim
    return self.reshape(shape[:dim] + list(sizes) + shape[dim + 1:])


def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **_):
    scale = 1.0 / (query.size(-1) ** 0.5) if scale is None else scale
    root = scale ** 0.5                  # the scale is split between q and k (as in the original attention implementations); identical result
    scores = torch.matmul(query * root, key.transpose(-2, -1) * root)
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool \
            else scores + attn_mask.to(scores.dtype)
    return torch.matmul(torch.softmax(scores, dim=-1).to(value.dtype), value)


@contextlib.contextmanager
def onnx_export_patches():
    orig = (torch.Tensor.unflatten, F.scaled_dot_product_attention)
    torch.Tensor.unflatten = _unflatten
    F.scaled_dot_product_attention = _sdpa
    try:
        yield
    finally:
        torch.Tensor.unflatten, F.scaled_dot_product_attention = orig


# --------------------------------------------------------------------------------------------------
def parameter_breakdown(model, arch: str) -> Dict[str, float]:
    enc = set()
    if hasattr(model, "encoder"):
        enc.update(id(p) for p in model.encoder.parameters())
    else:  # MobileNet-UNet: the MobileNetV2 stages are enc0..enc5
        for i in range(6):
            m = getattr(model, f"enc{i}", None)
            if m is not None:
                enc.update(id(p) for p in m.parameters())
    et = ett = dt = dtt = 0
    for p in model.parameters():
        if id(p) in enc:
            et += p.numel(); ett += p.numel() if p.requires_grad else 0
        else:
            dt += p.numel(); dtt += p.numel() if p.requires_grad else 0
    pct = lambda a, b: 100.0 * a / b if b else 0.0
    return {"params_total": et + dt, "params_trainable": ett + dtt,
            "encoder_trainable_pct": pct(ett, et), "decoder_trainable_pct": pct(dtt, dt),
            "overall_trainable_pct": pct(ett + dtt, et + dt)}


def _rank_spec(cfg: Dict, arch: str, rank: Optional[int]):
    specs = build_grid(cfg)
    rid = f"{arch}_lora_r{rank}" if rank is not None else f"{arch}_full_finetune"
    return resolve(specs, rid)


def parameters_by_rank(cfg: Dict, archs: List[str]) -> List[Dict[str, Any]]:
    """Trainable-parameter counts of the training configuration for every architecture and LoRA rank (structure only)."""
    from qpi.models import build_model
    rows = []
    for arch in archs:
        for r in cfg["lora"]["ranks"]:
            try:
                spec = _rank_spec(cfg, arch, int(r))
            except Exception:                      # rank not in the grid for this architecture
                continue
            rc = run_config(cfg, spec)
            seed_everything(cfg["seed"])
            model = build_model(rc, cfg["paths_abs"]["weights_dir"], pretrained=False)
            rows.append({"architecture": arch, "lora_rank": int(r), **parameter_breakdown(model, arch)})
            del model
    gc.collect()
    return rows


# --------------------------------------------------------------------------------------------------
# device helpers
# --------------------------------------------------------------------------------------------------
def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _device_used_mb(dev) -> Optional[float]:
    if dev.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(dev)
    return (total - free) / 2 ** 20


def _release(dev):
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(dev)


def gpu_state(dev) -> Optional[Dict[str, float]]:
    """GPU clock, temperature, power, utilisation and used memory from nvidia-smi (None when unavailable)."""
    if dev.type != "cuda":
        return None
    vis = _visible_physical_ids()
    return gpu_state_physical(vis[dev.index or 0] if vis else (dev.index or 0))


def gpu_state_physical(phys: int) -> Optional[Dict[str, float]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(phys), "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw,"
             "utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        v = [float(t) for t in out.stdout.strip().split(",")]
        return {"sm_clock_mhz": v[0], "sm_clock_max_mhz": v[1], "temperature_c": v[2], "power_w": v[3],
                "utilization_pct": v[4], "memory_used_mb": v[5], "physical_gpu": phys}
    except Exception:
        return None


def tf32_state() -> Dict[str, Any]:
    return {"torch_cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "torch_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
            "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
            "BENCH_TF32": os.environ.get("BENCH_TF32")}


def apply_tf32_policy() -> Dict[str, Any]:
    """FP32 means FP32 in both runtimes. On Ampere, PyTorch convolutions use TF32 by default (matmuls do not), while ONNX Runtime's CUDA
    provider uses TF32 for both, so an FP32 comparison would silently compare different arithmetic. Strict mode (default, BENCH_TF32 unset or 0)
    switches TF32 off in PyTorch and, through NVIDIA_TF32_OVERRIDE=0 (read by cuBLAS and cuDNN when they load, so it must be in the environment
    of the process: ops/run_benchmark.sh exports it), in ONNX Runtime too. BENCH_TF32=1 leaves TF32 on in both."""
    strict = os.environ.get("BENCH_TF32", "0") != "1"
    torch.backends.cudnn.allow_tf32 = not strict
    torch.backends.cuda.matmul.allow_tf32 = not strict
    torch.set_float32_matmul_precision("highest" if strict else "high")
    st = tf32_state()
    st["mode"] = "strict FP32 (TF32 off)" if strict else "TF32 on (torch convolutions and matmuls, ORT)"
    st["strict_effective"] = bool(strict and os.environ.get("NVIDIA_TF32_OVERRIDE") == "0")
    if strict and not st["strict_effective"]:
        print("[benchmark] WARNING: NVIDIA_TF32_OVERRIDE=0 is not set in the environment, so ONNX Runtime may still use TF32; "
              "start the benchmark through ops/run_benchmark.sh", flush=True)
    return st


def host_details() -> Dict[str, Any]:
    """CPU model, core count, memory, GPU driver and ORT providers: what a reader needs to reproduce or compare the timings."""
    out: Dict[str, Any] = {"cpu_logical_cores": os.cpu_count()}
    try:
        with open("/proc/cpuinfo") as fh:
            names = [l.split(":", 1)[1].strip() for l in fh if l.lower().startswith("model name")]
        out["cpu_model"] = names[0] if names else None
    except OSError:
        out["cpu_model"] = None
    try:
        with open("/proc/meminfo") as fh:
            out["ram_gb"] = round(int(fh.readline().split()[1]) / 2 ** 20, 1)
    except (OSError, ValueError):
        out["ram_gb"] = None
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        if q:
            f = [x.strip() for x in q[0].split(",")]
            out.update({"gpu_name": f[0], "gpu_driver": f[1], "gpu_memory_total_mb": float(f[2]),
                        "gpu_max_sm_clock_mhz": float(f[3]), "gpu_power_limit_w": float(f[4])})
    except Exception:
        pass
    try:
        import onnxruntime as ort
        out["ort_version"] = ort.__version__
        out["ort_available_providers"] = ort.get_available_providers()
    except ImportError:
        out["ort_available_providers"] = None
    return out


def warm_libraries(dev, scratch: Path) -> None:
    """Load the CUDA libraries (cuDNN, cuBLAS kernels, ONNX Runtime CUDA provider) before any memory baseline is taken, so that the
    first model of a process does not pay for them in its footprint."""
    if dev.type != "cuda":
        return
    conv = torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(8, 4, 1)).to(dev).eval()
    x = torch.randn(1, 1, 64, 64, device=dev)
    with torch.inference_mode():
        for _ in range(3):
            conv(x)
        torch.softmax(torch.matmul(x.flatten(2), x.flatten(2).transpose(1, 2)), dim=-1)
        torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / "_warm.onnx"
    _export_onnx(conv.cpu(), x.cpu(), path)
    sess = _make_session(path, dev)
    sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})
    del sess, conv, x
    path.unlink(missing_ok=True)
    _release(dev)


# --------------------------------------------------------------------------------------------------
# ONNX
# --------------------------------------------------------------------------------------------------
def _export_onnx(model, example, path: Path):
    model.eval()
    m, x = model.float(), example.float()
    # torch >= 2.5 needs dynamo=False to use the TorchScript exporter; older torch has no such argument
    extra = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    # nn.MultiheadAttention takes the inference fast path (aten::_native_multi_head_attention) when grad is disabled or no
    # parameter requires grad, and the ONNX exporter has no symbolic for that op. The fast path is skipped when grad is enabled
    # and a parameter requires grad, which yields the ordinary exportable graph (same function). Only the export is affected.
    mhas = [mod for mod in m.modules() if isinstance(mod, torch.nn.MultiheadAttention)]
    for mod in mhas:
        for prm in mod.parameters():
            prm.requires_grad_(True)
    grad_ctx = torch.enable_grad() if mhas else torch.no_grad()
    with onnx_export_patches(), torch.inference_mode(False), grad_ctx:
        torch.onnx.export(m, x, str(path), export_params=True, opset_version=17, do_constant_folding=True,
                          input_names=["input"], output_names=["output"], **extra)


def _make_session(path: Path, dev):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3                       # errors only
    if dev.type == "cuda":
        # kSameAsRequested: the ORT memory pool grows by what is requested instead of by powers of two, so the device
        # footprint reflects the model and not the pool growth policy
        providers = [("CUDAExecutionProvider", {"device_id": dev.index or 0, "arena_extend_strategy": "kSameAsRequested"}),
                     "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
    if dev.type == "cuda" and "CUDAExecutionProvider" not in sess.get_providers():
        raise RuntimeError(f"ONNX Runtime did not enable the CUDA provider (got {sess.get_providers()})")
    return sess


def compare_outputs(ref: torch.Tensor, got: np.ndarray) -> Dict[str, float]:
    r = ref.detach().float().cpu().numpy()
    g = np.asarray(got, dtype=np.float32)
    out = {"max_abs_logit_diff": float(np.max(np.abs(r - g))), "mean_abs_logit_diff": float(np.mean(np.abs(r - g)))}
    if r.ndim == 4 and r.shape[1] > 1:
        out["argmax_agreement_pct"] = float(100.0 * np.mean(r.argmax(1) == g.argmax(1)))
    return out


# --------------------------------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------------------------------
def _time_loop(fn, inputs: List, warmup: int, timed: int, dev) -> Dict[str, Any]:
    n = len(inputs)
    for i in range(warmup):
        fn(inputs[i % n])
    _sync(dev)
    t = []
    for i in range(timed):
        _sync(dev)
        t0 = time.perf_counter()
        fn(inputs[i % n])
        _sync(dev)
        t.append((time.perf_counter() - t0) * 1e3)
    return {"latencies_ms": [round(float(v), 4) for v in t], **latency_stats(t)}


def _first_tensor(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def benchmark_one(cfg: Dict, arch: str, backend: str, dev, scratch: Path, seed: int, build_rank: int) -> Dict[str, Any]:
    """One model in one runtime (FP32, trained checkpoint, LoRA merged, batch 1) for one seed."""
    from qpi.data.dataset import QPIDataset
    from qpi.models import build_model
    spec = _rank_spec(cfg, arch, build_rank)
    rc = run_config(cfg, spec)
    # deterministic=False: the training-time determinism switches (deterministic cuDNN/cuBLAS algorithms) restrict the kernels
    # PyTorch may choose and would slow the PyTorch path relative to ONNX Runtime; latency is measured with the library defaults
    seed_everything(seed, deterministic=False)
    model = build_model(rc, cfg["paths_abs"]["weights_dir"], pretrained=False).eval()
    weights_info = load_reference_weights(cfg, spec, model)      # trained best.pt; LoRA is merged after loading
    if hasattr(model, "merge_lora"):
        model.merge_lora()
    model = model.eval().float()

    bc = cfg["benchmark"]
    ds = QPIDataset(cfg["paths_abs"]["processed_root"], "val", rc["image_size"], [0], cfg["data"]["ignore_index"],
                    cfg["data"]["max_val_samples"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[: min(N_DEVICE_INPUTS, len(ds))]
    xs_cpu = [ds[int(i)]["phase"][None].float() for i in idx]

    row: Dict[str, Any] = {"architecture": arch, "precision": PRECISION, "backend": backend, "seed": seed,
                           "lora_rank": build_rank, "lora_merged": True, "input_size": rc["image_size"],
                           "batch_size": 1, "device": str(dev), "n_distinct_inputs": len(xs_cpu), **weights_info}
    warm, timed = bc["warmup_iterations"], bc["timed_iterations"]
    _release(dev)

    if backend == "pytorch":
        base = _device_used_mb(dev)
        model = model.to(dev).eval()
        xs = [x.to(dev) for x in xs_cpu]
        row["gpu_before"] = gpu_state(dev)
        with torch.inference_mode():
            stats = _time_loop(model, xs, warm, timed, dev)
        used = _device_used_mb(dev)
        row["peak_vram_mb"] = (used - base) if used is not None else None
        row["gpu_after"] = gpu_state(dev)
        row.update(stats)
        del model, xs
    elif backend == "onnx":
        import onnxruntime as ort
        model = model.to(dev).eval()
        xs = [x.to(dev) for x in xs_cpu]
        with torch.inference_mode():
            refs = [_first_tensor(model(x)).detach().float().cpu() for x in xs[:4]]
        path = scratch / f"{arch}_{PRECISION}_s{seed}.onnx"
        _export_onnx(model, xs[0], path)
        np_in = [x.detach().cpu().contiguous().numpy() for x in xs]
        del model, xs                                                       # free the torch copy before measuring ORT
        _release(dev)
        sess_stats: List[Dict[str, Any]] = []
        sess_mem: List[Optional[float]] = []
        comp: List[Dict[str, Any]] = []
        for si in range(ORT_SESSIONS_PER_PASS):
            base = _device_used_mb(dev)
            sess = _make_session(path, dev)                              # a fresh session: cuDNN picks its algorithms again
            in_name, out_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name
            dev_id = dev.index or 0
            binds = []
            for a in np_in:
                ov = ort.OrtValue.ortvalue_from_numpy(a, "cuda", dev_id)
                b = sess.io_binding()
                b.bind_ortvalue_input(in_name, ov)
                b.bind_output(out_name, "cuda", dev_id)
                binds.append((b, ov))

            def fn(t, sess=sess):
                sess.run_with_iobinding(t[0])
                t[0].synchronize_outputs()
            if si == 0:                                   # what is timed computes the same function as PyTorch (outside the timed loop)
                for (b, _), ref in zip(binds[:4], refs):
                    fn((b, None))
                    comp.append(compare_outputs(ref, b.copy_outputs_to_cpu()[0]))
                row["gpu_before"] = gpu_state(dev)
                row["ort_providers"] = sess.get_providers()
            sess_stats.append(_time_loop(fn, binds, warm, timed, dev))
            used = _device_used_mb(dev)
            sess_mem.append((used - base) if used is not None else None)
            if si == ORT_SESSIONS_PER_PASS - 1:
                row["gpu_after"] = gpu_state(dev)
            binds = b = ov = _ = None                    # IO bindings keep the session (and its memory arena) alive
            del sess, fn
            _release(dev)
        mem = [m for m in sess_mem if m is not None]
        row["peak_vram_mb"] = float(max(mem)) if mem else None
        row["peak_vram_mb_by_session"] = sess_mem
        row["n_sessions"] = ORT_SESSIONS_PER_PASS
        row["session_p50_ms"] = [s["latency_p50_ms"] for s in sess_stats]
        row["max_abs_logit_diff"] = max(c["max_abs_logit_diff"] for c in comp)
        row["argmax_agreement_pct"] = min(c["argmax_agreement_pct"] for c in comp)
        row.update(_combine_stats(sess_stats))
        path.unlink(missing_ok=True)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    _release(dev)
    return row


# --------------------------------------------------------------------------------------------------
# one pass = the six combinations once, for one seed
# --------------------------------------------------------------------------------------------------
def run_pass(cfg: Dict, dev, seed: int, pass_index: int, scratch: Path) -> Dict:
    archs = benchmark_architectures(cfg)
    k = pass_index % len(archs)
    archs = archs[k:] + archs[:k]                                           # rotate the order between passes
    build_rank = reference_rank(cfg)
    scratch.mkdir(parents=True, exist_ok=True)
    tf32 = apply_tf32_policy()
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = False
    start = gpu_state(dev)
    warm_libraries(dev, scratch)
    rows = []
    for arch in archs:
        for backend in BACKENDS:
            print(f"[benchmark] seed {seed} {arch} {PRECISION} {backend}", flush=True)
            rows.append(benchmark_one(cfg, arch, backend, dev, scratch, seed, build_rank))
    return {"seed": seed, "pass_index": pass_index, "gpu_at_start": start, "rows": rows, "arch_order": archs,
            "gpu_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            "environment": {**collect_environment(dev), "host_details": host_details(), "tf32": tf32}}


# --------------------------------------------------------------------------------------------------
# aggregation over seeds
# --------------------------------------------------------------------------------------------------
PER_SEED_KEYS = ("architecture", "precision", "backend", "seed", "input_size", "batch_size", "weights", "checkpoint_sha256_16",
                 "latency_p50_ms", "latency_p99_ms", "fps", "peak_vram_mb", "n_timed")


def _mean_sd(vals: List[Optional[float]]) -> Dict[str, Any]:
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None, "sd": None}
    a = np.asarray(v, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None}


def aggregate_seeds(passes: List[Dict]) -> Dict[str, Any]:
    """Per-seed rows and the summary over seeds (mean and sample SD of the per-seed P50, P99, FPS and Peak VRAM).

    The latency samples of different seeds are not pooled: every percentile is taken inside one seed, FPS = 1000 / that seed's P50."""
    per_seed, groups = [], {}
    for p in passes:
        for r in p["rows"]:
            if r["precision"] != PRECISION or r["backend"] not in BACKENDS:
                raise ValueError(f"unexpected benchmark row {r['architecture']}/{r['precision']}/{r['backend']}")
            per_seed.append({k: r[k] for k in PER_SEED_KEYS if k in r} |
                            {k: r[k] for k in ("session_p50_ms", "peak_vram_mb_by_session", "argmax_agreement_pct", "max_abs_logit_diff")
                             if k in r})
            groups.setdefault((r["architecture"], r["backend"]), []).append(r)
    arch_order = list(dict.fromkeys(r["architecture"] for p in passes for r in p["rows"]))
    rows = []
    for (arch, backend) in sorted(groups, key=lambda k: (arch_order.index(k[0]), BACKENDS.index(k[1]))):
        g = groups[(arch, backend)]
        out: Dict[str, Any] = {"architecture": arch, "precision": PRECISION, "backend": backend, "input_size": g[0]["input_size"],
                               "batch_size": 1, "weights": g[0]["weights"], "n_seeds": len(g), "seeds": [r["seed"] for r in g]}
        for k in METRICS:
            s = _mean_sd([r.get(k) for r in g])
            out[k], out[k + "_sd"], out[k + "_per_seed"] = s["mean"], s["sd"], [r.get(k) for r in g]
        if backend == "onnx":
            out["argmax_agreement_pct"] = min(r["argmax_agreement_pct"] for r in g)
            out["max_abs_logit_diff"] = max(r["max_abs_logit_diff"] for r in g)
        rows.append(out)
    return {"rows": rows, "per_seed": per_seed}


SUMMARY_CSV = ("architecture", "backend", "precision", "input_size", "batch_size", "n_seeds",
               "latency_p50_ms", "latency_p50_ms_sd", "latency_p99_ms", "latency_p99_ms_sd", "fps", "fps_sd", "peak_vram_mb", "peak_vram_mb_sd")
PER_SEED_CSV = ("architecture", "backend", "precision", "seed", "input_size", "batch_size", "latency_p50_ms", "latency_p99_ms", "fps",
                "peak_vram_mb", "n_timed")


def _write_csv(path: Path, keys, rows) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(keys), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_outputs(cfg: Dict, out_dir: Path, passes: List[Dict], params: List[Dict]) -> Dict:
    bc = cfg["benchmark"]
    agg = aggregate_seeds(passes)
    result = {"protocol": {"version": PROTOCOL_VERSION, "precision": PRECISION, "backends": list(BACKENDS),
                           "combinations_per_seed": len(benchmark_architectures(cfg)) * len(BACKENDS),
                           "batch_size": 1, "warmup_iterations": bc["warmup_iterations"], "timed_iterations": bc["timed_iterations"],
                           "seeds": [p["seed"] for p in passes],
                           "scope": "model forward pass only, deployment input size per architecture",
                           "weights": "trained reference checkpoint (best.pt of the selected rank), LoRA merged",
                           "ort_sessions_per_seed": ORT_SESSIONS_PER_PASS,
                           "ort_session_pooling": "inside one seed the timed runs of the fresh sessions are pooled",
                           "onnx_io_binding": True, "ort_arena_extend_strategy": "kSameAsRequested",
                           "accuracy_in_timed_loop": False,
                           "metrics": {"latency_p50_ms": "50th percentile of the per-inference latencies of one seed",
                                       "latency_p99_ms": "99th percentile of the per-inference latencies of one seed",
                                       "fps": "1000 / latency_p50_ms of the same seed",
                                       "peak_vram_mb": "device-level GPU memory used after the timed loop minus before the model was "
                                                       "placed (weights, activations, workspace, device inputs; CUDA context excluded); "
                                                       "ONNX Runtime: largest value over the fresh sessions of the seed"},
                           "aggregation": "per-seed values; summary = mean and sample SD (n - 1) over seeds of the per-seed values; "
                                          "latency samples of different seeds are not pooled",
                           "arch_order_per_seed": {str(p["seed"]): p["arch_order"] for p in passes},
                           "tf32": (passes[0].get("environment") or {}).get("tf32")},
              "gpu_name": passes[0].get("gpu_name"), "host_details": (passes[0].get("environment") or {}).get("host_details"),
              "gpu_at_start_of_seeds": {str(p["seed"]): p.get("gpu_at_start") for p in passes},
              "rows": agg["rows"], "per_seed": agg["per_seed"], "parameters_by_rank": params}
    write_json(out_dir / "benchmark.json", result)
    _write_csv(out_dir / "benchmark.csv", SUMMARY_CSV, agg["rows"])
    _write_csv(out_dir / "benchmark_per_seed.csv", PER_SEED_CSV, agg["per_seed"])
    if params:
        _write_csv(out_dir / "benchmark_parameters_by_rank.csv", list(params[0]), params)
    return result
