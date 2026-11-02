"""Benchmark protocol v5: FP32, PyTorch and ONNX Runtime, P50 / P99 / FPS / Peak VRAM, aggregation over seeds, equivalence gate."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import qpi.benchmark as bm
from qpi.benchmark import aggregate_seeds, compare_outputs, latency_stats


def _row(arch, backend, seed, lats, vram, **extra):
    st = latency_stats(lats)
    return {"architecture": arch, "precision": "fp32", "backend": backend, "seed": seed, "input_size": 256, "batch_size": 1,
            "weights": "trained", "checkpoint_sha256_16": "x" * 16, "latencies_ms": list(lats), "peak_vram_mb": vram, **st, **extra}


def test_latency_stats_definitions():
    a = np.arange(1, 101, dtype=float)
    st = latency_stats(a)
    assert st["latency_p50_ms"] == np.percentile(a, 50) and st["latency_p99_ms"] == np.percentile(a, 99)
    assert abs(st["fps"] - 1000.0 / st["latency_p50_ms"]) < 1e-12 and st["n_timed"] == 100


def test_aggregate_seeds_mean_sd_no_pooling():
    rng = np.random.default_rng(0)
    seeds = {42: 10.0, 43: 12.0, 44: 14.0}
    passes = [{"seed": s, "rows": [_row("edge_sam", "pytorch", s, base + rng.normal(0, 0.1, 200), 100.0 + s),
                                   _row("edge_sam", "onnx", s, base / 3 + rng.normal(0, 0.05, 200), 50.0,
                                        argmax_agreement_pct=99.99 + (s - 42) * 1e-3, max_abs_logit_diff=1e-4 * (s - 41))]}
              for s, base in seeds.items()]
    agg = aggregate_seeds(passes)
    assert [(r["backend"], r["n_seeds"], r["seeds"]) for r in agg["rows"]] == [("pytorch", 3, [42, 43, 44]), ("onnx", 3, [42, 43, 44])]
    r = agg["rows"][0]
    per_seed_p50 = [p["rows"][0]["latency_p50_ms"] for p in passes]
    per_seed_fps = [1000.0 / x for x in per_seed_p50]
    assert abs(r["latency_p50_ms"] - np.mean(per_seed_p50)) < 1e-12 and abs(r["latency_p50_ms_sd"] - np.std(per_seed_p50, ddof=1)) < 1e-12
    assert abs(r["fps"] - np.mean(per_seed_fps)) < 1e-9 and r["latency_p99_ms_per_seed"] == [p["rows"][0]["latency_p99_ms"] for p in passes]
    assert r["peak_vram_mb_per_seed"] == [142.0, 143.0, 144.0]
    pooled = np.percentile(np.concatenate([p["rows"][0]["latencies_ms"] for p in passes]), 50)
    assert abs(r["latency_p50_ms"] - pooled) > 1e-6                          # the mean of per-seed P50 is not the pooled P50
    o = agg["rows"][1]
    assert o["argmax_agreement_pct"] == 99.99 and abs(o["max_abs_logit_diff"] - 3e-4) < 1e-12
    assert len(agg["per_seed"]) == 6 and "latencies_ms" not in agg["per_seed"][0]
    assert set(("latency_p50_ms", "latency_p99_ms", "fps", "peak_vram_mb")) <= set(agg["per_seed"][0])


def test_single_seed_has_no_sd():
    agg = aggregate_seeds([{"seed": 42, "rows": [_row("a", "pytorch", 42, [1.0, 2.0, 3.0], 10.0)]}])
    assert agg["rows"][0]["latency_p50_ms_sd"] is None and agg["rows"][0]["n_seeds"] == 1


def test_aggregate_rejects_other_precision_or_runtime():
    bad = _row("a", "pytorch", 42, [1.0, 2.0], 1.0)
    bad["precision"] = "fp16"
    with pytest.raises(ValueError):
        aggregate_seeds([{"seed": 42, "rows": [bad]}])
    bad = _row("a", "tensorrt", 42, [1.0, 2.0], 1.0)
    with pytest.raises(ValueError):
        aggregate_seeds([{"seed": 42, "rows": [bad]}])


def test_combine_stats_pools_sessions_within_a_seed():
    def st(v):
        return {"latencies_ms": [v] * 10, **latency_stats([v] * 10)}
    c = bm._combine_stats([st(3.4), st(4.5)])
    assert c["n_timed"] == 20 and abs(c["latency_p50_ms"] - 3.95) < 1e-9 and abs(c["fps"] - 1000 / 3.95) < 1e-9
    s1 = st(3.4)
    assert bm._combine_stats([s1]) is s1


def test_compare_outputs():
    x = torch.randn(1, 4, 8, 8)
    c = compare_outputs(x, x.numpy() + 1e-4)
    assert c["max_abs_logit_diff"] < 2e-4 and c["argmax_agreement_pct"] == 100.0


def test_compare_logits_explains_near_ties_and_flags_real_differences():
    from qpi.equivalence import compare_logits
    ref = np.zeros((1, 3, 32, 32), dtype=np.float32)
    ref[:, 0] = 1.0
    got = ref.copy()
    ref[0, 1, 0, 0] = 0.999999                 # near-tie with class 0 in the reference
    got[0, 1, 0, 0] = 1.00001                  # flips the arg-max by a hair
    c = compare_logits(ref, got)
    assert c["n_disagree"] == 1 and c["all_disagreements_within_2x_max_logit_diff"] and c["explained_by_near_ties"]
    got2 = ref.copy()
    got2[0, 2, 1, 1] = 5.0                     # a real change
    c2 = compare_logits(ref, got2)
    assert c2["n_disagree"] == 1 and c2["max_abs_logit_diff"] == 5.0
    rng = np.random.default_rng(0)
    r = rng.normal(size=(1, 4, 16, 16)).astype(np.float32)
    g = r.copy()
    g[0, :, 3, 3] = np.inf
    c3 = compare_logits(r, g)
    assert c3["nonfinite_other"] == 4 and not c3["explained_by_near_ties"]
    c4 = compare_logits(r, rng.normal(size=r.shape).astype(np.float32) * 50)
    assert c4["disagree_pct"] > 50 and not c4["explained_by_near_ties"]


def test_quality_gate_thresholds_unchanged_and_applied():
    from qpi.equivalence import ACCEPTANCE, verdict_for
    assert ACCEPTANCE == {"argmax_agreement_pct_min": 99.9, "mean_dice_abs_change_max": 0.001,
                          "dry_mass_mean_abs_rel_error_max": 1e-3, "dry_mass_p99_rel_error_max": 5e-3}
    ok = {"argmax_agreement_pct": 99.95, "mean_dice_change": -0.0005, "dry_mass_mean_abs_rel_error": 5e-4, "dry_mass_p99_rel_error": 4e-3}
    assert verdict_for(ok, 66) == "accepted"
    for k, v in (("argmax_agreement_pct", 99.8), ("mean_dice_change", 0.002), ("dry_mass_mean_abs_rel_error", 2e-3),
                 ("dry_mass_p99_rel_error", 6e-3)):
        assert verdict_for({**ok, k: v}, 66) == "rejected", k
    assert verdict_for(ok, 0) == "rejected"


def test_trained_weights_missing_checkpoint_is_an_error(tmp_path):
    cfg = {"paths_abs": {"output_root": str(tmp_path)}}
    with pytest.raises(FileNotFoundError):
        bm.load_reference_weights(cfg, SimpleNamespace(run_id="edge_sam_lora_r8"), torch.nn.Linear(1, 1))


def test_trained_weights_load_and_hash(tmp_path):
    src, dst = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    ck = tmp_path / "runs" / "r1" / "checkpoints"
    ck.mkdir(parents=True)
    torch.save({"model_state": src.state_dict(), "epoch": 7}, ck / "best.pt")
    info = bm.load_reference_weights({"paths_abs": {"output_root": str(tmp_path)}}, SimpleNamespace(run_id="r1"), dst)
    assert info["weights"] == "trained" and info["checkpoint_epoch"] == 7 and len(info["checkpoint_sha256_16"]) == 16
    assert torch.equal(src.weight, dst.weight)


def test_onnx_export_and_cpu_session_match_pytorch(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    net = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(4, 3, 1)).eval()
    x = torch.randn(1, 1, 16, 16)
    path = tmp_path / "m.onnx"
    bm._export_onnx(net, x, path)
    sess = bm._make_session(path, torch.device("cpu"))
    got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    c = compare_outputs(net(x), got)
    assert c["max_abs_logit_diff"] < 1e-5 and c["argmax_agreement_pct"] == 100.0


def _cfg(archs=("edge_sam", "mobile_sam", "mobilenet_unet"), precisions=("fp32", "fp16")):
    return {"architectures": list(archs), "benchmark": {"warmup_iterations": 50, "timed_iterations": 500, "precisions": list(precisions),
                                                         "onnx_skip": ["mobile_sam"]},
            "lora": {"selected_rank": 8}}


def test_run_pass_runs_exactly_the_six_fp32_combinations_in_rotated_order(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(bm, "benchmark_one", lambda cfg, arch, backend, dev, scratch, seed, rank: calls.append((arch, backend, seed, rank))
                        or _row(arch, backend, seed, [1.0, 2.0], 1.0))
    monkeypatch.setattr(bm, "collect_environment", lambda dev: {})
    monkeypatch.setattr(bm, "host_details", lambda: {})
    res = bm.run_pass(_cfg(), torch.device("cpu"), 43, 1, tmp_path)      # config still lists fp16 and onnx_skip: both are ignored
    assert calls == [("mobile_sam", "pytorch", 43, 8), ("mobile_sam", "onnx", 43, 8), ("mobilenet_unet", "pytorch", 43, 8),
                     ("mobilenet_unet", "onnx", 43, 8), ("edge_sam", "pytorch", 43, 8), ("edge_sam", "onnx", 43, 8)]
    assert len(res["rows"]) == 6 and res["arch_order"] == ["mobile_sam", "mobilenet_unet", "edge_sam"]


def test_write_outputs_has_the_four_metrics_per_seed_and_in_the_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "benchmark_architectures", lambda cfg: ["edge_sam"])
    passes = []
    for i, s in enumerate((42, 43, 44)):
        rows = [_row("edge_sam", "pytorch", s, 9.0 + i + np.arange(100) * 0.01, 100.0),
                _row("edge_sam", "onnx", s, 3.0 + i * 0.1 + np.arange(100) * 0.01, 120.0, argmax_agreement_pct=100.0,
                     max_abs_logit_diff=1e-4)]
        passes.append({"seed": s, "rows": rows, "arch_order": ["edge_sam"], "gpu_name": "GPU", "environment": {"tf32": {"mode": "strict"}}})
    res = bm.write_outputs(_cfg(("edge_sam",)), tmp_path, passes, [{"architecture": "edge_sam", "lora_rank": 8}])
    assert res["protocol"]["version"] == 5 and res["protocol"]["precision"] == "fp32" and res["protocol"]["combinations_per_seed"] == 2
    assert "fps" in res["protocol"]["metrics"] and "1000 / latency_p50_ms" in res["protocol"]["metrics"]["fps"]
    import csv
    summ = list(csv.DictReader(open(tmp_path / "benchmark.csv")))
    per = list(csv.DictReader(open(tmp_path / "benchmark_per_seed.csv")))
    assert len(summ) == 2 and len(per) == 6
    for k in ("latency_p50_ms", "latency_p99_ms", "fps", "peak_vram_mb"):
        assert all(float(r[k]) > 0 for r in summ + per), k
    assert not (tmp_path / "benchmark_passes.csv").exists()


def _tex():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manuscript"))
    import make_tex
    return make_tex


def test_tex_benchmark_and_equivalence_sections():
    make_tex = _tex()
    D = object.__new__(make_tex.Data)
    D.bench = [{"architecture": "EdgeSAM", "backend": "onnx", "input_size": 256, "latency_p50_ms": 3.4, "latency_p50_ms_sd": 0.5,
                "latency_p99_ms": 5.0, "latency_p99_ms_sd": 0.2, "fps": 294.0, "fps_sd": 3.0, "peak_vram_mb": 118.0, "peak_vram_mb_sd": None}]
    D.bench_seeds = [{"architecture": "EdgeSAM", "backend": "onnx", "seed": 42, "latency_p50_ms": 3.4, "latency_p99_ms": 5.0, "fps": 294.0,
                      "peak_vram_mb": 118.0}]
    D.bench_protocol = {"seeds": [42, 43, 44], "warmup_iterations": 50, "timed_iterations": 500, "ort_sessions_per_seed": 3}
    D.host = {"gpu_name": "NVIDIA RTX A5000"}
    D.equiv = [{"architecture": "EdgeSAM", "rank": 8, "n_inputs": 16, "argmax_agreement_pct": 99.9999, "max_abs_logit_diff": 1.2e-4,
                "quality_argmax_agreement_pct": 99.9998, "mean_dice_pytorch": 0.7, "quality_mean_dice": 0.7, "quality_mean_dice_change": 0.0,
                "quality_dry_mass_mean_abs_rel_error": 1e-8, "quality_dry_mass_p99_rel_error": 4e-7, "quality_n_images": 66,
                "quality_verdict": "accepted"}]
    D.equiv_acc = {"argmax_agreement_pct_min": 99.9, "mean_dice_abs_change_max": 0.001, "dry_mass_mean_abs_rel_error_max": 0.001,
                   "dry_mass_p99_rel_error_max": 0.005}
    tex = make_tex.benchmark_section(D)
    for needle in ("P50 (ms)", "P99 (ms)", "FPS", "Peak VRAM (MB)", "FPS = 1000 / P50", "not pooled", "per seed", "RTX A5000"):
        assert needle in tex, needle
    for gone in ("FP16", "Real time", "GMACs", "CPU latency", "End-to-end"):
        assert gone not in tex, gone
    eq = make_tex.equivalence_section(D)
    assert "accepted" in eq and "66" in eq and "fixed before the first trial" in eq
