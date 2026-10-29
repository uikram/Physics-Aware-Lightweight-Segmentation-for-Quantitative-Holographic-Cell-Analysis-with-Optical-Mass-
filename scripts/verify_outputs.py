#!/usr/bin/env python
"""Final check that every expected artefact exists and is internally consistent. Exit status 1 on failure.

    python scripts/verify_outputs.py [--config-overlay smoke.yaml]
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qpi.config import load_config
from qpi.experiments import build_grid, trainable_specs
from qpi.utils.io import read_json
from qpi.utils.outputs import mode_of

FIGURES = ["fig3_per_class_dice", "fig4_lora_vs_full_per_class", "fig5_loss_ablation", "fig6_qualitative",
           "fig7_composition_over_storage", "fig8_dry_mass", "figS1_rank_vs_dice", "figS2_aggregate_radar",
           "figS3_accuracy_throughput", "figS4_gradient_alignment", "figS5_storage_examples", "figS6_ablation_by_rank"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-overlay", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config_overlay)
    mode = mode_of(cfg)
    root, man = Path(cfg["paths_abs"]["output_root"]), Path(cfg["paths_abs"]["manuscript_outputs"])
    errors, warnings = [], []

    def need(p: Path):
        if not p.exists() or (p.is_file() and p.stat().st_size == 0):
            errors.append(f"missing or empty: {p}")

    mark = root / "OUTPUT_MODE.json"
    need(mark)
    if mark.exists() and read_json(mark).get("mode") != mode:
        errors.append(f"{mark} has mode {read_json(mark).get('mode')} but configuration mode is {mode}")
    for f in ("grid.json", "run_metadata.json", "analysis/phase_integral.json", "analysis/drymass.json",
              "analysis/robustness.json", "benchmark/benchmark.json"):
        need(root / f)
    for s in trainable_specs(build_grid(cfg)):
        d = root / "runs" / s.run_id
        for f in ("config.json", "history.csv", "metrics.json", "checkpoints/last.pt", "checkpoints/best.pt",
                  "predictions_last.npz", "per_image_last.csv"):
            need(d / f)
        if (d / "metrics.json").exists():
            m = read_json(d / "metrics.json")
            if m.get("status") == "diverged":
                warnings.append(f"run {s.run_id} diverged (non-finite loss); result is recorded but needs attention")
            elif m.get("status") != "complete":
                errors.append(f"run {s.run_id} has status {m.get('status')}")
            if m.get("mode") != mode:
                errors.append(f"run {s.run_id} was produced in mode {m.get('mode')}")
    need(man / "results.json")
    for f in FIGURES:
        need(man / "figures" / f"{f}.png")
        need(man / "figures" / f"{f}.pdf")
    if (man / "results.json").exists():
        r = read_json(man / "results.json")
        if r["meta"]["mode"] != mode:
            errors.append("results.json mode mismatch")
        # benchmark (protocol v5, D-26): six FP32 combinations per seed, four metrics, results.json equal to benchmark.json
        bj, ej = root / "benchmark" / "benchmark.json", root / "benchmark" / "equivalence.json"
        if bj.exists():
            b = read_json(bj)
            pr = b.get("protocol", {})
            if pr.get("version") != 5:
                errors.append("benchmark.json is not protocol v5; run ops/run_benchmark.sh")
            archs = [x for x in cfg["architectures"] if not cfg.get("smoke") or x in cfg["smoke"]["architectures"]]
            want = {(a_, "fp32", be) for a_ in archs for be in ("pytorch", "onnx")}
            seeds = pr.get("seeds", [])
            if len(want) * len(seeds) != len(b.get("per_seed", [])) or {(x["architecture"], x["precision"], x["backend"])
                                                                         for x in b.get("per_seed", [])} != want:
                errors.append(f"benchmark.json must hold exactly the {len(want)} combinations for each of {len(seeds)} seeds")
            for x in b.get("per_seed", []):
                vals = [x.get(k) for k in ("latency_p50_ms", "latency_p99_ms", "fps", "peak_vram_mb")]
                tag = f"{x['architecture']}/{x['backend']}/seed {x['seed']}"
                if any(v is None or not math.isfinite(v) or v <= 0 for v in vals):
                    errors.append(f"benchmark per-seed row {tag} has a missing or non-positive P50, P99, FPS or Peak VRAM")
                elif abs(x["fps"] - 1000.0 / x["latency_p50_ms"]) > 1e-6 * x["fps"]:
                    errors.append(f"benchmark per-seed row {tag}: FPS is not 1000 / P50")
                elif x["latency_p99_ms"] < x["latency_p50_ms"]:
                    errors.append(f"benchmark per-seed row {tag}: P99 is below P50")
            lab = {"edge_sam": "EdgeSAM", "mobile_sam": "MobileSAM", "mobilenet_unet": "MobileNet-UNet"}
            have = {(x["architecture"], x["backend"]): x for x in r["tables"].get("table6_benchmark", [])}
            for x in b["rows"]:
                got = have.get((lab.get(x["architecture"], x["architecture"]), x["backend"]))
                if got is None or any(abs(got[k] - x[k]) > 1e-9 for k in ("latency_p50_ms", "latency_p99_ms", "fps", "peak_vram_mb")):
                    errors.append(f"results.json benchmark row {x['architecture']}/{x['backend']} differs from benchmark.json; rerun results.py")
            need(ej)
            if ej.exists():
                e = read_json(ej)
                if sorted(a_["architecture"] for a_ in e["architectures"]) != sorted(archs):
                    errors.append("equivalence.json does not cover every benchmarked architecture")
                n_val = r["meta"].get("n_validation_images")
                for a_ in e["architectures"]:
                    n_img = a_["quality"]["n_images"]
                    if n_val is not None and n_img != n_val:
                        errors.append(f"equivalence {a_['architecture']}: {n_img} images checked, the validation set has {n_val}")
                    for c in a_["quality"]["candidates"]:
                        if c["verdict"] == "rejected":
                            errors.append(f"equivalence {a_['architecture']}: {c['candidate']} fails the quality gate")
    # per-cell analysis of every architecture, rank and seed (scripts/run_cell_analysis.py, full mode only)
    if mode == "full":
        cj = Path(cfg["project_root"]) / "outputs" / "cell_analysis" / "cell_analysis.json"
        need(cj)
        if cj.exists():
            from qpi.analysis import cell_analysis as ca
            tables = read_json(man / "results.json")["tables"] if (man / "results.json").exists() else None
            errors += ca.verification_errors(read_json(cj), cfg, root, Path(cfg["project_root"]) / "outputs" / "multiseed", tables)
    # added layout (D-29): checked only where the compiled files exist; incomplete data is a warning, a stale or broken file an error
    if mode == "full":
        from qpi.analysis import compile as cp
        from qpi.utils.layout import compiled_path
        for kind in cp.COMPILERS:
            cpath = compiled_path(cp.NAMES[kind], Path(cfg["project_root"]))
            if not cpath.exists():
                continue
            doc = read_json(cpath)
            errors += [f"{cpath.name}: {p}" for p in cp.stale_problems(kind, Path(cfg["project_root"]))[:5]]
            if not doc.get("complete"):
                warnings.append(f"{cpath.name}: incomplete data, {len(doc.get('problems', []))} problem(s) listed in the file")
    for w in warnings:
        print(f"[WARN] {w}")
    for e in errors:
        print(f"[FAIL] {e}")
    print(f"verify_outputs: mode={mode} errors={len(errors)} warnings={len(warnings)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
