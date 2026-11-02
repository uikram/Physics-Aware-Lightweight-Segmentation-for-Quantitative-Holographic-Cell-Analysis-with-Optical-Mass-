"""Individual result files, resumable cell analysis, compilers, views and the one-command reorganiser (DECISIONS.md D-29)."""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_cell_analysis import OFFSET, _make_run, _val  # noqa: E402

from qpi.analysis import compile as cp  # noqa: E402
from qpi.analysis import migrate as mg  # noqa: E402
from qpi.analysis import robustness as rb  # noqa: E402
from qpi.utils.io import read_json, write_json  # noqa: E402
from qpi.utils.layout import analysis_dir, compiled_path  # noqa: E402
from qpi.utils.results_io import absolute_paths_in, read_unit, write_unit  # noqa: E402


# ------------------------------------------------------------------------------------------------ a tiny project
def _project(tmp_path: Path) -> Path:
    import tifffile
    proj = tmp_path / "proj"

    def skip(dirpath, names):
        drop = {n for n in names if n in ("__pycache__", ".pytest_cache")}
        if Path(dirpath) == ROOT:
            drop |= {n for n in names if n in ("to_delete", "outputs", "data")}
        return drop

    shutil.copytree(ROOT, proj, ignore=skip)
    val = _val()
    proc = proj / "data" / "processed"
    (proc / "val" / "phase").mkdir(parents=True)
    (proc / "val" / "mask").mkdir(parents=True)
    with open(proc / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "stem", "registration_ok", "storage_day", "phase_offset_rad"])
        w.writeheader()
        for stem, d in val.items():
            w.writerow({"split": "val", "stem": stem, "registration_ok": "True", "storage_day": d["day"], "phase_offset_rad": OFFSET})
            tifffile.imwrite(proc / "val" / "phase" / f"{stem}.tif", d["phase"].astype(np.float32))
            tifffile.imwrite(proc / "val" / "mask" / f"{stem}.tif", d["gt"].astype(np.uint8))
    _make_run(proj / "outputs" / "full", "edge_sam_lora_r8", val, seed=42)
    _make_run(proj / "outputs" / "multiseed", "edge_sam_lora_r8__s43", val, shift=2, seed=43)
    return proj


def _run(proj, *args):
    return subprocess.run([sys.executable, *args], cwd=proj, capture_output=True, text=True)


CELL = ["scripts/run_cell_analysis.py", "--archs", "edge_sam", "--ranks", "8", "--seeds", "42", "43"]


@pytest.fixture(scope="module")
def proj(tmp_path_factory):
    return _project(tmp_path_factory.mktemp("layout"))


# ------------------------------------------------------------------------------------------------ cell analysis: resume
def test_cell_analysis_writes_individual_files_and_resumes(proj):
    r = _run(proj, *CELL)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert r.stdout.strip().splitlines()[-1] == "COMPLETE"
    store = analysis_dir("cell", proj)
    units = sorted(p.name for p in store.glob("*.json") if not p.name.startswith("_"))
    assert units == ["edge_sam_lora_r8.json", "edge_sam_lora_r8__s43.json"]
    u = read_unit(store / units[0])
    assert u["status"] == "complete" and u["provenance"]["sources"] and not absolute_paths_in(u)
    assert (store / "_annotation.json").exists()
    legacy = read_json(proj / "outputs" / "cell_analysis" / "cell_analysis.json")      # the file the older scripts read
    assert legacy["meta"]["complete"] is True and set(legacy["runs"]) == {"edge_sam_lora_r8", "edge_sam_lora_r8__s43"}
    assert legacy["runs"]["edge_sam_lora_r8"]["run_dir"].startswith(str(proj))          # absolute on this machine, as before
    # second run: nothing is recomputed
    before = {p.name: p.stat().st_mtime_ns for p in store.glob("edge*.json")}
    r2 = _run(proj, *CELL)
    assert r2.returncode == 0 and r2.stdout.count("skipped (up to date)") == 2 and r2.stdout.strip().splitlines()[-1] == "COMPLETE"
    assert before == {p.name: p.stat().st_mtime_ns for p in store.glob("edge*.json")}
    # one individual file removed: only that run is recomputed
    (store / "edge_sam_lora_r8__s43.json").unlink()
    r3 = _run(proj, *CELL)
    assert r3.returncode == 0 and r3.stdout.count("skipped (up to date)") == 1 and r3.stdout.count(" done in ") == 1
    assert (store / "edge_sam_lora_r8__s43.json").exists()
    # predictions changed -> recomputed
    pred = proj / "outputs" / "multiseed" / "runs" / "edge_sam_lora_r8__s43" / "predictions_last.npz"
    data = dict(np.load(pred))
    np.savez(pred, **{k: np.roll(v, 5, axis=1) for k, v in data.items()})
    r4 = _run(proj, *CELL)
    assert r4.returncode == 0 and r4.stdout.count(" done in ") == 1
    # a failing/missing run: exit 1, no COMPLETE, stderr names the run
    r5 = _run(proj, *CELL[:-3], "--seeds", "42", "44")
    assert r5.returncode == 1 and "edge_sam_lora_r8__s44" in r5.stderr and "COMPLETE" not in r5.stdout


def test_compile_cell_reports_missing_seeds_instead_of_zero(proj):
    r = _run(proj, "scripts/compile_results.py", "cell")
    assert r.returncode == 1 and "COMPLETE" not in r.stdout           # strict: an incomplete data set is a failure here
    doc = read_json(compiled_path("cell_analysis", proj))
    c = doc["configurations"]["edge_sam|lora_r8"]
    assert c["n_seeds"] == 2 and c["seeds_found"] == [42, 43] and c["seeds_missing"] == [44]
    assert any("seed 44 missing" in p for p in doc["problems"]) and doc["complete"] is False
    assert c["metrics"]["mdare_pct"]["per_seed"].keys() == {"42", "43"}


# ------------------------------------------------------------------------------------------------ reorganiser
def test_organise_results_is_idempotent_and_matches_the_older_files(proj):
    import shutil as sh
    assert _run(proj, *CELL).returncode == 0                       # the older cell_analysis.json now holds both seeds again
    sh.rmtree(analysis_dir("cell", proj))
    r = _run(proj, "scripts/organise_results.py")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "COMPLETE"
    assert "check cell done" in r.stdout
    assert len(list(analysis_dir("cell", proj).glob("edge*.json"))) == 2
    r2 = _run(proj, "scripts/organise_results.py")
    assert r2.returncode == 0 and "2 already there" in r2.stdout
    # the older files are untouched
    assert (proj / "outputs" / "cell_analysis" / "cell_analysis.json").exists()


def test_organise_dry_run_writes_nothing(tmp_path):
    p = tmp_path / "p"
    (p / "outputs").mkdir(parents=True)
    r = _run(ROOT, "scripts/organise_results.py", "--dry-run", "--root", str(p))
    assert r.returncode == 0 and not (p / "outputs" / "analysis").exists()


# ------------------------------------------------------------------------------------------------ benchmark and robustness units
def _bench(path: Path):
    per = []
    for arch, base in (("edge_sam", 10.0), ("mobile_sam", 20.0)):
        for backend, f in (("pytorch", 1.0), ("onnx", 0.5)):
            for s, vr in zip((42, 43, 44), (100.0, 130.0, 110.0)):
                per.append({"architecture": arch, "precision": "fp32", "backend": backend, "seed": s, "input_size": 256,
                            "batch_size": 1, "weights": "trained", "checkpoint_sha256_16": "ab", "latency_p50_ms": base * f + s % 3,
                            "latency_p99_ms": base * f + 2, "fps": 1000 / (base * f), "peak_vram_mb": vr, "n_timed": 500})
    rows = []
    for arch in ("edge_sam", "mobile_sam"):
        for backend in ("pytorch", "onnx"):
            rr = [r for r in per if r["architecture"] == arch and r["backend"] == backend]
            row = {"architecture": arch, "backend": backend, "n_seeds": 3}
            for k in ("latency_p50_ms", "latency_p99_ms", "fps", "peak_vram_mb"):
                v = np.array([r[k] for r in rr])
                row[k], row[k + "_sd"], row[k + "_per_seed"] = float(v.mean()), float(v.std(ddof=1)), list(map(float, v))
            rows.append(row)
    write_json(path, {"protocol": {"version": 5}, "gpu_name": "x", "host_details": {}, "gpu_at_start_of_seeds": {"42": {"a": 1}},
                      "rows": rows, "per_seed": per, "parameters_by_rank": []})


def test_benchmark_units_compile_with_the_maximum_peak_vram(tmp_path):
    _bench(tmp_path / "outputs" / "full" / "benchmark" / "benchmark.json")
    r = mg.split_benchmark(tmp_path)
    assert r["written"] == 12
    doc = cp.compile_benchmark(tmp_path)
    assert doc["complete"] and all(row["peak_vram_mb_max"] == 130.0 for row in doc["rows"])
    assert doc["rows"][0]["peak_vram_mb"]["mean"] == pytest.approx(113.3333333)
    # a missing seed is listed, not averaged away
    (analysis_dir("benchmark", tmp_path) / "edge_sam__onnx__s44.json").unlink()
    doc = cp.compile_benchmark(tmp_path)
    assert not doc["complete"] and any("edge_sam onnx: seed 44 missing" in p for p in doc["problems"])


def test_robustness_rows_flag_the_offset_rows_and_legacy_file_is_rebuilt(tmp_path):
    import pandas as pd
    df = pd.DataFrame([{"run_id": "edge_sam_lora_r8", "distortion": d, "macro_dice": 0.7 + (0.001 if d == "tilt0.10" else 0),
                        "fg_iou": 0.8, "ipe_bgcorr_true_pct": 3.0, "ipe_uncorrected_seen_pct": 3.1}
                       for d in ("none", "offset+0.10", "tilt0.10")])
    rows = rb.unit_rows(df)
    flags = {r["distortion"]: r["invariant_by_construction"] for r in rows}
    assert flags == {"none": False, "offset+0.10": True, "tilt0.10": False}
    assert rows[2]["delta_macro_dice_vs_none"] == pytest.approx(0.001)
    store = tmp_path / "rob"
    store.mkdir()
    write_unit(store / "edge_sam_lora_r8.json", {"run_id": "edge_sam_lora_r8", "status": "complete", "checkpoint": "last",
                                                 "rows": rows, "checkpoint_file": {"sha256": None}})
    assert rb.write_legacy_robustness(store, ["edge_sam_lora_r8"], "last", tmp_path / "legacy")
    legacy = read_json(tmp_path / "legacy" / "robustness.json")
    assert set(legacy["rows"][0]) == set(rb.LEGACY_FIELDS) and len(legacy["rows"]) == 3
    assert not rb.write_legacy_robustness(store, ["edge_sam_lora_r8", "mobile_sam_lora_r8"], "last", tmp_path / "legacy2")
    ok, _ = rb.unit_is_current(read_unit(store / "edge_sam_lora_r8.json"), "last", tmp_path / "none.pt", [r["distortion"] for r in rows])
    assert ok                                                     # migrated unit: no hash was recorded


# ------------------------------------------------------------------------------------------------ small pieces
def test_parse_run_id_and_config_fields():
    assert cp.parse_run_id("edge_sam_lora_r32__s43") == {"arch": "edge_sam", "lora_rank": 32, "loss": "full", "seed": 43, "config": "edge_sam|lora_r32"}
    assert cp.parse_run_id("mobile_sam_abl_pmc_bga_r4__s44")["config"] == "mobile_sam|abl_pmc_bga_r4"
    assert cp.parse_run_id("mobilenet_unet_abl_dice_only") == {"arch": "mobilenet_unet", "lora_rank": 8, "loss": "dice_only", "seed": 42,
                                                              "config": "mobilenet_unet|abl_dice_only"}
    assert cp.parse_run_id("edge_sam_full_finetune")["loss"] == "full_finetune" and cp.parse_run_id("nonsense") == {}
    assert cp.config_fields("edge_sam|abl_pmc_r16") == {"arch": "edge_sam", "lora_rank": 16, "loss": "pmc"}


def test_stats_use_sample_sd_and_never_count_missing_as_zero():
    s = cp.stats({42: 1.0, 43: 3.0, 44: None})
    assert s["mean"] == 2.0 and s["n"] == 2 and s["sd"] == pytest.approx(np.sqrt(2.0)) and s["per_seed"]["44"] is None
    assert cp.stats({42: 1.0})["sd"] is None


def test_views_select_configurations_without_changing_the_compiled_file():
    doc = {"configurations": {
        "edge_sam|lora_r8": {"arch": "edge_sam", "lora_rank": 8, "loss": "full"},
        "edge_sam|lora_r32": {"arch": "edge_sam", "lora_rank": 32, "loss": "full"},
        "mobile_sam|lora_r8": {"arch": "mobile_sam", "lora_rank": 8, "loss": "full"},
        "mobile_sam|abl_pmc": {"arch": "mobile_sam", "lora_rank": 8, "loss": "pmc"}}}
    before = json.dumps(doc, sort_keys=True)
    same_rank = cp.apply_view(doc, {"rank": 8, "loss": "full"})
    assert [c["config"] for c in same_rank] == ["edge_sam|lora_r8", "mobile_sam|lora_r8"]
    preferred = cp.apply_view(doc, {"configs": [{"arch": "mobile_sam", "rank": 8, "loss": "full"}, {"arch": "edge_sam", "rank": 32, "loss": "full"}]})
    assert [c["config"] for c in preferred] == ["mobile_sam|lora_r8", "edge_sam|lora_r32"]
    assert json.dumps(doc, sort_keys=True) == before


def test_views_file_is_valid():
    views = cp.load_views()
    assert {"same_rank", "preferred", "rank_sweep", "loss_ablation"} <= set(views)
