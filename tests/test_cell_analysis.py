"""Cell analysis for all runs (qpi/analysis/cell_analysis.py): synthetic data only (no weights, no GPU, no real data)."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qpi.analysis import cell_analysis as ca  # noqa: E402
from qpi.analysis import drymass as dm  # noqa: E402
from qpi.config import load_config  # noqa: E402
from qpi.utils.classnames import LEGACY_NAMES, MANUSCRIPT_NAMES  # noqa: E402
from qpi.utils.io import read_json, write_json  # noqa: E402

SIZE = 192
CENTRES = [(50, 50, 1), (50, 140, 2), (140, 50, 3), (140, 140, 4)]      # one disc per class (mask values 1..4)
DAYS = [0, 0, 5, 5, 40, 40]
OFFSET = 0.5


def _val(wrapped_image=None):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    val = {}
    for i, day in enumerate(DAYS):
        gt = np.zeros((SIZE, SIZE), np.int64)
        phase = rng.normal(0, 0.01, (SIZE, SIZE))
        for cy, cx, cls in CENTRES:
            m = (yy - cy) ** 2 + (xx - cx) ** 2 <= 22 ** 2
            gt[m] = cls
            phase[m] += 1.2 + 0.1 * cls + 0.02 * day / 10
        if wrapped_image == i:
            phase[(yy - 50) ** 2 + (xx - 50) ** 2 <= 22 ** 2] = 6.2          # raw phase at the 2 pi ceiling
        val[f"20110511_{i}"] = {"phase": phase, "gt": gt, "valid": np.ones((SIZE, SIZE), bool), "day": day, "offset": OFFSET}
    return val


def _metrics(seed, dice_sphero=0.5):
    ev = {"mean_dice": 0.6, "aji": 0.3, "bf1": 0.7, "ipe_global": 0.03,
          "dice_discocyte": 0.9, "dice_echinocyte": dice_sphero,       # legacy name: mask value 2
          "dice_spherocyte": 0.01, "dice_stomatocyte": 0.4,           # legacy name: mask value 3
          "ipe_discocyte": 0.01, "ipe_echinocyte": 0.2, "ipe_spherocyte": 0.3, "ipe_stomatocyte": 0.4}
    return {"status": "complete", "seed": seed, "parameters": {"trainable": 10, "total": 100},
            "training": {"epochs_completed": 50}, "evaluation": {"last": ev}}


def _make_run(root: Path, run_id: str, val, shift=0, seed=42):
    d = root / "runs" / run_id
    d.mkdir(parents=True)
    write_json(d / "metrics.json", _metrics(seed))
    np.savez(d / "predictions_last.npz", **{k: np.roll(v["gt"], shift, axis=1).astype(np.uint8) for k, v in val.items()})
    return d


@pytest.fixture(scope="module")
def cfg():
    return load_config(None)


@pytest.fixture(scope="module")
def cfg_m(cfg):
    return ca.manuscript_cfg(cfg)


def test_manuscript_cfg_does_not_touch_the_original(cfg, cfg_m):
    assert list(cfg["data"]["class_names"]) == LEGACY_NAMES
    assert list(cfg_m["data"]["class_names"]) == MANUSCRIPT_NAMES


def test_locate_run_follows_the_run_layout(cfg, tmp_path):
    main, ms = tmp_path / "full", tmp_path / "multi"
    r = ca.locate_run(cfg, main, ms, "edge_sam", 32, "full", 42)
    assert r.run_id == "edge_sam_lora_r32" and r.run_dir == main / "runs" / "edge_sam_lora_r32" and r.config == "edge_sam|lora_r32"
    r = ca.locate_run(cfg, main, ms, "edge_sam", 32, "full", 43)
    assert r.run_id == "edge_sam_lora_r32__s43" and r.run_dir == ms / "runs" / "edge_sam_lora_r32__s43"
    assert ca.locate_run(cfg, main, ms, "mobile_sam", 8, "pmc", 42).run_id == "mobile_sam_abl_pmc"        # selected rank: main grid
    assert ca.locate_run(cfg, main, ms, "mobile_sam", 8, "pmc", 44).run_id == "mobile_sam_abl_pmc__s44"
    r = ca.locate_run(cfg, main, ms, "mobile_sam", 16, "pmc_bga", 42)                                   # other rank: multiseed only
    assert r.run_id == "mobile_sam_abl_pmc_bga_r16__s42" and r.run_dir.parent.parent == ms and r.config == "mobile_sam|abl_pmc_bga_r16"
    assert ca.locate_run(cfg, main, ms, "mobilenet_unet", None, "full_finetune", 42).run_id == "mobilenet_unet_full_finetune"
    assert ca.locate_run(cfg, main, ms, "mobilenet_unet", None, "full_finetune", 43) is None             # one seed only (D-B)


def test_default_scope_is_45_runs(cfg, tmp_path):
    refs = ca.expected_runs(cfg, tmp_path / "a", tmp_path / "b", list(cfg["architectures"]), cfg["lora"]["ranks"], [42, 43, 44])
    assert len(refs) == 45 and len({r.run_id for r in refs}) == 45
    assert sum(r.run_dir.parent.parent == tmp_path / "a" for r in refs) == 15                           # seed 42 in outputs/full


def test_helpers_equal_compare_architectures():
    spec = importlib.util.spec_from_file_location("compare_architectures", ROOT / "scripts" / "compare_architectures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(1)
    mk = lambda: pd.DataFrame({"storage_day": rng.choice([0, 5, 30], 60), "class_idx": rng.integers(1, 5, 60)})   # noqa: E731
    p, g = mk(), mk()
    assert ca.composition_error(p, g, 5) == mod.composition_error(p, g, 5)
    assert ca.class_count_ratio(p, g, MANUSCRIPT_NAMES) == mod.class_count_ratio(p, g, MANUSCRIPT_NAMES)


def test_end_to_end_two_seeds(cfg, cfg_m, tmp_path):
    val = _val(wrapped_image=2)
    main, ms, out = tmp_path / "full", tmp_path / "multi", tmp_path / "out"
    _make_run(main, "edge_sam_lora_r8", val, shift=0, seed=42)
    _make_run(ms, "edge_sam_lora_r8__s43", val, shift=3, seed=43)
    refs = [ca.locate_run(cfg, main, ms, "edge_sam", 8, "full", s) for s in (42, 43)]
    cells_gt, entries = ca.run_all(cfg_m, val, refs, "last", out, jobs=1, log=lambda s: None)
    assert [e["status"] for e in entries] == ["complete", "complete"]
    perfect, shifted = entries
    # class names are the manuscript order: mask value 2 is the spherocyte, and its Dice comes from the legacy 'echinocyte' key
    assert set(perfect["segmentation"]["dice_by_class"]) == set(MANUSCRIPT_NAMES[1:])
    assert perfect["segmentation"]["dice_by_class"]["spherocyte"] == 0.5
    assert perfect["segmentation"]["dice_by_class"]["echinocyte"] == 0.01
    assert perfect["segmentation"]["collapsed_classes"] == ["echinocyte"]
    comp = perfect["storage"]["per_day_pred_nowrap"][0]["composition_pct"]
    assert set(comp) == set(MANUSCRIPT_NAMES[1:])
    # perfect masks: exact dry mass agreement, every annotated cell matched
    ag = perfect["dry_mass"]["agreement_nowrap"]
    assert ag["mdare_pct"] == pytest.approx(0.0, abs=1e-9) and ag["pearson_r"] == pytest.approx(1.0)
    assert perfect["phase_integral"]["summary"]["matched_cells_pct"] == pytest.approx(100.0)
    assert perfect["phase_integral"]["summary"]["percell_abs_median_pct"] == pytest.approx(0.0, abs=1e-9)
    # one image has a cell at the 2 pi ceiling: flagged in prediction and annotation, excluded from the nowrap counts
    assert perfect["cells"]["wrapping"]["n_wrapped"] >= 1
    assert perfect["cells"]["predicted_nowrap"] == perfect["cells"]["predicted_all"] - perfect["cells"]["wrapping"]["n_wrapped"]
    assert perfect["background"]["median_abs_rad"] == pytest.approx(0.0, abs=0.01)
    # shifted masks are measurably worse
    assert shifted["dry_mass"]["agreement_nowrap"]["mdare_pct"] > 0
    assert shifted["phase_integral"]["summary"]["percell_abs_median_pct"] > 0
    # annotation block: offsets per file group, cells, wrapping
    ann = ca.annotation_block(cfg_m, cells_gt, val)
    assert ann["phase_offsets"]["by_group"]["date_named"]["n_images"] == 6
    assert ann["phase_offsets"]["by_group"]["date_named"]["median_rad"] == pytest.approx(OFFSET)
    assert ann["n_cells_nowrap"] == ann["n_cells_all"] - ann["wrapping"]["n_wrapped"]
    # flat table and mean +- SD over the two seeds
    row = ca.flatten(perfect)
    assert row["config"] == "edge_sam|lora_r8" and row["seed"] == 42 and row["mdare_pct"] == pytest.approx(0.0, abs=1e-9)
    agg = ca.aggregate(entries)["edge_sam|lora_r8"]
    assert agg["n_seeds"] == 2 and agg["seeds"] == [42, 43]
    m = agg["metrics"]["percell_abs_median_pct"]
    assert m["n"] == 2 and m["sd"] is not None and set(m["per_seed"]) == {"42", "43"}
    assert m["mean"] == pytest.approx(np.mean([m["per_seed"]["42"], m["per_seed"]["43"]]))
    # everything is strictly valid JSON
    doc = {"annotation": ann, "runs": {e["run_id"]: e for e in entries}, "by_configuration": ca.aggregate(entries)}
    write_json(tmp_path / "x.json", doc)
    json.loads((tmp_path / "x.json").read_text())
    assert (out / "runs" / "edge_sam_lora_r8" / "cells_pred.csv").exists()
    assert (out / "phase_integral" / "edge_sam_lora_r8__per_cell.csv").exists()


def test_missing_and_unfinished_runs_are_reported_not_hidden(cfg, cfg_m, tmp_path):
    val = _val()
    main, ms = tmp_path / "full", tmp_path / "multi"
    d = _make_run(main, "edge_sam_lora_r8", val)
    m = read_json(d / "metrics.json")
    m["status"] = "diverged"
    write_json(d / "metrics.json", m)
    refs = [ca.locate_run(cfg, main, ms, "edge_sam", 8, "full", 42), ca.locate_run(cfg, main, ms, "edge_sam", 8, "full", 43)]
    _, entries = ca.run_all(cfg_m, val, refs, "last", tmp_path / "o", log=lambda s: None)
    assert [e["status"] for e in entries] == ["diverged", "missing"]
    assert ca.aggregate(entries) == {}


def test_reference_check_agrees_with_the_selected_run_files_and_detects_a_stale_file(cfg, cfg_m, tmp_path):
    val = _val()
    main, ms = tmp_path / "full", tmp_path / "multi"
    _make_run(main, "edge_sam_lora_r8", val, shift=2)
    ref = ca.locate_run(cfg, main, ms, "edge_sam", 8, "full", 42)
    cells_gt, entries = ca.run_all(cfg_m, val, [ref], "last", tmp_path / "o", log=lambda s: None)
    # what scripts/run_analysis.py would have written for this run
    preds = {k: np.roll(v["gt"], 2, axis=1) for k, v in val.items()}
    cp = dm.cells_table(cfg_m, val, preds)
    nw_p, nw_g = cp[~cp.wrapped], cells_gt[~cells_gt.wrapped]
    write_json(main / "analysis" / "drymass.json", {"selected_run": "edge_sam_lora_r8", "agreement_nowrap": dm.agreement(nw_p, nw_g),
                                                    "agreement": dm.agreement(cp, cells_gt)})
    checks = ca.reference_checks(main, entries, 42, "edge_sam", 8)
    assert len(checks) > 5 and all(c["ok"] for c in checks)
    stale = read_json(main / "analysis" / "drymass.json")
    stale["agreement_nowrap"]["mdare_pct"] += 0.5
    write_json(main / "analysis" / "drymass.json", stale)
    bad = [c for c in ca.reference_checks(main, entries, 42, "edge_sam", 8) if not c["ok"]]
    assert [c["key"] for c in bad] == ["agreement_nowrap.mdare_pct"]


def test_freshness_detects_changed_predictions_and_new_runs(cfg, cfg_m, tmp_path):
    val = _val()
    main, ms = tmp_path / "full", tmp_path / "multi"
    d = _make_run(main, "edge_sam_lora_r8", val)
    ref = ca.locate_run(cfg, main, ms, "edge_sam", 8, "full", 42)
    _, entries = ca.run_all(cfg_m, val, [ref], "last", tmp_path / "o", log=lambda s: None)
    doc = {"meta": {"schema_version": ca.SCHEMA_VERSION, "checkpoint": "last",
                    "scope": {"archs": ["edge_sam"], "ranks": [8], "seeds": [42, 43], "losses": ["full"], "include_full_finetune": False}},
           "runs": {e["run_id"]: e for e in entries}}
    assert ca.freshness_problems(doc, main, ms, cfg) == []
    _make_run(ms, "edge_sam_lora_r8__s43", val, seed=43)                      # a run that finished after the analysis
    assert any("finished but not in the analysis" in p for p in ca.freshness_problems(doc, main, ms, cfg))
    np.savez(d / "predictions_last.npz", **{k: v["gt"].astype(np.uint8) for k, v in val.items()})   # predictions rewritten
    os.utime(d / "predictions_last.npz", (1, 1))
    assert any("changed after the analysis" in p for p in ca.freshness_problems(doc, main, ms, cfg))


def test_cli_end_to_end_in_a_copy_of_the_project(tmp_path):
    """scripts/run_cell_analysis.py on a tiny synthetic project: files, JSON content, exit status."""
    import csv
    import shutil
    import subprocess

    import tifffile

    proj = tmp_path / "proj"
    def skip(dirpath, names):                      # top-level data/outputs/to_delete only (qpi/data is code)
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
    cmd = [sys.executable, "scripts/run_cell_analysis.py", "--archs", "edge_sam", "--ranks", "8", "--seeds", "42", "43"]
    r = subprocess.run(cmd, cwd=proj, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = proj / "outputs" / "cell_analysis"
    doc = read_json(out / "cell_analysis.json")
    assert doc["meta"]["complete"] is True and doc["meta"]["n_runs_analysed"] == 2
    assert doc["meta"]["class_names_by_mask_value"] == MANUSCRIPT_NAMES
    assert doc["meta"]["n_validation_images"] == 6 and doc["meta"]["manifest_sha256"]
    assert set(doc["runs"]) == {"edge_sam_lora_r8", "edge_sam_lora_r8__s43"}
    assert doc["by_configuration"]["edge_sam|lora_r8"]["n_seeds"] == 2
    for f in ("cell_analysis_runs.csv", "cell_analysis_by_configuration.csv", "README.txt", "annotation/cells_gt.csv"):
        assert (out / f).exists(), f
    assert len(pd.read_csv(out / "cell_analysis_runs.csv")) == 2
    # a requested run that does not exist is a failure, and is listed
    r2 = subprocess.run(cmd[:-3] + ["--seeds", "42", "44"], cwd=proj, capture_output=True, text=True)
    assert r2.returncode == 1 and "edge_sam_lora_r8__s44" in r2.stderr
    assert read_json(out / "cell_analysis.json")["meta"]["complete"] is False
    # the stored file is current, then stale after the predictions are rewritten
    cfg = load_config(None, root=proj)
    d = proj / "outputs" / "cell_analysis" / "cell_analysis.json"
    r3 = subprocess.run(cmd, cwd=proj, capture_output=True, text=True)
    assert r3.returncode == 0
    doc = read_json(d)
    assert ca.freshness_problems(doc, proj / "outputs" / "full", proj / "outputs" / "multiseed", cfg) == []
    os.utime(proj / "outputs" / "full" / "runs" / "edge_sam_lora_r8" / "predictions_last.npz", (1, 1))
    assert ca.freshness_problems(doc, proj / "outputs" / "full", proj / "outputs" / "multiseed", cfg)


def test_reference_sensitivity_separates_the_three_references(cfg_m):
    val = _val()
    yy, xx = np.mgrid[:SIZE, :SIZE]
    big = {}
    for stem, d in val.items():                                           # predictions with larger cells (radius 25 instead of 22)
        pm = np.zeros((SIZE, SIZE), np.uint8)
        for cy, cx, cls in CENTRES:
            pm[(yy - cy) ** 2 + (xx - cx) ** 2 <= 25 ** 2] = cls
        big[stem] = pm
    s = ca.reference_sensitivity(cfg_m, val, big)
    ipe = s["ipe_global_pct"]
    assert s["n_images"] == 6 and all(v is not None and v > 0 for v in ipe.values())
    # the map carries a 0.5 rad offset: without a reference the extra predicted area adds offset x area to S_pred
    assert ipe["as_provided"] > ipe["annotation_background"]
    # perfect masks: every reference gives exactly zero error and the two background levels coincide
    perfect = ca.reference_sensitivity(cfg_m, val, {k: d["gt"].astype(np.uint8) for k, d in val.items()})
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in perfect["ipe_global_pct"].values())
    assert perfect["background_difference_prediction_minus_annotation_rad"]["max_abs"] == pytest.approx(0.0, abs=1e-12)


def test_annotation_reference_equals_the_processed_phase_ipe(cfg, cfg_m, tmp_path):
    val = _val()
    main, ms = tmp_path / "full", tmp_path / "multi"
    _make_run(main, "edge_sam_lora_r8", val, shift=3)
    _, (e,) = ca.run_all(cfg_m, val, [ca.locate_run(cfg, main, ms, "edge_sam", 8, "full", 42)], "last", tmp_path / "o", log=lambda s: None)
    # the processed map has its background at ~0, so referencing the provided map to the annotated background reproduces it
    assert e["reference_sensitivity"]["ipe_global_pct"]["annotation_background"] == pytest.approx(
        e["phase_integral"]["summary"]["ipe_global_raw_mean_pct"], rel=0.05, abs=0.05)


def test_verification_errors(cfg, cfg_m, tmp_path):
    val = _val()
    main, ms = tmp_path / "full", tmp_path / "multi"
    _make_run(main, "edge_sam_lora_r8", val)
    ref = ca.locate_run(cfg, main, ms, "edge_sam", 8, "full", 42)
    _, entries = ca.run_all(cfg_m, val, [ref], "last", tmp_path / "o", log=lambda s: None)
    doc = {"meta": {"schema_version": ca.SCHEMA_VERSION, "checkpoint": "last", "complete": True, "problems": [], "n_runs_analysed": 1,
                    "class_names_by_mask_value": MANUSCRIPT_NAMES, "reference_check": {"n_differences": 0},
                    "scope": {"archs": ["edge_sam"], "ranks": [8], "seeds": [42], "losses": ["full"], "include_full_finetune": False}},
           "runs": {e["run_id"]: e for e in entries}}
    tables = {"cell_analysis_runs": [ca.flatten(entries[0])]}
    assert ca.verification_errors(doc, cfg, main, ms, tables) == []
    assert any("holds 0" in e for e in ca.verification_errors(doc, cfg, main, ms, {"cell_analysis_runs": []}))
    bad = json.loads(json.dumps(doc))
    bad["meta"].update(complete=False, problems=[{"run_id": "x"}], class_names_by_mask_value=LEGACY_NAMES,
                       reference_check={"n_differences": 3})
    msgs = " | ".join(ca.verification_errors(bad, cfg, main, ms, tables))
    assert "incomplete" in msgs and "differs from the earlier analysis" in msgs and "manuscript class names" in msgs


def test_check_freshness_script(tmp_path):
    import shutil
    import subprocess

    def skip(dirpath, names):
        drop = {n for n in names if n in ("__pycache__", ".pytest_cache")}
        if Path(dirpath) == ROOT:
            drop |= {n for n in names if n in ("to_delete", "outputs", "data")}
        return drop

    proj = tmp_path / "proj"
    shutil.copytree(ROOT, proj, ignore=skip)
    run = proj / "outputs" / "full" / "runs" / "edge_sam_lora_r8"
    (run / "checkpoints").mkdir(parents=True)
    ana = proj / "outputs" / "full" / "analysis"
    ana.mkdir(parents=True)
    files = {run / "predictions_last.npz": 1000, run / "metrics.json": 1000, run / "checkpoints" / "last.pt": 1000,
             ana / "phase_integral.json": 1100,          # newer than its inputs
             ana / "drymass.json": 900}                  # older than the predictions it was made from
    for f, t in files.items():
        f.write_bytes(b"x")
        os.utime(f, (t, t))
    cmd = [sys.executable, "scripts/check_freshness.py"]
    r = subprocess.run(cmd, cwd=proj, capture_output=True, text=True)
    lines = {ln.split()[1]: ln.split()[0] for ln in r.stdout.splitlines() if ln[:5] in ("ok   ", "STALE", "MISSI")}
    assert r.returncode == 1, r.stdout + r.stderr
    assert lines["outputs/full/analysis/phase_integral.json"] == "ok"
    assert lines["outputs/full/analysis/drymass.json"] == "STALE"
    assert lines["outputs/cell_analysis/cell_analysis.json"] == "MISSING"
    assert "regenerate, in this order" in r.stdout and "./ops/run_cell_analysis.sh" in r.stdout
    assert subprocess.run(cmd + ["--warn-only"], cwd=proj, capture_output=True, text=True).returncode == 0


def test_results_collector_swaps_legacy_files_but_not_the_cell_analysis_file(tmp_path):
    sys.path.insert(0, str(ROOT / "manuscript"))                    # results.py imports its sibling modules as a script would
    spec = importlib.util.spec_from_file_location("ms_results", ROOT / "manuscript" / "results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    f = tmp_path / "x.json"
    write_json(f, {"dice_echinocyte": 1, "name": "spherocyte"})
    col = mod.Collector()
    assert col.load(f) == {"dice_spherocyte": 1, "name": "echinocyte"}              # legacy file: names exchanged once (D-25)
    assert col.load(f, swap=False) == {"dice_echinocyte": 1, "name": "spherocyte"}  # file that already has manuscript names


def test_parallel_analysis_gives_the_same_result_as_sequential(cfg, cfg_m, tmp_path):
    val = _val()
    main, ms = tmp_path / "full", tmp_path / "multi"
    _make_run(main, "edge_sam_lora_r8", val, shift=0, seed=42)
    _make_run(ms, "edge_sam_lora_r8__s43", val, shift=2, seed=43)
    _make_run(ms, "edge_sam_lora_r8__s44", val, shift=4, seed=44)
    refs = [ca.locate_run(cfg, main, ms, "edge_sam", 8, "full", s) for s in (42, 43, 44)]
    _, seq = ca.run_all(cfg_m, val, refs, "last", tmp_path / "a", jobs=1, log=lambda s: None)
    _, par = ca.run_all(cfg_m, val, refs, "last", tmp_path / "b", jobs=2, log=lambda s: None)
    assert [e["run_id"] for e in par] == [e["run_id"] for e in seq]
    from qpi.utils.io import sanitize
    # NaN != NaN, so compare the strictly valid JSON form (NaN -> null)
    assert sanitize([ca.flatten(e) for e in par]) == sanitize([ca.flatten(e) for e in seq])
