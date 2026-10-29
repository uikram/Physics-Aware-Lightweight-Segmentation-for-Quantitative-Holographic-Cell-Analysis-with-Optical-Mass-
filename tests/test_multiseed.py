"""Multi-seed extension: grid, sampler, aggregation and prerequisites (no GPU, no real data)."""
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qpi.analysis import multiseed as ms  # noqa: E402
from qpi.config import load_config  # noqa: E402
from qpi.data.dataset import QPIDataset, build_loaders  # noqa: E402
from qpi.experiments import build_grid, run_config, trainable_specs, up_to_stage  # noqa: E402


# ------------------------------------------------------------------------------------------------ grid
def test_multiseed_grid_counts_ids_stages_and_seeds():
    cfg = load_config("multiseed.yaml")
    g = build_grid(cfg)
    assert len(g) == 156 and len(trainable_specs(g)) == 156
    blocks = {}
    for s in g:
        blocks[s.block] = blocks.get(s.block, 0) + 1
    assert blocks == {f"{a}_{k}": n for a in ("edge_sam", "mobile_sam", "mobilenet_unet")
                      for k, n in (("main_grid_seeds", 16), ("ablation_other_ranks", 36))}
    assert [len(up_to_stage(g, n)) for n in (1, 2, 3)] == [52, 104, 156] and len(up_to_stage(g, None)) == 156
    assert {s.arch for s in up_to_stage(g, 1)} == {"edge_sam"} and {s.arch for s in g if s.stage == 2} == {"mobile_sam"}
    ids = [s.run_id for s in g]
    assert len(ids) == len(set(ids))
    assert all(s.run_id.endswith(f"__s{s.seed}") for s in g)
    # every configuration ends up with exactly seeds 42, 43, 44: main-grid configurations get 43, 44 (42 = outputs/full),
    # the ablations at ranks 2, 4, 16, 32 are not in the main grid and get 42, 43, 44
    by_base = {}
    for s in g:
        by_base.setdefault(s.base_id, set()).add(s.seed)
    main_ids = set(ms.main_reference_ids(cfg, g))
    for base, seeds in by_base.items():
        assert seeds == ({43, 44} if base in main_ids else {42, 43, 44}), base
    assert {s.group for s in g if s.seed == 42} == {"ablation_rank"} and all(s.sampler is None for s in g)
    assert all(s.alias_of is None for s in g)
    r = next(s for s in g if s.run_id == "mobile_sam_abl_pmc_bga_r16__s43")
    assert (r.lora_r, r.loss_type, r.base_id, r.stage, r.arch) == (16, "physics_aware", "mobile_sam_abl_pmc_bga_r16", 2, "mobile_sam")
    assert r.lambda_pmc > 0 and r.lambda_bga > 0 and r.lambda_pv == 0
    d = next(s for s in g if s.run_id == "mobilenet_unet_abl_dice_only_r4__s42")
    assert d.loss_type == "dice_only" and d.lora_r == 4 and d.stage == 3
    rc = run_config(cfg, d)
    assert rc["seed"] == 42 and rc["lora"]["r"] == 4 and rc["lora"]["alpha"] == 4.0 and rc["loss"]["type"] == "dice_only"
    rc = run_config(cfg, next(s for s in g if s.run_id == "edge_sam_lora_r8__s44"))
    assert rc["seed"] == 44 and rc["training"]["sampler"] == "shuffle" and rc["lora"]["r"] == 8


def test_main_reference_runs_per_stage():
    cfg = load_config("multiseed.yaml")
    g = build_grid(cfg)
    st1 = set(ms.main_reference_ids(cfg, up_to_stage(g, 1)))
    # EdgeSAM: the whole rank sweep and the rank-8 ablation runs (no full fine-tuning); nothing of the other architectures
    assert {f"edge_sam_lora_r{r}" for r in (2, 4, 8, 16, 32)} <= st1
    assert {"edge_sam_abl_dice_only", "edge_sam_abl_pmc", "edge_sam_abl_pmc_bga"} <= st1
    assert not any("full_finetune" in i for i in st1)
    assert not any(i.startswith(("mobile_sam", "mobilenet_unet")) for i in st1)
    st3 = set(ms.main_reference_ids(cfg, up_to_stage(g, 3)))
    assert len(st3) == 24 and "mobile_sam_lora_r4" in st3 and not any("full_finetune" in i for i in st3)


def test_main_grid_is_unchanged_by_the_extension():
    g = build_grid(load_config())
    assert len(g) == 30 and all(s.seed is None and s.sampler is None for s in g)
    assert run_config(load_config(), g[0])["seed"] == 42


def test_multiseed_grid_rejects_duplicates_aliases_unknown_runs_and_bad_ablation_blocks():
    cfg = load_config("multiseed.yaml")
    dup = json.loads(json.dumps(cfg["multiseed"]))
    dup["blocks"].append({"name": "dup", "seeds": [43], "runs": ["edge_sam_lora_r8"]})
    with pytest.raises(ValueError, match="occurs in blocks"):
        build_grid({**cfg, "multiseed": dup})
    for bad, msg in (("edge_sam_abl_full", "alias"), ("edge_sam_lora_r7", "not a run")):
        b = json.loads(json.dumps(cfg["multiseed"]))
        b["blocks"] = [{"name": "x", "seeds": [43], "runs": [bad]}]
        with pytest.raises(ValueError, match=msg):
            build_grid({**cfg, "multiseed": b})
    for ab, msg in (({"archs": ["edge_sam"], "ranks": [8], "losses": ["pmc"]}, "selected rank"),
                    ({"archs": ["edge_sam"], "ranks": [3], "losses": ["pmc"]}, "not in lora.ranks"),
                    ({"archs": ["edge_sam"], "ranks": [2], "losses": ["full"]}, "ablation loss"),
                    ({"archs": ["nope"], "ranks": [2], "losses": ["pmc"]}, "unknown architecture")):
        b = json.loads(json.dumps(cfg["multiseed"]))
        b["blocks"] = [{"name": "x", "seeds": [42], "ablation_by_rank": ab}]
        with pytest.raises(ValueError, match=msg):
            build_grid({**cfg, "multiseed": b})
    b = json.loads(json.dumps(cfg["multiseed"]))
    b["blocks"] = [{"name": "x", "seeds": [42], "runs": ["edge_sam_lora_r8"], "ablation_by_rank": {"archs": ["edge_sam"], "ranks": [2], "losses": ["pmc"]}}]
    with pytest.raises(ValueError, match="exactly one"):
        build_grid({**cfg, "multiseed": b})


def test_queue_order_is_stage_then_priority_then_cost():
    cfg = load_config("multiseed.yaml")
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_queue
    runs = sorted(trainable_specs(up_to_stage(build_grid(cfg), 2)),
                  key=lambda s: (s.stage, s.priority, -run_queue.COST.get(s.arch, 0), s.run_id))
    assert [s.stage for s in runs] == sorted(s.stage for s in runs)
    assert runs[0].block == "edge_sam_main_grid_seeds" and runs[16].block == "edge_sam_ablation_other_ranks"
    assert runs[52].arch == "mobile_sam" and runs[52].block == "mobile_sam_main_grid_seeds"


# ------------------------------------------------------------------------------------------------ sampler
def _processed_with_dominant_classes(tmp: Path, labels, n=32):
    for sp in ("train", "val"):
        (tmp / sp / "phase").mkdir(parents=True)
        (tmp / sp / "mask").mkdir(parents=True)
    rows = []
    rng = np.random.default_rng(0)
    for i, c in enumerate(labels):
        mk = np.zeros((n, n), np.uint8)
        mk[:10, :10] = c                      # dominant foreground class
        mk[20:22, 20:22] = 1 if c != 1 else 2   # a few pixels of another class
        mk[-1, :] = 255                       # ignored strip
        for sp in ("train", "val"):
            tifffile.imwrite(tmp / sp / "phase" / f"s{i}.tif", rng.normal(size=(n, n)).astype(np.float32))
            tifffile.imwrite(tmp / sp / "mask" / f"s{i}.tif", mk)
            rows.append({"split": sp, "stem": f"s{i}", "registration_ok": "True", "storage_day": 0})
    with open(tmp / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "stem", "registration_ok", "storage_day"])
        w.writeheader()
        w.writerows(rows)


def test_dominant_class_weights_equalise_label_shares(tmp_path):
    labels = [1] * 12 + [2] * 3 + [3] * 1
    _processed_with_dominant_classes(tmp_path, labels)
    ds = QPIDataset(tmp_path, "train", 32, [0, 1, 2, 3])
    assert ds.dominant_classes()[:4] == [1, 1, 1, 1] and len(ds.dominant_classes()) == 64
    w, info = ds.dominant_class_weights()
    assert info["items_per_label"] == {"0": 0, "1": 48, "2": 12, "3": 4, "4": 0}
    shares = info["expected_sampling_share_per_label"]
    assert shares["1"] == pytest.approx(1 / 3) and shares["2"] == pytest.approx(1 / 3) and shares["3"] == pytest.approx(1 / 3)
    assert float(w.sum()) == pytest.approx(3.0)


def test_weighted_loader_oversamples_rare_dominant_class(tmp_path):
    labels = [1] * 28 + [2] * 4
    _processed_with_dominant_classes(tmp_path, labels)
    cfg = load_config()
    cfg["paths_abs"]["processed_root"] = tmp_path
    cfg["data"]["num_workers"] = 0
    rc = {"image_size": 32, "batch_size": 8, "training": {"sampler": "weighted_dominant_class"}}
    tl, _ = build_loaders(cfg, rc, 7)
    assert tl.sampler_info["sampler"] == "weighted_dominant_class"
    stems = []
    for _ in range(5):                       # 5 epochs of sampling with replacement
        for b in tl:
            stems += [int(s.replace("rotation", "").lstrip("s").split("_")[0]) for s in b["sample_id"]]
    rare = np.mean([s >= 28 for s in stems])
    assert 0.35 < rare < 0.65, rare          # plain shuffling would give 4/32 = 0.125
    rc["training"]["sampler"] = "shuffle"
    tl0, _ = build_loaders(cfg, rc, 7)
    assert tl0.sampler_info == {"sampler": "shuffle"}
    rc["training"]["sampler"] = "bogus"
    with pytest.raises(ValueError):
        build_loaders(cfg, rc, 7)


# ------------------------------------------------------------------------------------------------ aggregation
def _fake_metrics(seed, dice, aji=0.3, bf1=0.7, ipe=0.03, collapsed=(), status="complete"):
    ev = {"mean_dice": dice, "mean_iou": dice * 0.8, "aji": aji, "bf1": bf1, "ipe_global": ipe}
    for c in ms.CLASSES:
        ev[f"dice_{c}"] = 0.0 if c in collapsed else dice
        ev[f"ipe_{c}"] = 1.0 if c in collapsed else ipe * 2
    m = {"status": status, "seed": seed, "parameters": {"trainable": 1000, "total": 5000},
         "training": {"epochs_completed": 50, "train_seconds": 100.0}}
    if status == "complete":
        m["evaluation"] = {"last": ev}
    return m


def _populate(root: Path, cfg, rank_dice=None, drop=None, collapse_seed=None, max_stage=None):
    rank_dice = rank_dice or {2: 0.70, 4: 0.72, 8: 0.73, 16: 0.74, 32: 0.745}
    main, out = root / "outputs/full/runs", root / "outputs/multiseed/runs"
    specs = up_to_stage(build_grid(cfg), max_stage)
    for s in specs:
        if s.run_id == drop:
            continue
        base = rank_dice.get(s.lora_r, 0.7) if s.lora_r else 0.65
        d = base + 0.01 * (s.seed - 44)
        col = ("echinocyte",) if collapse_seed == s.seed and s.base_id == "edge_sam_abl_pmc" else ()
        p = out / s.run_id
        p.mkdir(parents=True, exist_ok=True)
        (p / "metrics.json").write_text(json.dumps(_fake_metrics(s.seed, d, collapsed=col)))
    for b in ms.main_reference_ids(cfg, specs):
        p = main / b
        p.mkdir(parents=True, exist_ok=True)
        lr = int(b.rsplit("_r", 1)[1]) if "_lora_r" in b else None
        base = rank_dice.get(lr, 0.7) if lr else 0.65
        (p / "metrics.json").write_text(json.dumps(_fake_metrics(42, base + 0.01 * (42 - 44))))


def test_aggregation_statistics_provenance_and_missing(tmp_path):
    cfg = load_config("multiseed.yaml")
    _populate(tmp_path, cfg, collapse_seed=43)
    assert ms.write_outputs(cfg, tmp_path) == 0
    summ = {s["config"]: s for s in json.loads((tmp_path / "outputs/multiseed/summary/multiseed_results.json").read_text())["config_summary"]}
    r8 = summ["edge_sam|lora_r8"]
    assert r8["n"] == 3 and r8["seeds"] == [42, 43, 44]
    assert r8["mean_dice_mean"] == pytest.approx(0.73 - 0.01) and r8["mean_dice_sd"] == pytest.approx(np.std([-.02, -.01, 0], ddof=1))
    assert all(summ[f"{a}|lora_r{r}"]["n"] == 3 for a in ("edge_sam", "mobile_sam", "mobilenet_unet") for r in (2, 4, 8, 16, 32))
    assert summ["edge_sam|abl_pmc"]["n_runs_with_a_collapsed_class"] == 1 and summ["edge_sam|abl_pmc"]["collapsed_seeds"] == [43]
    assert summ["edge_sam|abl_pmc_r32"]["n"] == 3 and summ["mobile_sam|abl_dice_only_r2"]["seeds"] == [42, 43, 44]
    assert summ["mobilenet_unet|abl_pmc_bga_r16"]["n"] == 3 and "edge_sam|full_finetune" not in summ
    idx = list(csv.DictReader(open(tmp_path / "outputs/multiseed/summary/runs_index.csv")))
    n_main = len(ms.main_reference_ids(cfg, build_grid(cfg)))
    assert len(idx) == 156 + n_main
    row = next(r for r in idx if r["run_id"] == "edge_sam_lora_r8__s44")
    assert row["metrics_path"] == "outputs/multiseed/runs/edge_sam_lora_r8__s44/metrics.json" and len(row["metrics_sha256"]) == 64
    main_row = next(r for r in idx if r["run_id"] == "edge_sam_lora_r8" and r["seed"] == "42")
    assert main_row["origin"].startswith("outputs/full")
    paired = list(csv.DictReader(open(tmp_path / "outputs/multiseed/summary/paired_differences.csv")))
    p = next(r for r in paired if r["a"] == "edge_sam|lora_r8" and r["b"] == "edge_sam|lora_r16" and r["metric"] == "mean_dice")
    assert float(p["mean_diff"]) == pytest.approx(0.01) and int(p["n_pairs"]) == 3 and p["ci95_lo"] and float(p["sd_diff"]) == pytest.approx(0.0, abs=1e-12)
    tables = (tmp_path / "outputs/multiseed/summary/tables.md").read_text()
    for h in ("MS-1", "MS-2", "MS-4", "MS-6"):
        assert h in tables
    assert "MS-3" not in tables and "MS-5" not in tables and "Scope: all stages" in tables
    assert (tmp_path / "manuscript/outputs_multiseed/multiseed_results.json").exists()


def test_stage_one_aggregates_only_its_runs(tmp_path):
    cfg = load_config("multiseed.yaml")
    _populate(tmp_path, cfg, max_stage=1)
    assert ms.write_outputs(cfg, tmp_path, max_stage=1) == 0
    assert ms.write_outputs(cfg, tmp_path) == 1          # stages 2 and 3 are not done: reported as missing
    ms.write_outputs(cfg, tmp_path, max_stage=1)
    res = json.loads((tmp_path / "outputs/multiseed/summary/multiseed_results.json").read_text())
    assert res["missing"] == []
    summ = {s["config"]: s for s in res["config_summary"]}
    assert not any(k.startswith(("mobilenet_unet", "mobile_sam")) for k in summ)
    assert summ["edge_sam|lora_r8"]["n"] == 3 and "mobile_sam|lora_r32" not in summ
    tables = (tmp_path / "outputs/multiseed/summary/tables.md").read_text()
    assert "Scope: stages 1 to 1" in tables and "MS-6" in tables


def test_aggregation_flags_missing_runs_and_partial_mode(tmp_path):
    cfg = load_config("multiseed.yaml")
    _populate(tmp_path, cfg, drop="mobile_sam_lora_r8__s44")
    assert ms.write_outputs(cfg, tmp_path) == 1
    assert ms.write_outputs(cfg, tmp_path, allow_partial=True) == 0
    res = json.loads((tmp_path / "outputs/multiseed/summary/multiseed_results.json").read_text())
    assert any("mobile_sam_lora_r8__s44" in m for m in res["missing"])
    assert {s["config"]: s for s in res["config_summary"]}["mobile_sam|lora_r8"]["n"] == 2


def test_rank_selection_rule_picks_smallest_indistinguishable_rank(tmp_path):
    cfg = load_config("multiseed.yaml")
    # ranks 8, 16, 32 identical up to seed noise -> indistinguishable; ranks 2 and 4 clearly worse
    _populate(tmp_path, cfg, rank_dice={2: 0.60, 4: 0.65, 8: 0.74, 16: 0.74, 32: 0.74})
    ms.write_outputs(cfg, tmp_path)
    rs = json.loads((tmp_path / "outputs/multiseed/summary/multiseed_results.json").read_text())["rank_selection"]
    assert set(rs) == {"edge_sam", "mobile_sam", "mobilenet_unet"}
    d = rs["edge_sam"]["per_metric"]["mean_dice"]
    assert set(d["not_distinguishable_from_best"]) == {8, 16, 32} and d["smallest_not_distinguishable"] == 8
    assert d["selected_rank_not_distinguishable"] is True


def test_prereq_check_detects_changed_data_or_config(tmp_path):
    cfg = load_config("multiseed.yaml")
    (tmp_path / "configs").mkdir()
    for n in ("base.yaml", "experiments.yaml"):
        shutil.copy(ROOT / "configs" / n, tmp_path / "configs" / n)
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/processed/manifest.csv").write_text("split,stem\ntrain,a\n")
    _populate(tmp_path, cfg)
    meta = {"data_manifest_sha256": ms.sha256_file(tmp_path / "data/processed/manifest.csv"),
            "config_file_sha256": {n: ms.sha256_file(tmp_path / "configs" / n) for n in ("base.yaml", "experiments.yaml")}}
    (tmp_path / "outputs/full/run_metadata.json").write_text(json.dumps(meta))
    assert ms.check_prereqs(cfg, tmp_path) == []
    (tmp_path / "data/processed/manifest.csv").write_text("split,stem\ntrain,b\n")
    assert any("manifest.csv differs" in p for p in ms.check_prereqs(cfg, tmp_path))
    meta["data_manifest_sha256"] = ms.sha256_file(tmp_path / "data/processed/manifest.csv")
    meta["config_file_sha256"]["base.yaml"] = "0" * 64
    (tmp_path / "outputs/full/run_metadata.json").write_text(json.dumps(meta))
    assert any("base.yaml changed" in p for p in ms.check_prereqs(cfg, tmp_path))
    shutil.rmtree(tmp_path / "outputs/full/runs/edge_sam_lora_r8")
    assert any("seed-42 run edge_sam_lora_r8 is missing" in p for p in ms.check_prereqs(cfg, tmp_path))


def test_stage_two_has_edge_sam_and_mobile_sam_but_not_mobilenet_unet(tmp_path):
    cfg = load_config("multiseed.yaml")
    _populate(tmp_path, cfg, max_stage=2)
    assert ms.write_outputs(cfg, tmp_path, max_stage=2) == 0
    res = json.loads((tmp_path / "outputs/multiseed/summary/multiseed_results.json").read_text())
    assert set(res["rank_selection"]) == {"edge_sam", "mobile_sam"}
    summ = {s["config"]: s for s in res["config_summary"]}
    assert summ["mobile_sam|lora_r8"]["n"] == 3 and summ["mobile_sam|abl_pmc_r4"]["n"] == 3
    assert not any(k.startswith("mobilenet_unet") for k in summ)
    tables = (tmp_path / "outputs/multiseed/summary/tables.md").read_text()
    assert "### MobileSAM" in tables and "### EdgeSAM" in tables and "### MobileNet-UNet" not in tables
