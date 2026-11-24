#!/usr/bin/env python
"""Pre-flight checks (CPU, trains nothing). Run once on the server before the queue.
    python scripts/utilities/check_setup.py               # everything (data checks need data/prepared)
    python scripts/utilities/check_setup.py --no-data     # configuration, matrix and models only

1. physics constants and class mapping (qpicell/physics.py)
2. experiment matrix: every architecture x condition (r8, r16, r32, r64, fullft) x loss x seed present exactly once,
   queue order seed-major, every cIPP arm with eta = 0, global IPP and cIPP never combined
3. models: for every architecture and condition, the parameter report (total / trainable / adapter parameters,
   adapted layers); asserts that the adapter count grows with the rank on the SAME adapted layers, that the rank
   changes nothing else, and that full fine-tuning trains every parameter and has no adapter
4. data (unless --no-data): every prepared phase map float32 radians and unchanged since preparation (checksum), masks
   uint8 with labels 0-4 / 255, split sizes 141 / 41 / 66 and no overlap
Writes logs/setup_check.json; exits 1 on any failure."""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qpicell.cli import add_common, config  # noqa: E402
from qpicell.io import write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-data", action="store_true")
    add_common(ap)
    a = ap.parse_args(); cfg = config(a)
    out, ok = {"created": time.strftime("%Y-%m-%d %H:%M:%S")}, True

    def step(name, fn):
        nonlocal ok
        print(f"\n=== {name} ===", flush=True)
        try:
            out[name] = fn(); print(f"[ok] {name}", flush=True)
        except Exception as e:  # noqa: BLE001
            ok = False; out[name] = {"error": repr(e)}; print(f"[FAIL] {name}: {e!r}", flush=True)

    from qpicell.physics import assert_physics, dry_mass_factor_pg, optical_volume_factor_um3
    step("physics", lambda: (assert_physics(cfg), {"physics": cfg["physics"], "class_names": cfg["data"]["class_names"],
                                                   "dry_mass_pg_per_rad_px": dry_mass_factor_pg(cfg),
                                                   "optical_volume_um3_per_rad_px": optical_volume_factor_um3(cfg)})[1])

    def matrix():
        from qpicell.training.runs import conditions, grid
        runs = grid(cfg); ids = [r.run_id for r in runs]
        assert len(ids) == len(set(ids)), "duplicate run ids"
        A, C, L, S = list(cfg["models"]), conditions(cfg), list(cfg["arms"]), list(cfg["training"]["seeds"])
        assert C == ["r8", "r16", "r32", "r64", "fullft"], C
        assert len(L) == 7 and S == [42, 43, 44] and len(A) == 3
        expected = {(x, c, l, s) for x in A for c in C for l in L for s in S}
        got = {(r.arch, r.condition, r.arm, r.seed) for r in runs}
        assert got == expected, f"missing {expected - got}"
        seeds = [r.seed for r in runs]; assert seeds == sorted(seeds), "queue order is not seed-major"
        assert float(cfg["loss"]["cipp"]["eta"]) == 0.0, "eta must be 0"
        for k, v in cfg["arms"].items():
            assert not (v["ipp"] and v["cipp"]), k
        n_lora = sum(r.condition != "fullft" for r in runs)
        print(f"  {len(runs)} runs = {len(A)} architectures x {len(C)} conditions x {len(L)} losses x {len(S)} seeds "
              f"({n_lora} LoRA + {len(runs) - n_lora} full fine-tuning); first {ids[0]}, last {ids[-1]}")
        return {"n_runs": len(runs), "n_lora": n_lora, "n_full_finetune": len(runs) - n_lora, "architectures": A, "conditions": C,
                "losses": L, "seeds": S, "first_runs": ids[:10]}
    step("experiment_matrix", matrix)

    def models():
        from qpicell.models import build
        from qpicell.models.lora import parameter_report
        res = {}
        for arch in cfg["models"]:
            rows = {}
            for r in cfg["conditions"]["lora_ranks"]:
                rows[f"r{r}"] = parameter_report(build(cfg, arch, int(r), pretrained=False), int(r))
            rows["fullft"] = parameter_report(build(cfg, arch, None, pretrained=False), None)
            names = {k: tuple(v["lora_layer_names"]) for k, v in rows.items() if k != "fullft"}
            assert len(set(names.values())) == 1, f"{arch}: the adapted layers differ between ranks"
            ranks = [int(k[1:]) for k in names]; lp = [rows[f"r{k}"]["lora_parameters"] for k in ranks]
            assert all(x < y for x, y in zip(lp, lp[1:])), f"{arch}: adapter parameters do not grow with the rank: {lp}"
            per_rank = lp[0] / ranks[0]
            assert all(abs(p / k - per_rank) < 1e-6 for p, k in zip(lp, ranks)), f"{arch}: adapter parameters not linear in the rank"
            other = {rows[f"r{k}"]["trainable_non_lora"] for k in ranks}
            assert len(other) == 1, f"{arch}: the rank changes non-adapter trainable parameters"
            ft = rows["fullft"]; assert ft["trainable"] == ft["total"] and ft["lora_parameters"] == 0 and ft["lora_layers"] == 0
            print(f"  {arch}: {rows['r8']['lora_layers']} adapted layers; adapter parameters per unit rank {per_rank:,.0f}; "
                  f"other trainable {other.pop():,}")
            for k, v in rows.items():
                print(f"    {k:7s} total {v['total']:>12,}  trainable {v['trainable']:>12,} ({v['trainable_pct']:6.2f} %)  adapters {v['lora_parameters']:>10,}")
                v.pop("lora_layer_names") if k != "r8" else None
            res[arch] = rows
        return res
    step("models", models)

    if not a.no_data:
        def data():
            from qpicell.physics import split_stems, verify_prepared
            v = verify_prepared(cfg); sp = split_stems(cfg)
            exp = cfg["split"]["expected"]
            for k in ("train", "validation", "test"):
                assert len(sp[k]) == exp[k], f"{k}: {len(sp[k])} images, expected {exp[k]}"
            import numpy as np
            import tifffile
            from qpicell.physics import assert_mask
            P = Path(cfg["paths_abs"]["prepared"])
            for split, stems in (("train", sp["train"] + sp["validation"]), ("val", sp["test"])):
                for s in stems:
                    assert_mask(tifffile.imread(P / split / "mask" / f"{s}.tif"), cfg["data"]["num_classes"], cfg["data"]["ignore_index"], s)
            print(f"  {v['images']} prepared phase maps float32 and unchanged; split {[(k, len(x)) for k, x in sp.items()]}, disjoint")
            return {"prepared": v, "split_sizes": {k: len(x) for k, x in sp.items()}, "split": sp}
        step("data", data)
    write_json(Path(cfg["paths_abs"]["logs"]) / "setup_check.json", out)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} -> {Path(cfg['paths_abs']['logs']) / 'setup_check.json'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
