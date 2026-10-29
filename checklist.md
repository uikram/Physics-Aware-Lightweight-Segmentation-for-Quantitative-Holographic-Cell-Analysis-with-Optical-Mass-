# Checklist (updated 27 Oct 2026)

`[x]` = verified by execution (evidence given). `[ ]` = not done or not verified (reason given). Nothing is marked done on the strength of reading code alone.

## Pipeline and data
- [x] Masks (labels exactly 0 to 4) pair one to one with the 8-bit images; rotation convention proven (`rotation<stem>_<k>` equals `np.rot90(base, -k)`, `scripts/verify_dataset.py`); split leakage test (stem overlap and rotation-invariant content hash).
- [x] Float maps registered to the mask frame (ECC, scale + translation only); two borderline images handled by documented safeguards (D-16). `scripts/check_raw_scale.py` rejects maps confined to [0, pi] (the first delivery had half scale, D-20).
- [x] 182 of 256 base training images have float maps (728 views); the other 74 will not be supplied and are not used (D-05). All 66 validation images have float maps.
- [x] Mask and phase stay aligned after resize and evaluation upsampling (`nearest-exact`; regression test); all three models return logits at the input size; orientation checked through the real train and evaluation path.
- [x] Parameter counts at r = 8: EdgeSAM 534,205 trainable of 6,004,333 (8.90 %), MobileSAM 0.67 M, MobileNet-UNet 5.58 M.
- [x] Loss and training settings (class weights, lambdas 0.1 / 0.05 / 0.1, PMC margin 0.1 rad, AdamW + cosine, 50 epochs, per-architecture learning rate, batch size and input size) match `configs/experiments.yaml`. Physics terms act on the background-referenced radians; unannotated pixels are excluded.
- [ ] Phase zero level differs between the date-named and `X_*` file groups; handled by D-03 (per-image background referencing). Whether the offset is signal or an arbitrary baseline is not confirmed by the data provider (DECISIONS.md, Q-2).

## Experiments
- [x] Main experiment (27 trainings, 30 table entries, seed 42) finished 24 Oct on the corrected maps.
- [x] Multi-seed extension finished: 180 LoRA runs (3 architectures x 5 ranks x 4 losses x seeds 42, 43, 44) indexed with SHA-256 of every `metrics.json`; main grid unchanged by the extension (tests). Full fine-tuning is a single seed-42 run per architecture (D-24).
- [x] Wrapped-cell rule tested on synthetic cells and applied to the real masks (3.8 % of predicted, 3.0 % of annotated cells flagged, D-21).
- [ ] A stray partial run folder `edge_sam_full_finetune__s43` may exist outside the 156-run grid; check and remove.

## Benchmark (protocol v5, D-26)
- [x] Six combinations (three models x PyTorch / ONNX Runtime FP32), three seeds, reported run on physical GPU 2: `scripts/verify_outputs.py` errors = 0, warnings = 0.
- [x] ONNX FP32 against PyTorch FP32 on the 66 validation images: accepted for all three models by the fixed gate (agreement >= 99.9 %, |change in mean Dice| <= 0.001, dry-mass error mean <= 0.1 %, p99 <= 0.5 %).
- [x] Tests: 70 pass on the server before series C; 85 pass on a complete copy of the project after it (15 new in `tests/test_cell_analysis.py`).
- [ ] Server: copy `manuscript/make_tex.py` and rerun `./ops/make_manuscript_outputs.sh` (single maximum Peak VRAM display).
- [ ] Causes untested: the two speed levels of EdgeSAM, MobileSAM slower in ONNX Runtime, larger ONNX memory. Stale config keys `benchmark.precisions` and `benchmark.onnx_skip` are not read by the code.

## Per-cell analysis of all runs (series C)
- [x] `scripts/run_cell_analysis.py`, `ops/run_cell_analysis.sh`, `scripts/check_freshness.py` written; 15 tests in `tests/test_cell_analysis.py` pass on synthetic data (no real run was available when they were written).
- [ ] Run `./ops/run_cell_analysis.sh` on the server and read the reference check: the seed-42 numbers recomputed from the saved predictions must equal `analysis/drymass.json`, `phase_integral.json` and `architecture_comparison.json`.
- [ ] `python scripts/check_freshness.py` reports every result file ok.

## Manuscript outputs
- [x] `manuscript/results.py` writes one `results.json` (Tables 1 to 7, S-1, analyses, run flags, dataset report, configuration, environment, SHA-256 of every input); aborts if an input is missing, a run is unfinished or modes are mixed. `manuscript/figures.py` writes the figures from actual outputs. No number is typed by hand.
- [x] Class-name exchange (D-25) applied where results are read; `results.py` checks it against the raw `metrics.json` of every selected run.
- [ ] Manuscript, supplement and response letter are not edited yet; the fix list is in the project note `revised-ms-line-audit-2026-10-26.md`.

## Cleanup
- [ ] Move the old helper files and notes into `to_delete/` on the PC and the server (listed in the project note `benchmark-v5-fp32-reduction-2026-10-26.md`), then delete `to_delete/` at the end.
