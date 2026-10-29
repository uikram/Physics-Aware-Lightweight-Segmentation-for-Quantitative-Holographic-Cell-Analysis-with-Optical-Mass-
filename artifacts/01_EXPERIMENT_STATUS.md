# 01. Experiment status: what is finished, verified, missing

Checked on 29 Oct 2026 against the repository as pushed (commit "Results"). Nothing was retrained or rerun. Every item marked
**verified** was checked by `artifacts/code/verify_artifacts.py` or by the loaders in `artifacts/code/common.py` (they stop with an
error if a file is missing, incomplete or inconsistent).

## Summary

**All experiments the revision needs are finished. No experiment has to be rerun.** The only missing inputs are the processed
phase maps (`data/processed/`, not in the repository), which are needed for the three image figures only (Step 6 of the plan).

## Experiments

| Series | What | Runs | Status | Evidence |
|---|---|---|---|---|
| F1 | Main experiment on the corrected float32 maps, seed 42 (rank sweep, rank-8 loss ablation, full fine-tuning) | 27 trainings | finished, **verified** | `outputs/full/runs/*/metrics.json`, status complete, 50 of 50 epochs |
| M1 | Multi-seed extension: every LoRA configuration (3 architectures x 5 ranks x 4 losses) with seeds 42, 43, 44 | 180 LoRA runs (156 new + 24 seed-42 mirrors) | finished, **verified** | `outputs/multiseed/summary/runs_index.csv`: 180 rows, 0 missing; the SHA-256 of all 180 `metrics.json` files still match |
| FT | Full fine-tuning | 3 (seed 42 only, D-24) | finished, **verified** | `outputs/full/runs/*_full_finetune/metrics.json` |
| C | Per-cell analysis (dry mass, matching, per-cell IPE, wrapping, composition, storage statistics, background-reference sensitivity) of all 183 runs | 183 | finished, **verified** | `outputs/analysis/cell/*.json` (183 + annotation), `outputs/compiled/cell_analysis_multiseed.json` complete, 0 problems |
| R | Robustness to synthetic background perturbations, all 183 runs | 183 | finished, **verified** | `outputs/compiled/robustness_multiseed.json` complete, 0 problems |
| B | Inference benchmark, protocol v5 (FP32 PyTorch and ONNX Runtime, 3 seeds) + ONNX equivalence gate | 6 combinations x 3 seeds | finished, **verified**; all three ONNX models accepted by the gate | `outputs/compiled/benchmark_multiseed.json`, `outputs/full/benchmark/equivalence.json` |

Configuration integrity: `configs/base.yaml` and `configs/experiments.yaml` have the SHA-256 recorded in
`outputs/full/run_metadata.json` and `outputs/multiseed/run_metadata.json`, so the runs were trained with the configuration in the
repository. Data manifest SHA-256 (both series): `ab31b31b...e1`.

Unit tests: `python -m pytest -q tests` gives **98 passed, 2 skipped** (CPU, PyTorch 2.15; the vendored-model tests that need
weights are skipped).

## What the data set is (from the outputs)

* Training: **182 base images with a float32 map, 728 views** (four lossless 90 degree rotations). The 74 base images of the 8-bit
  set without a float map are not used (D-05).
* Validation = evaluation: **66 images**, 11 storage days x 6 images (days 0, 5, 8, 12, 15, 19, 23, 27, 30, 37, 47).
* Annotated validation cells (area 15 to 200 um2, not cut by the annotated field of view): **2,049**; **61 (3.0%)** flagged as
  phase-wrapped (D-21) and excluded from phase-derived results; **1,988** used.

## Consistency checks run for this revision

| Check | Result |
|---|---|
| metrics.json (class names swapped) gives the same Dice, AJI, BF1 and per-class Dice as the compiled cell file, all 183 runs | identical |
| Class-name exchange (D-25): compiled segmentation file (legacy names) vs compiled cell file (manuscript names), 180 runs | identical after the swap |
| MdARE recomputed from `cells_pred.csv` / `cells_gt.csv` vs the pipeline value (6 runs of F9) | identical (1e-6) |
| Every table in `artifacts/tables` equals a fresh build | yes (`verify_artifacts.py`: OK) |

## Not available in the repository, and what that affects

| Missing | Affects | What to do |
|---|---|---|
| `data/processed/` (float32 phase maps, masks, `manifest.csv`, `dataset_report.json`) | qualitative figures (old Fig. 6, S-4, S-5); the registration statistics (valid fraction of the float grid) | run `python artifacts/code/make_qualitative_figures.py` on the server (Step 6); copy `data/processed/dataset_report.json` into the SI notes |
| The point-by-point response letter | not part of this revision package (author's instruction) | none |

## Things found in the repository documentation that are out of date

| Where | What is stale | Correct value (source) |
|---|---|---|
| `DECISIONS.md` D-03 | "date-named training images and all 66 validation images have background between 0.4 and 1.1 rad" | 0.66 to 2.50 rad, median 1.50 rad for the 66 validation images (`annotation.phase_offsets` of the compiled cell file). The old range is from the first, half-scale delivery. |
| `DECISIONS.md` D-13 | "day-0 median dry mass 28.2 pg (predicted masks)" | true for EdgeSAM r8 seed 42 only; MobileSAM r8 28.21 ± 0.07 pg, EdgeSAM r32 27.88 ± 0.43 pg, annotation 27.83 pg |
| `configs/base.yaml` `analysis.selected_arch: edge_sam`, `selected_rank: 8`; `benchmark.precisions`, `benchmark.onnx_skip` | stale defaults of the older scripts (`manuscript/results.py`, `figures.py`); not read by the benchmark | do NOT edit `base.yaml` (its hash ties the runs to the configuration, item 1 of `verify_artifacts.py`); the revision tables come from `artifacts/code` and `configs/views.yaml` |
| `outputs/full/analysis/drymass.json`, `architecture_comparison.*` | single seed-42 run at rank 8, legacy class names | superseded for the manuscript by the 3-seed compiled file |
| `checklist.md` | open items about series C and freshness | done; see the updated checklist |
