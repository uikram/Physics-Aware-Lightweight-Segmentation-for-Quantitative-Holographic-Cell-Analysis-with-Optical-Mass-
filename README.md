# Physics-aware lightweight segmentation for quantitative holographic cell analysis

Code for the Applied Physics B manuscript "Physics-Aware Lightweight Segmentation for Quantitative Holographic
Cell Analysis with Optical Mass Preservation". Everything needed to regenerate the manuscript numbers and figures
from the float32 phase maps is here.

## Layout

```
configs/            base.yaml (all settings), experiments.yaml (architectures, LoRA ranks, ablation losses), smoke.yaml (overlay)
data/               raw data (input, read only):  dataset/{X,Y}_{train,val}, raw_data/ (date-named float32 maps, corrected delivery), X_train_raw/
data/processed/     built by scripts/prepare_dataset.py: float32 phase + mask in one frame, manifest.csv, dataset_report.json
qpi/data/           registration (ECC), dataset build, dataset class, verification
qpi/models/         EdgeSAM (vendored RepViT), MobileSAM, MobileNet-UNet, LoRA utilities
qpi/training/       physics-aware loss, trainer
qpi/evaluation/     segmentation / IPE metrics, checkpoint evaluation
qpi/analysis/       per-cell phase-integral analysis, dry mass, robustness
qpi/benchmark.py    inference benchmark, protocol v5: FP32 PyTorch and ONNX Runtime, P50 / P99 / FPS / Peak VRAM, 3 seeds
qpi/equivalence.py  ONNX FP32 against PyTorch FP32 (logits, Dice, per-image dry mass) with a fixed acceptance gate
scripts/            prepare_dataset, verify_dataset, run_queue, run_analysis, run_cell_analysis, check_freshness, run_benchmark, check_equivalence, verify_outputs, record_run
manuscript/results.py   ONE json with every number used in the manuscript (from actual outputs only)
manuscript/benchmarking_protocol.md   what the benchmark measures and how (no results)
manuscript/figures.py   manuscript figures from actual outputs (independent of training and evaluation)
tests/              unit tests
run_full_experiment.sh  the full pipeline (CUDA_VISIBLE_DEVICES=2,3): main experiment, analysis, benchmark, figures
run_multiseed.sh        multi-seed extension, STAGE=1|2|3 (configs/multiseed.yaml, outputs/multiseed); needs outputs/full first
run_all.sh              main experiment, then multi-seed stage 1, then comparison with old_run/ (not needed again)
run_smoke_test.sh       the same pipeline on a tiny budget with separate settings and outputs
ops/                    server helpers: run_benchmark.sh (benchmark, equivalence gate, manuscript outputs), make_manuscript_outputs.sh,
                        check_multiseed.sh, pack_results.sh (results without checkpoints), start_multiseed.sh, progress.sh, status.sh,
                        stop.sh, clean.sh, tidy.sh
tools_pc/               PowerShell helpers for the PC, numbered: 1_upload, 2_clean_server, 3_run, 4_watch, 5_download, 6_plots (README_PC.md)
EXPERIMENTS.md      registry: every experiment series, run id, output location and the table it feeds
DECISIONS.md        every methodological decision, with the reason, and the open questions
checklist.md        verification status of each pipeline step
```

## Running on the server

```bash
pip install -r requirements.txt          # CUDA builds of torch/torchvision, onnxruntime-gpu, mobile_sam
bash run_smoke_test.sh                      # ~ minutes; writes outputs/smoke, manuscript/outputs_smoke, data/processed_smoke
bash run_full_experiment.sh                 # full run; sets CUDA_VISIBLE_DEVICES=2,3 itself (refuses any other value)
```

* GPU policy: only physical GPUs 2 and 3 are used. Inside the process they are `cuda:0` and `cuda:1`; all code
  addresses the visible indices. `enforce_gpu_policy` aborts if `CUDA_VISIBLE_DEVICES` contains anything else.
* Resumable: finished runs are skipped. To redo a run delete `outputs/full/runs/<run_id>` (or `main.py run --force`).
* Smoke and full outputs cannot mix: `OUTPUT_MODE.json` binds an output directory to one mode.
* Outputs: `outputs/full/runs/<run_id>/{config.json,history.csv,metrics.json,checkpoints/,predictions_*.npz,per_image_*.csv}`,
  `outputs/full/analysis/`, `outputs/full/benchmark/`, `outputs/full/run_metadata.json`, `manuscript/outputs/results.json`,
  `manuscript/outputs/figures/`.

## Status (27 Oct 2026)

* Data: the date-named float maps were re-delivered at the correct scale (D-20) and phase-wrapped cells are excluded from the
  quantitative phase results (D-21). Everything from the first delivery is in `old_run/` and is not used.
* Main experiment (seed 42, 27 trainings) and the multi-seed extension (every LoRA configuration with seeds 42, 43, 44: 3 architectures x
  5 ranks x 4 losses = 180 runs, 156 of them new) are finished. Full fine-tuning is a single seed-42 run per architecture (D-24).
  Results: `outputs/full/`, `outputs/multiseed/summary/{tables.md,runs_index.csv,config_summary.csv,paired_differences.csv,multiseed_results.json}`.
  What each series is for and which table it feeds: `EXPERIMENTS.md`.
* Class names (D-25): the manuscript order is mask value 2 = spherocyte, 3 = echinocyte. Raw outputs and `configs/base.yaml` keep the earlier
  names; the two names are exchanged where results are read (`qpi/utils/classnames.py`), so `results.json`, tables and figures use the manuscript order.

## Re-running pieces

```bash
python -m pytest -q tests                                    # no GPU needed
./ops/make_manuscript_outputs.sh                             # results.json, figures, tex report, verification (no GPU)
./ops/run_cell_analysis.sh                                   # per-cell analyses (resumable, one file per run) -> outputs/analysis/cell/ and outputs/cell_analysis/, then organise + freshness (no GPU); ARGS="--all" for all losses
./ops/organise_results.sh                                    # ONE command: individual files + outputs/compiled/*_multiseed.json (mean, SD, per seed, missing seeds) + check against the older files
python scripts/run_robustness.py --scope all                 # robustness of every run -> outputs/analysis/robustness/ (GPU)
python scripts/check_freshness.py                            # which result file is older than its inputs, and the command that regenerates it
GPU=1 SEEDS="42 43 44" ./ops/run_benchmark.sh                # benchmark (FP32, PyTorch and ONNX Runtime) + equivalence gate + manuscript outputs
./ops/start_multiseed.sh N / ./ops/progress.sh --watch       # only to rerun or extend the multi-seed series (finished runs are skipped)
```

From the PC (`tools_pc/README_PC.md`): `1_upload.ps1 -Stop`, `3_run.ps1`, `5_download.ps1`, and `6_plots.ps1` for tables and figures only.
Benchmark definition and what may be claimed from it: `manuscript/benchmarking_protocol.md`.

## Reproducibility

Seed 42 for every run of the main experiment (the multi-seed extension adds seeds 43 and 44 and seed 42 for the ablations at ranks 2 and 32; Python, NumPy, PyTorch, data loader generator); deterministic algorithms are requested
(operations without a deterministic CUDA kernel only warn, so bitwise equality across GPU models is not guaranteed).
`run_metadata.json` records configuration hashes, data manifest hash, package versions, GPU names and timestamps.
No split leakage: train and validation come from the provided split and are checked by content hash in
`scripts/verify_dataset.py`.

Read `DECISIONS.md` before interpreting results: it lists where the implementation follows the data provider's float
maps rather than the 8-bit pipeline described in an earlier manuscript version.
