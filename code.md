# code.md: current state of the code (for Claude; keep short, no history)

Update this file in the same step as any code change; delete lines for removed code.

## What the project does
Lightweight segmentation (EdgeSAM, MobileSAM with LoRA; MobileNet-UNet) of quantitative-phase holograms of red blood cells, with a
physics-aware loss, then per-cell dry-mass / phase-integral analysis. Input: float32 phase maps (rad) and 3-class masks
(1 discocyte, 2 spherocyte, 3 echinocyte in the manuscript order; raw outputs use the earlier names, `qpi/utils/classnames.py` exchanges 2 and 3 on read).

## Pipeline (corrected, D-03 ... D-12)
- `scripts/prepare_dataset.py` -> `data/processed/{train,val}/{phase,mask}/*.tif`, `manifest.csv` (split, stem, registration_ok, storage_day, phase_offset_rad). 768x768 float32; 182 base training images (728 views), 66 validation images.
- Per image: subtract the background level (background referencing), then z-score the network input. Loss = Dice + 0.1 PMC + 0.05 BGA + 0.1 IPP (`qpi/training/losses.py`). Checkpoint used everywhere = last epoch (`last.pt`, D-08).
- Grid (`configs/experiments.yaml`): 3 archs x LoRA ranks 2,4,8,16,32 x losses full/dice_only/pmc/pmc_bga x seeds 42,43,44; plus full fine-tuning (seed 42). Config `configs/base.yaml` (do not overwrite on the server), `multiseed.yaml`, `smoke.yaml`, `views.yaml`.
- Seed 42 lives in `outputs/full/runs/<id>`; seeds 43/44 in `outputs/multiseed/runs/<id>__s43|__s44`. Ablation losses at ranks other than 8 exist only in multiseed. Run id: `<arch>_lora_r<k>`, `<arch>_abl_<loss>[_r<k>]`, `<arch>_full_finetune`.
- Training: `scripts/run_queue.py`, `run_full_experiment.sh`, `run_multiseed.sh`, `ops/start_multiseed.sh`. Needs the vendored model code in `qpi/models/vendor` (not in the PC copy).

## Packages (`qpi/`)
- `data/` registration, dataset build, verification. `models/` three networks + `lora_utils` (LoRA merges into the weights at deploy, so cost is rank independent). `training/` loss, trainer. `evaluation/` metrics (Dice, AJI, IPE) and checkpoint evaluation.
- `analysis/cell_analysis.py` per-cell analysis of one run from `predictions_last.npz`: dry mass, matching, phase integral, wrapping, composition, storage trends, background-reference sensitivity. `run_units()` is the resumable runner (a stored unit is reused only if status, schema, checkpoint tag, manifest sha and predictions sha256 all match). `run_all()` is the older runner used by tests.
- `analysis/robustness.py` distortion tests (offset, tilt, bowl, astig). Offset rows are invariant by construction (added before the z-score) and flagged `invariant_by_construction`. `run_robustness_units()` is resumable; `write_legacy_robustness()` rebuilds `robustness.json`.
- `analysis/multiseed.py` aggregator for segmentation metrics (`rank_selection` stays; its result is a fact table, not a rule, D-27).
- `analysis/compile.py` builds the compiled multi-seed JSONs from the individual files; `stale_problems()` (sha256 comparison); `load_views`/`apply_view`. `analysis/migrate.py` splits the older JSONs into individual files.
- `benchmark.py`, `equivalence.py` protocol v5 (FP32 PyTorch and ONNX Runtime, 3 seeds, D-26).
- `utils/layout.py` all paths of the added layout; `utils/results_io.py` write/read unit files with provenance (relative paths, sha256, config hashes); `utils/progress.py` the progress / exit contract; `utils/io.py` atomic JSON writes.

## Result layout (D-29)
- Older files, still written and read by the older scripts: `outputs/full/runs/*`, `outputs/multiseed/runs/*` and `summary/*`, `outputs/full/analysis/*.json`, `outputs/full/benchmark/*`, `outputs/cell_analysis/cell_analysis.json` (+csv), `manuscript/outputs/results.json`, `manuscript/outputs_multiseed/multiseed_results.json` (written by the aggregator).
- Individual: `outputs/analysis/cell/<run_id>.json`, `robustness/<run_id>.json`, `benchmark/<arch>__<backend>__s<seed>.json`, `cell/_annotation.json` (helper). Each has `unit_schema`, `status`, `provenance`.
- Compiled (`outputs/compiled/`): `segmentation_`, `cell_analysis_`, `robustness_`, `benchmark_multiseed.json`. Per configuration: per-seed values, mean, sample SD (n-1), min, max, seeds expected/found/missing, `inputs` (sha256), `problems`, `complete`. A missing seed is listed, never counted as zero. Benchmark headline Peak VRAM = max over seeds (`peak_vram_mb_max`).
- Choosing configurations happens last: `configs/views.yaml` (same_rank, preferred = MobileSAM r8 + EdgeSAM r32 + MobileNet-UNet r8, rank_sweep, loss_ablation) through `apply_view`. Compiled files never hold a selection.

## Scripts and how to run (all print progress in the foreground; last line `COMPLETE` only after real success, exit 1 otherwise)
- `python scripts/organise_results.py [--dry-run] [--force]` (or `ops/organise_results.sh`): ONE command. Stages: split older files -> compile 4 files -> check every number against the older files. Incomplete data is a NOTE (listed in `problems`), broken files are failures.
- `python scripts/run_cell_analysis.py [--all] [--jobs N] [--force] [--archs --ranks --seeds --losses --include-full-finetune --trust-missing-sources]`: resumable; CPU only; writes units, then rebuilds the older `cell_analysis.json`, CSVs, README. `ops/run_cell_analysis.sh` = this + organise + freshness (no results.json/figures/tex).
- `python scripts/run_robustness.py --scope selected|all [--limit N --gpu G ...]`: GPU; units in `outputs/analysis/robustness/`; rebuilds `robustness.json` when the six selected runs exist. `scripts/run_analysis.py` (older) uses the same unit code for the selected runs.
- `python scripts/compile_results.py all|segmentation|cell|robustness|benchmark [--expect-all-cells]`: strict (exit 1 on any listed problem).
- `scripts/run_benchmark.py` (also writes the individual timing files), `check_equivalence.py`, `aggregate_multiseed.py`, `compare_architectures.py`, `verify_outputs.py`, `check_freshness.py` (compiled files compared by sha256), `verify_dataset.py`.
- Manuscript side (not touched now): `manuscript/results.py`, `make_tex.py`, `figures.py`, `ops/make_manuscript_outputs.sh`.

## Tests (`python -m pytest -q tests`, no GPU)
`test_results_layout.py` (units, resume, compile, views, organise, benchmark/robustness units), `test_progress.py`, `test_merge_rank_independence.py` (per-architecture parts skip without the vendored code), `test_cell_analysis.py`, `test_multiseed.py`, `test_benchmark.py`, `test_core.py` and others. Without `qpi/models/vendor` the 3 `test_core.py::test_models_return_logits_at_input_size` tests fail; that is expected on the PC copy.

## Cleanup
`python scripts/tidy_stale.py --dry-run`, then `python scripts/tidy_stale.py` (works on the PC without bash; the list of stale paths lives there). `ops/tidy.sh` (also run by `ops/clean.sh`, server) calls it and also tidies old launcher copies. It moves stale files into `_to_delete/` with their folder structure kept (never deletes; delete that folder yourself): old v2 `scripts/run_benchmark`, `tests/DECISIONS.md`, the half-scale old-run tools and comparison (`scripts/compare_with_old_run.py`, `tests/test_old_run_tools.py`, `outputs/full/analysis/comparison_with_old_run.*`), `ops/legacy/`, and `old_run/`, `bench_run.out`, `study_run.out`, root `run_benchmark.py`, `multiseed_results.pdf` where present.
Left in place on purpose: `manuscript/outputs_multiseed/*` (aggregator writes it, manuscript tools read it), `outputs/multiseed/summary_manuscript_names/`, the older result files, sampler code and stale config keys (`selected_arch/rank`, `benchmark.precisions/onnx_skip`, `rank_selection`), until the final numbers are written into `manuscript/outputs`.
