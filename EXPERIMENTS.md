# Experiment registry: what was run, where it lives, which table it feeds

One place that answers "which run produced this number?". Every number in a manuscript table or in a rebuttal must be
traceable to a row below. If you add a run, add a row here in the same commit. No em dashes on purpose.

**Rules that keep the runs apart**

1. A run is identified by its directory name. Outputs of different experiments never share an output root.
2. Seed 42 of every configuration is the run in `outputs/full` (series F1). It is never retrained; the multi-seed series
   (M1) reads it, read only, and records the SHA-256 of every `metrics.json` it used in `runs_index.csv`.
3. The multi-seed series refuses to start (`aggregate_multiseed.py --check-prereqs --max-stage N`, a step of `run_multiseed.sh`) if the
   data manifest, `configs/base.yaml` or `configs/experiments.yaml` differ from what `outputs/full` was trained with.
4. Primary checkpoint everywhere = last epoch (D-08). Metrics at the native 768 grid (D-12).
5. Training/validation data: 182 base training images (728 views), 66 validation images (D-05), float32 phase, background
   referenced (D-03). Nothing in a series below was trained on 8-bit images.

## Series overview

| Series | What | Output root | Seeds | Runs | Launch | Status |
|---|---|---|---|---|---|---|
| **O**  | Original manuscript runs (8-bit era, v3) | not in this repository | 42 | unknown set | old scripts | superseded; do not mix with anything below |
| **F0** | Main experiment on the FIRST delivery of the float maps (date-named maps at half scale) | `old_run/outputs/full` | 42 | 27 | `run_full_experiment.sh` | **superseded 24 Oct 2026 (D-20)**; kept for the run-by-run comparison only |
| **M0** | Multi-seed extension on the first delivery (incl. a sampler block, not repeated) | `old_run/outputs/multiseed` | 42-46 | 51 | `run_multiseed.sh` | **superseded (D-20)**; its interpretation is in `old_run/docs/MULTISEED_INTERPRETATION.md` and describes the old numbers only |
| **F1** | Main experiment on the corrected maps | `outputs/full` | 42 | 27 trained, 30 table entries (`abl_full` = alias of `lora_r8`, D-09) | `run_full_experiment.sh` | finished 24 Oct 2026 |
| **M1** | Multi-seed extension on the corrected maps, in three stages = architectures (D-22) | `outputs/multiseed` | 43, 44; 42 also for the ablations at ranks 2, 4, 16, 32 | 156 new (52 per architecture); with the mirrored seed-42 runs 180 LoRA runs | `./ops/start_multiseed.sh N` | finished 25 Oct 2026 (180 of 180 indexed) |
| **B** | Inference benchmark, protocol v5 (D-26): FP32 PyTorch and ONNX Runtime, 3 seeds, reported run on physical GPU 2 | `outputs/full/benchmark` | 42, 43, 44 | 6 combinations x 3 seeds | `GPU=2 ./ops/run_benchmark.sh` | finished 26 Oct 2026; protocol in `manuscript/benchmarking_protocol.md` |
| **C** | Per-cell analyses (dry mass, matching, wrapping, composition, storage trends, background-reference sensitivity) of every architecture x rank x seed from the saved predictions; no training | `outputs/cell_analysis/cell_analysis.json` | 42, 43, 44 | 45 runs (full objective; ablations and full fine-tuning optional) | `./ops/run_cell_analysis.sh` | written by the script; class names in the manuscript order (D-25) |

Rule: only F1 and M1 may feed the manuscript. F0 and M0 stay in `old_run/`. `outputs/full/analysis/comparison_with_old_run.md`
(written by `scripts/compare_with_old_run.py`) lists every run against its old counterpart.

## F1 (and F0): main experiment (seed 42), run ids in `outputs/full/runs/` (F0: `old_run/outputs/full/runs/`)

Run id = `<arch>_<variant>`. Architectures: `edge_sam`, `mobile_sam`, `mobilenet_unet`.

| Variant | Meaning | Archs | Feeds |
|---|---|---|---|
| `lora_r2`, `lora_r4`, `lora_r8`, `lora_r16`, `lora_r32` | LoRA rank sweep, full physics-aware loss | all three | Table 1 (r8), rank-sweep table/figure |
| `full_finetune` | all encoder weights trained, full loss (seed 42 only, D-24) | all three | LoRA vs full fine-tuning table |
| `abl_dice_only`, `abl_pmc`, `abl_pmc_bga` | loss ablation (Dice only; Dice+PMC; Dice+PMC+BGA) | all three | ablation table |
| `abl_full` | Dice+PMC+BGA+IPP; **alias of `lora_r8`**, not a separate run (D-09) | all three | ablation table |

Downstream outputs of F1: `outputs/full/analysis/` (per-cell IPE, dry mass, robustness), `outputs/full/benchmark/`,
`manuscript/outputs/results.json` (all manuscript numbers), `manuscript/outputs/figures/`.

## M1 (and M0): multi-seed extension, run ids in `outputs/multiseed/runs/` (M0: `old_run/outputs/multiseed/runs/`)

Run id = `<base run id>__s<seed>`. Ablation at a rank other than 8: `<arch>_abl_<loss>_r<rank>__s<seed>`.
Configuration key used in tables: `<arch>|<variant>`, e.g. `edge_sam|lora_r8`, `mobile_sam|abl_pmc_r32`.

**Rule: every configuration has exactly 3 seeds (42, 43, 44)**, so that no comparison has to be re-run later and every claim can be
checked against seed variation. Seed 42 of the main-grid configurations is the finished run in `outputs/full`; it is never retrained and
is mirrored (hard links, `scripts/link_seed42.py`) as `outputs/multiseed/runs/<id>__s42`, so that all seeds of a configuration sit side by
side. Only the loss ablations at ranks 2, 4, 16 and 32 are new at seed 42 (the main run has the ablation at rank 8 only).

**Stages = architectures.** `STAGE=N` runs the blocks of stages 1..N and skips finished runs.

| Stage | Architecture | Block | Content | Seeds | New runs |
|---|---|---|---|---|---|
| 1 | EdgeSAM | `edge_sam_main_grid_seeds` | LoRA rank sweep r = 2, 4, 8, 16, 32 (full objective); rank-8 ablation (Dice only, + PMC, + PMC + BGA); no full fine-tuning (D-24) | 43, 44 (42 = F1) | 16 |
| 1 | EdgeSAM | `edge_sam_ablation_other_ranks` | the three ablation losses at r = 2, 4, 16, 32 | 42, 43, 44 | 36 |
| 2 | MobileSAM | `mobile_sam_main_grid_seeds`, `mobile_sam_ablation_other_ranks` | same as above | same | 16 + 36 |
| 3 | MobileNet-UNet | `mobilenet_unet_main_grid_seeds`, `mobilenet_unet_ablation_other_ranks` | same as above | same | 16 + 36 |
| | | | | **total** | **156** |

The full objective (Dice + PMC + BGA + IPP) at each rank is the rank-sweep run of that rank. Per architecture and rank the comparison is
therefore: Dice only, + PMC, + PMC + BGA and full, each with 3 seeds, paired by seed.
Queue order: stage, then block priority, then longest run first. The weighted-sampler block of M0 was dropped (D-22).

### Tables written by the aggregator (`outputs/multiseed/summary/tables.md`)

| Table | Content | Replaces / complements in the manuscript |
|---|---|---|
| MS-1 | Rank sweep per architecture (3 seeds for every architecture once its stage has run), paired differences to r8 with 95% t-intervals, rank rule statements | complements the rank-sweep table/figure; gives the rank-8 argument |
| MS-2 | Architectures at r8 (3 seeds) | complements Table 1 |
| MS-3 | (not produced: full fine-tuning is a single seed-42 run, D-24; the LoRA vs full fine-tuning comparison stays the F1 table, worded as single runs) | |
| MS-4 | Loss ablation at rank 8 per architecture, paired differences to the full loss (3 seeds) | complements Table 3 (and removes the aliased-row problem, D-09) |
| MS-6 | Loss ablation by rank (2, 4, 8, 16, 32) per architecture; within-rank differences to the full objective, paired by seed | tells whether the PMC/BGA/IPP claims hold beyond rank 8 |

Other files in `outputs/multiseed/summary/`: `runs_index.csv` (one row per configuration and seed: run id, which
directory the number came from, SHA-256 of its `metrics.json`, collapse flag), `config_summary.csv`,
`paired_differences.csv` (every paired comparison, all metrics), `multiseed_results.json` (everything, also copied to
`manuscript/outputs_multiseed/`). Console log: `outputs/multiseed/logs/pipeline.log`.

### Interpretation rules (fixed before the runs, so they cannot be tuned to the outcome)

* **Rank 8.** For each of Dice, AJI, BF1 and global IPE the best rank by mean is found. A rank whose paired difference to
  the best has a 95% interval containing 0 is "not distinguishable from the best". Report: which ranks are
  indistinguishable, and whether r8 is one of them. The claim allowed by the result:
  * r8 indistinguishable from the best on the headline metrics and smaller than the best: "rank 8 is a
    parameter-efficient choice within seed variation", never "optimal".
  * r8 clearly worse than a larger rank on Dice: say so, state the cost (parameters, latency), and choose by the
    accuracy/efficiency trade-off or change the default rank.
* **Collapse.** A run with any class Dice below 0.05 counts as a collapse. Collapse frequency per configuration is reported
  next to the means, because a mean over a collapsed and a healthy run hides the failure.
* **Comparisons (MS-2 to MS-4).** A difference is reported as "larger than seed variation" only if the 95% interval of
  the paired difference excludes 0; with n = 3 the interval is wide, so the wording stays descriptive.
* **Ablation by rank (MS-6).** Every rank has 3 seeds, so within-rank differences carry 95% t-intervals (wide with n = 3). A statement such
  as "the effect of PMC/BGA/IPP does not depend on the rank" is allowed only if the sign of the paired difference to the full objective is the
  same at all five ranks, for both SAM models, and the interval excludes 0 at the ranks where it is claimed; otherwise the manuscript says
  where it holds ("at rank 8", "at ranks 8 to 32") and reports the rest descriptively. A collapsed class in a run is reported as a collapse, not averaged.
* **Sampler.** No sampler experiment in M1 (D-22). The manuscript says plain shuffling with class-weighted Dice (D-04).

## Where the knobs are

| Thing | File |
|---|---|
| seeds, blocks, which runs | `configs/multiseed.yaml` |
| expansion of blocks into run ids | `qpi/experiments.py::build_multiseed_grid` |
| sampler implementation (unused since D-22) | `qpi/data/dataset.py` (`dominant_class_weights`, `build_loaders`), D-18 |
| aggregation, statistics, tables | `qpi/analysis/multiseed.py`, `scripts/aggregate_multiseed.py` |
| launch | PC: `tools_pc\1_upload.ps1`, `2_clean_server.ps1`, `3_run.ps1`, `4_watch.ps1`, `5_download.ps1`, `6_plots.ps1` (tables and figures only). Server: `./ops/check_multiseed.sh N`, `[GPUS=free] ./ops/start_multiseed.sh N`, `./ops/progress.sh --watch`, `./ops/clean.sh`, `./ops/stop.sh`, `GPU=n ./ops/run_benchmark.sh` |
| moving the old results away; tidying | `ops/legacy/move_old_run.sh` (done), `ops/tidy.sh` |
| wrapped-cell rule | `analysis.wrapping` in `configs/base.yaml`, `qpi/analysis/wrapping.py`, D-21 |
| benchmark | `qpi/benchmark.py`, `qpi/equivalence.py`, `scripts/run_benchmark.py`, `scripts/check_equivalence.py`, `manuscript/benchmarking_protocol.md` |
| tests | `tests/test_multiseed.py`, `tests/test_benchmark.py` |

## Adding a run later

1. Add it to `configs/multiseed.yaml` as a new block (never edit a finished block's seeds).
2. `python main.py list-runs --config-overlay multiseed.yaml | wc -l` shows the new total.
3. Give the block a `stage`, rerun `./ops/start_multiseed.sh <stage>` (finished runs are skipped), then add a row to the M1 table above.
