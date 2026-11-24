# paper/ — published tables and figures

Copies of the results behind the article, for reading without running anything.

## tables/

These are the outputs of `additional_analysis/tables_for_manuscript.py` for the published build (21 November 2026):

- one CSV per table;
- `all_tables.xlsx`, with every table on its own sheet;
- `provenance.json`, with every result file read and its SHA-256;
- `flags.md`, which lists the values that could not be traced to a result file (none).

Every number in these CSVs appears in the final manuscript or Supplementary Information. Supplementary Tables S5
and S9 are printed there with a slightly rearranged column layout; the values are the same.

| Table | File | Content |
|---|---|---|
| 1 | `Table_1_representative_configurations.csv` | Selected configuration per architecture under the proposed objective |
| 2 | `Table_2_objective_comparison_per_architecture.csv` | Proposed objective against each other loss configuration (12 paired LoRA runs, Holm-adjusted Wilcoxon) |
| 3 | `Table_3_touching_isolated_and_outcomes.csv` | Touching and isolated cells; matched, missed, merged and fragmented cells |
| 4 | `Table_4_inference_cost.csv` | Forward-pass latency, throughput and GPU memory (PyTorch and ONNX Runtime) |
| 5 | `Table_5_dry_mass_and_storage.csv` | Single-cell dry-mass agreement and storage-time trends |
| S1 | `Table_S1_test_set_per_day.csv` | Test set per storage day: reference and scored cells, touching cells, cell pixels, median area |
| S2 | `Table_S2_phase_range.csv` | Range of the phase per image and per reference cell (phase wrapping) |
| S3 | `Table_S3_background_invariance.csv` | Effect of added background terms on the corrected maps and on integrated phase |
| S4 | `Table_S4_timing_per_pass.csv` | Benchmark values of each timing pass |
| S5 | `Table_S5_loss_ablation.csv` | Seven loss configurations at every rank and with full fine-tuning |
| S6 | `Table_S6_per_architecture_further_metrics.csv` | Per-architecture tests of further metrics and of the checkpoint choice |
| S7 | `Table_S7_full_cipp_vs_full_global_ipp.csv` | Full objective with cIPP against the full objective with global IPP |
| S8 | `Table_S8_cell_level_paired.csv` | Cell-level paired comparisons at the selected configurations |
| S9 | `Table_S9_rank_sweep.csv` | Every LoRA rank and full fine-tuning under the proposed objective, with parameter counts |
| S10 | `Table_S10_per_class.csv` | Per-class Dice, IoU and class-specific IPE at the selected configurations |
| S11 | `Table_S11_missed_pixels_by_phase_quartile.csv` | Annotated pixels predicted as background, by phase quartile |
| S12 | `Table_S12_storage_trends_per_seed.csv` | Storage trends per seed, date-clustered tests |
| S13 | `Table_S13_per_day_composition.csv` | Per-day matched fraction, median area and dry mass, and class composition |
| S14 | `Table_S14_dry_mass_outliers.csv` | Matched cells with a dry-mass error above 10 pg |
| S15 | `Table_S15_class_confusion.csv` | Class confusion of the scored cells |
| S16 | `Table_S16_signed_errors.csv` | Signed and absolute errors of matched cells, touching and isolated |

## figures/

These are the figures as printed in the article, rendered from the SVGs embedded in the final manuscript and SI.
`manuscript/make_all.py` regenerates Figs. 3–8 and S1–S7 from the result files. Fig. 1 (framework overview) and
Fig. 2 (optical setup) are drawings.
