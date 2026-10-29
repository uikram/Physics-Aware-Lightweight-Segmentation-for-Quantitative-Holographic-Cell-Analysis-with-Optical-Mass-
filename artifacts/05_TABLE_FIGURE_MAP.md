# 05. Tables and figures: old to new, with file paths

Every new item is a generated file. Tables: `artifacts/tables/{main,si}/<id>.csv` (values) and `<id>.md` (caption notes); all in
one Word file: `artifacts/tables/ALL_TABLES.docx`. Figures: `artifacts/figures/{main,si}/<id>.png` (300 dpi) and `.pdf`.
Class names carry their mask value (2 = spherocyte, 3 = echinocyte).

## Main manuscript

| Old | Old content | New number | New file | Notes |
|---|---|---|---|---|
| Fig. 1 | framework | Fig. 1 | redraw by hand | background-referenced phase to the loss, z-scored phase to the network; no SAM prompt encoder / mask decoder (03, Sec. 2.1) |
| Fig. 2 | DHM setup | Fig. 2 | unchanged | |
| Table 1 | segmentation at r8, single run | **Table 1** | `tables/main/T2_preferred` | preferred MobileSAM r8, EdgeSAM r32, MobileNet-UNet r8 |
| (none) | | **Table 2** | `tables/main/T3_same_rank` | same rank r8, with paired differences |
| Fig. 3 | per-class Dice at r8 | **Fig. 3** | `figures/main/F3_per_class_dice` | preferred + EdgeSAM r8, every seed shown |
| (none) | | **Table 3** | `tables/main/T4_training_stability` | lost classes per architecture, rank, seed |
| (Fig. S-1) | EdgeSAM rank line | **Fig. 4** | `figures/main/F4_rank_sweep` | all architectures, Dice / AJI / BF1 / IPE |
| (none) | | **Fig. 5** | `figures/main/F5_training_curves` | instability visible per seed |
| Table 3 + Table 4 | loss ablation (single run) + EdgeSAM per-class | **Table 4** | `tables/main/T5_loss_ablation` | old Table 4 dropped (class detail in S4/S7) |
| Fig. 5 | ablation BF1 bars + IPE line (dual axis) | **Fig. 6** | `figures/main/F6_loss_ablation` | IPE and Dice in separate rows, no dual axis |
| Table 2 | LoRA vs full FT | **Table 5** | `tables/main/T6_lora_vs_full_ft` | 3-seed LoRA rows + seed-42 LoRA rows + FT |
| Fig. 4 | LoRA vs FT per class | SI | `figures/main/F7_lora_vs_full_finetuning` | move to SI (Table 5 carries the numbers) |
| Table 5 | IPE global + class | **Table 6** | `tables/main/T7_phase_and_mass` | global + per-cell IPE + dry-mass agreement; class IPE to S7 |
| Fig. 8a | predicted vs annotation dry mass scatter | **Fig. 7** | `figures/main/F9_dry_mass_agreement` | scatter + Bland-Altman, 3 seeds |
| Table 6 | benchmark FP32/FP16 | **Table 7** | `tables/main/T8_efficiency` | FP32 only, PyTorch and ONNX, 3 seeds |
| Fig. S-3 | accuracy-throughput | SI | `figures/main/F8_accuracy_vs_latency` | move to SI |
| Fig. 6 | qualitative EdgeSAM r8 | **Fig. 8** | `figures/main/FQ1_qualitative_segmentation` | **not generated yet** (needs data/processed): `make_qualitative_figures.py` |
| Fig. 7 | predicted composition EdgeSAM r8 | **Fig. 9** | `figures/main/F11_composition` | annotation + both preferred models |
| Table 7 | per-day predicted cells, EdgeSAM r8 | **Table 8** | `tables/main/T9_storage_per_day` | annotation + both preferred models |
| (text) | storage statistics | **Table 9** (or text only) | `tables/main/T10_storage_statistics` | annotation vs predicted |
| Fig. 8b | normalised area and dry mass vs day | **Fig. 10** | `figures/main/F10_storage_trends` | per-image medians, annotation vs predicted |

## Supplementary Information

| Old | New file | New content |
|---|---|---|
| Table S-1 | `tables/si/S1_rank_sweep` | rank sweep, all architectures, mean ± SD |
| | `tables/si/S2_rank_sweep_per_seed` | every seed, lost classes named |
| | `tables/si/S3_rank_paired` | paired rank differences (to r8 and r32) |
| | `tables/si/S4_ablation_all_ranks` | ablation at every rank |
| | `tables/si/S5_ablation_paired` | paired differences + direction summary |
| | `tables/si/S6_all_runs.csv` (+ `tables/raw/all_runs.csv`) | all 183 runs (data file, not a printed table) |
| (old Table 4, 5 class columns) | `tables/si/S7_per_class_ipe` | class IPE and Dice |
| | `tables/si/S8_parameters_by_rank` | trainable parameters |
| | `tables/si/S9_benchmark_per_seed` | per-seed timing + ONNX equivalence |
| (old 4.3 robustness text) | `tables/si/S10_robustness` | perturbations |
| | `tables/si/S11_wrapping` | wrapped cells |
| (old 4.3 IPP bias text) | `tables/si/S12_error_strata` | error strata + IPP and low-phase misses |
| | `tables/si/S13_cell_metrics_all_configs` | dry-mass bias in every configuration |
| | `tables/si/S14_reference_sensitivity` | background reference |
| | `tables/main/T1_dataset` (use as Table S0) | data set summary |
| Fig. S-1 | `figures/si/FS1_rank_sweep_per_class` | per-class Dice vs rank |
| | `figures/si/FS2_ablation_ipe_heatmap` | ablation IPE, all ranks |
| | `figures/si/FS3_percell_ipe_cdf` | per-cell IPE distribution |
| | `figures/si/FS4_robustness` | robustness |
| | `figures/si/FS5_parameters_vs_rank` | parameters vs rank |
| | `figures/si/FS6_error_strata` | error strata |
| | `figures/si/FS7_training_curves_all_ranks` | all training curves |
| | `figures/si/FS8_annotation_trends` | annotation per-image area and dry mass vs day |
| Fig. S-2 (radar) | delete | redundant and misleading |
| Fig. S-3 | `figures/main/F8_accuracy_vs_latency` | |
| Fig. S-4 | `figures/si/FQS1_gradient_alignment` | **server** |
| Fig. S-5 | `figures/si/FQS2_storage_examples` | **server** |
| (old Fig. 4) | `figures/main/F7_lora_vs_full_finetuning` | |

## Final numbering (fill in while editing)

| Artifact id | Final number in manuscript / SI |
|---|---|
| T2 | Table 1 |
| ... | ... |
