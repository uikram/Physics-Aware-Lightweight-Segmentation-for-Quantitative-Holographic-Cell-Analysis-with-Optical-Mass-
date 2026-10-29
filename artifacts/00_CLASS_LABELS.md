# 00. Class labels: read this first

## The only correct mapping (confirmed by the author)

| Mask value | Class |
|---|---|
| 0 | background |
| 1 | discocyte |
| **2** | **spherocyte** |
| **3** | **echinocyte** |
| 4 | stomatocyte |

Everything in `artifacts/` uses this mapping, and every table and figure writes the mask value next to the name, e.g.
"Dice spherocyte (2)", "sphero(2)". If a number in the manuscript names a class without that check, take it from `artifacts/`.

## Why there was confusion

The pipeline was written with the two names the other way round (2 = "echinocyte", 3 = "spherocyte"). The trained models, the
mask values and every number are unaffected: only the **names** attached to mask values 2 and 3 were swapped. The raw outputs were
deliberately not rewritten, so some files still carry the old names. `qpi/utils/classnames.py` swaps them where results are read.

**Rule: never copy a class-specific number from a file marked "OLD names" below. Use `artifacts/tables` instead.**

| File(s) | Names | Use for the manuscript? |
|---|---|---|
| `artifacts/tables/**`, `artifacts/figures/**`, `artifacts/numbers.json` | correct, with mask value | **yes (the only source)** |
| `outputs/compiled/cell_analysis_multiseed.json`, `outputs/analysis/cell/*.json`, `outputs/cell_analysis/**` (incl. `cells_gt.csv`, `cells_pred.csv`) | correct | yes, through `artifacts/code` |
| `outputs/multiseed/summary_manuscript_names/*` | correct | yes |
| `outputs/{full,multiseed}/runs/*/metrics.json`, `per_image_*.csv` | **OLD names** (key `dice_echinocyte` = mask value 2 = spherocyte) | no |
| `outputs/compiled/segmentation_multiseed.json`, `outputs/multiseed/summary/*`, `manuscript/outputs_multiseed/multiseed_results.json` | **OLD names** | no |
| `outputs/full/analysis/*` (`drymass.json`, `drymass/*.csv`, `phase_integral*`, `architecture_comparison.*`, `robustness.*`) | **OLD names**, single seed 42, rank 8 | no |
| `configs/base.yaml` `data.class_names` | **OLD names** | do not edit (its SHA-256 ties the runs to the configuration; `manuscript/results.py` relies on it) |
| The uploaded manuscript and SI (text, Tables 1 to 7, Figs. 3, 4, 6, 7) | **OLD names in many places** | rewrite with this mapping (03_MANUSCRIPT_CHANGES.md) |

## How it is checked (automatically)

`python artifacts/code/verify_artifacts.py`, check 6, ties names to mask values using only the raw predictions: for every run, a
class reported with Dice 0 has no predicted pixel of its mask value, and a detected class has some. Example: MobileNet-UNet r8
seed 42 predicts 0 pixels of mask value 2 and 0 of mask value 4, so it loses **spherocytes** and stomatocytes. In the old file
`metrics.json` the same run reads `dice_echinocyte: 0.0` because there mask value 2 is called "echinocyte". Check 4 confirms the
swap is consistent between the old-name and new-name files for all 180 LoRA runs.

## Consequences for the text (all class statements change)

| Statement | Old manuscript | Correct |
|---|---|---|
| Classes MobileNet-UNet never detects | echinocytes and stomatocytes | **spherocytes (2)** and stomatocytes (4) |
| Class EdgeSAM loses in unstable seeds / full fine-tuning | echinocyte | **spherocyte (2)** |
| Classes MobileSAM full fine-tuning loses | echinocyte, stomatocyte | spherocyte (2), echinocyte (3), stomatocyte (4) |
| Dominant class at day 47 (annotation) | (old Table 7: spherocytes 42%, echinocytes 36%, predicted) | spherocytes 59%, echinocytes 30% |
| Table 1 / 2 column "Dice_echi" | echinocyte | rename per the new tables; show all four classes with mask values |
| Table 4 / 5 class columns, Fig. 3 / 4 / 6 / 7 legends | old names | regenerate from `artifacts/` |
