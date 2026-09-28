# visualization/

Regenerates every **data-driven** figure in the manuscript from small JSON files with a
single command. No checkpoints, dataset or CSVs are needed.

```bash
# from the repository root
python visualization/visualization.py
```

PNGs are written to `visualization/figures/`, which is created automatically and ignored by git.
The final manuscript versions of all figures are in [`docs/figures/`](../docs/figures).

## What is covered

| Manuscript | Data file | Output file | Content |
|---|---|---|---|
| Fig. 2  | `figure02.json` | `per_class_dice.png` | per-class Dice at r = 8 |
| Fig. 3  | `figure03.json` | `radar_chart.png` | aggregate performance radar |
| Fig. 4  | `figure04.json` | `ablation_curve.png` | EdgeSAM LoRA rank sweep |
| Fig. 5  | `figure05.json` | `pareto_frontier.png` | accuracy vs. throughput |
| Fig. 6  | `figure06.json` | `lora_vs_full_finetune.png` | LoRA vs. full fine-tuning |
| Fig. 8  | `figure08.json` | `loss_ablation.png` | loss-component ablation |
| Fig. 10 | `figure10.json` | `population_shift.png` | morphology composition over storage |
| Fig. 12 | `figure12.json` | `drymass_validation.png` | dry-mass agreement and trajectory |

**Not covered**, because they do not come from tabulated data:

- Fig. 1 is the framework diagram.
- Figs. 7, 9 and 11 are image montages built from model predictions by `extract_visual.py` and
  `analysis/plot_trends.py`.

Each JSON file has a `description` field and, where relevant, a `source` field naming the
table in `results/` that the numbers come from. Values are stored exactly as plotted. The only
quantity computed at draw time is the radar normalisation (each axis divided by the best
model), so the raw values can still be inspected.

## Restyling

All visual parameters live in the `STYLE` dictionary at the top of `visualization.py`: colours
per architecture and per class, fonts and font sizes, line widths, markers, bar widths, grid,
figure widths and DPI. Nothing in the plotting functions hard-codes a colour, size or font.

## Reproducibility notes

- Figure widths are the true IEEE column widths (3.50 in single, 7.16 in double).
- Plotting runs inside `plt.style.context("default")`, so a user `matplotlibrc` or seaborn theme
  cannot change the result.

## Requirements

```bash
pip install matplotlib numpy
```
