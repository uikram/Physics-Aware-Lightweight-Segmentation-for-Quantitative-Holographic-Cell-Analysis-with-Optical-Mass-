# Editable figures (PowerPoint)

`Figures_revised.pptx` (built from `Figures_original.pptx` by `artifacts/code/build_figures_pptx.py`):

| Slide | Figure | Content |
|---|---|---|
| 1 | Fig. 1 | framework, corrected labels; image panels are placeholders |
| 2 | Fig. 1 objects | the same four placeholders, for staging |
| 3 | Fig. S-4 | phase, phase-gradient magnitude, EdgeSAM (r=32) contours (full 768 × 768 px field) |
| 4 | Fig. 6 | 4 classes × 6 columns, headers, row labels, 20 µm bars, legend |
| 5 | Fig. S-5 | five storage days, phase and EdgeSAM (r=32) contours (decorative title removed) |

## Filling the placeholders

On the machine with `data/processed`, from the repository root:

    python artifacts/code/make_figure_objects.py --seed 42

This writes every panel to `artifacts/figures/objects/<Fig1|FigS4|Fig6|FigS5>/` with the file name printed on its
placeholder. In PowerPoint: right-click the placeholder > Change Picture > From a File. Size, position and the 20 µm bars
stay as they are.

`artifacts/figures/objects/objects.json` gives, per figure, the chosen image and window, the storage day of each Fig. 6
row, and the colour-bar limits of Fig. 1. Update the numbers next to the Fig. 1 colour bars (slide 1: 6.3 / 0 / -2.5 and
3 / 0 / -3) to the values in `objects.json` (`p1_phase_colorbar_rad`; the z-score bar is fixed at -3 to 3).

## Fig. 1 corrections (slide 1)

- network input: per-image z-score (zero mean, unit variance), not RGB pretraining statistics
- the loss uses the background-referenced phase, not the raw phase
- class legend: mask value 2 = spherocyte, 3 = echinocyte (was swapped); colours as in Figs. 6 and 7
- readout: S = Σ(φᵢ − φ_bg)·Δx² and m = (λ/2πα)·S (Eqs. 9 and 10); "Integrated phase and dry mass"
- LoRA placement ("attention, linear and convolution layers"), gradients update "LoRA adapters, stem, normalisation and
  decoder"; trainable share given for EdgeSAM (r = 32); "Ttrainable" typo fixed
- the edited equation boxes are now plain text (Cambria Math characters) and render the same in every viewer

LibreOffice does not draw some of the white header labels (also in the original file); PowerPoint shows them.
