# Editable figures (PowerPoint)

`Figures_revised.pptx` (built from `Figures_original.pptx` by `artifacts/code/build_figures_pptx.py`):

| Slide | Figure | Content |
|---|---|---|
| 1 | Fig. 1 | framework, corrected labels; image panels are placeholders |
| 2 | Fig. 1 objects | the same four placeholders, for staging |
| 3 | Fig. S-4 | as in the SI: phase (rad), phase-gradient magnitude, predicted contours of EdgeSAM (r=32) (blue) and MobileSAM (r=8) (orange), 256 × 256 px window, legend, 20 µm bars (54.2% of the panel) |
| 4 | Fig. 6 | 4 classes × 6 columns, headers, row labels, 20 µm bars, legend |
| 5 | Fig. S-5 | five storage days, phase and EdgeSAM (r=32) contours (decorative title removed) |

## Filling the placeholders

On the machine with `data/processed`, from the repository root:

    python artifacts/code/make_figure_objects.py --seed 42

This writes every panel to `artifacts/figures/objects/<Fig1|FigS4|Fig6|FigS5>/` with the file name printed on its
placeholder. In PowerPoint: right-click the placeholder > Change Picture > From a File. Size, position and the 20 µm bars
stay as they are.

The Fig. 1 phase, z-score and volume objects contain their colour bar with the values printed next to it (phase: upper
and lower limit of the window, and 0 when it lies inside the range; z-score: 3 / 0 / −3; volume: high / low), so nothing
has to be aligned by hand. The placeholders on slides 1 and 2 have the same aspect ratio as these images.
`artifacts/figures/objects/objects.json` gives, per figure, the chosen image and window, the storage day of each Fig. 6
row, and the colour-bar limits of Fig. 1.

## Fig. 1 corrections (slide 1)

- network input: per-image z-score (zero mean, unit variance), not RGB pretraining statistics
- the loss uses the background-referenced phase, not the raw phase
- class legend: mask value 2 = spherocyte, 3 = echinocyte (was swapped); colours as in Figs. 6 and 7
- readout formulas as Eqs. 7–10: A = N_Ω·Δx_eff², C = 4πA/P², S = Σ_{i∈Ω}(φᵢ − φ_bg)·Δx_eff², m = (λ/2πα)·S; S is
  labelled "Integrated phase" (was "Phase-integrated optical quantity"); "Integrated phase and dry mass"
- LoRA placement ("attention, linear and convolution layers"), gradients update "LoRA adapters, stem, normalisation and
  decoder"; trainable share given for EdgeSAM (r = 32); "Ttrainable" typo fixed
- the edited equation boxes are now plain text (Cambria Math characters) and render the same in every viewer

LibreOffice does not draw some of the white header labels (also in the original file); PowerPoint shows them.
