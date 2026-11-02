# artifacts/

Everything that goes into the manuscript and the response to reviewers: the figures, the tables, and
the code that builds them from `outputs/`. Nothing here is computed on the fly — each file is
written by a script in `code/` from the results already in `outputs/` and
`manuscript/outputs/results.json`.

```
figures/
  manuscript/   the figures as they appear in the manuscript (PNG + PDF): Fig1, Fig3, Fig4, Fig5,
                Fig7, Fig8, FigS1-FigS3
  main/         the full set of main-text result figures (F3-F11)
  si/           supplementary figures FS1-FS8
  pptx/         Figures.pptx, the editable sources; see pptx/README.md
tables/
  main/         T1-T10, each as .md and .csv
  si/           S1-S14, each as .md and .csv
  raw/          all_runs.csv (every run, every metric), cell_analysis_long.csv (per cell)
  ALL_TABLES.md, ALL_TABLES.docx   every table concatenated
code/           the builders (see below)
```

## Class labels

Classes carry their mask value throughout: **1 = discocyte, 2 = spherocyte, 3 = echinocyte,
4 = stomatocyte**, 0 = background. Raw outputs and `configs/base.yaml` keep an earlier naming in
which 2 and 3 are exchanged; the swap is applied once, where results are read
(`qpi/utils/classnames.py`), so everything in this directory is already in the manuscript order and
must not be swapped again.

## Rebuilding

From the repository root, after `outputs/` and `manuscript/outputs/results.json` are up to date:

```bash
python artifacts/code/make_tables.py               # tables/main, tables/si, ALL_TABLES.*
python artifacts/code/make_manuscript_figures.py   # figures/manuscript
python artifacts/code/make_figures.py              # figures/main, figures/si
python artifacts/code/make_qualitative_figures.py  # qualitative panels (needs data/processed)
python artifacts/code/build_figures_pptx.py        # figures/pptx/Figures.pptx
python artifacts/code/verify_artifacts.py          # checks every table and figure against results.json
python artifacts/code/audit_claims.py              # checks the manuscript text against results.json
```

`make_figure_objects.py` writes the individual image panels used to fill the placeholders in
`Figures.pptx`; it needs `data/processed`, so it only runs on the machine that holds the phase maps.

`python scripts/check_freshness.py` reports any file here that is older than the results it was
built from, together with the command that regenerates it.
