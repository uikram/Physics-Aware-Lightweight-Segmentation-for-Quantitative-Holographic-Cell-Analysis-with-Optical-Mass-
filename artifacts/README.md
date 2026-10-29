# Revision artifacts (Applied Physics B resubmission)

Start with [`00_START_HERE.md`](00_START_HERE.md) (step-by-step plan) and [`00_CLASS_LABELS.md`](00_CLASS_LABELS.md)
(mask value 2 = spherocyte, 3 = echinocyte).

Everything here is generated from `outputs/` (read only) by the scripts in `code/`; no number is typed by hand.

```
pip install -r requirements.txt python-docx        # matplotlib, pandas, scipy, scikit-image, torch (CPU is enough)
python artifacts/code/make_tables.py               # artifacts/tables, artifacts/numbers.json
python artifacts/code/make_figures.py              # artifacts/figures (all but the image figures)
python artifacts/code/make_qualitative_figures.py  # image figures; needs data/processed (server)
python artifacts/code/verify_artifacts.py          # checks; last line OK
```

| Guide | Purpose |
|---|---|
| `00_START_HERE.md` | ordered plan with checkboxes |
| `00_CLASS_LABELS.md` | class names and which files use the old names |
| `01_EXPERIMENT_STATUS.md` | what was run and verified |
| `02_RESULTS_AND_CLAIMS.md` | findings and claim audit |
| `03_MANUSCRIPT_CHANGES.md` | manuscript edits, paragraph by paragraph, with drafts |
| `04_SI_CHANGES.md` | SI edits with drafts |
| `05_TABLE_FIGURE_MAP.md` | old to new tables and figures |
| `06_METHODS_AS_IMPLEMENTED.md` | the pipeline as coded, with draft method text |
| `07_REMAINING_WORK.md` | open items |
