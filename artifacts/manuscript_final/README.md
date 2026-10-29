# Manuscript and SI: current versions (internal folder)

**To send to Prof. Moon: only the files in `for_submission/`** (the two Word files and their PDFs). Everything else in
this folder is internal working material (change log with file paths, build notes) and should not be sent.

| File | Content |
|---|---|
| `for_submission/revised_manuscript_final.docx` | main text; revised content highlighted yellow |
| `for_submission/supplementary_information_final.docx` | SI, sections S-I to S-IX, Tables S-1 to S-7, Figs. S-1 to S-6 |
| `revised_manuscript_final.docx`, `supplementary_information_final.docx` | working copies (identical to `for_submission/`) |
| `change_log.md` | internal: every edit (old -> new) and the source file/key of every number |

The figures that need the processed phase maps (Fig. 6, Figs. S-4 to S-6) were generated with
`artifacts/code/make_data_figures.py --seed 42` and inserted by hand. Fig. S-6 must be redrawn once more after the y-axis fix:

    python artifacts/code/make_data_figures.py --seed 42 --only-figS6

then replace the Fig. S-6 picture in the SI (right-click > Change Picture).

Text and table corrections are applied with `python artifacts/code/refresh_final_docs.py`, which keeps every inserted
picture; do not rerun `build_final_docs.py`, which would restore placeholders.
