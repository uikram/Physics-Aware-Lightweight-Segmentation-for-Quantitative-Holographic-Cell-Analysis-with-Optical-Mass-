# Publication versions of the manuscript and SI

Built from the final-review Word files (`artifacts/manuscript_in_prog/Claudes_Update/`) by

    python artifacts/code/audit_claims.py          # recomputes the checkable claims -> artifacts/audit/claims_check.json
    python artifacts/code/make_manuscript_figures.py
    python artifacts/code/patch_static_figures.py  # Fig. 1 label corrections
    python artifacts/code/build_final_docs.py

| File | Content |
|---|---|
| `revised_manuscript_final.docx` | main text; every changed or new run highlighted yellow (earlier yellow kept) |
| `supplementary_information_final.docx` | rebuilt SI, sections S-I to S-IX with descriptive titles |
| `*_review_copy.docx` | same documents with this round's changes in green, so they can be told apart from earlier revisions |
| `Supplementary_Data_1.xlsx` | per-seed, per-day values cited as Supplementary Data 1 |
| `change_log.md` | every edit (old -> new) and the source file/key of every number written |
| `*.pdf` | LibreOffice renderings for proofreading |

## What still needs your data (data/processed)

Fig. 6 and Figs. S-4, S-5 and S-6 are placeholders. On the machine that has `data/processed`, run

    python artifacts/code/make_data_figures.py --seed 42

It writes `artifacts/figures/manuscript/{Fig6_qualitative,FigS4_gradient_alignment,FigS5_storage_images,FigS6_component_fate}.png`
(in Word: right-click the placeholder > Change Picture) and `artifacts/audit/server_numbers.json`, which recomputes the
numbers listed at the end of `change_log.md` that cannot be checked from `outputs/` (unannotated fraction, day-15
foreground fraction, pixel confusion, bridged cells, day-15 component fate). Update those sentences if the values differ.

## Main-text structure after this revision

Tables: 1 same-rank comparison (+ EdgeSAM (r=32)); 2 LoRA vs full fine-tuning (seed 42); 3 loss ablation;
4 benchmark (rank 8); 5 dry-mass agreement and storage trends. The former per-class tables (old Tables 4, 5) are Table S-3,
the per-day storage table (old Table 7) is Table S-4, MobileSAM storage is Table S-5, date-clustered statistics Table S-6.
Figures 1-8 keep their numbers; Fig. 5 no longer uses a second y-axis; the SI radar chart was removed.
