> **SUPERSEDED (29 Oct 2026).** The current manuscript and SI are in `artifacts/manuscript_final/`. This file predates the final audit and contains statements that are WRONG and must not be reused: (1) a "small storage-dependent dry-mass bias": the date-clustered test does not support it (EdgeSAM r=32 p = 0.08 to 0.19, MobileSAM r=8 p = 0.63 to 0.98; the p = 0.008 to 0.05 values come from a test that treats the 66 images as independent); (2) "dry mass remained constant": a decrease is only *not statistically supported*; (3) latency of EdgeSAM paired with r = 32: only the rank-8 checkpoints were timed. See `artifacts/manuscript_final/change_log.md`.

# START HERE: step-by-step revision plan

The corrected code and the verified outputs are the source of truth; the uploaded manuscript is not. Everything you need is in this
folder. Work through the steps in order and tick them off. Each step names the file that tells you exactly what to do.

```
artifacts/
  00_START_HERE.md              this plan
  00_CLASS_LABELS.md            2 = spherocyte, 3 = echinocyte; which files use old names (read first, 2 min)
  01_EXPERIMENT_STATUS.md       what was run and verified (all done; nothing to rerun)
  02_RESULTS_AND_CLAIMS.md      the 10 findings + audit of 31 old claims (keep / revise / remove)
  03_MANUSCRIPT_CHANGES.md      paragraph-by-paragraph edits with draft text (the main work file)
  04_SI_CHANGES.md              SI edits with draft text
  05_TABLE_FIGURE_MAP.md        old -> new table/figure numbers and file paths
  06_METHODS_AS_IMPLEMENTED.md  preprocessing, background correction, loss, models, dry mass exactly as coded (+ draft text)
  07_REMAINING_WORK.md          what is left, decisions for the professor, what could not be verified
  numbers.json                  headline numbers with their source file
  tables/                       main/ (T1-T10), si/ (S1-S14), raw/, ALL_TABLES.md, ALL_TABLES.docx
  figures/                      main/ (F3-F11), si/ (FS1-FS8)   (+ FQ* after Step 6)
  code/                         common.py, make_tables.py, make_figures.py, make_qualitative_figures.py, verify_artifacts.py
```

## The story in five lines (agree on this before editing)

1. Contribution: a physics-aware loss that treats the predicted boundary as the measurement domain of the integrated phase, on
   background-referenced 32-bit phase maps, with LoRA-adapted lightweight encoders.
2. Evidence: the loss lowers the integrated-phase error vs Dice-only in all 15 architecture x rank combinations, most where
   Dice-only is unstable, and reduces runs that lose a class.
3. MobileSAM (r8) vs EdgeSAM (r32): MobileSAM more accurate and stable (Dice 0.810, IPE 1.32%); EdgeSAM better instance separation
   (AJI 0.335) at half the latency and one eighth of the memory. Same-rank comparison and every seed reported.
4. Optical mass: uncalibrated per-image dry mass within 2.9 to 3.9% of the annotation-derived value.
5. Biology: stored cells lose 44% of projected area but no dry mass (annotation); predicted masks reproduce area and composition
   but add a small dry-mass bias from merged cells.

## Steps

- [ ] **Step 0. Labels (5 min).** Read `00_CLASS_LABELS.md`. From now on, take class-specific numbers only from `artifacts/`.
- [ ] **Step 1. Understand the results (30 min).** Read `02_RESULTS_AND_CLAIMS.md` section A, then look at Figs. F3, F4, F5, F6,
      F9, F10 in `figures/main`. Read section B (claim audit) so you know which old sentences go.
- [ ] **Step 2. Professor sign-off (meeting).** Show the five-line story above and the decisions in `07_REMAINING_WORK.md` B.
      Do not polish text before this; the decisions change the Discussion.
- [ ] **Step 3. Methods (Sections 2 and 3).** Follow `03_MANUSCRIPT_CHANGES.md`, "2. Method" and "3. Experiments", using the
      draft paragraphs in `06_METHODS_AS_IMPLEMENTED.md`. Priorities: 2.2 (32-bit data, registration, background referencing,
      no flips), 2.3 (encoders + task decoders, not SAM's decoder), 2.4 (α = r, layers, merge), 2.5 (remove "complementary"),
      2.6 (no MCH calibration, wrapped cells), 3.1 (182 + 66 images, delete "no cell exceeded 2π"), 3.2 (no sampler, seeds,
      last epoch), 3.4 (FP32 only, MobileSAM ONNX works), 3.5, 3.6.
- [ ] **Step 4. Results (Section 4).** Rebuild the section in the new order (`03`, "4. Results" mapping table). For each
      subsection: paste the table from `tables/ALL_TABLES.docx`, insert the figure PNG/PDF, paste and edit the draft paragraph,
      delete the old paragraphs listed. Do 4.1 to 4.6 and 4.8 now; 4.7 after Step 6.
- [ ] **Step 5. Supplementary Information.** Follow `04_SI_CHANGES.md` (new structure S-I to S-XI). Delete old Table S-1 and
      Figs. S-1 to S-3. Attach `tables/si/S6_all_runs.csv` as a data file.
- [ ] **Step 6. Server job (only thing that needs the server).** On the machine with `data/processed/`:
      `python artifacts/code/make_qualitative_figures.py`, then commit the FQ1, FQS1, FQS2 files. While there, answer
      `07` C items Q-D, Q-E, Q-F, Q-G. Then write Sec. 4.7 from FQ1.
- [ ] **Step 7. Discussion and Conclusion.** Drafts D1 to D6 and the Conclusion in `03`. Adjust to the professor's decisions.
- [ ] **Step 8. Abstract and Introduction last.** Draft abstract and the contributions paragraph in `03`. The abstract must match
      the final numbers of Sections 4.1 to 4.8.
- [ ] **Step 9. Fig. 1.** Redraw (two phase branches; no SAM prompt encoder / mask decoder), `07` A5.
- [ ] **Step 10. Consistency pass.** Run the string search at the end of `03` on the Word file; every remaining number must be
      in `tables/` or `numbers.json`. Renumber tables and figures; fill in the "Final numbering" table in `05`.
      Run `python artifacts/code/verify_artifacts.py` (must end with OK).

## If you change anything in outputs/ or the code

```
python artifacts/code/make_tables.py        # tables, ALL_TABLES.md/.docx, numbers.json   (seconds, CPU)
python artifacts/code/make_figures.py       # all figures that do not need the phase maps
python artifacts/code/verify_artifacts.py   # provenance, label and freshness checks; last line OK
```
Never edit a table by hand: `verify_artifacts.py` reports any table that differs from a fresh build.
