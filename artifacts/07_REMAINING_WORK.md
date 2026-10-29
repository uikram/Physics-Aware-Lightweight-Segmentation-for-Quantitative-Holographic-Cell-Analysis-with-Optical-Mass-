> **SUPERSEDED (29 Oct 2026).** The current manuscript and SI are in `artifacts/manuscript_final/`. This file predates the final audit and contains statements that are WRONG and must not be reused: (1) a "small storage-dependent dry-mass bias": the date-clustered test does not support it (EdgeSAM r=32 p = 0.08 to 0.19, MobileSAM r=8 p = 0.63 to 0.98; the p = 0.008 to 0.05 values come from a test that treats the 66 images as independent); (2) "dry mass remained constant": a decrease is only *not statistically supported*; (3) latency of EdgeSAM paired with r = 32: only the rank-8 checkpoints were timed. See `artifacts/manuscript_final/change_log.md`.

# 07. What remains to be done, and what could not be verified

Nothing here requires retraining. Items are ordered by when you need them.

## A. Must do (the revision is not complete without these)

| # | Task | Where | How |
|---|---|---|---|
| A1 | Generate the three image figures (old Fig. 6, S-4, S-5) | server with `data/processed/` | `python artifacts/code/make_qualitative_figures.py` (writes FQ1, FQS1, FQS2 into `artifacts/figures`). Then look at FQ1 and write Sec. 4.7 from what it shows. |
| A2 | Edit the manuscript text | Word | `03_MANUSCRIPT_CHANGES.md`, in the order of `00_START_HERE.md` |
| A3 | Replace all tables and figures | Word | `05_TABLE_FIGURE_MAP.md`; copy tables from `tables/ALL_TABLES.docx` |
| A4 | Rewrite the SI | Word | `04_SI_CHANGES.md` |
| A5 | Redraw Fig. 1 | drawing tool | two phase branches (background-referenced rad to the loss, z-scored to the network); remove SAM prompt encoder / mask decoder |
| A6 | Final string search and verification | | list at the end of `03`; `python artifacts/code/verify_artifacts.py` must print OK |
| A7 | Ask the professor to approve the changed storyline (B below) before polishing the text | | |

## B. Decisions for the professor (the corrected results change the message)

1. **Main narrative.** The old paper selected EdgeSAM as "the most balanced" model. The corrected results do not support that:
   MobileSAM is more accurate and more stable at r2 to r16; EdgeSAM is cheaper and separates cells better. The drafts present both
   (preferred configurations) without a winner. Confirm.
2. **The physics-aware loss claim** becomes "lower IPE than Dice-only in all 15 configurations, large where Dice-only is unstable,
   small for MobileSAM; individual terms not separable with three seeds". Confirm that this is acceptable as the main evidence.
3. **Biology.** The old "dry-mass decline explained by area" is withdrawn. The new message is "area decreases, dry mass is
   conserved (annotation); predicted masks add a small bias". Confirm, and check the interpretation with references [24], [25],
   [44] (spherocytic transformation without loss of haemoglobin).
4. **Disclosure of the rank choice.** The drafts say that the preferred ranks were chosen after viewing the rank sweep on the
   evaluation images. Confirm the wording.
5. **Table count.** Main text would have 8 or 9 tables and 10 figures. If too many: T10 into the text, F11 or F5 to the SI.

## C. Could not be verified from the repository (ask the data provider or check on the server)

| # | Item | Why it matters | What to do |
|---|---|---|---|
| Q-A | Pixel pitch and field of view: Sec. 3.1 says 1024 x 1024 camera pixels, field of view about 140 um; the maps are 768 x 768 and the code uses 0.1441 um/px (field about 111 um) | dry mass scales with the pixel area | The day-0 annotation median (27.8 pg, 0.93 x MCH) and cell area (46.8 um2) are plausible with 0.1441 um, which argues for the current value. Ask the data provider for the pixel pitch of the 768 maps and correct the Sec. 3.1 sentence. |
| Q-B | Why the zero level differs between the date-named and the `X_*` maps (D-03, Q-2, still open) | justifies the background referencing | Ask; the manuscript can describe the referencing without the answer. |
| Q-C | "Laterally touching cells and unresolved clusters were most frequent at day 15" (Sec. 3.1) | used to explain the day-15 dry-mass bias | Check on FQ1 / the annotation (the highest annotated cell-mean area, 59.3 um2, is at day 15, T9, which is consistent). |
| Q-D | LoRA module list of MobileSAM (fused qkv) | Sec. 2.4 wording | On the server: build `MobileSAMSeg`, call `inject_lora(8, 8)`, print the names of the replaced modules once. |
| Q-E | Encoder output resolution of EdgeSAM at 256 px input | possible explanation of its larger boundary error (S12) | Print the encoder output shape; only then mention resolution in the Discussion. |
| Q-F | Missing file `outputs/multiseed/runs/edge_sam_abl_dice_only_r4__s43/predictions_last.npz` | the cell analysis of this run exists and is used; only the label check skips it | Copy it from the server (SHA-256 recorded in `outputs/analysis/cell/edge_sam_abl_dice_only_r4__s43.json`: `904b93ec...2bb8`). |
| Q-G | Registration statistics (valid fraction of the float grid, `dataset_report.json`) | SI data section | Copy `data/processed/dataset_report.json` from the server. |

## D. Known limitations of the results (state them, do not hide them)

* Validation = evaluation set; preferred ranks chosen after inspection; no independent test cohort.
* n = 3 seeds: paired 95% intervals are wide; most differences are descriptive.
* Full fine-tuning: one seed.
* 74 of 256 training images had no float map; masks were transferred from 8-bit exports by registration.
* 3.0% of annotated cells excluded as phase-wrapped.
* One GPU (RTX A5000), batch 1, MobileSAM at 1024 px vs 256 px for the others.
* The benchmark and the ONNX gate used the seed-42 rank-8 `best.pt` (D-26); for EdgeSAM it is the last epoch (epoch 50), for
  MobileSAM epoch 38. Timing does not depend on the weights; the gate only compares ONNX with PyTorch. Mention only if asked.

## E. Optional, if time allows (not needed for the claims)

* An independent selection split (for example by holding out some training images) would remove the rank-choice caveat; this
  would require retraining and is not part of this revision.
* More seeds for EdgeSAM r8 and MobileSAM r32 would quantify the failure rate better.
