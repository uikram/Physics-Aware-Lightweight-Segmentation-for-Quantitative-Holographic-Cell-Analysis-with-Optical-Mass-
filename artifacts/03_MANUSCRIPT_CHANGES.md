> **SUPERSEDED (29 Oct 2026).** The current manuscript and SI are in `artifacts/manuscript_final/`. This file predates the final audit and contains statements that are WRONG and must not be reused: (1) a "small storage-dependent dry-mass bias": the date-clustered test does not support it (EdgeSAM r=32 p = 0.08 to 0.19, MobileSAM r=8 p = 0.63 to 0.98; the p = 0.008 to 0.05 values come from a test that treats the 66 images as independent); (2) "dry mass remained constant": a decrease is only *not statistically supported*; (3) latency of EdgeSAM paired with r = 32: only the rank-8 checkpoints were timed. See `artifacts/manuscript_final/change_log.md`.

# 03. Manuscript: what to change, where, and with what (paragraph by paragraph)

Applies to `revised_manuscript_fv_-_ui.docx` (the uploaded file). Locations are given as **section, paragraph number within the
section, and the first words of the paragraph**, so they can be found with Ctrl+F. Every number in the replacement text comes from
`artifacts/tables` (table id in brackets). Class names: mask value 2 = spherocyte, 3 = echinocyte (`00_CLASS_LABELS.md`).

Action codes: **KEEP** (no change), **EDIT** (small change, listed), **REWRITE** (replace with the draft text), **DELETE**,
**MOVE** (to the SI or another section), **NEW** (insert).

**Start with the QUICK CHECKLIST below (everything to delete and modify); the detailed sections after it give the replacement text.**

Drafts are deliberately plain; edit them for style, but keep the numbers and the hedging ("not separable", "in this data set").

---

## QUICK CHECKLIST: everything to DELETE and everything to MODIFY

A one-page list of every change, in manuscript order. Tick each box in your copy. The sections further down give the location
(first words of the paragraph), the replacement text and the source table for each item. Numbers in brackets are artifact ids
(T = main table, S = SI table, F = figure, `05_TABLE_FIGURE_MAP.md`).

### Part 1. DELETE (these statements are wrong or unsupported with the corrected data)

**Abstract**
- [ ] The whole old abstract (every result sentence in it is outdated). Replacement: draft abstract below.

**Section 2 (Method)**
- [ ] 2.2: "converted individually to 8-bit grey-level images, which constitute the dataset used in this study".
- [ ] 2.2: "horizontal and vertical flipping, and translation" (only the three 90 degree rotations are used).
- [ ] 2.2: "Spatial resizing and cropping" (there is no cropping; keep only resizing).
- [ ] 2.3: "MobileSAM and EdgeSAM retain the mask-decoding pipeline of the original SAM" (only the image encoders are used).
- [ ] 2.5.5: "As demonstrated by the ablation experiments, the individual constraints are complementary: PMC provides ..., BGA
      anchors ..., and IPP ensures ...".
- [ ] 2.6: "with the phase scale calibrated to the mean corpuscular hemoglobin (MCH) reference".
- [ ] 2.6: "No background correction is applied when computing the segmentation loss or the global Integrated-Phase Error (IPE);
      both operate directly on the raw phase values. ..." (whole sentence pair).

**Section 3 (Experiments)**
- [ ] 3.1: "The dataset comprises 1,090 densely annotated images" and "divided into 1,024 training images and 66 validation images".
- [ ] 3.1: "The quantitative phase maps were used directly, without further normalization, for integrated-phase and dry-mass
      calculations".
- [ ] 3.1 (second paragraph): "and no annotated cell exceeded a phase of 2π" (61 of 2,049 annotated cells, 3.0%, do).
- [ ] 3.2: both WeightedRandomSampler sentences ("At the data-loading level, a WeightedRandomSampler ...").
- [ ] 3.3.2: "For the selected EdgeSAM configuration (r = 8), the per-cell median IPE was 3.04% (90th percentile 11.57%), with 73.7%
      of ground-truth cells successfully matched" (new values go to Results 4.5).
- [ ] 3.4: "under FP32 and FP16 precision".
- [ ] 3.4: "MobileSAM was excluded from ONNX benchmarking ... Native PyTorch inference was therefore used for MobileSAM ..." (MobileSAM
      is exported and passes the equivalence gate).

**Section 4.1 (segmentation accuracy)**
- [ ] Every sentence selecting EdgeSAM: "most balanced", "the only model that simultaneously maintained ...", "EdgeSAM was selected
      as the primary architecture ...", "particularly suitable for the proposed framework".
- [ ] "MobileSAM ... showed relatively poor instance separation and boundary quality, with an AJI of 0.249 and Boundary F1 of 0.644".
- [ ] "MobileNet-UNet completely failed to detect echinocytes and stomatocytes" (it is spherocytes (2) and stomatocytes (4)).
- [ ] "Visual comparisons further corroborated ... MobileSAM frequently merged neighboring cells ..., EdgeSAM provided
      individualized cellular contours" (rewrite only after looking at the new FQ1).
- [ ] The radar-chart paragraph ("The aggregate comparison across criteria is shown in Fig. S-2 ...").
- [ ] "LoRA rank was selected for EdgeSAM from a sweep ... Rank 8 gave the best balance ...".
- [ ] Old Table 1 and old Fig. 3.

**Section 4.2 (LoRA vs full fine-tuning)**
- [ ] "For EdgeSAM, full fine-tuning reduced the mean Dice from 0.727 to 0.557 and caused complete collapse of the echinocyte class
      (Dice: 0.776 to 0.000), while increasing the IPE from 3.63% to 5.08%".
- [ ] "Among the evaluated configurations, LoRA-adapted EdgeSAM provided the most favorable balance ...".
- [ ] Old Table 2; old Fig. 4 (moves to the SI as F7).

**Section 4.3 (loss ablation)**
- [ ] The whole paragraph "Note that the EdgeSAM r = 8 values in Table 3 (Dice 0.740, IPE 3.04%) differ slightly ..." (run aliasing,
      MobileSAM non-convergence).
- [ ] Table 3 caption sentence "The ablation runs were trained independently; therefore, comparisons are made within this table".
- [ ] Old Table 3 and old Table 4 (per-class EdgeSAM ablation, including "Dice = 0.018" and "1073.58").
- [ ] "Adding PMC alone substantially degraded both boundary fidelity and integrated-phase preservation, whereas the addition of BGA
      restored ..." and "restored its Dice score from 0.018 to 0.643".
- [ ] "The complete objective incorporating IPP achieved the lowest global IPE (3.04%) and the highest Boundary F1 (0.953) ... IPE
      decreased from 15.52% ... to 3.01%".
- [ ] "For MobileSAM, the Dice+PMC+BGA configuration achieved a lower IPE (2.37%) ... did not converge (best checkpoint at epoch 1)".
- [ ] The phase-quartile paragraph ("Misses concentrated in the lowest-phase quartile for all models (18-33%) ... 33.3% to 28.7% ...
      approximately 30%").
- [ ] The robustness paragraph ("EdgeSAM Dice 0.739 to 0.720, 0.735, and 0.706 ... 3.55% to 3.92% ...") (new version goes to 4.5).
- [ ] The closing sentence "Overall, the ablation results demonstrate complementary roles of the three physics-aware components".
- [ ] Old Fig. 5 (dual axis).

**Section 4.4 (integrated phase)**
- [ ] "EdgeSAM achieved a global IPE of 3.63%, while MobileSAM and MobileNet-UNet achieved 2.57% and 2.60%".
- [ ] "MobileNet-UNet completely failed to detect echinocytes and stomatocytes, resulting in 100% class-specific IPE".
- [ ] "MobileSAM ... frequently merging adjacent cells ... Among the evaluated architectures, EdgeSAM therefore provided the most
      balanced measurement domain".
- [ ] Old Table 5.

**Section 4.5 (efficiency)**
- [ ] "reducing the median latency from 13.36 ms in PyTorch FP32 to 4.33 ms in ONNX FP16 and increasing throughput from 74.82 to
      228.33 FPS, while requiring only 16.4 MB".
- [ ] "MobileNet-UNet achieved the highest throughput (376.10 FPS with ONNX FP16)".
- [ ] "MobileSAM ... could not be reliably accelerated using ORT because of unsupported decoder operations".
- [ ] "EdgeSAM provided the most balanced overall performance among the evaluated architectures".
- [ ] Old Table 6 (all FP16 rows, 16.4 MB, the MobileSAM-not-in-ONNX caption) and the Fig. S-3 sentence.

**Sections 4.6 to 4.8 (qualitative, storage)**
- [ ] 4.6: "the LoRA-adapted EdgeSAM successfully delineated individual RBCs" and "the predicted masks provided individualized cellular
      regions rather than merged foreground regions" (write again from FQ1).
- [ ] 4.6: "MobileNet-UNet recovered the discocytes and spherocytes but missed echinocytes and stomatocytes" (wrong class names).
- [ ] 4.7: "Geometric and phase-derived quantities were extracted ... selected LoRA-adapted EdgeSAM (r = 8)".
- [ ] 4.7: "Stomatocytes remained a substantial minority class, rising from 10% at day 0 to a maximum of 26% at day 19 and accounting
      for 15% at day 47 (Table 7)".
- [ ] 4.8: "the projected cell area ... increased from 63.1 µm² at day 0 to a peak of 83.6 µm² at day 15, before declining to 43.4
      µm² by day 47 (Spearman ρ = −0.67 ...)".
- [ ] 4.8: "Phase-derived dry-mass estimates likewise fluctuated ... significant overall decline ... (Spearman ρ = −0.40, p = 0.0009);
      a partial correlation ... (partial ρ = −0.15, p = 0.23) ...".
- [ ] 4.8: "the substantial change in the two-dimensional cellular footprint was not accompanied by a proportional change in
      phase-derived dry mass" (rewrite with the annotation result).
- [ ] 4.8: "(r = 0.966, CCC = 0.96, MdARE = 3.35%, Bland-Altman bias = −0.12 pg)".
- [ ] Old Table 7 (incl. "1,974 cells"); old Figs. 6, 7, 8.

**Section 5 (Discussion)**
- [ ] "reduced the global IPE from 5.22% with Dice-only optimization to 3.04%, while increasing Boundary F1 from 0.938 to 0.953 ...
      MobileNet-UNet ... from 15.52% to 3.01%".
- [ ] "In particular, PMC alone degraded boundary accuracy and minority-class segmentation, whereas the addition of BGA restored
      boundary localization".
- [ ] "For MobileSAM, the full-objective run did not converge, so the effect of IPP on this architecture could not be established"
      (paragraphs 2 and 7).
- [ ] "(r = 0.966, CCC = 0.96, MdARE = 3.35%, Bland-Altman bias = −0.12 pg) therefore provides direct evidence ...".
- [ ] "LoRA-adapted EdgeSAM maintained non-zero segmentation ... adequate instance separation, and low IPE".
- [ ] "The LoRA rank analysis further showed that r = 8 provided the best overall balance ...".
- [ ] The whole efficiency paragraph ("ONNX FP16 inference ... 4.33 ms ... 228.3 FPS ... 16.4 MB ... limited ONNX compatibility").
- [ ] "Phase-derived dry mass fluctuated with a significant overall decline that was largely ... explained by the concurrent area
      change" and "with the phase scale calibrated to the MCH reference (Section 2.6)".
- [ ] "The full physics-aware objective also reduced instance-separation performance relative to Dice-only optimization for EdgeSAM"
      (not true at r = 32: AJI 0.335 vs 0.329).

**Section 6 (Conclusion)**
- [ ] "for MobileSAM, the full-objective run did not converge, so no benefit could be established".
- [ ] "Among the evaluated architectures, LoRA-adapted EdgeSAM provided the most balanced performance ...".
- [ ] "with strong agreement between predicted-mask and annotation-derived measurements (r = 0.966)".

**Supplementary Information**
- [ ] S-I: the whole text ("rank 8 provided the best overall balance ... selected as the optimal LoRA rank"), Table S-1, Fig. S-1.
- [ ] S-II: the radar chart Fig. S-2 and its paragraph; Fig. S-3 (FP16, "dashed ring marks the selected configuration") and its
      paragraph.
- [ ] S-III: "EdgeSAM maintained high Boundary F1 scores at ranks of 8 and above, ranging from 0.946 to 0.952, whereas ... 0.905 at
      r = 2 and 0.918 at r = 4 ..." and the conclusion drawn from it.

### Part 2. MODIFY (keep the content, change it as stated)

**Title, abstract, keywords**
- [ ] Title: keep.
- [ ] Abstract: replace with the draft abstract below (numbers from T2, T3, T5/S5, T7, T8, T10).
- [ ] Index terms: keep; optionally add "Low-Rank Adaptation".

**Class names everywhere (do this as one pass)**
- [ ] Every statement, table header and figure legend that names spherocytes or echinocytes: use mask value 2 = spherocyte,
      3 = echinocyte (`00_CLASS_LABELS.md`). In particular: MobileNet-UNet loses **spherocytes and stomatocytes**; EdgeSAM loses
      **spherocytes** in its failed seeds and in full fine-tuning; MobileSAM full fine-tuning loses spherocytes, echinocytes and
      stomatocytes.
- [ ] Replace the "Dice_echi / Dice_stom" columns by all four class columns with mask values (T2, T3, T6).

**1. Introduction**
- [ ] Paragraphs 1 to 3 (holography history): optional condensing into one paragraph (redundant, not wrong).
- [ ] Paragraph 7: describe MobileSAM and EdgeSAM only as distilled SAM image encoders [33, 34].
- [ ] Paragraph 11: "Raw, unnormalized phase values are supplied directly to this objective ..." -> "The objective acts on the 32-bit
      phase in radians after per-image background referencing, never on the standardised network input ...".
- [ ] Paragraph 12: replace by the contributions paragraph (states the main contribution as the professor asked).

**2. Method**
- [ ] 2.1 text: "standardization of the quantitative phase maps" -> "background referencing and standardisation of the 32-bit phase maps".
- [ ] Fig. 1 + caption: two phase branches (background-referenced rad -> loss; z-scored -> network); no SAM prompt encoder or mask
      decoder; "raw unnormalized phase" -> "background-referenced phase (rad)".
- [ ] 2.2: rewrite with the 06 draft: float32 maps 768 x 768, ECC registration of the 8-bit masks, unannotated pixels excluded,
      per-image background referencing (0.66 to 2.50 rad offsets), per-image z-score for the input, only 90 degree rotations.
- [ ] 2.3: rewrite with the 06 draft: pretrained encoders (TinyViT, RepViT) + task decoders trained from scratch; 1-channel stem
      from the mean of the RGB filters.
- [ ] 2.4: after Eq. (1) add α = r; LoRA layers as implemented (MobileSAM fused qkv projections; EdgeSAM and MobileNet-UNet all
      non-grouped encoder convolutions / linear layers); trainable stem, normalisation, decoder; merge at deployment (cost
      independent of r); "ranks specified in the experimental protocol" -> r ∈ {2, 4, 8, 16, 32}.
- [ ] 2.5.1: add "pixels without annotation are excluded from all sums".
- [ ] 2.5.2: margin = 0.1 rad; "phase" -> "background-referenced phase".
- [ ] 2.5.3: "raw phase map" -> "background-referenced phase map"; forward differences; average over annotated pixels.
- [ ] 2.5.4: φ = background-referenced phase in rad; ε = 10⁻⁴; add the c × A sentence (why referencing matters for S).
- [ ] 2.5.5: add λ1 = 0.1, λ2 = 0.05, λ3 = 0.1; terms averaged over images with at least one annotated cell, computed in float32;
      end with "The contribution of each term is examined in Section 4.3."
- [ ] 2.6: Eq. (9) φ = background-referenced; last paragraph replaced by the draft (area 15 to 200 µm², Crofton perimeter,
      background median > 5 px from the mask, physical constants only, no calibration, wrapped-cell rule).

**3. Experiments**
- [ ] 3.1 paragraph 1: 182 training images with float32 maps (728 views with rotations) + 66 validation images (6 per day, 11 days);
      2,049 annotated cells, 61 (3.0%) wrapped and excluded from phase-derived results [T1]; class list by mask value.
- [ ] 3.1 paragraph 2: check pixel count / field of view with the data provider (768 x 768 at 0.1441 µm = about 111 µm vs "1024 x
      1024 ... about 140 µm"; `07` Q-A).
- [ ] 3.2: rewrite with the 06 draft: AdamW + cosine, 50 epochs, learning rates and batch sizes, FP16 forward / float32 loss, plain
      shuffling, seeds 42/43/44 for 180 LoRA runs, full fine-tuning one seed, last-epoch checkpoint (no selection on the evaluation
      images).
- [ ] 3.3.1: Dice/IoU pooled over pixels then macro-averaged; native 768 grid; definition of a lost class (Dice < 0.05); mean ± SD
      over three seeds.
- [ ] 3.3.2: IPE on background-referenced phase; per-cell IPE definition only (IoU ≥ 0.5 matching, background plane, wrapped cells
      excluded); numbers move to 4.5.
- [ ] 3.4: protocol v5 (batch 1, native input sizes, strict FP32, 50 + 500 passes, 3 seeds, 3 ONNX sessions, P50/P99/FPS/Peak
      VRAM); 1024 vs 256 px caveat; ONNX equivalence gate passed by all models.
- [ ] 3.5: rewrite with the draft (three experiment groups, all with three seeds; preferred MobileSAM r8 and EdgeSAM r32 + same
      rank r8; ablation at every rank; full objective = rank-sweep run; ranks chosen after inspection, no separate selection set).
- [ ] 3.6: add early = days 0 to 8 (18 images), late = days 37 and 47 (12 images); seed-paired differences with 95% t-intervals
      used descriptively; r and CCC reported but agreement judged by MdARE, bias and limits of agreement.

**4. Results (reorganise into 4.1 to 4.8; new tables and figures)**
- [ ] Results intro: rewrite (what is compared, three seeds, final-epoch models, preferred + same-rank + baseline).
- [ ] 4.1 Segmentation accuracy: Table 1 = T2, Table 2 = T3, Fig. 3 = F3; draft text.
- [ ] 4.2 LoRA rank and training stability (new): Table 3 = T4, Fig. 4 = F4, Fig. 5 = F5; draft text.
- [ ] 4.3 Loss ablation: Table 4 = T5, Fig. 6 = F6; draft text (15/15 direction, 3/15 interval, IPP 13/15, fewer lost classes).
- [ ] 4.4 LoRA vs full fine-tuning: Table 5 = T6 (F7 to the SI); draft text.
- [ ] 4.5 Integrated phase and optical mass: Table 6 = T7, Fig. 7 = F9; draft text (IPE, per-cell IPE, dry-mass MdARE / bias /
      LoA, why r is low, background reference, robustness, error strata).
- [ ] 4.6 Computational efficiency: Table 7 = T8 (F8 to the SI); draft text (FP32, PyTorch and ONNX, rank independent).
- [ ] 4.7 Qualitative results: Fig. 8 = FQ1 (generate on the server first); describe what the figure shows, three models.
- [ ] 4.8 Storage: Table 8 = T9 (+ T10 as Table 9 or in the text), Fig. 9 = F11, Fig. 10 = F10; draft text (annotation: area
      −44%, dry mass unchanged; models reproduce area and composition but add a dry-mass bias at days 5 to 19).

**5. Discussion and 6. Conclusion**
- [ ] Discussion: replace paragraphs 1 to 6 by drafts D1 to D6; keep the amplitude-image paragraph of the limitations; paragraph 8
      keep with "establish" -> "support".
- [ ] Conclusion: replace by the draft.

**Back matter**
- [ ] Author contributions: check "benchmarking experiments" wording.
- [ ] Data availability: add the code / results repository if it will be public.
- [ ] References: check [24], [25], [44] for the "area lost, haemoglobin retained" interpretation; remove references that become
      uncited.

**Supplementary Information** (details in `04_SI_CHANGES.md`)
- [ ] New structure S-I to S-XI with tables S1 to S14 and figures FS1 to FS8, F7, F8.
- [ ] S-III keep the first two sentences (why boundaries matter); replace the rest by the error-strata text (S-V).
- [ ] S-IV text keep; Fig. S-5 replaced by FQS2; Fig. S-4 replaced by FQS1 (server).
- [ ] Attach `tables/si/S6_all_runs.csv` as a data file.

**Last**
- [ ] Renumber tables and figures; fill in the table in `05_TABLE_FIGURE_MAP.md`.
- [ ] Run the string search at the end of this file; run `python artifacts/code/verify_artifacts.py` (OK).


---

## Title, abstract, keywords

| Where | Action | What |
|---|---|---|
| Title | KEEP | "Physics-Aware Lightweight Segmentation for Quantitative Holographic Cell Analysis with Optical Mass Preservation" is still supported (global IPE 1.3 to 3.0%, image-level dry-mass MdARE 2.9 to 3.9%). |
| Abstract | REWRITE | Every result sentence of the old abstract is wrong or unsupported (02 B-1 to B-7). Use the draft below. |
| Index Terms | KEEP | Optionally add "Low-Rank Adaptation". |

**Draft abstract (about 250 words).**
> Digital holographic microscopy provides label-free quantitative phase images from which cellular dry mass is obtained by
> integrating the phase over each cell. Segmentation networks trained with geometric overlap losses do not constrain this integral,
> so boundary errors propagate into optical-mass estimates. We propose a physics-aware lightweight segmentation framework in which
> the predicted cell boundary is treated as the measurement domain of the integrated phase. A Physics-Aware Phase Consistency Loss
> adds phase-mask contrast, boundary-gradient alignment and integrated-phase preservation terms, computed on background-referenced
> 32-bit phase maps, to a class-weighted Dice loss, and low-rank adaptation (LoRA) adapts lightweight foundation-model encoders to
> single-channel phase images. On red blood cell phase maps acquired over 47 days of storage, MobileSAM and EdgeSAM were compared at
> five LoRA ranks with three random seeds per configuration (180 training runs including the loss ablation). Relative to Dice-only
> training, the physics-aware objective lowered the global integrated-phase error in all 15 architecture-rank combinations and
> reduced the number of runs that lost a morphology class. MobileSAM (rank 8) reached a mean Dice of 0.810 ± 0.006, a boundary F1 of
> 0.802 and a global integrated-phase error of 1.32 ± 0.09%; EdgeSAM (rank 32) gave better instance separation (aggregated Jaccard
> index 0.335 vs 0.262) at about half the latency (10.6 vs 20.4 ms) and one eighth of the GPU memory. Without calibration, per-image
> dry mass from predicted masks agreed with annotation-derived values within a median absolute relative error of 2.9% (MobileSAM)
> and 3.9% (EdgeSAM). Annotation-derived dry mass showed no storage trend while projected cell area decreased by 44%; the predicted
> masks reproduced the area change and the morphological composition but added a small storage-dependent dry-mass bias, which
> identifies instance separation as the remaining limitation for optical-mass analysis.

Sources: T2, T3, T4, T5/S5, T7, T8, T10. Check after editing: 0.810 ± 0.006, 0.802, 1.32 ± 0.09%, 0.335 vs 0.262, 10.6 vs 20.4 ms,
2.9%, 3.9%, 44%.

---

## 1. Introduction

| Para | First words | Action | What |
|---|---|---|---|
| 1-3 | "Holography was introduced by Dennis Gabor" / "Digital recording followed" / "Applied to microscopy, DH" | EDIT (optional, **redundant**) | About 400 words of holography history before the problem is stated. Condense paragraphs 1 to 3 into one paragraph if the journal asks for length; nothing in them is wrong. |
| 4 | "Quantitative phase imaging (QPI) uses the refractive index" | KEEP | The dry-mass rationale is the core of the paper. |
| 5 | "This is particularly valuable in hematological analysis" | KEEP | Literature statement (discocytes to echinocytes to spherocytes) is consistent with the corrected annotation (spherocytes 0% at day 0, 59% at day 47; T9). |
| 6-8 | "Segmentation is regarded as mandatory" / "Lightweight segmentation models" / "Because full-network adaptation" | EDIT | Para 7: "MobileSAM and EdgeSAM retain the mask decoder" is not how they are used here; say only that MobileSAM and EdgeSAM provide distilled SAM image encoders [33, 34]. |
| 9-10 | "Despite these advances" / "A second limitation" | KEEP | |
| 11 | "This work addresses these limitations" | EDIT | Replace "Raw, unnormalized phase values are supplied directly to this objective so that the preserved physical quantity is not altered by input normalization." with: "The objective acts on the 32-bit phase in radians after per-image background referencing, never on the standardised network input, so the preserved physical quantity is not altered by normalisation or by the arbitrary zero level of each reconstruction." |
| 12 | "Three representative lightweight architectures" | REWRITE | Replace by the contributions paragraph below (states the main contribution as the professor asked). |

**Draft for Introduction paragraph 12 (contributions).**
> The contributions of this work are as follows. (i) We formulate segmentation for QPI as the estimation of a measurement domain
> and introduce a Physics-Aware Phase Consistency Loss that constrains the integrated phase enclosed by the predicted boundary,
> computed on background-referenced 32-bit phase maps. (ii) We adapt the image encoders of two lightweight foundation models,
> MobileSAM [33] and EdgeSAM [34], to single-channel phase images with LoRA and compare them objectively, with a convolutional
> MobileNet-UNet [32] as baseline, in terms of segmentation accuracy, integrated-phase preservation, computational efficiency and
> training stability, using three random seeds for every configuration. (iii) We show on red blood cells imaged over 47 days of
> storage that dry mass estimated from the predicted measurement domains agrees with annotation-derived dry mass within a few
> percent without calibration, and we identify where the remaining errors arise.

---

## 2. Method

| Section / para | First words | Action | What |
|---|---|---|---|
| 2.1 para 1 | "The proposed framework treats the predicted cell boundary" | EDIT | "four main stages: standardization of the quantitative phase maps" -> "background referencing and standardisation of the 32-bit phase maps". |
| Fig. 1 | framework figure | EDIT (figure) | Show two branches from the phase map: (a) background-referenced phase in rad -> PMC/BGA/IPP; (b) per-image z-score -> network. Replace "raw unnormalized phase" in the figure and caption with "background-referenced phase (rad)". Do not draw a SAM prompt encoder or SAM mask decoder (not used). |
| 2.2 para 1 | "The input data consist of single-channel quantitative phase maps" | REWRITE | The old text says the maps were converted to 8-bit and the 8-bit images are the data set. Wrong now. Use the 2.2 draft in `06_METHODS_AS_IMPLEMENTED.md` section 1. |
| 2.2 para 2 | "To adapt the single-channel phase maps" | REWRITE | Merged into the same draft (per-image z-score for the input; background-referenced radians for the loss; bilinear resize to 256 or 1024 px; no cropping). |
| 2.2 para 3 | "Data augmentation is restricted" | REWRITE | Only the three 90° rotations; **delete "horizontal and vertical flipping, and translation"**. Covered by the draft. |
| 2.3 | "Three lightweight architectures are evaluated" | REWRITE | **Wrong now:** "MobileSAM and EdgeSAM retain the mask-decoding pipeline of the original SAM". Only the image encoders are used, each with a task decoder trained from scratch. Use the 2.3 draft in `06_METHODS_AS_IMPLEMENTED.md` section 2. Keep the last sentence (EdgeSAM evaluated as a computationally efficient architecture, not as embedded deployment). |
| 2.4 para 1 + Eq. (1) | "LoRA is employed to adapt" | EDIT | After Eq. (1) add: "We used α = r, so the update enters with unit scaling at every rank." Replace "rank-sweep experiments are performed using the ranks specified in the experimental protocol" by "a rank sweep r ∈ {2, 4, 8, 16, 32} was performed for every architecture". |
| 2.4 para 2 | "LoRA modules are inserted into architecture-dependent layers" | REWRITE | "query and value projection layers" is not what the code does (MobileSAM: fused qkv projections; EdgeSAM and MobileNet-UNet: all non-grouped encoder convolutions / linear layers). Use the 2.4 draft in 06, section 2. Add the merge sentence (inference cost independent of r). |
| 2.5 intro | "Conventional segmentation objectives optimize" | KEEP | |
| 2.5.1 | "To mitigate class imbalance among rare morphology" | EDIT | Add "Pixels without annotation (outside the field of view of the original annotation) are excluded from all sums." Keep the weights 0.5, 1.0, 1.5, 2.0, 2.0 and the +1 smoothing. |
| 2.5.2 | "To ensure the predicted cell region" | EDIT | State the margin: 0.1 rad. "phase values" -> "background-referenced phase values". |
| 2.5.3 | "Because probabilistic network outputs" | EDIT | "raw phase map" -> "background-referenced phase map". Add "gradients by forward differences; the L1 difference is averaged over annotated pixels". |
| 2.5.4 | "The key quantitative constraint" | EDIT | "raw quantitative phase map φ" -> "background-referenced phase map φ (rad)"; ε = 10⁻⁴. Add: "Because φ is referenced to the background, S_GT measures the cell only; with an arbitrary constant offset c, S would contain an additional c × A term." Keep the proportionality-to-dry-mass sentence. |
| 2.5.5 | "The final objective combines" | EDIT | Add λ1 = 0.1, λ2 = 0.05, λ3 = 0.1. Add "Physics terms are averaged over the images of a mini-batch that contain at least one annotated cell and are computed in float32." **Delete** "As demonstrated by the ablation experiments, the individual constraints are complementary: PMC provides coarse ..., BGA anchors ..., and IPP ensures ..." (not supported, 02 B-4, B-15). Replace with: "The contribution of each term is examined by the ablation in Section 4.3." |
| 2.6 Eqs. (7)-(10) | "The principal purpose of the proposed physics-aware segmentation" | KEEP | Equations unchanged. In Eq. (9) define φ as background-referenced. |
| 2.6 last para | "Per-cell measurements are obtained through connected-component labeling" | REWRITE | **Delete** "with the phase scale calibrated to the mean corpuscular hemoglobin (MCH) reference" and **delete** "No background correction is applied when computing the segmentation loss or the global IPE; both operate directly on the raw phase values ..." Both are wrong now. Replace by the paragraph below. |

**Draft for Sec. 2.6, last paragraph.**
> Per-cell measurements are obtained from connected components of the predicted foreground at the native 768 × 768 resolution.
> Components with a projected area outside 15 to 200 µm² (isolated pixels and unresolved clusters) and components that touch an
> unannotated image region are excluded; the perimeter is computed with the Crofton estimator. For dry mass, the background level is
> the median phase of the non-cellular pixels farther than 5 pixels from the analysed mask and is subtracted from the phase of each
> cell. Because the phase maps are in radians, Eq. (10) is applied with the physical constants only (λ = 666 nm, α = 0.2 mL/g,
> Δx_eff = 0.1441 µm); no calibration to a reference haemoglobin content is used. Cells whose phase reached the 2π limit of the
> reconstruction or showed a wrap discontinuity (phase within 0.3 rad of 2π, or a jump larger than π between neighbouring pixels,
> within 2 pixels of the cell) are excluded from all phase-derived results; this rule was fixed before the results were computed.

---

## 3. Experiments

| Section / para | First words | Action | What |
|---|---|---|---|
| 3.1 para 1 | "The experimental evaluation was designed" | EDIT | Replace "The dataset comprises 1,090 densely annotated images ..." and "divided into 1,024 training images and 66 validation images" by: "Float32 phase maps were available for 182 of the 256 annotated training images and for all 66 validation images (six images at each of the eleven storage time points: days 0, 5, 8, 12, 15, 19, 23, 27, 30, 37 and 47). Each training image was used with its three 90° rotations (728 training views). The validation images were used for evaluation; they contain 2,049 annotated cells (area 15 to 200 µm²), of which 61 (3.0%) were excluded from phase-derived results because of phase wrapping (Table S11)." [T1] Fix the class list: "(1) background, (2) discocyte, (3) spherocyte, (4) echinocyte, (5) stomatocyte" **only if** you number classes by mask value + 1; safer: "mask values 0 to 4: background, discocyte, spherocyte, echinocyte, stomatocyte". **Delete** the last sentence ("The quantitative phase maps were used directly, without further normalization, for integrated-phase and dry-mass calculations") and refer to Sec. 2.2 (background referencing). Keep the IRB sentence and the annotation/separation sentences; the clause "Laterally touching cells ... most frequent at day 15, are discussed in Section 4.8" stays valid (see 4.8). |
| 3.1 para 2 | "Holograms were recorded with an off-axis transmission" | EDIT | **Delete** "and no annotated cell exceeded a phase of 2π" (wrong: 3.0% of annotated cells meet the wrapping rule, 02 B-23). Check with the data provider: the text gives 1024 × 1024 sensor pixels and a field of view of about 140 µm, but the phase maps are 768 × 768 at 0.1441 µm (about 111 µm) (07, item Q-A). |
| 3.2 | "All networks were optimized using the AdamW optimizer" | REWRITE | **Delete the WeightedRandomSampler sentences** (not used, D-04). Use the 3.2 draft in 06, section 4 (seeds, last-epoch checkpoint, 180 runs, full fine-tuning single seed). |
| 3.3.1 | "Spatial agreement between predicted and ground-truth masks" | EDIT | Add: "Dice and IoU of each class are computed from all validation pixels pooled, then macro-averaged over the four foreground classes; metrics are computed at the native 768 × 768 grid after bilinear upsampling of the logits. A run in which any class reaches a Dice below 0.05 is reported as having lost that class (training instability); results are given as mean ± standard deviation over three seeds, with every seed reported in the Supplementary Information." |
| 3.3.2 | "The physical fidelity of the predicted measurement boundaries" | EDIT | In the IPE definition replace "raw phase" logic by "background-referenced phase". **Replace** "For the selected EdgeSAM configuration (r = 8), the per-cell median IPE was 3.04% (90th percentile 11.57%), with 73.7% of ground-truth cells successfully matched" by moving the numbers to Results 4.5 (Table 6): MobileSAM r8 1.11% (P90 8.3%, 66.6% matched), EdgeSAM r32 1.88% (P90 16.2%, 66.3% matched) [T7]. Methods should only define the per-cell IPE. Add "wrapped cells excluded". |
| 3.3.3 | "Computational efficiency was characterized" | KEEP | |
| 3.4 para 1 | "Computational benchmarking was performed on an NVIDIA RTX A5000" | REWRITE | Use the protocol summary in 06, section 7. **Delete** "under FP32 and FP16 precision". |
| 3.4 para 2 | "The purpose of this experiment was to establish" | EDIT | Keep the first two sentences. **Delete** "MobileSAM was excluded from ONNX benchmarking ... Native PyTorch inference was therefore used for MobileSAM" (wrong now, 02 B-20). Add: "MobileSAM is timed at its 1024-pixel input and the other models at 256 pixels, so latency differences between architectures combine architecture and input size. Before any ONNX Runtime result was reported, the exported models were checked against PyTorch on every validation image (all three passed; Table S9)." |
| 3.5 | "Three groups of ablation experiments" | REWRITE | See draft below. |
| 3.6 | "Statistical analyses were performed" | EDIT | Keep (image as the unit, per-image medians, Spearman, Mann-Whitney, partial Spearman, Pearson, CCC, MdARE, Bland-Altman). Add: "Early = days 0 to 8 (18 images), late = days 37 and 47 (12 images)." Add a seed paragraph: "Differences between configurations were computed per seed (paired by seed) and are reported with 95% t-intervals; with three seeds these intervals are wide and are used descriptively, not as significance tests. Because the annotated per-image dry mass varies little between images, Pearson r and CCC are reported for completeness, and agreement is assessed by MdARE, bias and the 95% limits of agreement." |

**Draft for Sec. 3.5.**
> Three groups of experiments were performed for all three architectures, each configuration with three random seeds (42, 43, 44).
> First, a LoRA rank sweep (r = 2, 4, 8, 16, 32) characterised the relation between adaptation capacity, segmentation accuracy,
> integrated-phase preservation and training stability (Supplementary Section S-I). From this sweep, MobileSAM at r = 8 and EdgeSAM
> at r = 32 are presented as the preferred configurations, and all architectures are additionally compared at the same rank
> (r = 8). Second, LoRA was compared with full fine-tuning of each network under the same objective and schedule (one seed for full
> fine-tuning). Third, the Physics-Aware Phase Consistency Loss was ablated by progressively adding PMC, BGA and IPP to the Dice
> loss, at every rank (60 configurations, 180 runs); the full objective is the rank-sweep run of the same rank and seed. The
> preferred ranks were chosen after inspection of the rank sweep on the evaluation images; no separate selection set was
> available (Section 5).

---

## 4. Results (new structure)

Old to new mapping. The new section order puts the professor's three comparison axes (accuracy, efficiency, stability) and the
main contribution (integrated phase / optical mass) in separate subsections.

| New section | Content | Tables (main) | Figures (main) | SI |
|---|---|---|---|---|
| 4 intro | one paragraph: what is compared, seeds, preferred and same-rank | | | |
| 4.1 Segmentation accuracy | preferred configurations and same rank | **Table 1** = T2, **Table 2** = T3 | **Fig. 3** = F3 | S1, S2 |
| 4.2 LoRA rank and training stability | rank sweep, lost classes, training curves | **Table 3** = T4 | **Fig. 4** = F4, **Fig. 5** = F5 | S3, FS1, FS7 |
| 4.3 Physics-aware loss ablation | four objectives, every rank | **Table 4** = T5 | **Fig. 6** = F6 | S4, S5, FS2, S12 (IPP and low-phase misses) |
| 4.4 LoRA vs full fine-tuning | | **Table 5** = T6 | (F7 in SI) | F7, S8 |
| 4.5 Integrated-phase and optical-mass preservation | global and per-cell IPE, dry-mass agreement, robustness | **Table 6** = T7 | **Fig. 7** = F9 | S7, S10, S12, S14, FS3, FS4, FS6 |
| 4.6 Computational efficiency | | **Table 7** = T8 | (F8 in SI) | S8, S9, F8 |
| 4.7 Qualitative results | | | **Fig. 8** = FQ1 (run on the server, 07) | FQS1 |
| 4.8 Storage: morphology and dry mass | composition, area, dry mass, annotation vs prediction | **Table 8** = T9 (+ T10 as Table 9 or in the text) | **Fig. 9** = F11, **Fig. 10** = F10 | S11, S13, FS8, FQS2 |

Old tables/figures that disappear: old Table 4 (per-class IPE/Dice for EdgeSAM ablation; replaced by S7 and S4), old Table 5
(merged into Table 6 = T7, class IPE to S7), old Fig. 4 (moved to SI as F7), old Fig. 5 (replaced by F6, no dual axis).

### 4 intro, "The proposed lightweight, physics-aware segmentation framework was evaluated" : REWRITE

> The framework was evaluated on segmentation accuracy, integrated-phase and optical-mass preservation, computational efficiency and
> training stability. Unless stated otherwise, values are means ± standard deviation over three seeds of the final-epoch models,
> computed on the 66 validation images at their native resolution. MobileSAM at r = 8 and EdgeSAM at r = 32 are presented as the
> preferred configurations, MobileNet-UNet at r = 8 as a convolutional baseline, and all architectures are also compared at the
> same rank (r = 8).

### 4.1 Segmentation accuracy : REWRITE all four paragraphs, replace Table 1 and Fig. 3

Delete every sentence that selects EdgeSAM as "the primary architecture", "most balanced", "the only model that ...", and the
sentence "LoRA rank was selected for EdgeSAM ... Rank 8 gave the best balance" (02 B-8, B-11). Delete the radar-chart reference
(Fig. S-2 is removed, 04).

> Table 1 lists the preferred configurations and Table 2 the comparison at the same rank. MobileSAM (r = 8) reached the highest mean
> Dice (0.810 ± 0.006) and boundary F1 (0.802 ± 0.011) and the highest Dice for every morphology class, including echinocytes
> (0.760 ± 0.006) and stomatocytes (0.708 ± 0.020) (Fig. 3). EdgeSAM (r = 32) reached a mean Dice of 0.738 ± 0.006 and separated
> individual cells better than MobileSAM (AJI 0.335 ± 0.020 vs 0.262 ± 0.017). MobileNet-UNet reached a high boundary F1 (0.814)
> and the highest AJI (0.551), but it did not predict a single spherocyte or stomatocyte pixel in any seed, so its mean Dice was
> 0.412; aggregate boundary and instance metrics alone do not reveal the loss of whole morphology classes.
>
> At the same rank (r = 8, Table 2), MobileSAM exceeded EdgeSAM in boundary F1 (paired difference +0.194, 95% interval +0.145 to
> +0.243). The differences in mean Dice (+0.164, −0.133 to +0.461) and global IPE (−2.49 percentage points, −7.15 to +2.17) were not
> separable from seed variation, because one of the three EdgeSAM seeds lost the spherocyte class at r = 8 (Section 4.2).

Table 1 caption: "Segmentation at the preferred LoRA configurations (full physics-aware objective). Mean ± SD over three seeds
(42, 43, 44); final-epoch models; native 768 × 768 grid. Class names with mask value. 'Runs with a lost class': number of seeds in
which a class had Dice < 0.05." Table 2 caption analogous, plus the paired-difference rows.

### 4.2 LoRA rank and training stability : NEW section (old rank sentences of 4.1 and S-I move here)

> Across the rank sweep (Fig. 4, Table S1), MobileSAM was stable from r = 2 to r = 16 (mean Dice 0.808 to 0.813, seed SD ≤ 0.006).
> At r = 32 one seed lost the spherocyte and stomatocyte classes; its validation Dice dropped after the first epoch and did not
> recover (Fig. 5). EdgeSAM improved with rank in AJI and boundary F1 and was stable at r = 4 and r = 32, but at r = 2, 8 and 16
> the same seed (44) lost the spherocyte class (Table 3). EdgeSAM converged late, with a step increase of the validation Dice
> between about epochs 15 and 25; in the unstable runs the Dice stayed near 0.5 and this step did not occur (Fig. 5). Between EdgeSAM r = 32 and r = 8 the mean Dice
> difference was +0.091 (−0.196 to +0.379) and the AJI difference +0.086 (+0.026 to +0.146) (Table S3). MobileNet-UNet lost the
> spherocyte class in all 15 runs, which is a systematic failure rather than instability. Increasing the rank therefore did not
> improve MobileSAM, whereas EdgeSAM benefited from the larger adaptation capacity; we present MobileSAM at r = 8 and EdgeSAM at
> r = 32 for this reason, without claiming an optimal rank. LoRA is merged into the weights for inference, so the rank changes the
> number of trainable parameters (Table S8) but not the inference cost.

### 4.3 Physics-aware loss ablation : REWRITE (all paragraphs, Tables 3 and 4, Fig. 5)

Delete: the paragraph on run aliasing and the non-converged MobileSAM run ("Note that the EdgeSAM r = 8 values in Table 3 ...");
"The ablation runs were trained independently" (caption); old Table 4; the PMC-alone stomatocyte statement (Dice 0.018); the
MobileSAM non-convergence sentence; the old robustness paragraph (moves to 4.5); the "complementary roles" closing sentence.

> Table 4 and Fig. 6 compare the four objectives for the preferred configurations, EdgeSAM at r = 8 and MobileNet-UNet; Tables S4
> and S5 and Fig. S2 give all ranks. Adding the physics-aware terms lowered the mean global IPE relative to Dice-only training in all
> 15 architecture-rank combinations; with three seeds the 95% interval excluded zero in three of them (Table S5). The reduction was
> large where Dice-only training was unstable: for EdgeSAM at r = 32 the IPE fell from 11.20 ± 6.46% to 3.02 ± 0.84%, and for
> MobileNet-UNet at r = 8 from 15.48 ± 12.73% to 2.60 ± 0.07%. For MobileSAM at r = 8, whose Dice-only IPE was already low, the
> reduction was small (1.53 ± 0.27% to 1.32 ± 0.09%; −0.21 percentage points, −0.75 to +0.33).
>
> The physics terms also reduced training failures: over the rank sweep, runs that lost a class decreased from 3 of 15 to 1 of 15
> for MobileSAM and from 6 of 15 to 3 of 15 for EdgeSAM (Table 3). The contributions of the individual terms could not be separated
> reliably with three seeds. Adding IPP to Dice + PMC + BGA lowered the mean IPE in 13 of 15 combinations, most clearly for EdgeSAM
> at r = 32 (−0.63 percentage points, −0.75 to −0.51), whereas PMC and BGA alone had no consistent effect (8 and 11 of 15). Mean
> Dice was not systematically changed by the physics terms. IPP did not consistently increase the misses of low-phase pixels: the
> false-negative rate in the lowest phase quartile changed by −4.0 (−5.3 to −2.8) percentage points for EdgeSAM at r = 32 and by
> +2.4 (−1.2 to +5.9) for MobileSAM at r = 8 (Table S12).

Table 4 caption: "Ablation of the Physics-Aware Phase Consistency Loss. Each block adds one term to the class-weighted Dice loss.
The full objective is the rank-sweep run of the same rank and seeds. Mean ± SD over three seeds; lower block: seed-paired
differences with 95% t-intervals."

### 4.4 LoRA versus full fine-tuning (old 4.2) : REWRITE, replace Table 2 by Table 5 (T6); old Fig. 4 to SI (F7)

> Full fine-tuning (one seed) updated 5.7 to 7.6 M parameters and lost morphology classes in all three architectures (Table 5): the
> spherocyte class for EdgeSAM (mean Dice 0.559, global IPE 8.31%), the spherocyte, echinocyte and stomatocyte classes for MobileSAM
> (mean Dice 0.217) and, as with LoRA, the spherocyte and stomatocyte classes for MobileNet-UNet. With the same seed, the LoRA models
> of EdgeSAM (r = 32) and MobileSAM (r = 8) detected all four classes. With a single full fine-tuning run per architecture this
> comparison is indicative; it agrees with the view that constraining the update to a low-rank subspace helps to retain minority
> classes on this small data set.

Delete "For EdgeSAM, full fine-tuning reduced the mean Dice from 0.727 to 0.557 and caused complete collapse of the echinocyte
class" and "LoRA-adapted EdgeSAM provided the most favorable balance" (old names, old numbers, selection language).

### 4.5 Integrated-phase and optical-mass preservation (old 4.4) : REWRITE, Table 6 = T7, Fig. 7 = F9; old Table 5 removed

> Table 6 summarises the measurement-related errors. The global IPE was 1.32 ± 0.09% for MobileSAM (r = 8) and 3.02 ± 0.84% for
> EdgeSAM (r = 32). For annotated cells that were matched one-to-one to a predicted cell (IoU ≥ 0.5; 66.6% and 66.3% of the cells),
> the median per-cell integrated-phase error was 1.11 ± 0.09% and 1.88 ± 0.24%, and 90% of matched cells had errors below 8.3% and
> 16.2%, respectively (Fig. S3). Unmatched cells were mostly merged or missed cells: both SAM models produced about two thirds as
> many cell components as were annotated (0.67 and 0.70), consistent with their AJI.
>
> Dry mass was computed without calibration. The median day-0 dry mass was 27.8 pg from the annotation masks, 0.93 of a typical
> mean corpuscular haemoglobin of 30 pg, and 28.2 pg and 27.9 pg from the MobileSAM and EdgeSAM masks. Per image, dry mass from the
> predicted masks agreed with the annotation-derived dry mass within a median absolute relative error of 2.87 ± 0.21% (MobileSAM)
> and 3.93 ± 0.52% (EdgeSAM), with biases of +0.66 and +0.46 pg and 95% limits of agreement of about −5 to +6 pg (Fig. 7). The
> Pearson correlation was low (0.24 and 0.38) because the annotation-derived per-image dry mass varied little (28.7 ± 1.5 pg); for
> such a narrow range, the error-based measures describe the agreement better than correlation.
>
> Global IPE depended on the background referencing: without it, the IPE of MobileSAM rose from 1.32% to 2.06% (Table S14).
> Estimating the background from the prediction instead of the annotation changed the IPE by less than 0.01 percentage points.
> Segmentation was insensitive to synthetic background distortions (tilt, defocus-like and astigmatism-like fields up to 0.5 rad):
> the Dice changed by at most 0.003 (Table S10, Fig. S4); constant offsets are removed by the per-image standardisation of the input
> and are therefore not a test of robustness. Segmentation errors concentrated within 2 pixels of the annotated boundary and in the
> lowest-phase quartile of the cells (false-negative rate 23 to 28% vs below 6% in the other quartiles; Table S12, Fig. S6).

Check before use: "below 6% in the other quartiles": MobileSAM Q2 to Q4 < 1%, EdgeSAM r32 Q2 5.6%. OK as written.

### 4.6 Computational efficiency (old 4.5) : REWRITE, Table 7 = T8; old Table 6 and Fig. S-3 removed

> Inference cost is independent of the LoRA rank after merging, so one measurement per architecture is reported (Table 7). In
> PyTorch FP32, EdgeSAM needed 10.6 ± 1.5 ms per image (81 to 103 frames per second) and 42 MB of GPU memory, MobileSAM 20.4 ±
> 0.2 ms (49 to 50 FPS) and 358 MB, and MobileNet-UNet 5.7 ± 0.1 ms and 52 MB. ONNX Runtime reduced the latency of EdgeSAM to 4.1 ±
> 0.4 ms (229 to 277 FPS) and of MobileNet-UNet to 2.5 ms, but increased that of MobileSAM to 25.4 ms; it used more memory for all
> models (132 to 1,178 MB). MobileSAM processes 1024-pixel inputs and the other models 256-pixel inputs, so these differences
> combine architecture and resolution. All ONNX models reproduced the PyTorch predictions (arg-max agreement ≥ 99.99%, change in mean
> Dice ≤ 10⁻⁶; Table S9). These measurements characterise a desktop GPU, not an embedded device.

Delete: ONNX FP16, 228.3 FPS, 16.4 MB, "MobileSAM ... could not be reliably accelerated using ORT", "EdgeSAM provided the most
balanced overall performance".

### 4.7 Qualitative results (old 4.6) : EDIT, Fig. 8 = FQ1

Replace "the LoRA-adapted EdgeSAM successfully delineated ..." by a description of FQ1 after you have generated it (07, step 6):
which classes each model gets right, MobileNet-UNet missing spherocytes (2) and stomatocytes (4), merged neighbouring cells. Do
not claim "individualised cellular regions rather than merged foreground regions" without looking at the figure: the AJI and the
cell-count ratio (0.67 to 0.70) say merging happens for both SAM models. Caption: "MobileSAM (r = 8), EdgeSAM (r = 32) and
MobileNet-UNet (r = 8), seed 42".

### 4.8 Storage: morphology and dry mass (old 4.7 and 4.8) : REWRITE, Table 8 = T9 (+T10), Fig. 9 = F11, Fig. 10 = F10

Delete: the area peak at day 15 (83.6 µm²) narrative, the dry-mass decline (ρ = −0.40) and partial correlation (−0.15)
narrative, r = 0.966 / CCC 0.96 / MdARE 3.35%, "1,974 cells", the stomatocyte 10% to 26% sentence, old Table 7 (old names,
predicted cells only, single seed). This section now reports the annotation-derived reference and how well each model reproduces it.

> The annotation shows the expected storage lesion (Fig. 9, Table 8): discocytes made up 93% of the cells at day 0 and 11% at day
> 47, while spherocytes rose from 0% to 59% and echinocytes from 2% to 30%. Both preferred models reproduced this composition (mean
> absolute difference per day 1.7 and 2.8 percentage points for MobileSAM and EdgeSAM).
>
> With the image as the unit of analysis (Table 9), the projected area of the annotated cells decreased with storage (Spearman ρ =
> −0.53, p < 0.001; −44% from days 0 to 8 to days 37 and 47), whereas their dry mass did not change (ρ = −0.11, p = 0.36; −0.8%;
> Mann-Whitney p = 0.21) (Fig. 10). The cells therefore lost projected area without losing dry mass, consistent with the spherocytic
> transformation in which membrane is lost while the haemoglobin content is retained [24, 25, 44]. The predicted masks reproduced the
> decrease in area (MobileSAM −40.6%, EdgeSAM −44.3%) but showed a small decrease in dry mass that the annotation does not show
> (MobileSAM −2.6 ± 0.4%, not significant; EdgeSAM −6.1 ± 0.3%, p between 0.008 and 0.05 across seeds). Fig. 10 shows that this
> bias arises at days 5 to 19, where the predicted per-image dry mass exceeds the annotation-derived value, and is largest at day 15,
> the time point with the most touching cells. The same bias, of different size, occurred for all 15 rank-sweep configurations
> (Table S13). Population-level dry-mass trends from predicted masks should therefore be read against this segmentation bias.

Check after generating FQ1: "the time point with the most touching cells" comes from the old manuscript (3.1); keep only if still
true for the annotation (07, Q-C). Day 15 also has the highest annotated cell-mean area (59.3 µm², T9), which supports it.

---

## 5. Discussion : REWRITE (all seven paragraphs)

| Old para | First words | Action |
|---|---|---|
| 1 | "The results demonstrate that physics-aware lightweight segmentation" | REWRITE: draft D1 (old numbers 5.22 to 3.04, 0.938 to 0.953, 15.52 to 3.01 are wrong) |
| 2 | "The three components of the proposed physics-aware objective" | REWRITE: draft D2 (no "complementary" claim; MobileSAM non-convergence sentence deleted) |
| 3 | "Preserving integrated phase is particularly important" | REWRITE: draft D3 (r = 0.966 removed) |
| 4 | "The comparison among lightweight architectures and adaptation strategies" | REWRITE: draft D4 (objective comparison, stability; "r = 8 best overall balance" deleted) |
| 5 | "Computational benchmarking also showed" | REWRITE: merged into D4 (FP16, 228.3 FPS, 16.4 MB deleted) |
| 6 | "The biological analysis further demonstrates" | REWRITE: draft D5 (no dry-mass decline, no MCH calibration) |
| 7 | "Several limitations remain" | REWRITE: draft D6 (keep the unwrapping and amplitude sentences, update the rest) |
| 8 | "Overall, the present results establish" | EDIT: keep, soften "establish" to "support" |

> **D1.** Treating the predicted boundary as the measurement domain of the integrated phase changes what a segmentation model is
> optimised for. With the Physics-Aware Phase Consistency Loss, the global integrated-phase error was lower than with Dice-only
> training in every architecture and rank examined, and it was 1.3% for MobileSAM and 3.0% for EdgeSAM at the preferred
> configurations. The gain was largest where geometric training alone was unstable (EdgeSAM, MobileNet-UNet) and small where it was
> already accurate (MobileSAM), which indicates that the physics terms mainly protect the measurement domain against failure modes
> that overlap metrics do not penalise strongly.
>
> **D2.** The ablation supports the full objective as a whole rather than a ranking of its terms. With three seeds, the full objective
> gave the lowest and least variable IPE and fewer runs that lost a class; IPP had its clearest effect for EdgeSAM at r = 32, and the
> separate effects of PMC and BGA were not consistent across ranks. Because all three terms act on the class-agnostic foreground, they
> cannot by themselves prevent the confusion between morphologies; the class-weighted Dice term carries that task.
>
> **D3.** The dry-mass results show why the integrated phase is the right quantity to preserve. Without any calibration, the dry mass
> from the predicted masks agreed with the annotation-derived dry mass within a median of 2.9 to 3.9% per image, and the day-0 value
> (27.8 pg from the annotation) was consistent with a typical mean corpuscular haemoglobin. The low correlation between predicted and
> annotation-derived per-image dry mass is a consequence of the narrow range of the reference, not of large errors; Bland-Altman limits
> of about ±6 pg describe the remaining per-image error, which is dominated by a few images with merged or missed cells.
>
> **D4.** MobileSAM and EdgeSAM offer different trade-offs. MobileSAM was the more accurate model (higher Dice, boundary F1 and class
> Dice, lower IPE) and was stable at ranks 2 to 16, but it is twice as slow in PyTorch, needs about eight times more GPU memory, and
> separated touching cells less well. EdgeSAM was faster and lighter, separated cells better, and needed a larger rank (r = 32) to
> train reliably; at smaller ranks one of three seeds failed to learn the spherocyte class. Full fine-tuning lost minority classes in
> all three architectures. Reporting every seed was essential: a single seed would have hidden the failure of EdgeSAM at r = 8
> (seed 44) or of MobileSAM at r = 32 (seed 43). These benchmarks describe a desktop GPU; embedded deployment was not tested.
>
> **D5.** In the annotated cells, projected area decreased by 44% over 47 days of storage while dry mass remained constant, the
> signature of the spherocytic transformation in which membrane, not haemoglobin, is lost [24, 25]. This contrast is exactly the
> kind of information QPI adds to morphology, and it depends on the preservation of the integrated phase: the predicted masks
> reproduced the area change and the composition, but introduced a dry-mass bias of 2.6 to 6.1% between early and late storage that
> arose mainly from images with many touching cells. Instance separation, not boundary placement, is therefore the main limitation of
> the present models for population-level optical-mass analysis.
>
> **D6.** Several limitations remain. The validation images were also the evaluation images: no checkpoint was selected on them, but
> the preferred ranks were chosen after inspecting the rank sweep, and an independent test cohort was not available. Three seeds give
> wide confidence intervals; full fine-tuning was run once. Float32 phase maps were available for 182 of 256 training images, and the
> annotations were transferred from 8-bit exports by registration. Cells reaching the 2π limit of the reconstruction (3.0% of the
> annotated cells; 12% of echinocytes and 10% of stomatocytes) were excluded from phase-derived results; specimens with larger optical
> thickness would require phase unwrapping before this framework is applied. The framework was evaluated on one RBC data set and one
> GPU. [Keep the amplitude paragraph as it is.] Future work will address instance separation within the physics-aware objective,
> independent QPI data sets and cell types, and embedded platforms.

Wrapped share by class (S11): echinocyte (3) 12.2%, stomatocyte (4) 10.0%, spherocyte (2) 5.4%, discocyte (1) 0.6%. "dominated by a few images with merged or missed cells" is an interpretation of F9 (outliers above
+10 pg); verify on FQ1 or drop the clause.

## 6. Conclusion : REWRITE

> This study presented a physics-aware lightweight segmentation framework for quantitative holographic cell analysis in which the
> predicted cell boundary is the measurement domain of the integrated phase. On background-referenced 32-bit phase maps of red blood
> cells, the Physics-Aware Phase Consistency Loss lowered the integrated-phase error relative to Dice-only training for every
> architecture and LoRA rank examined and reduced training failures. With LoRA, MobileSAM (r = 8) gave the most accurate
> segmentation and the lowest integrated-phase error (1.3%), and EdgeSAM (r = 32) the best combination of instance separation and
> computational cost. Dry mass estimated from the predicted masks without calibration agreed with annotation-derived values within a
> median of 3 to 4% per image, and the annotation showed that stored cells lose projected area but not dry mass. Improving the
> separation of touching cells within the physics-aware objective, and validation on independent data sets, cell types and embedded
> hardware, are the next steps.

---

## Declarations and back matter

| Item | Action |
|---|---|
| Author contributions | KEEP (check "benchmarking experiments" still accurate) |
| Data availability | EDIT: add "Code, configuration files and all per-run results are available at <repository URL / DOI>" if you will publish the repository |
| References | KEEP; check that [24], [25], [44] support "membrane loss with retained haemoglobin"; [38] etc. unchanged. Remove refs that become uncited (check [31], [41], [42] after editing). |

## Final consistency pass (do last)

Search the finished manuscript for each string below; none should remain: `0.966`, `0.96`, `3.35`, `5.22`, `3.04`, `15.52`,
`3.01`, `0.952`, `0.953`, `228`, `16.4 MB`, `FP16` (allowed only for "FP16 mixed-precision training"), `1,024 training`,
`1,090`, `1,974`, `83.6`, `43.4`, `WeightedRandomSampler`, `calibrated`, `exceeded a phase of 2`, `query and value`, `did not
converge`, `trained independently`, `optimal`, `most balanced`, `echi` in a table header without the mask value. Then run
`python artifacts/code/verify_artifacts.py`.
