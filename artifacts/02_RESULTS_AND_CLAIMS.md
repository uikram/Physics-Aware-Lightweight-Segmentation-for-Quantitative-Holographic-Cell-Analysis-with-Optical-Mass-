> **SUPERSEDED (29 Oct 2026).** The current manuscript and SI are in `artifacts/manuscript_final/`. This file predates the final audit and contains statements that are WRONG and must not be reused: (1) a "small storage-dependent dry-mass bias": the date-clustered test does not support it (EdgeSAM r=32 p = 0.08 to 0.19, MobileSAM r=8 p = 0.63 to 0.98; the p = 0.008 to 0.05 values come from a test that treats the 66 images as independent); (2) "dry mass remained constant": a decrease is only *not statistically supported*; (3) latency of EdgeSAM paired with r = 32: only the rank-8 checkpoints were timed. See `artifacts/manuscript_final/change_log.md`.

# 02. What the corrected results show, and what the manuscript may claim

All numbers: mean ± SD over seeds 42, 43, 44 (last-epoch checkpoint, native 768 x 768 grid), from `artifacts/tables`. Paired
differences are seed-matched with a 95% t-interval (n = 3, so the intervals are wide and descriptive only). "Lost class" = a
class with Dice < 0.05 in a run. Class names follow the manuscript (mask value 2 = spherocyte, 3 = echinocyte, D-25).

## A. The ten findings that carry the revised paper

1. **Integrated phase is preserved within a few percent by both SAM models.** Global IPE: MobileSAM r8 1.32 ± 0.09%, EdgeSAM r32
   3.02 ± 0.84%. Per-cell median |IPE| of matched cells: 1.11 ± 0.09% and 1.88 ± 0.24% (T2, T7).
2. **The physics-aware terms lower the integrated-phase error relative to Dice-only training in every one of the 15
   architecture x rank combinations** (direction 15/15), but with three seeds the 95% interval excludes zero in only 3 of 15 (S5).
   The gain is large where Dice-only training is unstable (EdgeSAM r32: 11.20 ± 6.46% to 3.02 ± 0.84%; MobileNet-UNet r8:
   15.48 ± 12.73% to 2.60 ± 0.07%) and small where Dice-only is already good (MobileSAM r8: 1.53 ± 0.27% to 1.32 ± 0.09%,
   difference −0.21 [−0.75, +0.33] pp) (T5, F6).
3. **The physics terms also stabilise training.** Runs that lost a class, rank sweep: Dice-only vs full objective, MobileSAM 3/15
   vs 1/15, EdgeSAM 6/15 vs 3/15 (T4). Wording: "fewer runs lost a minority class", not "prevents collapse".
4. **IPP alone (full vs Dice + PMC + BGA)** lowers mean IPE in 13 of 15 combinations; the interval excludes zero only for EdgeSAM r32
   (−0.63 [−0.75, −0.51] pp) (S5). PMC alone and BGA alone have no consistent effect (8/15 and 11/15). The text must not call the
   three terms "complementary" as a demonstrated fact; it may say the full objective gives the lowest and most stable IPE.
5. **MobileSAM vs EdgeSAM, objectively** (same rank r8 and preferred ranks):
   * Accuracy: MobileSAM is higher on Dice (0.810 vs 0.738 at the preferred ranks), boundary F1 (0.802 vs 0.642) and every class Dice;
     EdgeSAM r32 is higher on AJI (0.335 vs 0.262), i.e. better instance separation (T2). Same rank r8: boundary F1 +0.194 [+0.145,
     +0.243] for MobileSAM; the Dice and IPE intervals include zero because EdgeSAM r8 seed 44 lost the spherocyte class (T3).
   * Efficiency: EdgeSAM is about 2x faster in PyTorch (10.6 vs 20.4 ms P50) and about 6x in ONNX Runtime (4.1 vs 25.4 ms), with
     8.5x less PyTorch memory (42 vs 358 MB); MobileSAM runs at 1024 px, EdgeSAM at 256 px, so the gap mixes architecture and
     input size (T8). Cost is rank independent after the LoRA merge.
   * Stability: MobileSAM is stable at r2 to r16 (seed SD of Dice ≤ 0.006) and lost two classes in one seed at r32 (collapse at
     epoch 2, F5); EdgeSAM lost the spherocyte class in seed 44 at r2, r8 and r16 and was stable at r4 and r32; it converges late
     (step increase around epochs 15 to 25, absent in the failed runs, F5).
6. **Preferred configurations: MobileSAM r8 and EdgeSAM r32** are supported as a presentation choice: MobileSAM r8 is inside the
   0.808 to 0.813 Dice plateau of r2 to r16 with no lost class; EdgeSAM r32 has the highest Dice, AJI and BF1 of its sweep and no lost
   class. EdgeSAM r32 minus r8: Dice +0.091 [−0.196, +0.379] (not separable), AJI +0.086 [+0.026, +0.146] (S3). Not a selection
   rule (D-27): the ranks were chosen after the results were seen on the evaluation images; this must be stated.
7. **LoRA vs full fine-tuning:** full fine-tuning (single seed-42 run) lost spherocytes in EdgeSAM (Dice 0.559), all three minority
   classes in MobileSAM (Dice 0.217) and spherocytes and stomatocytes in MobileNet-UNet (as did its LoRA runs) (T6, F7).
8. **Dry mass from predicted masks agrees with annotation-derived dry mass to a few percent at the image level**: MdARE 2.87 ±
   0.21% (MobileSAM r8) and 3.93 ± 0.52% (EdgeSAM r32), bias +0.66 and +0.46 pg, limits of agreement about −5 to +6 pg (T7, F9).
   Pearson r (0.24, 0.38) and CCC (0.21, 0.32) are low **because the reference hardly varies** (annotation per-image median dry
   mass 28.66 ± 1.55 pg, range 25.0 to 33.0 pg): correlation is not a useful agreement measure here and must not be the headline.
   Day-0 median dry mass without any calibration: annotation 27.83 pg (0.93 x MCH 30 pg), MobileSAM 28.21, EdgeSAM 27.88 pg.
9. **Storage biology (annotation masks): projected area falls by 44% while dry mass is conserved.** Annotation per-image medians:
   area ρ = −0.53 (p = 4e−6), late vs early −43.9%; dry mass ρ = −0.11 (p = 0.36), −0.8% (Mann-Whitney p = 0.21); circularity no
   trend. The spherocyte share rises from 0% (day 0) to 59% (day 47) (T9, T10, F10, F11). This is the physically meaningful result:
   cells lose surface area (spherocytic shift) without losing haemoglobin mass.
10. **Predicted masks reproduce the area trend and the composition but add a small spurious dry-mass decline**: MobileSAM r8 −2.6 ±
    0.4% (ρ −0.17, p 0.11 to 0.27, not significant), EdgeSAM r32 −6.1 ± 0.3% (ρ −0.27, p 0.008 to 0.05), MobileNet-UNet −8.2%.
    All 15 rank-sweep configurations show a larger decline than the annotation (S13). F10 shows where it comes from: predicted
    per-image dry mass is too high at days 5 to 19 (largest at day 15), not too low late. A plausible cause (not tested) is merging of
    touching cells: the SAM models produce only about 0.67 to 0.70 as many cells as annotated (T7).

## B. Claim audit: every claim of the submitted revision that depends on results

Status: **KEEP** (still true), **REVISE** (true in a changed form), **REMOVE** (not supported). Section numbers refer to the
uploaded revised manuscript.

| # | Where | Old claim | Status | What the corrected results support | Source |
|---|---|---|---|---|---|
| 1 | Abstract, Disc., Concl. | Full objective reduces global IPE 5.22% to 3.04% (EdgeSAM) and 15.52% to 3.01% (MobileNet-UNet) vs Dice only | REVISE | EdgeSAM r32 11.20 ± 6.46% to 3.02 ± 0.84%; MobileNet-UNet r8 15.48 ± 12.73% to 2.60 ± 0.07%; MobileSAM r8 1.53 to 1.32%; same direction in 15/15 configurations, interval excludes 0 in 3/15 | T5, S5 |
| 2 | Abstract | "while improving boundary fidelity" | REMOVE | BF1 full vs Dice only is not consistently higher (MobileSAM r8 0.802 vs 0.808; EdgeSAM r32 0.642 vs 0.519 with 2 lost-class runs in Dice only) | T5, S5 |
| 3 | Abstract, 4.5, Disc. | Selected EdgeSAM: BF1 0.952, 228.3 FPS with ONNX FP16 | REMOVE | FP16 is not part of the benchmark (D-26). EdgeSAM ONNX FP32: 4.11 ± 0.43 ms, 229 to 277 FPS; BF1 0.642 ± 0.005 (r32) | T8, T2 |
| 4 | Abstract, 4.8 | 1,974 segmented RBCs over 47 days | REVISE | 1,988 annotated non-wrapped cells (2,049 incl. 61 wrapped); predicted cells per run about 1,300 to 1,450 | T1, T9 |
| 5 | Abstract, 4.8, Disc. | Non-monotonic area peaking at day 15 (83.6 um2) then 43.4 um2 at day 47 | REVISE | Per-image median area is flat (about 44 to 48 um2) to day 27 and falls to about 23 um2 at day 47 (annotation, F10); the day-15 high cell **mean** (59.3 um2, T9) comes from large clustered components, not from a real peak | T9, F10, FS8 |
| 6 | Abstract, 4.8, Disc. | Dry-mass decline, largely explained by the area change (ρ −0.40, partial ρ −0.15) | REMOVE | Annotation-derived dry mass does not change (ρ −0.11, p 0.36; −0.8%); the decline seen with predicted masks is a segmentation bias | T10, S13, F10 |
| 7 | Abstract, 4.8, Disc., Concl. | Predicted vs annotation dry mass: r = 0.966, CCC 0.96, MdARE 3.35%, bias −0.12 pg | REVISE | MdARE 2.87% / 3.93%, bias +0.66 / +0.46 pg, LoA about −5 to +6 pg; r and CCC low because the reference range is narrow; report agreement, not correlation | T7, F9 |
| 8 | 4.1, 4.4, 4.5, Disc., Concl. | EdgeSAM selected as the single primary architecture; "most balanced", "only model with non-degenerate segmentation, high boundary fidelity and adequate instance separation" | REMOVE | Objective comparison: MobileSAM better on accuracy, BF1, IPE and stability at r2 to r16; EdgeSAM better on AJI and cost; both are presented (preferred configurations) | T2, T3, T4, T8 |
| 9 | 4.1 | MobileSAM highest Dice 0.812 but poor boundary quality (BF1 0.644) | REVISE | MobileSAM has the highest Dice (0.810) **and** the highest boundary F1 of the SAM models (0.802); its weakness is instance separation (AJI 0.262) | T2 |
| 10 | 4.1, Table 1 | MobileNet-UNet BF1 0.971, AJI 0.634; loses echinocytes and stomatocytes | REVISE | BF1 0.814, AJI 0.551 ± 0.053; loses **spherocytes** and stomatocytes (D-25 class order) in every run | T2, T4 |
| 11 | 4.1, S-I | Rank 8 gives the best balance for EdgeSAM ("optimal rank") | REMOVE | EdgeSAM r32 highest on all metrics; r8 lost a class in one seed. Never "optimal" (D-27) | S1, S3 |
| 12 | 4.2 | LoRA more stable than full fine-tuning; FT collapses echinocytes in EdgeSAM | REVISE | Still true, now: FT loses spherocytes (EdgeSAM), three classes (MobileSAM), two classes (MobileNet-UNet); FT is a single seed | T6, F7 |
| 13 | 4.3 | MobileSAM full-objective run did not converge (best checkpoint at epoch 1), effect of IPP not established | REMOVE | That run belonged to the superseded pipeline. MobileSAM full objective now converges in all three seeds at r8 | T4, F5 |
| 14 | 4.3, Table 3 | "The ablation runs were trained independently" and the r8 aliasing paragraph | REMOVE | The full-objective row is the rank-sweep run (D-09); every row has three seeds | T5 |
| 15 | 4.3 | PMC alone degrades BF1 and stomatocytes (Dice 0.018); BGA restores them | REMOVE | Not reproduced: Dice + PMC is not consistently worse; PMC effect 8/15 configurations | S4, S5 |
| 16 | 4.3 | IPP does not increase low-phase misses (EdgeSAM 33.3% to 28.7%) | REVISE | Lowest-phase-quartile false-negative rate with vs without IPP: EdgeSAM r32 −4.0 [−5.3, −2.8] pp, MobileSAM r8 +2.4 [−1.2, +5.9] pp; misses concentrate in the lowest quartile for all models (23 to 28%) | S12 |
| 17 | 4.3 | Robustness: EdgeSAM Dice 0.739 to 0.706 under the largest astigmatism | REVISE | |ΔDice| ≤ 0.0034 for all preferred configurations up to 0.5 rad tilt, defocus and astigmatism; the offset rows are invariant by construction (per-image z-score) and must be described as such | S10, FS4 |
| 18 | 4.4, Table 5 | MobileSAM lowest global IPE 2.57%; EdgeSAM 3.63% | REVISE | MobileSAM r8 1.32%, EdgeSAM r32 3.02%, EdgeSAM r8 3.81%, MobileNet-UNet 2.60% | T7 |
| 19 | 3.3.2 | Per-cell IPE median 3.04%, P90 11.57%, 73.7% matched (EdgeSAM r8) | REVISE | MobileSAM r8 1.11% / 8.3% / 66.6%; EdgeSAM r32 1.88% / 16.2% / 66.3% | T7 |
| 20 | 4.5, Table 6 | ONNX FP16; 16.4 MB peak VRAM; MobileSAM cannot be exported to ONNX | REMOVE | FP32 only; Peak VRAM 42 to 1,178 MB; MobileSAM exported and accepted by the equivalence gate (slower in ONNX Runtime: 25.4 vs 20.4 ms) | T8, S9 |
| 21 | S-I, Table S-1 | Latency and VRAM per rank | REVISE | LoRA is merged at deployment, cost is rank independent; one timing per architecture | T8, S8 |
| 22 | 4.7, Table 7 | Composition: stomatocytes 10% to 26% (day 19), 15% at day 47; echinocytes 36% at day 47 | REVISE | Annotation: stomatocytes 5% (day 0), 15% (day 19), 0% (day 47); spherocytes 59% and echinocytes 30% at day 47; predicted composition within 1.7 pp (MobileSAM) / 2.8 pp (EdgeSAM) mean absolute error | T9, F11 |
| 23 | 3.1 | No annotated cell exceeded 2π | REMOVE | 61 of 2,049 annotated cells (3.0%) meet the wrapping rule (raw phase within 0.3 rad of 2π or a neighbour jump > π) and are excluded from phase-derived results | T1, S11 |
| 24 | 2.6, Disc. | Phase scale calibrated to MCH | REMOVE | No calibration (D-13); day-0 median 27.8 pg = 0.93 x MCH is a plausibility check | T7 |
| 25 | 2.6 | No background correction for the loss and the global IPE | REMOVE | Every map is referenced to its own background median before training and evaluation (D-03); global IPE as provided is higher (MobileSAM 2.06%) | S14, 06_METHODS |
| 26 | 3.2 | WeightedRandomSampler | REMOVE | Plain shuffling; imbalance handled by the class-weighted Dice (D-04) | 06_METHODS |
| 27 | 3.1 | 1,090 images, 1,024 training + 66 validation | REVISE | 182 base training images (728 views) + 66 validation images | T1 |
| 28 | 2.2 | Flips and translation offline; float maps converted to 8-bit | REVISE | Only lossless 90 degree rotations; the float32 maps are the network input; the 8-bit exports are used only to carry the masks onto the float grid | 06_METHODS |
| 29 | 2.4 | LoRA on query and value projections | REVISE | MobileSAM: the fused qkv projections of the TinyViT attention; EdgeSAM, MobileNet-UNet: all non-grouped convolution / linear layers of the encoder; α = r | 06_METHODS |
| 30 | Disc. | Full objective reduced AJI for EdgeSAM vs Dice only (trade-off) | REVISE | EdgeSAM r32: AJI full 0.335 vs Dice only 0.329, no trade-off; r8: 0.249 vs 0.334 | S4 |
| 31 | Limitations | Rank and loss selected on the validation set; no independent test cohort | KEEP and extend | Also: preferred ranks chosen after viewing the evaluation results; full fine-tuning single seed; one GPU | 03 |

## C. Words to avoid in the revised text

"optimal rank", "selected by", "demonstrates" (for n = 3 differences whose interval includes 0), "strong agreement (r = ...)",
"prevents collapse", "complementary constraints (shown by ablation)", "real-time / edge deployment", "FP16", "calibrated to MCH",
"no annotated cell exceeded 2π", "trained independently".
