# 04. Supplementary Information: what to change, where, and with what

Applies to `supplementry_info_fv_-_ui.docx`. The old SI has four sections (S-I rank ablation for EdgeSAM only, S-II accuracy and
efficiency trade-off, S-III boundary quality, S-IV storage images), Table S-1 and Figs. S-1 to S-5. **Every numerical item in it is
superseded.** The professor asked for the complete rank sweep with all three seeds and the training instability in the SI, so the
SI becomes the place for the full results.

## Old item by item

| Old item | First words / content | Action | Replacement |
|---|---|---|---|
| S-I heading | "Low-Rank Adaptation (LoRA) Rank Ablation for EdgeSAM" | REWRITE | "S-I. Complete LoRA rank sweep, all architectures and seeds" |
| S-I text | "The effect of LoRA rank was systematically evaluated for EdgeSAM ... r = 8 was selected as the optimal LoRA rank" | DELETE | Every number is wrong (old pipeline, single seed) and "optimal" is not supported. Draft S-I text below. |
| Table S-1 | EdgeSAM rank ablation, single run, with p50 / VRAM per rank | DELETE | new Table S1 (all architectures, mean ± SD), S2 (every seed), S3 (paired differences), S8 (parameters by rank). Latency per rank is meaningless after the LoRA merge. |
| Fig. S-1 | EdgeSAM Dice line and parameter bars (dual axis) | DELETE | Fig. 4 of the main text (F4) shows the sweep; new FS1 (per class) and FS5 (parameters vs rank). No dual axis. |
| S-II text + Fig. S-2 | radar chart "Aggregate performance at r = 8" | DELETE (**redundant and misleading**) | The radar chart normalises each axis to the best model and hides lost classes (it said so itself). Its content is in Tables 1, 2, 7. |
| S-II text + Fig. S-3 | accuracy-throughput frontier, FP16, "dashed ring marks the selected configuration" | REPLACE | F8 (accuracy vs latency, FP32, PyTorch and ONNX panels, both EdgeSAM ranks). Draft caption below. No "selected configuration". |
| S-III text | "Accurate boundary localization is particularly important ... EdgeSAM maintained high Boundary F1 scores at ranks of 8 and above, ranging from 0.946 to 0.952" | REWRITE | BF1 values are wrong (EdgeSAM 0.598 to 0.642). Keep the first two sentences (why boundaries matter). Replace the rest with the S-V text below (error strata). |
| Fig. S-4 | phase, gradient, EdgeSAM r8 contours | REPLACE | FQS1 (both preferred models' contours), generate on the server. |
| S-IV text + Fig. S-5 | representative storage images | KEEP text, REPLACE figure | FQS2 (same selection rule), generate on the server; caption add day numbers. |

## New SI structure (with the tables and figures that already exist)

| New section | Content | Tables | Figures |
|---|---|---|---|
| S-I Complete LoRA rank sweep (3 architectures, 5 ranks, 3 seeds) | mean ± SD; every seed; paired differences to r8 and r32; parameters by rank | S1, S2, S3, S8 | FS1, FS5 |
| S-II Training stability | training curves of every rank and seed; which seeds lost which class | (T4 in main text) | FS7 |
| S-III Loss ablation at every rank | 60 configurations; paired differences; direction summary | S4, S5 | FS2 |
| S-IV LoRA vs full fine-tuning per class | | (T6 in main text) | F7 |
| S-V Integrated phase: class-specific IPE, error strata, IPP and low-phase misses, per-cell IPE distribution | | S7, S12 | FS3, FS6 |
| S-VI Background reference and robustness | IPE with/without background referencing; synthetic tilt, defocus, astigmatism; offsets invariant by construction | S14, S10 | FS4 |
| S-VII Phase wrapping | wrapped cells by class and day | S11 | |
| S-VIII Dry mass and storage: all configurations, annotation reference | the dry-mass bias exists for every configuration | S13 | FS8 |
| S-IX Computational benchmark details | per-seed timing, ONNX equivalence | S9 | F8 |
| S-X Representative images | | | FQS1, FQS2 (server) |
| S-XI Data and preprocessing summary | dataset counts, registration, wrapping rule | T1 (as Table S0) | |
| (all runs, machine-readable) | 183 runs, every metric | S6 (CSV; attach as a data file, not as a printed table) | |

## Draft text

**S-I (complete rank sweep).**
> LoRA ranks r = 2, 4, 8, 16 and 32 were trained for all three architectures with the full physics-aware objective and three seeds
> each (Table S1; every seed in Table S2). MobileSAM was stable from r = 2 to r = 16 (mean Dice 0.808 to 0.813, seed SD at most
> 0.006); at r = 32, seed 43 lost the spherocyte and stomatocyte classes. EdgeSAM improved with rank in AJI and boundary F1; at r = 2,
> 8 and 16 seed 44 lost the spherocyte class, while r = 4 and r = 32 were stable in all seeds. MobileNet-UNet lost the spherocyte class
> in every run. Seed-paired differences between ranks are given in Table S3; with three seeds their 95% intervals are wide and are
> reported descriptively. The trainable parameters grow with the rank (Table S8, Fig. S5), whereas the inference cost does not,
> because the low-rank matrices are merged into the frozen weights before deployment.

**S-II (training stability).**
> Fig. S7 shows the validation Dice per epoch for every rank and seed. Two failure modes occurred. In MobileSAM at r = 32 (seed 43),
> the Dice fell after the first epoch and never recovered (loss of two classes). In EdgeSAM, the Dice of all runs increased in a step
> between about epochs 15 and 25; in the failed runs (seed 44 at r = 2, 8 and 16) this step did not occur and the spherocyte class
> was never learned. A class is counted as lost when its Dice is below 0.05.

**S-III (ablation at every rank).**
> Table S4 lists the four objectives at every rank and Table S5 the seed-paired differences. The full objective had a lower mean
> global IPE than Dice-only training in all 15 architecture-rank combinations (95% interval excluding zero in 3); adding IPP to
> Dice + PMC + BGA lowered the mean IPE in 13 of 15 (interval excluding zero in 1); PMC and BGA alone changed the IPE in no
> consistent direction (8 and 11 of 15). Fig. S2 summarises the mean IPE and the number of runs that lost a class.

**S-V (where errors occur).**
> For the preferred configurations and MobileNet-UNet, segmentation errors were concentrated at the cell boundary (pixel error rate 28 to 35% within 2 pixels of the annotated boundary,
> below 4% beyond 4 pixels) and among the low-phase pixels of each cell (false-negative rate 23 to 28% in the lowest phase quartile,
> below 6% in the others) (Table S12, Fig. S6). Adding IPP did not change this low-phase deficit consistently (EdgeSAM r = 32: −4.0
> percentage points; MobileSAM r = 8: +2.4, interval including zero). Class-specific IPE (Table S7) is dominated by confusion between
> morphologies; the class-agnostic global IPE measures the preservation of the integrated phase.

**S-VI (background reference and robustness).**
> The as-provided validation maps had background levels between 0.66 and 2.50 rad. Without background referencing the global IPE of
> MobileSAM (r = 8) would be 2.06% instead of 1.32% (Table S14); estimating the background from the predicted instead of the
> annotated mask changed it by less than 0.01 percentage points. Adding synthetic tilt, defocus-like and astigmatism-like fields of
> up to 0.5 rad changed the mean Dice by at most 0.003 and the background-corrected IPE by at most 0.1 percentage points (Table S10,
> Fig. S4). Constant offsets are removed by the per-image standardisation of the network input, so they do not test robustness.

**S-VIII (dry mass, all configurations).**
> Table S13 repeats the optical-mass and storage analysis for every rank of every architecture. Annotation-derived dry mass showed no
> storage trend (ρ = −0.11, p = 0.36), but all 15 configurations showed a larger dry-mass decrease from early to late storage than
> the annotation, so the bias described in Section 4.8 of the main text is a property of the predicted masks, not of the preferred
> configurations alone. Fig. S8 shows the annotation-derived per-image medians.

**Caption for F8 (accuracy vs latency).**
> Mean Dice (± SD over three seeds) against median latency (± SD over three seeds) for PyTorch FP32 (left) and ONNX Runtime FP32
> (right) on an NVIDIA RTX A5000, batch 1. Filled markers: preferred configurations; open marker: EdgeSAM at r = 8. LoRA is merged
> before inference, so both EdgeSAM ranks share the same latency. MobileSAM runs at 1024 px, the other models at 256 px.

Check before use: "below 4% beyond 4 pixels" (S12: MobileSAM 0.5%, EdgeSAM 3.5%, MobileNet 0.9% in 4-8 px); "at most 0.1
percentage points" for IPE under distortion (EdgeSAM r32 range 2.99 to 3.13 vs 3.04 none: max change 0.09). OK.

## Numbering

After the content is fixed, renumber tables and figures in order of first mention. The ids used here (S1 to S14, FS1 to FS8, F7, F8,
FQS1, FQS2, T1) are file names in `artifacts/`; keep a note of the final number of each in `05_TABLE_FIGURE_MAP.md`.
