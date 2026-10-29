"""Paragraph-by-paragraph revision of the Supplementary Information (source of EDIT_GUIDE_SI.md and REVISED_SI.docx).

Same format as manuscript_spec.py. Table/figure numbers follow SI_TABLES / SI_FIGS of manuscript_spec.py, so the manuscript and the SI
refer to the same numbers. {{STABLE:x}} inserts the full SI table x from artifacts/tables (all rows).
"""

E = lambda find, action, new="", why="", **kw: dict(find=find, action=action, new=new.strip(), why=why, **kw)

ENTRIES = [
    E(("Supplementary Information", "Corresponding author"), "KEEP"),

    # ------------------------------------------------------------------------------------------------ S-I rank sweep
    E("S-I. Low-Rank Adaptation (LoRA) Rank Ablation for EdgeSAM", "REPLACE",
      "**S-I. Complete LoRA Rank Sweep for All Architectures and Seeds**",
      why="The old S-I covered EdgeSAM only, one run per rank; the professor asked for the complete sweep with all three seeds."),
    E("The effect of LoRA rank was systematically evaluated for EdgeSAM", "REPLACE", r"""
LoRA ranks *r* = 2, 4, 8, 16, and 32 were trained for all three architectures with the full physics-aware objective and three seeds
each ([[stab:S1]]; every seed in [[stab:S2]]). MobileSAM was stable from *r* = 2 to *r* = 16 (mean Dice 0.808 to 0.813, seed SD at most
0.006); at *r* = 32, seed 43 lost the spherocyte and stomatocyte classes. EdgeSAM improved with rank in AJI and boundary F1-score; at
*r* = 2, 8, and 16, seed 44 lost the spherocyte class, while *r* = 4 and *r* = 32 were stable in all seeds. MobileNet-UNet lost the
spherocyte class in every run. Seed-paired differences between ranks are given in [[stab:S3]]; with three seeds their 95% intervals are
wide and are reported descriptively. The trainable parameters grow with the rank ([[stab:S8]], [[sfig:FS5]]), whereas the inference cost
does not, because the low-rank matrices are merged into the frozen weights before deployment. Per-class Dice against rank is shown in
[[sfig:FS1]].
""", why="Removes 'r = 8 provided the best overall balance ... selected as the optimal LoRA rank' and every old number (single run, old pipeline)."),
    E(("TABLE S-1", "32             1.47M"), "REPLACE", r"""
**[[STAB:T1]]** Dataset after the corrected preprocessing (32-bit phase, background referenced).

{{STABLE:T1}}

**[[STAB:S1]]** Complete LoRA rank sweep, all architectures (full physics-aware objective, mean ± SD over seeds 42, 43, 44; numbers in
parentheses are mask values).

{{STABLE:S1}}

**[[STAB:S2]]** Rank sweep: every seed (full objective). Lost classes are named per seed.

{{STABLE:S2}}

**[[STAB:S3]]** Rank sweep: seed-paired differences to *r* = 8 and to *r* = 32 (mean [95% t-interval], n = 3; * interval excludes 0).

{{STABLE:S3}}

**[[STAB:S8]]** Trainable parameters by LoRA rank.

{{STABLE:S8}}
""", why="Old Table S-1 (EdgeSAM single run, latency per rank) is replaced; latency per rank is meaningless after the LoRA merge."),
    E(("![image](media/image2.png)", "Fig. S-1"), "REPLACE", r"""
{{FIG:FS1}}

**[[SFIG:FS1]]** Per-class Dice against LoRA rank (full objective). Lines: mean over three seeds; circles: seeds; crosses: seeds in which the class
was lost.

{{FIG:FS7}}

**[[SFIG:FS7]]** Validation Dice during training for every rank and seed (full objective). Dashed lines: runs that lost a class.

{{FIG:FS5}}

**[[SFIG:FS5]]** Trainable parameters against LoRA rank.
""", why="Old Fig. S-1 (EdgeSAM Dice line with parameter bars on a second axis) is replaced."),
    E(None, "NEW", "**S-II. Training Stability**"),
    E(None, "NEW", r"""
[[sfig:FS7]] shows the validation Dice per epoch for every rank and seed. Two failure modes occurred. In MobileSAM at *r* = 32 (seed 43),
the Dice fell after the first epoch and never recovered (loss of two classes). In EdgeSAM, the Dice of all runs increased in a step
between about epochs 15 and 25; in the failed runs (seed 44 at *r* = 2, 8, and 16) this step did not occur and the spherocyte class was
never learned. A class is counted as lost when its Dice is below 0.05; Table 2 of the main text lists every lost class.
"""),

    # ------------------------------------------------------------------------------------------------ S-III ablation (replaces S-II trade-off)
    E("S-II. Accuracy-Efficiency Trade-off", "REPLACE", "**S-III. Physics-Aware Loss Ablation at Every Rank**",
      why="The old S-II (radar chart and FP16 accuracy-throughput frontier) is removed; the accuracy-latency plot moves to S-IX."),
    E("Fig. S-2 summarizes the overall trade-off among segmentation accuracy", "REPLACE", r"""
[[stab:S4]] lists the four objectives at every rank and [[stab:S5]] the seed-paired differences. The full objective had a lower mean
global IPE than Dice-only training in all 15 architecture-rank combinations (95% interval excluding zero in 3); adding IPP to
Dice + PMC + BGA lowered the mean IPE in 13 of 15 (interval excluding zero in 1); PMC and BGA alone changed the IPE in no consistent
direction (8 and 11 of 15). [[sfig:FS2]] summarizes the mean IPE and the number of runs that lost a class. [[sfig:F7]] shows the
per-class comparison of LoRA and full fine-tuning.

**[[STAB:S4]]** Loss ablation at every LoRA rank (mean ± SD over three seeds).

{{STABLE:S4}}

**[[STAB:S5]]** Loss ablation: seed-paired differences within each rank, and the direction of the IPE change over the 15
architecture-rank combinations.

{{STABLE:S5}}
""", why="The radar chart hid lost classes (it said so itself) and is removed (redundant with Tables 1, 5, 6)."),
    E(("![image](media/image3.png)", "Fig. S-2."), "REPLACE", r"""
{{FIG:FS2}}

**[[SFIG:FS2]]** Global IPE (mean over three seeds) of every architecture, objective, and rank; in brackets, the number of runs that lost a
class.

{{FIG:F7}}

**[[SFIG:F7]]** Per-class Dice of LoRA (preferred rank, three seeds) and full fine-tuning (seed 42). Crosses: lost classes.
"""),
    E("Fig. S-3 compares the three architectures in terms of segmentation accuracy", "DELETE",
      why="FP16 throughput, 'MobileSAM could not be accelerated with ONNX Runtime' and the EdgeSAM selection are no longer supported."),
    E(("![image](media/image4.png)", "Fig. S-3"), "DELETE", why="Replaced by Fig. S10 (FP32, PyTorch and ONNX Runtime)."),

    # ------------------------------------------------------------------------------------------------ S-IV integrated phase
    E("S-III. Boundary Quality and Physical Gradient Alignment", "REPLACE", "**S-IV. Integrated-Phase Errors: Classes, Strata, and Individual Cells**"),
    E("Accurate boundary localization is particularly important", "REPLACE", r"""
Accurate boundary localization is particularly important in quantitative phase imaging (QPI) because dilation or erosion of the
predicted cellular region directly affects the integrated phase used for subsequent quantitative analysis. For the preferred
configurations and MobileNet-UNet, segmentation errors were concentrated at the cell boundary (pixel error rate 28 to 35% within
2 pixels of the annotated boundary, below 4% beyond 4 pixels) and among the low-phase pixels of each cell (false-negative rate 23 to 28%
in the lowest phase quartile, below 6% in the others) ([[stab:S12]], [[sfig:FS6]]). Adding IPP did not change this low-phase deficit
consistently (EdgeSAM *r* = 32: −4.0 percentage points; MobileSAM *r* = 8: +2.4, interval including zero). Class-specific IPE
([[stab:S7]]) is dominated by confusion between morphologies; the class-agnostic global IPE measures the preservation of the integrated
phase. The distribution of the per-cell IPE is shown in [[sfig:FS3]], and [[sfig:FQS1]] shows predicted contours on the phase-gradient
map.

**[[STAB:S7]]** Class-specific integrated-phase error (%) and Dice.

{{STABLE:S7}}

**[[STAB:S12]]** Where segmentation errors occur (pixel error by distance to the annotated boundary; false-negative rate by phase quartile),
and the effect of IPP on low-phase misses.

{{STABLE:S12}}
""", why="Keeps the first sentence of the old S-III; removes 'Boundary F1 ... 0.946 to 0.952' (old pipeline)."),
    E(("![](media/image5.png)", "Fig. S-4"), "REPLACE", r"""
{{FIG:FS3}}

**[[SFIG:FS3]]** Cumulative distribution of the per-cell |IPE| of matched, non-wrapped cells (three seeds pooled).

{{FIG:FS6}}

**[[SFIG:FS6]]** Pixel error rate by distance from the annotated boundary (left) and false-negative rate by phase quartile inside annotated cells
(right).

{{FIG:FQS1}}

**[[SFIG:FQS1]]** Alignment of predicted cell contours with quantitative phase gradients. Left: background-referenced phase map; center:
phase-gradient magnitude; right: contours of MobileSAM (*r* = 8) and EdgeSAM (*r* = 32) on the gradient map.
""", why="Old Fig. S-4 (EdgeSAM r = 8 contours) is regenerated with both preferred models (needs the phase maps; run on the server)."),

    # ------------------------------------------------------------------------------------------------ S-V .. S-IX (new) and storage images
    E(None, "NEW", "**S-V. Background Reference and Robustness to Background Phase Variations**"),
    E(None, "NEW", r"""
The validation phase maps had background levels between 0.66 and 2.50 rad as provided. Without background referencing, the global IPE
of MobileSAM (*r* = 8) would be 2.06% instead of 1.32% ([[stab:S14]]); estimating the background from the predicted instead of the
annotated mask changed it by less than 0.01 percentage points. Adding synthetic tilt, defocus-like, and astigmatism-like fields of up
to 0.5 rad changed the mean Dice by at most 0.003 and the background-corrected IPE by at most 0.1 percentage points ([[stab:S10]],
[[sfig:FS4]]). Constant offsets are removed by the per-image standardization of the network input, so they do not test robustness.

**[[STAB:S14]]** Global IPE under different background references.

{{STABLE:S14}}

**[[STAB:S10]]** Robustness to synthetic background-phase perturbations.

{{STABLE:S10}}

{{FIG:FS4}}

**[[SFIG:FS4]]** Mean Dice under synthetic tilt, defocus-like, and astigmatism-like background distortions.
"""),
    E(None, "NEW", "**S-VI. Phase Wrapping**"),
    E(None, "NEW", r"""
A cell was flagged as phase-wrapped when, within 2 pixels of the cell, the phase as provided was within 0.3 rad of 2π or two neighboring
pixels differed by more than π. [[stab:S11]] lists the flagged cells by class and storage day. Flagged cells are kept for segmentation
training and scoring and excluded from dry mass and per-cell phase results.

**[[STAB:S11]]** Phase-wrapped cells by class and storage day.

{{STABLE:S11}}
"""),
    E(None, "NEW", "**S-VII. Dry Mass and Storage Trends for All Configurations**"),
    E(None, "NEW", r"""
[[stab:S13]] repeats the optical-mass and storage analysis for every rank of every architecture. Annotation-derived dry mass showed no
storage trend (ρ = −0.11, *p* = 0.36), but all 15 configurations showed a larger dry-mass decrease from early to late storage than the
annotation, so the bias described in Section 4.9 of the main text is a property of the predicted masks rather than of the preferred
configurations alone. [[sfig:FS8]] shows the annotation-derived per-image medians.

**[[STAB:S13]]** Optical-mass and storage metrics of every rank-sweep configuration.

{{STABLE:S13}}

{{FIG:FS8}}

**[[SFIG:FS8]]** Annotation-derived per-image median projected area (left) and dry mass (right) against storage day.
"""),
    E(None, "NEW", "**S-VIII. Computational Benchmark Details**"),
    E(None, "NEW", r"""
[[stab:S9]] gives the benchmark of every seed and the comparison of ONNX Runtime with PyTorch on all validation images. [[sfig:F8]]
shows mean Dice against median latency.

**[[STAB:S9]]** Benchmark per seed and ONNX Runtime equivalence.

{{STABLE:S9}}

{{FIG:F8}}

**[[SFIG:F8]]** Mean Dice (± SD over three seeds) against median latency (± SD over three seeds), PyTorch FP32 (left) and ONNX Runtime FP32
(right), NVIDIA RTX A5000, batch 1. Filled markers: preferred configurations; open marker: EdgeSAM at *r* = 8. LoRA is merged before
inference, so both EdgeSAM ranks share the same latency. MobileSAM runs at 1024 px, the other models at 256 px.
"""),
    E("S-IV. Representative Quantitative Phase Images over the Storage Period", "REPLACE",
      "**S-IX. Representative Quantitative Phase Images over the Storage Period**"),
    E("Fig. S-5 shows representative quantitative phase images", "REPLACE", r"""
[[sfig:FQS2]] shows representative quantitative phase images from different points of the 47-day storage period. Cells at the earliest
time point are mostly biconcave discocytes, while images from later storage show a growing share of cells with altered shapes.
""", why="Only the figure number changes."),
    E(("![](media/image7.png)", "Fig. S-5."), "REPLACE", r"""
{{FIG:FQS2}}

**[[SFIG:FQS2]]** Representative quantitative phase images (background referenced) at five storage time points.
"""),
    E(None, "NEW", "**Data S1.** All 183 training runs (180 LoRA, 3 full fine-tuning) with every metric per seed: artifacts/tables/si/S6_all_runs.csv "
                   "(attach as a supplementary data file)."),
]
