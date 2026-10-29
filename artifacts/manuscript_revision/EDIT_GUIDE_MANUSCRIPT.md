# Manuscript: line-by-line edit guide

Source file: `artifacts/manuscript_in_prog/revised_manuscript_fv - ui.docx` (your current version). Every paragraph of that file appears below, in order, with one of four actions:

* **KEEP**: leave the paragraph exactly as it is.
* **REPLACE**: select the whole paragraph in Word (find it with the quoted first words) and paste the complete new text. Never delete only part of a sentence: every new paragraph is complete and reads correctly.
* **DELETE**: remove the whole paragraph (or table / figure).
* **NEW**: insert the new text at this position (after the previous paragraph of this list).

Counts: 68 keep, 85 replace, 4 delete, 9 new. The finished result of all edits is `REVISED_MANUSCRIPT.docx` in this folder (changed text highlighted in yellow), so you can also work from that file instead. 'R1-4' = reviewer 1, comment 4 (for the response letter).

## Overview of all changes (tick when done)

| Done | Id | Action | Where | First words |
|---|---|---|---|---|
| [ ] | M004 | REPLACE | Abstract | Digital holographic microscopy provides label-free quantitative phase images ... |
| [ ] | M005 | REPLACE | Abstract | Index Terms---Quantitative Digital Holographic Microscopy, Quantitative Phase Imaging, ... |
| [ ] | M017 | REPLACE | 1. Introduction | This work addresses these limitations with a physics-aware ... |
| [ ] | M018 | REPLACE | 1. Introduction | Three representative lightweight architectures, MobileNet-UNet [32], MobileSAM [33], ... |
| [ ] | M021 | REPLACE | 2.1 Overall Framework | The proposed framework treats the predicted cell boundary ... |
| [ ] | M023 | REPLACE | 2.1 Overall Framework | Fig. 1 Overview of the proposed physics-aware lightweight ... |
| [ ] | M025 | REPLACE | 2.2 QPI Preprocessing | The input data consist of single-channel quantitative phase ... |
| [ ] | M026 | REPLACE | 2.2 QPI Preprocessing | To adapt the single-channel phase maps to segmentation ... |
| [ ] | M027 | REPLACE | 2.2 QPI Preprocessing | Data augmentation is restricted to transformations that preserve ... |
| [ ] | M029 | REPLACE | 2.3 Lightweight Segmentation Models | Three lightweight architectures are evaluated, covering convolutional, transformer-based, ... |
| [ ] | M033 | REPLACE | 2.4 LoRA-Based Lightweight Adaptation | where W is the frozen pretrained weight matrix, ... |
| [ ] | M034 | REPLACE | 2.4 LoRA-Based Lightweight Adaptation | LoRA modules are inserted into architecture-dependent layers. For ... |
| [ ] | M040 | REPLACE | 2.5.1 Multi-Class Dice Loss | where $w_{c} in { 0.5,1.0,1.5,2.0,2.0}$ correspond to the ... |
| [ ] | M042 | REPLACE | 2.5.2 Phase-Mask Contrast Term | To ensure the predicted cell region physically maintains ... |
| [ ] | M044 | REPLACE | 2.5.2 Phase-Mask Contrast Term | where $mu_{cell}$ and $mu_{bg}$ denote the mean phase ... |
| [ ] | M046 | REPLACE | 2.5.3 Boundary-Gradient Alignment Term | Because probabilistic network outputs $lbrack 0,1rbrack$ and physical ... |
| [ ] | M048 | REPLACE | 2.5.3 Boundary-Gradient Alignment Term | This term geometrically anchors the predicted measurement boundary ... |
| [ ] | M050 | REPLACE | 2.5.4 Integrated-Phase Preservation Term | The key quantitative constraint of the proposed framework ... |
| [ ] | M052 | REPLACE | 2.5.4 Integrated-Phase Preservation Term | The relative formulation improves numerical stability during mixed-precision ... |
| [ ] | M056 | REPLACE | 2.5.5 Final Loss Function | where $lambda_{1}$, $lambda_{2}$, and $lambda_{3}$ control the relative ... |
| [ ] | M064 | REPLACE | $C = frac{4pi A}{P{2}}$ , (8) | Here, S~φ~ represents the area-integrated quantitative phase associated ... |
| [ ] | M067 | REPLACE | $C = frac{4pi A}{P{2}}$ , (8) | Per-cell measurements are obtained through connected-component labeling of ... |
| [ ] | M070 | REPLACE | 3.1 Dataset and Imaging Configuration | The experimental evaluation was designed to assess three ... |
| [ ] | M071 | REPLACE | 3.1 Dataset and Imaging Configuration | Holograms were recorded with an off-axis transmission digital ... |
| [ ] | M075 | REPLACE | 3.2 Training Configuration | All networks were optimized using the AdamW optimizer ... |
| [ ] | M077 | REPLACE | 3.3 Evaluation Metrics | The proposed framework was evaluated using three complementary ... |
| [ ] | M079 | REPLACE | 3.3.1 Segmentation Performance and Training Stability | Spatial agreement between predicted and ground-truth masks was ... |
| [ ] | M081 | REPLACE | 3.3.2 Quantitative Phase Preservation | The physical fidelity of the predicted measurement boundaries ... |
| [ ] | M083 | REPLACE | 3.3.2 Quantitative Phase Preservation | IPE was evaluated both globally and for individual ... |
| [ ] | M087 | REPLACE | 3.4 Computational Benchmarking Protocol | Computational benchmarking was performed on an NVIDIA RTX ... |
| [ ] | M088 | REPLACE | 3.4 Computational Benchmarking Protocol | The purpose of this experiment was to establish ... |
| [ ] | M090 | REPLACE | 3.5 Ablation Studies | Three groups of ablation experiments were designed to ... |
| [ ] | M092 | REPLACE | 3.6 Statistical Analysis | Statistical analyses were performed to evaluate storage-associated changes ... |
| [ ] | M094 | REPLACE | 4. Results | The proposed lightweight, physics-aware segmentation framework was evaluated ... |
| [ ] | M095 | REPLACE | 4.1 Segmentation Accuracy | Segmentation Accuracy Comparison |
| [ ] | M096 | REPLACE | 4.1 Segmentation Accuracy | Three lightweight architectures---EdgeSAM, MobileSAM, and MobileNet-UNet---were first evaluated ... |
| [ ] | M097 | REPLACE | 4.1 Segmentation Accuracy | TABLE 1 Segmentation performance at the selected LoRA ... |
| [ ] | M098 | REPLACE | 4.1 Segmentation Accuracy | ![image](media/image5.png){width="3.6023622047244093in" height="2.47244094488189in"} |
| [ ] | M099 | REPLACE | 4.1 Segmentation Accuracy | Fig. 3 Per-class Dice scores of MobileNet-UNet, MobileSAM, ... |
| [ ] | M100 | REPLACE | 4.1 Segmentation Accuracy | The per-class Dice scores shown in Fig. 3 ... |
| [ ] | M101 | REPLACE | 4.1 Segmentation Accuracy | Among the evaluated architectures, EdgeSAM was the only ... |
| [ ] | M102 | DELETE | 4.1 Segmentation Accuracy | The aggregate comparison across criteria is shown in ... |
| [ ] | M103 | DELETE | 4.1 Segmentation Accuracy | LoRA rank was selected for EdgeSAM from a ... |
| [ ] | M104 | NEW | 4.2 LoRA Rank and Training Stability | ## 4.2 LoRA Rank and Training Stability |
| [ ] | M105 | NEW | 4.2 LoRA Rank and Training Stability | Across the rank sweep (Fig. 4, Table S2), ... |
| [ ] | M106 | NEW | 4.2 LoRA Rank and Training Stability | Increasing the rank therefore did not improve MobileSAM, ... |
| [ ] | M107 | NEW | 4.2 LoRA Rank and Training Stability | TABLE 2 Training stability: number of seeds (of ... |
| [ ] | M108 | NEW | 4.2 LoRA Rank and Training Stability | {{FIG:F4}} Fig. 4 LoRA rank sweep with the ... |
| [ ] | M109 | NEW | 4.2 LoRA Rank and Training Stability | {{FIG:F5}} Fig. 5 Validation Dice during training for ... |
| [ ] | M110 | REPLACE | 4.3 Parameter-Efficient Adaptation versus Full Fine-Tuning | Parameter-Efficient Adaptation versus Full Fine-Tuning |
| [ ] | M111 | REPLACE | 4.3 Parameter-Efficient Adaptation versus Full Fine-Tuning | To evaluate the effectiveness of parameter-efficient adaptation, LoRA ... |
| [ ] | M112 | REPLACE | 4.3 Parameter-Efficient Adaptation versus Full Fine-Tuning | Fig. 4 further illustrates the advantage of LoRA ... |
| [ ] | M113 | REPLACE | 4.3 Parameter-Efficient Adaptation versus Full Fine-Tuning | TABLE 2 Comparison of LoRA (r = 8) ... |
| [ ] | M114 | DELETE | 4.3 Parameter-Efficient Adaptation versus Full Fine-Tuning | ![image](media/image6.png){width="5.830708661417323in" height="2.4921259842519685in"} |
| [ ] | M115 | REPLACE | 4.4 Physics-Aware Loss Ablation | Physics-Aware Loss Ablation |
| [ ] | M116 | REPLACE | 4.4 Physics-Aware Loss Ablation | To evaluate the contribution of each physics-aware constraint, ... |
| [ ] | M117 | REPLACE | 4.4 Physics-Aware Loss Ablation | TABLE 3 Component-wise ablation of the Physics-Aware Phase ... |
| [ ] | M118 | DELETE | 4.4 Physics-Aware Loss Ablation | TABLE 4 Per-class IPE and Dice scores for ... |
| [ ] | M119 | REPLACE | 4.4 Physics-Aware Loss Ablation | For EdgeSAM, the Dice-only baseline achieved a mean ... |
| [ ] | M120 | REPLACE | 4.4 Physics-Aware Loss Ablation | The complete objective incorporating IPP achieved the lowest ... |
| [ ] | M121 | REPLACE | 4.4 Physics-Aware Loss Ablation | ![](media/image7.png){width="6.573708442694663in" height="2.4285126859142605in"} |
| [ ] | M122 | REPLACE | 4.4 Physics-Aware Loss Ablation | Fig. 5 Component-wise loss ablation at the selected ... |
| [ ] | M123 | REPLACE | 4.5 Integrated-Phase and Optical-Mass Preservation | Integrated-Phase and Optical-Mass Preservation Analysis |
| [ ] | M124 | REPLACE | 4.5 Integrated-Phase and Optical-Mass Preservation | The preservation of phase-integrated optical information was evaluated ... |
| [ ] | M125 | REPLACE | 4.5 Integrated-Phase and Optical-Mass Preservation | TABLE 5 Integrated-phase preservation analysis at the selected ... |
| [ ] | M126 | NEW | 4.5 Integrated-Phase and Optical-Mass Preservation | Dry mass was computed without calibration. The median ... |
| [ ] | M127 | NEW | 4.5 Integrated-Phase and Optical-Mass Preservation | {{FIG:F9}} Fig. 7 Dry-mass agreement between predicted and ... |
| [ ] | M128 | NEW | 4.5 Integrated-Phase and Optical-Mass Preservation | The global IPE depended on the background referencing ... |
| [ ] | M129 | REPLACE | 4.6 Computational Efficiency | Computational Efficiency Analysis |
| [ ] | M130 | REPLACE | 4.6 Computational Efficiency | Computational efficiency was evaluated on an NVIDIA RTX ... |
| [ ] | M131 | REPLACE | 4.6 Computational Efficiency | TABLE 6 Computational benchmarking on an NVIDIA RTX ... |
| [ ] | M132 | REPLACE | 4.7 Visualization of Segmentation Results | Visualization of Segmentation and Morphology Classification Results |
| [ ] | M133 | REPLACE | 4.7 Visualization of Segmentation Results | Qualitative segmentation results further support the quantitative evaluation. ... |
| [ ] | M134 | REPLACE | 4.7 Visualization of Segmentation Results | ![](media/image8.png){width="6.532890419947506in" height="1.7261253280839894in"} |
| [ ] | M135 | REPLACE | 4.7 Visualization of Segmentation Results | Fig. 6 Qualitative segmentation results of LoRA-adapted MobileNet-UNet, ... |
| [ ] | M136 | REPLACE | 4.8 Storage-Induced Morphology Changes | Storage-Induced Morphology Degradation |
| [ ] | M137 | REPLACE | 4.8 Storage-Induced Morphology Changes | To evaluate the applicability of the proposed framework ... |
| [ ] | M138 | REPLACE | 4.8 Storage-Induced Morphology Changes | The predicted RBC morphology distributions showed systematic changes ... |
| [ ] | M139 | REPLACE | 4.8 Storage-Induced Morphology Changes | ![](media/image9.png){width="3.65748031496063in" height="2.34251968503937in"} |
| [ ] | M140 | REPLACE | 4.8 Storage-Induced Morphology Changes | Fig. 7 Predicted RBC morphology composition over the ... |
| [ ] | M141 | REPLACE | 4.8 Storage-Induced Morphology Changes | Representative quantitative phase images in Fig. S-5 further ... |
| [ ] | M142 | REPLACE | 4.9 Quantitative Phase-Derived Cellular Analysis Across Storage | Quantitative Phase-Derived Cellular Analysis Across Storage |
| [ ] | M143 | REPLACE | 4.9 Quantitative Phase-Derived Cellular Analysis Across Storage | Quantitative cellular measurements were extracted from the EdgeSAM-predicted ... |
| [ ] | M144 | REPLACE | 4.9 Quantitative Phase-Derived Cellular Analysis Across Storage | To evaluate the reliability of the predicted segmentation ... |
| [ ] | M145 | REPLACE | 4.9 Quantitative Phase-Derived Cellular Analysis Across Storage | TABLE 7 Per-cell quantitative morphology and phase-derived dry-mass ... |
| [ ] | M146 | REPLACE | 4.9 Quantitative Phase-Derived Cellular Analysis Across Storage | ![](media/image10.png){width="6.172784339457568in" height="2.4681681977252845in"} |
| [ ] | M147 | REPLACE | 4.9 Quantitative Phase-Derived Cellular Analysis Across Storage | Fig. 8 Phase-derived dry-mass analysis. (a) Per-image median ... |
| [ ] | M149 | REPLACE | 5. Discussion | The results demonstrate that physics-aware lightweight segmentation can ... |
| [ ] | M150 | REPLACE | 5. Discussion | The three components of the proposed physics-aware objective ... |
| [ ] | M151 | REPLACE | 5. Discussion | Preserving integrated phase is particularly important because segmentation ... |
| [ ] | M152 | REPLACE | 5. Discussion | The comparison among lightweight architectures and adaptation strategies ... |
| [ ] | M153 | REPLACE | 5. Discussion | Computational benchmarking also showed that the selected EdgeSAM ... |
| [ ] | M154 | REPLACE | 5. Discussion | The biological analysis further demonstrates the utility of ... |
| [ ] | M155 | REPLACE | 5. Discussion | Several limitations remain. The LoRA rank and loss ... |
| [ ] | M156 | REPLACE | 5. Discussion | Overall, the present results establish optical-mass preservation as ... |
| [ ] | M158 | REPLACE | 6. Conclusion | This study presented a physics-aware lightweight segmentation framework ... |
| [ ] | M159 | REPLACE | 6. Conclusion | Among the evaluated architectures, LoRA-adapted EdgeSAM provided the ... |
| [ ] | M160 | REPLACE | 6. Conclusion | Overall, the proposed approach demonstrates that optical-mass preservation ... |

## Paragraph by paragraph


## Physics-Aware Lightweight Segmentation for Quantitative Holographic Cell Analysis with Optical Mass Preservation

**M002 KEEP** "Usama Ikram1, Youhyun Kim2, and Inkyu Moon1,2, 1. Department of Artificial Intelligence, DGIST, Daegu ..."

**M003 KEEP** "Abstract"

### M004 REPLACE: Abstract

**Find:** "Digital holographic microscopy provides label-free quantitative phase images from which ..."

**Original:**

> Digital holographic microscopy provides label-free quantitative phase images from which cellular dry mass is obtained by integrating the phase over each cell. Segmentation networks trained with geometric overlap losses do not constrain this integral, so boundary errors propagate into optical-mass estimates. We propose a physics-aware lightweight segmentation framework in which the predicted cell boundary is treated as the measurement domain of the integrated phase. A Physics-Aware Phase Consistency Loss adds phase-mask contrast, boundary-gradient alignment and integrated-phase preservation terms, computed on background-referenced 32-bit phase maps, to a class-weighted Dice loss, and low-rank adaptation (LoRA) adapts lightweight foundation-model encoders to single-channel phase images. On red blood cell phase maps acquired over 47 days of storage, MobileSAM and EdgeSAM were compared at five LoRA ranks with three random seeds per configuration (180 training runs including the loss ablation). Relative to Dice-only training, the physics-aware objective lowered the global integrated-phase error in all 15 architecture-rank combinations and reduced the number of runs that lost a morphology class. MobileSAM (rank 8) reached a mean Dice of 0.810 ± 0.006, a boundary F1 of 0.802 and a global integrated-phase error of 1.32 ± 0.09%; EdgeSAM (rank 32) gave better instance separation (aggregated Jaccard index 0.335 vs 0.262) at about half the latency (10.6 vs 20.4 ms) and one eighth of the GPU memory. Without calibration, per-image dry mass from predicted masks agreed with annotation-derived values within a median absolute relative error of 2.9% (MobileSAM) and 3.9% (EdgeSAM). Annotation-derived dry mass showed no storage trend while projected cell area decreased by 44%; the predicted masks reproduced the area change and the morphological composition but added a small storage-dependent dry-mass bias, which identifies instance separation as the remaining limitation for optical-mass aalysis.

**New text (complete):**

Digital holographic microscopy provides label-free quantitative phase images from which cellular dry mass is obtained by integrating the phase over each cell. Segmentation networks trained with geometric overlap losses do not constrain this integral, so boundary errors propagate into optical-mass estimates. We propose a physics-aware lightweight segmentation framework in which the predicted cell boundary is treated as the measurement domain of the integrated phase. A Physics-Aware Phase Consistency Loss adds phase-mask contrast, boundary-gradient alignment, and integrated-phase preservation terms, computed on background-referenced 32-bit phase maps, to a class-weighted Dice loss, and low-rank adaptation (LoRA) adapts lightweight foundation-model encoders to single-channel phase images. On red blood cell phase maps acquired over 47 days of storage, MobileSAM and EdgeSAM were compared at five LoRA ranks with three random seeds per configuration (180 training runs including the loss ablation). Relative to Dice-only training, the physics-aware objective lowered the global integrated-phase error in all 15 architecture-rank combinations and reduced the number of runs that lost a morphology class. MobileSAM (rank 8) reached a mean Dice of 0.810 ± 0.006, a boundary F1-score of 0.802, and a global integrated-phase error of 1.32 ± 0.09%; EdgeSAM (rank 32) gave better instance separation (aggregated Jaccard index 0.335 vs 0.262) at about half the latency (10.6 vs 20.4 ms) and one eighth of the GPU memory. Without calibration, per-image dry mass from predicted masks agreed with annotation-derived values within a median absolute relative error of 2.9% (MobileSAM) and 3.9% (EdgeSAM). Annotation-derived dry mass showed no storage trend while projected cell area decreased by 44%; the predicted masks reproduced the area change and the morphological composition but added a small storage-dependent dry-mass bias, which identifies instance separation as the remaining limitation for optical-mass analysis.

**Why:** Your current abstract is already this text; corrected the typo 'aalysis' and the stray 'f[rom' highlight. Numbers: Tables 1, 5, 6, 8.

---

### M005 REPLACE: Abstract

**Find:** "Index Terms---Quantitative Digital Holographic Microscopy, Quantitative Phase Imaging, Physics-Aware Segmentation, ..."

**Original:**

> **Index Terms---**Quantitative Digital Holographic Microscopy, Quantitative Phase Imaging, Physics-Aware Segmentation, Optical Mass Preservation, Lightweight Deep Learning.

**New text (complete):**

**Index Terms**---Quantitative Digital Holographic Microscopy, Quantitative Phase Imaging, Physics-Aware Segmentation, Optical Mass Preservation, Lightweight Deep Learning, Low-Rank Adaptation.

**Why:** Optional: 'Low-Rank Adaptation' added because LoRA is now a main axis of the comparison.

---


## 1. Introduction

**M007 KEEP** "Holography was introduced by Dennis Gabor in 1947 as a two-step process of recording ..."  
*Note:* Added for R1-1 (general background on holography); keep.

**M008 KEEP** "Digital recording followed advances in electronic detection and computation. Goodman and Lawrence demonstrated digital ..."  
*Note:* R1-1; keep.

**M009 KEEP** "Applied to microscopy, DH is known as digital holographic microscopy (DHM) [9], which combines ..."  
*Note:* R1-1; keep. First expansion of DHM.

**M010 KEEP** "Quantitative phase imaging (QPI) uses the refractive index of the specimen as an endogenous ..."  
*Note:* Dry-mass rationale; still correct.

**M011 KEEP** "This is particularly valuable in hematological analysis, where red blood cell (RBC) morphology reports ..."  
*Note:* Literature statement (discocytes to echinocytes to spherocytes) agrees with the corrected annotation (spherocytes 0% at day 0, 59% at day 47).

**M012 KEEP** "Segmentation is regarded as mandatory for the quantitative analysis of QPI data [27]. Conventional ..."

**M013 KEEP** "Lightweight segmentation models are attractive for quantitative biomedical imaging because they reduce inference latency ..."  
*Note:* Describes the published MobileSAM and EdgeSAM models; how they are used here is stated in Section 2.3.

**M014 KEEP** "Because full-network adaptation of large, pre-trained models is computationally expensive and prone to overfitting ..."

**M015 KEEP** "Despite these advances, an important limitation remains. Cross-Entropy and Dice, the standard objectives for ..."

**M016 KEEP** "A second limitation concerns domain adaptation and computational cost. Lightweight and foundation vision models ..."

### M017 REPLACE: 1. Introduction

**Find:** "This work addresses these limitations with a physics-aware lightweight segmentation ..."

**Original:**

> This work addresses these limitations with a physics-aware lightweight segmentation framework. We introduce a Physics-Aware Phase Consistency Loss that incorporates quantitative phase information into the segmentation objective through three complementary constraints: Phase-Mask Contrast (PMC), a global constraint on foreground localization based on cell-to-background phase contrast; Boundary-Gradient Alignment (BGA), which encourages predicted boundaries to follow spatial phase transitions; and Integrated-Phase Preservation (IPP), which explicitly constrains the quantitative phase integrated within the predicted cellular region. Raw, unnormalized phase values are supplied directly to this objective so that the preserved physical quantity is not altered by input normalization. LoRA then adapts lightweight models pretrained on natural images to the holographic phase domain with most parameters frozen, which also regularizes training on a small, imbalanced dataset.

**New text (complete):**

This work addresses these limitations with a physics-aware lightweight segmentation framework whose main contribution is a segmentation objective that preserves the integrated quantitative phase within the predicted cells, so that the predicted boundary can be used directly for optical-mass estimation. We introduce a Physics-Aware Phase Consistency Loss that incorporates quantitative phase information into the segmentation objective through three constraints: Phase-Mask Contrast (PMC), a global constraint on foreground localization based on cell-to-background phase contrast; Boundary-Gradient Alignment (BGA), which encourages predicted boundaries to follow spatial phase transitions; and Integrated-Phase Preservation (IPP), which explicitly constrains the quantitative phase integrated within the predicted cellular region. The constraints act on the original 32-bit phase maps in radians after each map has been referenced to its own background, never on the normalized network input, so that the preserved physical quantity is altered neither by input normalization nor by the arbitrary zero level of the reconstruction. LoRA then adapts lightweight models pretrained on natural images to the holographic phase domain with most parameters frozen, which also regularizes training on a small, imbalanced dataset.

**Why:** States the main contribution (professor's point 1) inside the existing paragraph, not as a separate contribution list (R1-2 objected to such a list). 'Raw, unnormalized phase' is replaced because the phase is now background-referenced (R3-5).

---

### M018 REPLACE: 1. Introduction

**Find:** "Three representative lightweight architectures, MobileNet-UNet [32], MobileSAM [33], and EdgeSAM ..."

**Original:**

> Three representative lightweight architectures, MobileNet-UNet \[32\], MobileSAM \[33\], and EdgeSAM \[34\], are evaluated on quantitative phase images of red blood cells recorded over a 47-day storage period, so that segmentation accuracy, integrated-phase preservation, and computational efficiency can be compared across convolutional, transformer-based, and hybrid designs.

**New text (complete):**

The image encoders of two lightweight foundation models, MobileSAM [33] and EdgeSAM [34], and a convolutional MobileNet-UNet [32] baseline are adapted with LoRA and evaluated on quantitative phase images of red blood cells recorded over a 47-day storage period. Because the stability of parameter-efficient adaptation on a small dataset cannot be judged from a single training run, every configuration is trained with three random seeds, and the models are compared in terms of segmentation accuracy, integrated-phase and optical-mass preservation, computational efficiency, and training stability.

**Why:** Professor's point 2 (objective comparison of accuracy, efficiency and stability) and point 4 (three seeds).

---


## 2. Method


## 2.1 Overall Framework

### M021 REPLACE: 2.1 Overall Framework

**Find:** "The proposed framework treats the predicted cell boundary as a ..."

**Original:**

> The proposed framework treats the predicted cell boundary as a quantitative measurement domain rather than as a purely geometric contour. As illustrated in Fig. 1, the pipeline consists of four main stages: standardization of the quantitative phase maps, segmentation by a LoRA-adapted lightweight network, optimization with the Physics-Aware Phase Consistency Loss, and extraction of phase-derived cellular measurements.

**New text (complete):**

The proposed framework treats the predicted cell boundary as a quantitative measurement domain rather than as a purely geometric contour. As illustrated in Fig. 1, the pipeline consists of four main stages: background referencing and standardization of the 32-bit quantitative phase maps, segmentation by a LoRA-adapted lightweight network, optimization with the Physics-Aware Phase Consistency Loss, and extraction of phase-derived cellular measurements.

**Why:** 'standardization of the quantitative phase maps' -> 'background referencing and standardization of the 32-bit ...' (preprocessing as implemented).

---

**M022 KEEP** "![](media/image1.png){width="7.255905511811024in" height="3.909448818897638in"}"  
*Note:* Fig. 1 must be REDRAWN: draw two branches from the phase map (background-referenced phase in rad -> PMC/BGA/IPP; per-image standardized phase -> network); do not draw the SAM prompt encoder or mask decoder (not used).

### M023 REPLACE: 2.1 Overall Framework

**Find:** "Fig. 1 Overview of the proposed physics-aware lightweight segmentation framework ..."

**Original:**

> **Fig. 1** Overview of the proposed physics-aware lightweight segmentation framework for quantitative holographic cell analysis. LoRA-adapted lightweight networks perform cell segmentation from quantitative phase images, while raw unnormalized phase information is directly incorporated into the Physics-Aware Phase Consistency Loss through the PMC, BGA, and IPP constraints. The resulting segmentation masks serve as physically meaningful measurement domains that preserve phase-integrated optical information for subsequent quantitative cellular analysis.

**New text (complete):**

**Fig. 1** Overview of the proposed physics-aware lightweight segmentation framework for quantitative holographic cell analysis. LoRA-adapted lightweight networks segment cells from the standardized phase image, while the background-referenced phase in radians enters the Physics-Aware Phase Consistency Loss through the PMC, BGA, and IPP constraints. The resulting segmentation masks serve as measurement domains that preserve phase-integrated optical information for subsequent quantitative cellular analysis.

**Why:** 'raw unnormalized phase' -> background-referenced phase; matches the redrawn figure.

---


## 2.2 QPI Preprocessing

### M025 REPLACE: 2.2 QPI Preprocessing

**Find:** "The input data consist of single-channel quantitative phase maps, reconstructed ..."

**Original:**

> The input data consist of single-channel quantitative phase maps, reconstructed as 32-bit floating-point arrays in radians. These maps represent the quantitative phase delay associated with the optical path difference introduced by the cells. Because the measured phase depends jointly on refractive-index contrast and specimen thickness, it is treated as an optical phase quantity rather than as a direct measurement of geometric cell thickness or volume. Ground-truth semantic masks contain five discrete classes: background, discocyte, echinocyte, spherocyte, and stomatocyte.

**New text (complete):**

The input data consist of single-channel quantitative phase maps reconstructed as 32-bit floating-point arrays in radians (768 × 768 pixels). These maps represent the quantitative phase delay associated with the optical path difference introduced by the cells. Because the measured phase depends jointly on refractive-index contrast and specimen thickness, it is treated as an optical phase quantity rather than as a direct measurement of geometric cell thickness or volume. Ground-truth semantic masks contain five classes, with mask values 0 to 4 assigned to background, discocyte, spherocyte, echinocyte, and stomatocyte, respectively. The masks had been drawn on 8-bit grey-level exports of the phase maps; they were transferred to the 32-bit maps by an affine enhanced-correlation-coefficient registration of each phase map to its 8-bit export (the estimated transforms consisted of scale and translation only) with nearest-neighbor resampling of the labels. Pixels of a phase map outside the field of view of its 8-bit export carry no annotation and are excluded from every loss term and every metric.

**Why:** Your current first sentence is kept. Added: map size, class order by mask value (2 = spherocyte, 3 = echinocyte; the old list had the two swapped), how the masks were carried onto the 32-bit maps, and the unannotated pixels (all as implemented, DECISIONS D-01, D-02).

---

### M026 REPLACE: 2.2 QPI Preprocessing

**Find:** "To adapt the single-channel phase maps to segmentation networks originally ..."

**Original:**

> To adapt the single-channel phase maps to segmentation networks originally pretrained on RGB images, phase normalization is applied strictly for numerical stability during forward propagation. The raw phase values are retained separately and are never replaced by the normalized values when evaluating the physics-aware constraints. Spatial resizing and cropping are performed to standardize the network input dimensions.

**New text (complete):**

The reconstructed phase maps do not share a common zero level: the background phase of the validation maps lay between 0.66 and 2.50 rad. Because an arbitrary constant c adds c·A to the phase integrated over a region of area A, every map was referenced to its own background before training and evaluation: the median phase of the annotated background pixels more than 5 pixels from any cell was subtracted, so that the background is 0 rad and the integrated phase of a cell measures the cell only. The same background-referenced phase is used for the physics-aware loss terms (PMC, BGA, and IPP), for the integrated-phase error, and for the dry-mass estimation. For the network input only, the referenced map is resized to the input size of each model (256 × 256 pixels for MobileNet-UNet and EdgeSAM, 1024 × 1024 pixels for MobileSAM) and standardized per image to zero mean and unit variance; the physics-aware terms receive the background-referenced phase in radians at the same resolution and never the standardized values.

**Why:** Background phase correction described as implemented (professor's point 5; R2-1, R3-5: the same correction is applied to the loss, the IPE and dry mass). Removes 'Spatial resizing and cropping' (there is no cropping). Offsets: Table S10.

---

### M027 REPLACE: 2.2 QPI Preprocessing

**Find:** "Data augmentation is restricted to transformations that preserve the quantitative ..."

**Original:**

> Data augmentation is restricted to transformations that preserve the quantitative phase values and cellular morphology. Rotation, horizontal and vertical flipping, and translation are therefore applied offline before training. Intensity perturbations, nonlinear contrast transformations, and phase-value modifications are excluded because they could alter the quantitative optical information that the proposed framework is designed to preserve.

**New text (complete):**

Data augmentation is restricted to transformations that preserve the quantitative phase values and cellular morphology. Each training image is used in its original orientation and in its three lossless 90° rotations. Intensity perturbations, nonlinear contrast transformations, and phase-value modifications are excluded because they could alter the quantitative optical information that the proposed framework is designed to preserve.

**Why:** Only 90° rotations are used (no flips, no translation); the paragraph is rewritten so it reads correctly.

---


## 2.3 Lightweight Segmentation Models

### M029 REPLACE: 2.3 Lightweight Segmentation Models

**Find:** "Three lightweight architectures are evaluated, covering convolutional, transformer-based, and hybrid ..."

**Original:**

> Three lightweight architectures are evaluated, covering convolutional, transformer-based, and hybrid designs. MobileNet-UNet pairs an ImageNet-pretrained MobileNetV2 encoder with a U-Net-style decoder with skip connections, which retains the fine spatial information required for accurate cellular boundary delineation \[32\]. MobileSAM and EdgeSAM retain the mask-decoding pipeline of the original SAM, with the heavyweight ViT-H encoder replaced by a distilled TinyViT backbone and a RepViT backbone respectively \[33, 34\]. EdgeSAM is evaluated as a computationally efficient architecture rather than as evidence of embedded deployment.

**New text (complete):**

Three lightweight architectures are evaluated, covering convolutional, transformer-based, and hybrid designs. MobileNet-UNet pairs an ImageNet-pretrained MobileNetV2 encoder with a U-Net-style decoder with skip connections, which retains the fine spatial information required for accurate cellular boundary delineation [32]. For MobileSAM and EdgeSAM, only the pretrained image encoders are used: the distilled TinyViT encoder of MobileSAM [33] and the RepViT encoder of EdgeSAM [34]. Because dense five-class segmentation requires no prompts, each encoder is combined with a compact decoder trained from scratch. In MobileSAM, a learned prompt embedding is added to five class tokens that attend to the image embedding through one transformer decoder layer, and each class map is the inner product of its token with the upsampled embedding; in EdgeSAM, four transposed-convolution stages upsample the embedding to the input resolution, followed by a 1 × 1 classifier. In all three networks, the first convolution is replaced by a single-channel convolution initialized with the mean of the pretrained RGB filters. EdgeSAM is evaluated as a computationally efficient architecture rather than as evidence of embedded deployment.

**Why:** The old text said MobileSAM and EdgeSAM 'retain the mask-decoding pipeline of the original SAM'; the code uses only their encoders with task decoders (qpi/models/mobile_sam.py, edge_sam.py). Kept short (R1-3).

---


## 2.4 LoRA-Based Lightweight Adaptation

**M031 KEEP** "LoRA is employed to adapt pretrained lightweight segmentation networks to the distinct statistical and ..."

**M032 KEEP** "$W' = W + (B cdot A) times frac{alpha}{r}$ , (1)"

### M033 REPLACE: 2.4 LoRA-Based Lightweight Adaptation

**Find:** "where W is the frozen pretrained weight matrix, B and ..."

**Original:**

> where *W* is the frozen pretrained weight matrix, *B* and *A* are trainable low-rank matrices, *r* is the rank of the low-dimensional adaptation space, and *α* is a scaling factor. To evaluate the balance between adaptation capacity and parameter efficiency, LoRA rank-sweep experiments are performed using the ranks specified in the experimental protocol.

**New text (complete):**

where *W* is the frozen pretrained weight matrix, *B* and *A* are trainable low-rank matrices, *r* is the rank of the low-dimensional adaptation space, and *α* is a scaling factor. We set *α* = *r*, so the low-rank update enters with unit scaling at every rank, and *B* is initialized to zero so that training starts from the pretrained model. To evaluate the balance between adaptation capacity, parameter efficiency, and training stability, a rank sweep *r* ∈ {2, 4, 8, 16, 32} is performed for every architecture.

**Why:** α = r as implemented (configs/experiments.yaml); the ranks are now stated.

---

### M034 REPLACE: 2.4 LoRA-Based Lightweight Adaptation

**Find:** "LoRA modules are inserted into architecture-dependent layers. For transformer-based encoders, ..."

**Original:**

> LoRA modules are inserted into architecture-dependent layers. For transformer-based encoders, low-rank matrices are introduced into the query and value projection layers, whereas convolutional bottleneck layers are adapted for CNN-based backbones. To address the structural domain shift from three-channel natural images to single-channel quantitative phase maps, the modified single-channel input stem and normalization parameters are fully unfrozen. The segmentation decoder and final projection heads also remain trainable to generate the five-class RBC segmentation output. Because the decoder and output heads remain trainable, the fraction of trainable parameters differs among architectures.

**New text (complete):**

LoRA modules are inserted into architecture-dependent layers. In the TinyViT encoder of MobileSAM, they are added to the fused query-key-value projections of the attention blocks; in the convolutional encoders of EdgeSAM and MobileNet-UNet, they are added to every non-grouped convolution and linear layer, while depthwise convolutions remain frozen. To address the structural domain shift from three-channel natural images to single-channel quantitative phase maps, the single-channel input stem and the normalization parameters are fully unfrozen. The decoder and the output head also remain trainable to generate the five-class RBC segmentation output, so the fraction of trainable parameters differs among architectures (Table S5). For inference, the low-rank matrices are merged into the frozen weights (*W*′ = *W* + *BA*), so latency and memory do not depend on the rank.

**Why:** 'query and value projection layers' is not what the code does (qpi/models/lora_utils.py). VERIFY on the server once that the MobileSAM LoRA layers are the qkv projections (07_REMAINING_WORK Q-D).

---


## 2.5 Physics-Aware Phase Consistency Loss

**M036 KEEP** "Conventional segmentation objectives optimize geometric agreement between predicted and annotated masks but do not ..."


## 2.5.1 Multi-Class Dice Loss

**M038 KEEP** "To mitigate class imbalance among rare morphology degradation types (e.g., stomatocytes) and ensure numerical ..."

**M039 KEEP** "$L_{Dice} = frac{1}{sum_{c = 0}{4}w_{c}}sum_{c = 0}{4}w_{c}left( 1 - frac{2|P_{c} cap G_{c}| + 1}{|P_{c}| ..."

### M040 REPLACE: 2.5.1 Multi-Class Dice Loss

**Find:** "where $w_{c} in { 0.5,1.0,1.5,2.0,2.0}$ correspond to the background and ..."

**Original:**

> where $w_{c} \in \{ 0.5,1.0,1.5,2.0,2.0\}$ correspond to the background and the four morphology classes, respectively. A Laplace smoothing factor of $+ 1$ prevents zero division on background-only patches.

**New text (complete):**

where $w_{c} \in \{ 0.5,1.0,1.5,2.0,2.0\}$ correspond to the background and the four morphology classes, respectively, $P_{c}$ and $G_{c}$ are the predicted probability map and the annotation of class *c*, and pixels without annotation are excluded from all sums. A Laplace smoothing factor of $+ 1$ prevents division by zero for images in which a class is absent. All loss terms are computed in 32-bit floating point, whereas the network forward pass uses FP16 mixed precision.

**Why:** Adds the exclusion of unannotated pixels and the float32 loss computation (as implemented).

---


## 2.5.2 Phase-Mask Contrast Term

### M042 REPLACE: 2.5.2 Phase-Mask Contrast Term

**Find:** "To ensure the predicted cell region physically maintains higher phase ..."

**Original:**

> To ensure the predicted cell region physically maintains higher phase values than the surrounding medium, the contrast term penalizes predictions where the mean background phase exceeds the mean cell phase by a predefined margin:

**New text (complete):**

To ensure that the predicted cell region maintains higher phase values than the surrounding medium, the contrast term penalizes predictions in which the mean cell phase does not exceed the mean background phase by at least a predefined margin:

**Why:** Wording made consistent with Eq. (3) (the hinge is active when the cell-background contrast is below the margin).

---

**M043 KEEP** "$L_{PMC} = text{max}(0,mu_{bg} - mu_{cell} + text{margin})$, (3)"

### M044 REPLACE: 2.5.2 Phase-Mask Contrast Term

**Find:** "where $mu_{cell}$ and $mu_{bg}$ denote the mean phase values inside ..."

**Original:**

> where $\mu_{cell}$ and $\mu_{bg}$ denote the mean phase values inside the predicted cellular foreground and the surrounding background, respectively. This term provides a global physical constraint on foreground localization but does not by itself guarantee accurate cellular boundaries.

**New text (complete):**

where $\mu_{cell}$ and $\mu_{bg}$ denote the mean background-referenced phase inside the predicted cellular foreground and in the surrounding background, respectively, each weighted by the soft foreground probability, and the margin is 0.1 rad. This term provides a global physical constraint on foreground localization but does not by itself guarantee accurate cellular boundaries.

**Why:** States the phase used and the margin value (configs/base.yaml).

---


## 2.5.3 Boundary-Gradient Alignment Term

### M046 REPLACE: 2.5.3 Boundary-Gradient Alignment Term

**Find:** "Because probabilistic network outputs $lbrack 0,1rbrack$ and physical phase measurements ..."

**Original:**

> Because probabilistic network outputs $\lbrack 0,1\rbrack$ and physical phase measurements exist on fundamentally incompatible scales, spatial gradient magnitudes for both the predicted mask ($\nabla M$) and the raw phase map ($\nabla\Phi$) are first max-normalized per-sample to a $\lbrack 0,1\rbrack$ scale. The segmentation boundary is then structurally aligned with strong phase gradients via an $L_{1}$ penalty:

**New text (complete):**

Because probabilistic network outputs $\lbrack 0,1\rbrack$ and physical phase measurements exist on incompatible scales, the spatial gradient magnitudes of the predicted mask ($\nabla M$) and of the background-referenced phase map ($\nabla\Phi$), computed by forward differences, are first max-normalized per sample to a $\lbrack 0,1\rbrack$ scale. The segmentation boundary is then aligned with strong phase gradients via an $L_{1}$ penalty:

**Why:** 'raw phase map' -> background-referenced phase map (R3-5).

---

**M047 KEEP** "$L_{BGA} = left. parallelfrac{nabla M}{text{max}(nabla M) + epsilon} - frac{nablaPhi}{text{max}(nablaPhi) + epsilon} right.parallel_{1}$. (4)"

### M048 REPLACE: 2.5.3 Boundary-Gradient Alignment Term

**Find:** "This term geometrically anchors the predicted measurement boundary to phase ..."

**Original:**

> This term geometrically anchors the predicted measurement boundary to phase transitions present in the measured QPI data.

**New text (complete):**

The $L_{1}$ difference is averaged over annotated pixels. This term anchors the predicted measurement boundary to phase transitions present in the measured QPI data.

---


## 2.5.4 Integrated-Phase Preservation Term

### M050 REPLACE: 2.5.4 Integrated-Phase Preservation Term

**Find:** "The key quantitative constraint of the proposed framework is preservation ..."

**Original:**

> The key quantitative constraint of the proposed framework is preservation of the phase signal integrated over the segmented cellular domain. For a raw quantitative phase map *φ~i~*, predicted foreground probability *M~i~*, and ground-truth foreground mask *G~i~*, the predicted and reference integrated phase quantities are $S_{pred} = \sum_{i}^{}{M_{i}\varphi_{i}},$ $S_{GT} = \sum_{i}^{}{G_{i}\varphi_{i}}$. The IPP loss is then defined as the ground-truth-normalized relative error:

**New text (complete):**

The key quantitative constraint of the proposed framework is preservation of the phase signal integrated over the segmented cellular domain. For a background-referenced quantitative phase map *φ~i~*, predicted foreground probability *M~i~*, and ground-truth foreground mask *G~i~*, the predicted and reference integrated phase quantities are $S_{pred} = \sum_{i}^{}{M_{i}\varphi_{i}},$ $S_{GT} = \sum_{i}^{}{G_{i}\varphi_{i}}$. The IPP loss is then defined as the ground-truth-normalized relative error:

**Why:** 'raw quantitative phase map' -> background-referenced (R3-5).

---

**M051 KEEP** "$L_{IPP} = frac{left| S_{pred} - S_{GT} right|}{left| S_{GT} right| + epsilon}.$ (5)"

### M052 REPLACE: 2.5.4 Integrated-Phase Preservation Term

**Find:** "The relative formulation improves numerical stability during mixed-precision training and ..."

**Original:**

> The relative formulation improves numerical stability during mixed-precision training and makes the penalty insensitive to the absolute scale of the phase integral. The IPP term directly constrains the integrated phase quantity used in subsequent optical-mass analysis. Under the conventional specific refractive increment assumption, this quantity is proportional to cellular dry mass; therefore, preserving the integrated phase during segmentation directly improves the reliability of downstream dry-mass estimation.

**New text (complete):**

The relative formulation ($\epsilon = 10^{- 4}$) makes the penalty insensitive to the absolute scale of the phase integral. Because *φ* is referenced to the background, $S_{GT}$ measures the cells only; without this referencing, a residual offset *c* would add *c* times the foreground area to both integrals. Because the sums run over the whole foreground of an image, IPP constrains the total integrated phase of the image; the error of individual cells is evaluated separately (Section 3.3.2). The IPP term directly constrains the integrated phase quantity used in subsequent optical-mass analysis. Under the conventional specific refractive increment assumption, this quantity is proportional to cellular dry mass; preserving the integrated phase during segmentation therefore targets the quantity required for downstream dry-mass estimation.

**Why:** R2-1 / R3-5 (offset sensitivity explained), R3-2 (IPP is image-level, per-cell error reported separately). 'directly improves the reliability' softened to 'targets'.

---


## 2.5.5 Final Loss Function

**M054 KEEP** "The final objective combines the biological segmentation task with the three physics-aware constraints:"

**M055 KEEP** "$L_{total} = L_{Dice} + lambda_{1}L_{PMC} + lambda_{2}L_{BGA} + lambda_{3}L_{IPP}$ , (6)"

### M056 REPLACE: 2.5.5 Final Loss Function

**Find:** "where $lambda_{1}$, $lambda_{2}$, and $lambda_{3}$ control the relative contributions of ..."

**Original:**

> where $\lambda_{1}$, $\lambda_{2}$, and $\lambda_{3}$ control the relative contributions of the PMC, BGA, and IPP terms, respectively. All physics-aware terms are evaluated using the binary cellular foreground probability rather than individual morphology classes. This separates the morphological classification objective from the physical requirement that the total cellular measurement domain be accurately preserved. As demonstrated by the ablation experiments, the individual constraints are complementary: PMC provides coarse phase-based foreground localization, BGA anchors the predicted boundary to phase transitions, and IPP ensures preservation of the integrated quantitative phase information.

**New text (complete):**

where $\lambda_{1} = 0.1$, $\lambda_{2} = 0.05$, and $\lambda_{3} = 0.1$ control the relative contributions of the PMC, BGA, and IPP terms, respectively. All physics-aware terms are evaluated using the binary cellular foreground probability $M = 1 - p_{background}$ rather than individual morphology classes, and each term is averaged over the images of a mini-batch that contain at least one annotated cell. This separates the morphological classification objective from the physical requirement that the total cellular measurement domain be accurately preserved. The contribution of each term is examined by the ablation in Section 4.4.

**Why:** Weights stated; the claim 'As demonstrated by the ablation experiments, the individual constraints are complementary ...' is removed because the corrected ablation does not separate the terms (Table S7).

---


## 2.6 Quantitative Phase-Derived Cellular Measurements

**M058 KEEP** "The principal purpose of the proposed physics-aware segmentation is to enable reliable quantitative cellular ..."

**M059 KEEP** "$A = N_{Omega},Delta x_{text{eff}}{2}$ , (7)"

**M060 KEEP** "where N~Ω~ is the number of pixels in Ω , and ∆x~eff~ is the ..."  
*Note:* Pixel pitch 0.1441 µm is the value used by the code; see Section 3.1 note.

**M061 KEEP** "$C = frac{4pi A}{P{2}}$ , (8)"

**M062 KEEP** "where P denotes the perimeter of the segmented component. Circularity characterizes the two-dimensional projected ..."

**M063 KEEP** "$S_{varphi} = sum_{i in Omega}{}varphi_{i},Delta x_{text{eff}}{2}$ , (9)"

### M064 REPLACE: $C = frac{4pi A}{P{2}}$ , (8)

**Find:** "Here, S~φ~ represents the area-integrated quantitative phase associated with the ..."

**Original:**

> Here, *S~φ~* represents the area-integrated quantitative phase associated with the cell, incorporating the coupled contributions of refractive-index contrast and optical thickness. It serves as the primary phase-derived optical quantity whose preservation is explicitly enforced by the IPP loss. Under the conventional assumption that cellular refractive-index increment is approximately proportional to dry-mass concentration, cellular dry mass can be estimated from the integrated phase as:

**New text (complete):**

Here, *φ~i~* is the phase of pixel *i* after subtraction of the local background level described below, and *S~φ~* represents the area-integrated quantitative phase associated with the cell, incorporating the coupled contributions of refractive-index contrast and optical thickness. It serves as the primary phase-derived optical quantity whose preservation is explicitly enforced by the IPP loss. Under the conventional assumption that the cellular refractive-index increment is approximately proportional to dry-mass concentration, cellular dry mass can be estimated from the integrated phase as:

**Why:** Defines φ as background-corrected (R3-5).

---

**M065 KEEP** "$m = frac{lambda}{2pialpha}S_{varphi}$ , (10)"

**M066 KEEP** "where λ is the illumination wavelength and α is the specific refractive increment of ..."

### M067 REPLACE: $C = frac{4pi A}{P{2}}$ , (8)

**Find:** "Per-cell measurements are obtained through connected-component labeling of the predicted ..."

**Original:**

> Per-cell measurements are obtained through connected-component labeling of the predicted foreground at the native patch resolution. Components outside the predefined projected-area range are excluded to remove isolated pixel artifacts and unresolved multi-cell clusters. Cell perimeter is calculated using the Crofton estimator to reduce sensitivity to pixelation. For dry-mass estimation, the background phase level is estimated from the non-cellular region of each quantitative phase map and subtracted to remove residual image-wise phase offsets. The integrated phase is converted to cellular dry mass using Eq. (10), with the phase scale calibrated to the mean corpuscular hemoglobin (MCH) reference. No background correction is applied when computing the segmentation loss or the global Integrated-Phase Error (IPE); both operate directly on the raw phase values. Background-phase compensation is applied at the dry-mass estimation stage described above and in the per-cell IPE statistic of Section 3.3.2.

**New text (complete):**

Per-cell measurements are obtained from connected components of the predicted foreground at the native 768 × 768 resolution. Components with a projected area outside 15–200 µm², which correspond to isolated pixel artifacts or unresolved multi-cell clusters, and components that touch an unannotated image region are excluded. The cell perimeter is calculated with the Crofton estimator to reduce sensitivity to pixelation. For dry-mass estimation, the background level is estimated as the median phase of the non-cellular pixels farther than 5 pixels from the analyzed mask and subtracted from the phase of each cell, so that each measurement uses only the mask under evaluation (predicted or annotated). Because the phase maps are in radians, Eq. (10) is applied with the physical constants only; no calibration to a reference hemoglobin content is used. Cells whose phase reaches the 2π upper limit of the reconstructed maps or shows a wrap discontinuity (phase within 0.3 rad of 2π, or a jump larger than π between neighboring pixels, within 2 pixels of the cell) are excluded from all phase-derived results; this rule was fixed before the results were computed.

**Why:** Removes 'phase scale calibrated to the MCH reference' (no calibration, DECISIONS D-13) and 'No background correction is applied when computing the segmentation loss or the global IPE' (now applied everywhere, R3-5). Adds the wrapped-cell rule (R2-3).

---


## 3. Experiments


## 3.1 Dataset and Imaging Configuration

### M070 REPLACE: 3.1 Dataset and Imaging Configuration

**Find:** "The experimental evaluation was designed to assess three core aspects ..."

**Original:**

> The experimental evaluation was designed to assess three core aspects of the proposed framework: (1) multi-class segmentation accuracy across different RBC morphologies, (2) preservation of quantitative phase information and optical-mass-related measurements, and (3) computational efficiency of the lightweight segmentation architectures. The framework was evaluated using a dataset of single-channel quantitative phase images acquired with a DHM system during long-term blood storage experiments \[26, 44\]. Blood samples (2 mL each) were collected from three healthy donors under a protocol approved by the Institutional Review Board of DGIST (Approval No. DGIST-180713-BR-012-01), with the same samples imaged repeatedly over the storage period. The quantitative phase maps were numerically reconstructed from off-axis holograms and represent the phase delay associated with the optical path difference introduced by the RBCs. The dataset comprises 1,090 densely annotated images acquired at eleven time points over 47 days of blood storage (days 0, 5, 8, 12, 15, 19, 23, 27, 30, 37, and 47), thereby capturing progressive storage-induced morphological changes in RBCs. The ground-truth semantic masks contain five classes: (1) background, (2) discocyte, (3) echinocyte, (4) spherocyte, and (5) stomatocyte. Cells were manually annotated and classified by an expert biologist using representative example morphologies as classification criteria. Cells were segmented into multi-label masks, and overlapping cells were separated into individual instances using a distance transform of the mask thresholded at 0.3 to construct binary marker maps. No cells superposed along the optical axis were identified among the annotated cells: the cells with the highest phase per pixel relative to the typical cell in the same image were single, optically dense cells rather than superposed pairs. Laterally touching cells and unresolved multi-cell clusters, which were most frequent at day 15, are discussed in Section 4.8. To provide sufficient training data while maintaining a separate validation set for evaluation, the dataset was divided into 1,024 training images and 66 validation images, corresponding to six validation patches at each storage time point. The quantitative phase maps were used directly, without further normalization, for integrated-phase and dry-mass calculations.

**New text (complete):**

The experimental evaluation was designed to assess four aspects of the proposed framework: (1) multi-class segmentation accuracy across different RBC morphologies, (2) preservation of quantitative phase information and optical-mass-related measurements, (3) computational efficiency, and (4) training stability of the lightweight segmentation architectures. The framework was evaluated using single-channel quantitative phase images acquired with a DHM system during long-term blood storage experiments [26, 44]. Blood samples (2 mL each) were collected from three healthy donors under a protocol approved by the Institutional Review Board of DGIST (Approval No. DGIST-180713-BR-012-01), with the same samples imaged repeatedly over the storage period. The quantitative phase maps were numerically reconstructed from off-axis holograms and represent the phase delay associated with the optical path difference introduced by the RBCs. Images were acquired at eleven time points over 47 days of storage (days 0, 5, 8, 12, 15, 19, 23, 27, 30, 37, and 47). Cells were manually annotated and classified by an expert biologist into discocytes, spherocytes, echinocytes, and stomatocytes using representative example morphologies as classification criteria. Cells were segmented into multi-label masks, and overlapping cells were separated into individual instances using a distance transform of the mask thresholded at 0.3 to construct binary marker maps. No cells superposed along the optical axis were identified among the annotated cells: the cells with the highest phase per pixel relative to the typical cell in the same image were single, optically dense cells rather than superposed pairs. Laterally touching cells and unresolved multi-cell clusters, which were most frequent at day 15, are discussed in Section 4.9. The provided split comprises 256 training images and 66 validation images (six per time point). Because 32-bit phase maps were available for 182 of the training images and for all validation images, the models were trained on 182 images, each with its three 90° rotations (728 training views), and evaluated on the 66 validation images. These contain 2,049 annotated cells with a projected area of 15–200 µm², of which 61 (3.0%) met the phase-wrapping criterion of Section 2.6 and were excluded from phase-derived results (Table S1).

**Why:** Replaces '1,090 images ... 1,024 training' (now 182 training images with 32-bit maps, 728 views, + 66 validation; D-05) and the last sentence about using the maps 'without further normalization'. Keeps the donors and IRB (R3-1), data source [26, 44] (R1-4) and the superposed-cell statement (R2-4). Adds 'training stability' as the fourth aspect (professor's point 2).

---

### M071 REPLACE: 3.1 Dataset and Imaging Configuration

**Find:** "Holograms were recorded with an off-axis transmission digital holographic microscope ..."

**Original:**

> Holograms were recorded with an off-axis transmission digital holographic microscope based on a Mach-Zehnder interferometer \[26, 44\], as shown in Fig. 2. The light source was a 666 nm laser diode. The sample was imaged with a 40×/0.75 NA microscope objective onto a CCD sensor (11.3 mm × 7.1 mm, 1920 × 1200 pixels of 5.86 µm × 5.86 µm, of which 1024 × 1024 pixels were used). The field of view was approximately 140 µm, with an effective pixel size at the sample of 0.1441 µm. The object wave was reconstructed numerically: the +1 diffraction order was isolated by spatial filtering in the Fourier domain to remove the zero order and twin image, and the phase was recovered using a Fresnel-approximation numerical diffraction calculation. Background phase aberrations were corrected using a numerical phase mask fitted to a fringe-free reference region of each hologram. Phase unwrapping was applied during reconstruction, and no annotated cell exceeded a phase of 2π.

**New text (complete):**

Holograms were recorded with an off-axis transmission digital holographic microscope based on a Mach-Zehnder interferometer [26, 44], as shown in Fig. 2. The light source was a 666 nm laser diode. The sample was imaged with a 40×/0.75 NA microscope objective onto a CCD sensor (11.3 mm × 7.1 mm, 1920 × 1200 pixels of 5.86 µm × 5.86 µm, of which 1024 × 1024 pixels were used). The field of view was approximately 140 µm, with an effective pixel size at the sample of 0.1441 µm. The object wave was reconstructed numerically: the +1 diffraction order was isolated by spatial filtering in the Fourier domain to remove the zero order and twin image, and the phase was recovered using a Fresnel-approximation numerical diffraction calculation. Background phase aberrations were corrected using a numerical phase mask fitted to a fringe-free reference region of each hologram. The reconstructed phase maps used in this study comprise 768 × 768 pixels and are confined to the interval [0, 2π] rad. Relative to the background, the peak phase of the annotated cells had a median of 2.15 rad (95th percentile 4.16 rad); cells whose phase reached the 2π limit were identified and excluded from phase-derived results as described in Section 2.6.

**Why:** Removes 'Phase unwrapping was applied during reconstruction, and no annotated cell exceeded a phase of 2π' (61 annotated cells, 3.0%, reach the limit). Answers R2-3 with the measured phase range (Table S1). Keeps the optical details (R1-4, R2-6). VERIFY with the data provider: 1024 × 1024 sensor pixels / 140 µm vs the 768 × 768 maps (07_REMAINING_WORK Q-A); the dry-mass check supports 0.1441 µm.

---

**M072 KEEP** "![](media/image3.png){width="7.259722222222222in" height="2.8409722222222222in"}"

**M073 KEEP** "Fig. 2 Schematic of the off-axis transmission Mach-Zehnder DHM setup used to acquire the ..."


## 3.2 Training Configuration

### M075 REPLACE: 3.2 Training Configuration

**Find:** "All networks were optimized using the AdamW optimizer with a ..."

**Original:**

> All networks were optimized using the AdamW optimizer with a cosine annealing learning-rate schedule. Mixed-precision (FP16) training was employed to reduce memory consumption and improve computational efficiency. A major challenge in longitudinal RBC morphology analysis is the substantial class imbalance caused by the relatively low prevalence of certain morphological classes, particularly at specific storage stages. To mitigate this imbalance, a two-level strategy was employed. At the data-loading level, a WeightedRandomSampler dynamically assigned sampling probabilities according to the inverse frequency of the constituent classes, thereby increasing the exposure of underrepresented RBC morphologies during training. At the optimization level, the class-weighted Multi-Class Dice Loss described in Section 2.5.1 was applied. LoRA rank-sweep experiments were conducted using $r \in \{ 2,4,8,16,32\}$ to determine an appropriate balance between adaptation capacity, segmentation performance, and parameter efficiency. All configurations reported in this study---including the LoRA rank sweep, comparison with full fine-tuning, and physics-aware loss ablation---were trained using an identical optimization schedule and evaluated on the same validation split to ensure a controlled comparison.

**New text (complete):**

All networks were trained for 50 epochs with the AdamW optimizer and a cosine annealing learning-rate schedule (initial learning rate 5 × 10⁻⁴ for MobileSAM and EdgeSAM and 10⁻³ for MobileNet-UNet; batch size 8 for MobileSAM and 16 for the other models). Mixed precision (FP16) was used in the forward pass to reduce memory consumption, while all loss terms were computed in 32-bit floating point. The substantial class imbalance of longitudinal RBC morphology data was addressed by the class-weighted Multi-Class Dice Loss of Section 2.5.1; mini-batches were drawn by uniform shuffling. Every LoRA configuration (three architectures, five ranks, and four loss variants; Section 3.5) was trained with three random seeds (42, 43, and 44), giving 180 training runs; full fine-tuning was trained once per architecture (seed 42). All configurations used the same optimization schedule and data. Because the validation images also serve as the evaluation images, no checkpoint was selected on them: every reported result is from the model after the final epoch.

**Why:** Removes the WeightedRandomSampler (not used, DECISIONS D-04). Adds learning rates, batch sizes, seeds, run count and the final-epoch checkpoint (professor's point 4).

---


## 3.3 Evaluation Metrics

### M077 REPLACE: 3.3 Evaluation Metrics

**Find:** "The proposed framework was evaluated using three complementary categories of ..."

**Original:**

> The proposed framework was evaluated using three complementary categories of metrics: segmentation performance, quantitative phase preservation, and computational efficiency.

**New text (complete):**

The proposed framework was evaluated using four categories of metrics: segmentation performance, quantitative phase preservation, computational efficiency, and training stability.

---


## 3.3.1 Segmentation Performance and Training Stability

### M079 REPLACE: 3.3.1 Segmentation Performance and Training Stability

**Find:** "Spatial agreement between predicted and ground-truth masks was evaluated using ..."

**Original:**

> Spatial agreement between predicted and ground-truth masks was evaluated using the Dice coefficient and Intersection-over-Union (IoU). Both metrics were macro-averaged exclusively over the four foreground RBC morphology classes. A morphology class that was not predicted by the model was assigned a score of zero rather than being excluded from the average. This evaluation strategy prevents the dominant background class from inflating the aggregate performance and enables failures in minority morphology classes to be directly reflected in the reported scores. To additionally evaluate instance-level separation in densely distributed cellular regions, the Aggregated Jaccard Index (AJI) was calculated.

**New text (complete):**

Spatial agreement between predicted and ground-truth masks was evaluated using the Dice coefficient and Intersection-over-Union (IoU) at the native 768 × 768 resolution, after bilinear upsampling of the network output. For each morphology class, Dice and IoU were computed from all validation pixels pooled and then macro-averaged over the four foreground classes. A morphology class that was not predicted by the model received a score of zero rather than being excluded from the average, so that the dominant background class cannot inflate the aggregate performance and failures in minority classes are reflected directly in the reported scores. Instance-level separation in densely populated regions was evaluated with the Aggregated Jaccard Index (AJI) on the connected components of the binary foreground. Training stability was assessed by repeating every configuration with three seeds: a run in which any morphology class reached a Dice below 0.05 is reported as having lost that class. Results are given as mean ± standard deviation over the three seeds, and every seed is reported in the Supplementary Information.

**Why:** States pooling, native grid, and the lost-class definition (professor's point 4). Heading renamed to include training stability.

---


## 3.3.2 Quantitative Phase Preservation

### M081 REPLACE: 3.3.2 Quantitative Phase Preservation

**Find:** "The physical fidelity of the predicted measurement boundaries was evaluated ..."

**Original:**

> The physical fidelity of the predicted measurement boundaries was evaluated using the IPE. This metric quantifies the relative difference between the phase-integrated optical quantities obtained from the predicted and ground-truth cellular regions, consistent with the integrated-phase definition in Eq. (9). For a predicted integrated phase *S~pred~* and the corresponding ground-truth value *S~GT~*, IPE is defined as

**New text (complete):**

The physical fidelity of the predicted measurement boundaries was evaluated using the integrated-phase error (IPE). This metric quantifies the relative difference between the phase-integrated optical quantities obtained from the predicted and ground-truth cellular regions on the background-referenced phase, consistent with the integrated-phase definition in Eq. (9). For a predicted integrated phase $S_{pred}$ and the corresponding ground-truth value $S_{GT}$, IPE is defined as

**Why:** IPE expanded at its first use (R1-11); background-referenced phase (R3-5).

---

**M082 KEEP** "$IPE(%) = frac{left| S_{pred} - S_{GT} right|}{left| S_{GT} right| + epsilon} times 100.$ (11)"

### M083 REPLACE: 3.3.2 Quantitative Phase Preservation

**Find:** "IPE was evaluated both globally and for individual RBC morphology ..."

**Original:**

> IPE was evaluated both globally and for individual RBC morphology classes. For each image, $S_{pred}$ and $S_{GT}$ are accumulated by summing the phase over all pixels in the relevant foreground region (the full foreground for the global formulation, or the corresponding class mask for class-specific IPE), and Eq. (11) is evaluated per image; the reported global IPE is the mean of these per-image values, and class-specific IPE is the mean over the images that contain that class. Because Eq. (11) operates on $\left| S_{pred} - S_{GT} \right|$, IPE is an absolute (unsigned) error, so over- and under-estimation in different images cannot cancel in the reported mean; within one image, however, the phase is summed over all cells before the absolute value is taken, which is why the per-cell IPE is also reported below. A morphology class with zero detections yields $S_{pred} = 0$ for that class, which is not excluded but is instead reflected directly in Eq. (11) as an IPE approaching 100%, consistent with the complete-detection-failure cases reported in Section 4.4. In addition to the global formulation in Eq. (11), a per-cell IPE was computed by matching each ground-truth cell (projected area 15 to 200 µm²) one-to-one with a predicted cell (IoU ≥ 0.5) and evaluating the relative phase-integral error for that cell individually, using phase after subtraction of a least-squares background plane; unmatched cells were excluded from this per-cell statistic. For the selected EdgeSAM configuration (*r* = 8), the per-cell median IPE was 3.04% (90th percentile 11.57%), with 73.7% of ground-truth cells successfully matched to a predicted cell. A lower IPE indicates that the predicted segmentation boundary more accurately preserves the phase-integrated optical information associated with the cell and therefore provides a more reliable measurement domain for subsequent optical-mass and dry-mass analysis. Boundary accuracy was additionally evaluated using the boundary F1-score with a 2-pixel tolerance. The globally pooled boundary F1-score quantifies the spatial agreement between predicted and annotated cellular contours and complements IPE by evaluating geometric boundary fidelity independently of the integrated phase quantity. Together, boundary F1 and IPE provide complementary measures of geometric boundary accuracy and optical-mass preservation, respectively.

**New text (complete):**

IPE was evaluated both globally and for individual RBC morphology classes. For each image, $S_{pred}$ and $S_{GT}$ are accumulated by summing the background-referenced phase over all annotated pixels in the relevant region (the full foreground for the global formulation, or the corresponding class mask for class-specific IPE), and Eq. (11) is evaluated per image; the reported global IPE is the mean of these per-image values, and class-specific IPE is the mean over the images that contain that class. Because Eq. (11) operates on $\left| S_{pred} - S_{GT} \right|$, IPE is an absolute (unsigned) error, so over- and underestimation in different images cannot cancel in the reported mean; within one image, however, the phase is summed over all cells before the absolute value is taken. A morphology class with zero detections yields $S_{pred} = 0$ for that class, which is not excluded but is reflected directly in Eq. (11) as an IPE of 100%. To assess the error of individual cells, a per-cell IPE was also computed by matching each ground-truth cell (projected area 15–200 µm²) one-to-one with a predicted cell (IoU ≥ 0.5) and evaluating the relative phase-integral error for that cell, using phase after subtraction of a least-squares background plane; unmatched and phase-wrapped cells were excluded from this per-cell statistic, and the fraction of matched cells is reported. A lower IPE indicates that the predicted segmentation boundary more accurately preserves the phase-integrated optical information associated with the cell. Boundary accuracy was additionally evaluated using the boundary F1-score with a 2-pixel tolerance, pooled over all images, excluding boundary pixels adjacent to unannotated regions. Together, the boundary F1-score and IPE measure geometric boundary accuracy and optical-mass preservation, respectively.

**Why:** Keeps the aggregation explanation (R3-4) and the per-cell IPE (R3-2); the old EdgeSAM r = 8 numbers (3.04%, 11.57%, 73.7%) move to Section 4.5 with the corrected values.

---


## 3.3.3 Computational Efficiency

**M085 KEEP** "Computational efficiency was characterized using median inference latency, tail latency, throughput in frames per ..."


## 3.4 Computational Benchmarking Protocol

### M087 REPLACE: 3.4 Computational Benchmarking Protocol

**Find:** "Computational benchmarking was performed on an NVIDIA RTX A5000 GPU ..."

**Original:**

> Computational benchmarking was performed on an NVIDIA RTX A5000 GPU to characterize the relative inference efficiency of the evaluated lightweight architectures. Profiling was conducted independently of the primary training procedure to avoid interference from training-related computational overhead. For stable latency measurement, each model underwent 50 unrecorded warm-up iterations followed by 500 synchronized forward passes. Inference performance was evaluated under FP32 and FP16 precision using both native PyTorch execution and, where supported, static-graph execution through ONNX Runtime (ORT) with the CUDA Execution Provider.

**New text (complete):**

Computational benchmarking was performed on an NVIDIA RTX A5000 GPU to characterize the relative inference efficiency of the evaluated lightweight architectures, independently of training. Each model was evaluated in strict 32-bit floating point (TF32 disabled) with batch size 1 at its native input size, both in native PyTorch and through ONNX Runtime (ORT) with the CUDA Execution Provider. For each runtime, 50 unrecorded warm-up iterations were followed by 500 synchronized forward passes on real validation phase maps kept on the GPU; for ORT, three independent sessions were timed. The procedure was repeated with three seeds. We report the median (P50) and 99th-percentile (P99) latency, the throughput (FPS = 1000/P50), and the peak GPU memory, measured as the device memory in use after the timed loop minus that before the model was loaded.

**Why:** FP16 removed (not part of the corrected benchmark, manuscript/benchmarking_protocol.md); memory definition changed (the old 16.4 MB was not ORT memory).

---

### M088 REPLACE: 3.4 Computational Benchmarking Protocol

**Find:** "The purpose of this experiment was to establish the relative ..."

**Original:**

> The purpose of this experiment was to establish the relative computational efficiency of the candidate architectures under a controlled GPU environment rather than to demonstrate actual embedded deployment. The resulting latency, throughput, memory, and parameter measurements therefore characterize the computational requirements of the proposed framework and its potential for future implementation on resource-constrained hardware. MobileSAM was excluded from ONNX benchmarking because structural operations within its cross-attention mask decoder were not fully supported by the ORT CUDA Execution Provider, resulting in partial CPU fallback. Native PyTorch inference was therefore used for MobileSAM when comparing computational performance.

**New text (complete):**

The purpose of this experiment was to establish the relative computational efficiency of the candidate architectures under a controlled GPU environment rather than to demonstrate actual embedded deployment. Because LoRA is merged into the weights before inference, latency and memory are independent of the rank and were measured once per architecture. MobileSAM is timed at its 1024-pixel input and the other models at 256 pixels, so latency differences between architectures combine architecture and input size. Before any ORT result was reported, the exported models were compared with the PyTorch models on all validation images under acceptance criteria fixed in advance (argmax agreement ≥ 99.9%, change in mean Dice ≤ 0.001, and mean per-image dry-mass difference ≤ 0.1%); all three models passed (Table S14).

**Why:** Removes 'MobileSAM was excluded from ONNX benchmarking ...' (MobileSAM is exported and passes the check).

---


## 3.5 Ablation Studies

### M090 REPLACE: 3.5 Ablation Studies

**Find:** "Three groups of ablation experiments were designed to systematically evaluate ..."

**Original:**

> Three groups of ablation experiments were designed to systematically evaluate the contribution of the major components of the proposed framework. First, a LoRA rank-sweep analysis was conducted using $r \in \{ 2,4,8,16,32\}$ to characterize the relationship among adaptation capacity, segmentation accuracy, quantitative phase preservation, and parameter efficiency. This experiment was used to identify an appropriate LoRA configuration for each lightweight architecture. The results are given in the Supplementary Information (Section S-I). Second, LoRA adaptation at the selected rank was directly compared with full fine-tuning of the corresponding network. The comparison was performed for all three architectures under the same training objective, optimization schedule, and validation split. This experiment evaluates whether parameter-efficient adaptation can preserve segmentation performance and minority morphology classes while substantially reducing the number of trainable parameters. Third, a component-wise ablation of the Physics-Aware Phase Consistency Loss was performed across all three lightweight architectures. Starting from the Multi-Class Dice-only baseline, the PMC, BGA, and IPP terms were progressively introduced. The resulting models were compared using segmentation metrics, boundary F1-score, and IPE. This ablation was designed to determine the individual and complementary contributions of the three physical constraints. In particular, it evaluates whether the proposed physics-aware objective improves not only geometric segmentation accuracy but also the preservation of the phase-integrated optical information required for reliable optical-mass and dry-mass analysis.

**New text (complete):**

Three groups of experiments were performed for all three architectures, each configuration with three random seeds. First, a LoRA rank sweep (*r* = 2, 4, 8, 16, and 32) characterized the relation among adaptation capacity, segmentation accuracy, integrated-phase preservation, and training stability; the complete sweep is given in the Supplementary Information. From this sweep, MobileSAM at *r* = 8 and EdgeSAM at *r* = 32 are presented as the preferred configurations, and all architectures are additionally compared at the same rank (*r* = 8). Second, LoRA was compared with full fine-tuning of each network under the same objective and schedule. Third, the Physics-Aware Phase Consistency Loss was ablated by progressively adding the PMC, BGA, and IPP terms to the Dice loss at every rank (60 configurations); the full objective is the rank-sweep run of the same rank and seed, so every comparison within a rank is paired by seed. The preferred ranks were chosen after inspection of the rank sweep on the evaluation images, because no separate selection set was available (Section 5).

**Why:** Professor's point 3 (preferred configurations + same-rank comparison + full sweep in the SI). Discloses how the ranks were chosen.

---


## 3.6 Statistical Analysis

### M092 REPLACE: 3.6 Statistical Analysis

**Find:** "Statistical analyses were performed to evaluate storage-associated changes in RBC ..."

**Original:**

> Statistical analyses were performed to evaluate storage-associated changes in RBC morphology and phase-derived quantitative measurements. Because cells segmented from the same QPI share common acquisition conditions, individual cells were not treated as statistically independent observations. The image was therefore used as the unit of analysis, with each image represented by the median of its detected cells. Cell-level distributions were reported as descriptive summaries. Monotonic associations between quantitative measurements and storage duration were evaluated using Spearman rank correlation. Differences between the early discocyte-dominated storage period and the terminal storage period were evaluated using a two-sided Mann-Whitney test. For phase-integrated quantities that co-vary with projected area, partial Spearman correlation controlling for area was used to determine whether an independent association with storage duration remained. Agreement between measurements derived from predicted and manually annotated boundaries was assessed using Pearson correlation of the per-image medians, Lin's concordance correlation coefficient (CCC) \[45\], the median absolute relative error (MdARE), and Bland-Altman bias \[46\]. MdARE was computed as the median, across images, of the absolute difference between predicted-mask and annotation-mask dry mass divided by the annotation-mask dry mass, using per-image median dry-mass values. Bland-Altman bias is the mean predicted-minus-annotation difference in pg.

**New text (complete):**

Statistical analyses were performed to evaluate storage-associated changes in RBC morphology and phase-derived quantitative measurements. Because cells segmented from the same QPI share common acquisition conditions, individual cells were not treated as statistically independent observations. The image was therefore used as the unit of analysis, with each image represented by the median of its detected cells. Cell-level distributions were reported as descriptive summaries. Monotonic associations between quantitative measurements and storage duration were evaluated using Spearman rank correlation. Differences between the early discocyte-dominated storage period (days 0–8, 18 images) and the terminal storage period (days 37 and 47, 12 images) were evaluated using a two-sided Mann-Whitney test. For phase-integrated quantities that co-vary with projected area, partial Spearman correlation controlling for area was used to determine whether an independent association with storage duration remained. Agreement between dry-mass estimates derived from predicted and manually annotated boundaries was assessed on per-image median dry mass by the median absolute relative error (MdARE: the median, across images, of the absolute difference between predicted-mask and annotation-mask dry mass divided by the annotation-mask dry mass), the Bland-Altman bias (mean predicted-minus-annotation difference) with 95% limits of agreement [46], Pearson correlation, and Lin's concordance correlation coefficient (CCC) [45]. Because correlation coefficients depend on the spread of the reference values, agreement was judged primarily by MdARE, bias, and the limits of agreement. Differences between model configurations were computed per seed (paired by seed) and are reported with 95% t-intervals; with three seeds these intervals are wide and are used descriptively rather than as significance tests.

**Why:** Keeps MdARE (R3-3); adds early/late definitions, limits of agreement, and the seed statistics (professor's point 4).

---


## 4. Results

### M094 REPLACE: 4. Results

**Find:** "The proposed lightweight, physics-aware segmentation framework was evaluated using the ..."

**Original:**

> The proposed lightweight, physics-aware segmentation framework was evaluated using the DHM dataset to assess three principal aspects: multi-class segmentation performance, preservation of phase-integrated optical information for quantitative cell analysis, and computational efficiency. Particular emphasis was placed on determining whether accurate segmentation boundaries could simultaneously support reliable optical-mass-related measurements across different RBC morphologies.

**New text (complete):**

The framework was evaluated on segmentation accuracy, training stability, integrated-phase and optical-mass preservation, and computational efficiency. Unless stated otherwise, values are means ± standard deviation over three seeds of the final-epoch models, computed on the 66 validation images at their native resolution. MobileSAM at *r* = 8 and EdgeSAM at *r* = 32 are presented as the preferred configurations and MobileNet-UNet at *r* = 8 as a convolutional baseline, and all architectures are also compared at the same rank (*r* = 8).

---


## 4.1 Segmentation Accuracy

### M095 REPLACE: 4.1 Segmentation Accuracy

**Find:** "Segmentation Accuracy Comparison"

**Original:**

> 1.  **Segmentation Accuracy Comparison**

**New text (complete):**

## 4.1 Segmentation Accuracy

---

### M096 REPLACE: 4.1 Segmentation Accuracy

**Find:** "Three lightweight architectures---EdgeSAM, MobileSAM, and MobileNet-UNet---were first evaluated to identify ..."

**Original:**

> Three lightweight architectures---EdgeSAM, MobileSAM, and MobileNet-UNet---were first evaluated to identify a suitable model for quantitative holographic cell segmentation. LoRA was applied to adapt the pretrained architectures to the distinct characteristics of single-channel QPI phase maps. The models were evaluated in terms of multi-class segmentation accuracy, instance separation, boundary fidelity, and trainable parameter efficiency. As summarized in Table 1, the three architectures exhibited substantially different performance characteristics, indicating that no single aggregate segmentation metric was sufficient for model selection. MobileSAM achieved the highest mean Dice score (0.812) but showed relatively poor instance separation and boundary quality, with an AJI of 0.249 and Boundary F1 of 0.644. Qualitative inspection further showed that adjacent cells in densely populated regions were frequently merged into single components.

**New text (complete):**

Table 1 lists the preferred configurations and the comparison at the same rank. MobileSAM (*r* = 8) reached the highest mean Dice (0.810 ± 0.006) and boundary F1-score (0.802 ± 0.011) and the highest Dice for every morphology class, including echinocytes (0.760 ± 0.006) and stomatocytes (0.708 ± 0.020) (Fig. 3). EdgeSAM (*r* = 32) reached a mean Dice of 0.738 ± 0.006 and separated individual cells better than MobileSAM (AJI 0.335 ± 0.020 vs 0.262 ± 0.017).

**Why:** Every number of the old paragraph is outdated; EdgeSAM is no longer 'selected' (professor's point 2: objective comparison).

---

### M097 REPLACE: 4.1 Segmentation Accuracy

**Find:** "TABLE 1 Segmentation performance at the selected LoRA rank (r = 8). Best value ..." (the old table: caption and all 3 table block(s))

**Original caption:**

> **TABLE 1** Segmentation performance at the selected LoRA rank (*r* = 8). Best value per column is shown in bold. Mean Dice is macro-averaged over the four foreground morphology classes. Dice~echi~ and Dice~stom~ denote per-class Dice scores for echinocytes and stomatocytes, respectively; 0.000 indicates complete class-detection failure, for which aggregate boundary metrics require cautious interpretation.

**New text (complete):**

**TABLE 1** Segmentation performance of the preferred LoRA configurations (MobileSAM *r* = 8, EdgeSAM *r* = 32) and of all architectures at the same rank (*r* = 8), full physics-aware objective. Mean ± SD over three seeds (42, 43, 44); final-epoch models; native 768 × 768 grid. "Runs with a lost class": seeds in which a class had Dice < 0.05 (sphero(2) = spherocyte, stomato(4) = stomatocyte). Per-class Dice: Fig. 3 and Table S2.

| Architecture | LoRA rank | Role | Trainable params | Mean Dice | Mean IoU | AJI | Boundary F1 | Global IPE (%) | Runs with a lost class |
|---|---|---|---|---|---|---|---|---|---|
| MobileSAM | 8 | preferred; same rank | 0.67 M (10.0%) | 0.810 ± 0.006 | 0.690 ± 0.008 | 0.262 ± 0.017 | 0.802 ± 0.011 | 1.32 ± 0.09 | 0/3 |
| EdgeSAM | 32 | preferred | 1.47 M (21.2%) | 0.738 ± 0.006 | 0.598 ± 0.007 | 0.335 ± 0.020 | 0.642 ± 0.005 | 3.02 ± 0.84 | 0/3 |
| EdgeSAM | 8 | same rank | 0.53 M (8.9%) | 0.647 ± 0.114 | 0.514 ± 0.095 | 0.249 ± 0.043 | 0.608 ± 0.009 | 3.81 ± 1.82 | 1/3 (s44: sphero(2)) |
| MobileNet-UNet | 8 | baseline; same rank | 5.58 M (71.8%) | 0.412 ± 0.004 | 0.353 ± 0.006 | 0.551 ± 0.053 | 0.814 ± 0.003 | 2.60 ± 0.07 | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2), stomato(4)) |

Seed-paired differences at *r* = 8 (mean [95% t-interval]):

| Comparison (seed-paired, r = 8) | Δ Mean Dice | Δ AJI | Δ Boundary F1 | Δ Global IPE (pp) |
|---|---|---|---|---|
| MobileSAM minus EdgeSAM | +0.164 [-0.133, +0.461] | +0.012 [-0.088, +0.113] | +0.194 [+0.145, +0.243] | -2.49 [-7.15, +2.17] |
| MobileNet-UNet minus EdgeSAM | -0.235 [-0.508, +0.038] | +0.301 [+0.133, +0.469] | +0.206 [+0.181, +0.231] | -1.21 [-5.70, +3.28] |

**Why:** Old Table 1 (single run, old class names) is replaced; generated from artifacts/tables/main/T2_preferred and T3_same_rank.

---

### M098 REPLACE: 4.1 Segmentation Accuracy

**Find:** "![image](media/image5.png){width="3.6023622047244093in" height="2.47244094488189in"}"

**Original:**

> ![image](media/image5.png){width="3.6023622047244093in" height="2.47244094488189in"}

**New text (complete):**

*[insert figure file artifacts/figures/main/F3_per_class_dice.png]*

---

### M099 REPLACE: 4.1 Segmentation Accuracy

**Find:** "Fig. 3 Per-class Dice scores of MobileNet-UNet, MobileSAM, and EdgeSAM ..."

**Original:**

> **Fig. 3 Per-class** Dice scores of MobileNet-UNet, MobileSAM, and EdgeSAM at the selected LoRA rank (*r* = 8). Zero-height bars indicate complete failure to detect the corresponding morphology class.

**New text (complete):**

**Fig. 3** Per-class Dice of MobileSAM (*r* = 8), EdgeSAM (*r* = 32 and *r* = 8), and MobileNet-UNet (*r* = 8). Bars: mean over three seeds; circles: individual seeds; crosses: seeds in which the class was lost (Dice < 0.05). Numbers in parentheses are mask values.

---

### M100 REPLACE: 4.1 Segmentation Accuracy

**Find:** "The per-class Dice scores shown in Fig. 3 further reveal ..."

**Original:**

> The per-class Dice scores shown in Fig. 3 further reveal substantial differences in morphology-specific segmentation performance. MobileNet-UNet completely failed to detect echinocytes and stomatocytes, yielding a Dice score of zero for both classes, despite its nominally high Boundary F1 (0.971) and AJI (0.634). In contrast, MobileSAM and EdgeSAM maintained non-zero Dice scores across all four RBC morphologies. These results demonstrate that aggregate boundary and instance metrics alone can obscure complete segmentation failure for minority morphology classes. EdgeSAM provided the most balanced performance across the evaluated criteria. It achieved a mean Dice of 0.727, AJI of 0.484, and Boundary F1 of 0.952 while maintaining non-zero segmentation performance for all four RBC morphologies. In particular, as illustrated in Fig. 3, EdgeSAM achieved Dice scores of 0.776 and 0.536 for echinocytes and stomatocytes, respectively. These results are particularly relevant because both classes are less represented in the dataset and are therefore more susceptible to degradation during model adaptation. For quantitative holographic cell analysis, balanced segmentation performance is important because the predicted cellular boundary defines the spatial measurement domain over which the quantitative phase is subsequently integrated. Reliable quantitative analysis therefore requires not only high region overlap but also complete morphology coverage, accurate boundary localization, and adequate separation of individual cells. A model that completely misses a particular morphology or merges adjacent cells cannot provide reliable per-cell phase-integrated measurements, even when its aggregate segmentation metrics appear favorable.

**New text (complete):**

MobileNet-UNet reached a high boundary F1-score (0.814) and the highest AJI (0.551), but it did not predict a single spherocyte or stomatocyte pixel in any seed, so its mean Dice was 0.412. Boundary F1 and AJI are computed on the binary foreground, which MobileNet-UNet delineates well; they are blind to the class assigned to each cell. Aggregate boundary and instance metrics alone therefore do not reveal the loss of whole morphology classes, and per-class results must be read together with them. For quantitative holographic analysis, complete morphology coverage matters because the predicted region of each class defines where its phase is integrated.

**Why:** R1-8 (why MobileNet-UNet has high values despite missing classes). Class names corrected: it loses spherocytes (2) and stomatocytes (4), not echinocytes.

---

### M101 REPLACE: 4.1 Segmentation Accuracy

**Find:** "Among the evaluated architectures, EdgeSAM was the only model that ..."

**Original:**

> Among the evaluated architectures, EdgeSAM was the only model that simultaneously maintained non-degenerate segmentation across all four RBC morphologies, high boundary fidelity, and adequate instance separation while requiring only 0.53 M trainable parameters. MobileSAM required substantially more trainable parameters and showed limited instance separation, whereas MobileNet-UNet, despite its high nominal Boundary F1 and AJI, failed to identify two morphology classes. Visual comparisons further corroborated these quantitative findings. MobileNet-UNet produced empty prediction masks for several minority-class cells, whereas MobileSAM frequently merged neighboring cells into single segmented regions. In contrast, EdgeSAM provided individualized cellular contours across the evaluated RBC morphologies, including the more complex boundaries associated with echinocytic cells.

**New text (complete):**

At the same rank (*r* = 8), MobileSAM exceeded EdgeSAM in boundary F1-score (paired difference +0.194, 95% interval +0.145 to +0.243). The differences in mean Dice (+0.164, −0.133 to +0.461) and in global IPE (−2.49 percentage points, −7.15 to +2.17) were not separable from seed variation, because one of the three EdgeSAM seeds lost the spherocyte class at *r* = 8 (Section 4.2).

**Why:** Removes 'EdgeSAM was the only model ...' (not supported). Same-rank comparison (professor's point 3); values from Table 1 source (T3).

---

### M102 DELETE: 4.1 Segmentation Accuracy

**Find:** "The aggregate comparison across criteria is shown in Fig. S-2; ..."

**Original:**

> The aggregate comparison across criteria is shown in Fig. S-2; because it does not encode morphology-class coverage, the high Boundary F1 and AJI of MobileNet-UNet must be read together with its complete failure on echinocytes and stomatocytes (Fig. 3). MobileSAM provided the highest mean Dice but was limited by poor instance separation and a substantially larger trainable parameter count. EdgeSAM, in contrast, provided the most balanced overall profile across the criteria required for quantitative holographic cell analysis. Based on this combined evaluation, EdgeSAM was selected as the primary architecture for subsequent quantitative phase preservation, physics-aware loss, and computational efficiency analyses. Its ability to maintain segmentation across all RBC morphologies while providing accurate cellular boundaries and adequate instance separation makes it particularly suitable for the proposed framework, in which segmentation masks serve as measurement domains for preserving phase-integrated optical information and supporting subsequent optical-mass and dry-mass analysis.

**Delete the whole block.**

**Why:** The radar chart (old Fig. S-2) is removed (it hid lost classes); the EdgeSAM selection statements are no longer supported.

---

### M103 DELETE: 4.1 Segmentation Accuracy

**Find:** "LoRA rank was selected for EdgeSAM from a sweep over ..."

**Original:**

> LoRA rank was selected for EdgeSAM from a sweep over *r* = 2, 4, 8, 16, and 32. Rank 8 gave the best balance of segmentation accuracy, instance separation, IPE, and trainable parameters (0.53 M), and was used in all subsequent analyses (Table S-1, Fig. S-1).

**Delete the whole block.**

**Why:** Moved and rewritten in the new Section 4.2 (rank 8 is not the best EdgeSAM rank).

---


## 4.2 LoRA Rank and Training Stability

### M104 NEW: 4.2 LoRA Rank and Training Stability

**Insert:**

## 4.2 LoRA Rank and Training Stability

**Why:** New subsection for the rank sweep and training instability (professor's points 3 and 4).

---

### M105 NEW: 4.2 LoRA Rank and Training Stability

**Insert:**

Across the rank sweep (Fig. 4, Table S2), MobileSAM was stable from *r* = 2 to *r* = 16 (mean Dice 0.808 to 0.813, seed SD at most 0.006). At *r* = 32, one seed lost the spherocyte and stomatocyte classes; its validation Dice dropped after the first epoch and did not recover (Fig. 5). EdgeSAM improved with rank in AJI and boundary F1-score and was stable at *r* = 4 and *r* = 32, but at *r* = 2, 8, and 16 the same seed (44) lost the spherocyte class (Table 2). EdgeSAM converged late, with a step increase of the validation Dice between about epochs 15 and 25; in the unstable runs the Dice stayed near 0.5 and this step did not occur (Fig. 5). Between EdgeSAM *r* = 32 and *r* = 8, the mean Dice difference was +0.091 (−0.196 to +0.379) and the AJI difference +0.086 (+0.026 to +0.146) (Table S4). MobileNet-UNet lost the spherocyte class in all 15 runs, which is a systematic failure rather than instability.

---

### M106 NEW: 4.2 LoRA Rank and Training Stability

**Insert:**

Increasing the rank therefore did not improve MobileSAM, whereas EdgeSAM benefited from the larger adaptation capacity; we present MobileSAM at *r* = 8 and EdgeSAM at *r* = 32 for this reason, without claiming an optimal rank. LoRA is merged into the weights for inference, so the rank changes the number of trainable parameters (Table S5) but not the inference cost.

---

### M107 NEW: 4.2 LoRA Rank and Training Stability

**Insert:**

**TABLE 2** Training stability: number of seeds (of three) in which a morphology class was lost (Dice < 0.05), with the seed and the lost classes, for every architecture and LoRA rank, with the full physics-aware objective and with the Dice loss only.

| Objective | Architecture | r = 2 | r = 4 | r = 8 | r = 16 | r = 32 |
|---|---|---|---|---|---|---|
| Full physics-aware | MobileSAM | 0/3 | 0/3 | 0/3 | 0/3 | 1/3 (s43: sphero(2), stomato(4)) |
| Full physics-aware | EdgeSAM | 1/3 (s44: sphero(2)) | 0/3 | 1/3 (s44: sphero(2)) | 1/3 (s44: sphero(2)) | 0/3 |
| Full physics-aware | MobileNet-UNet | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2), stomato(4)) | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2), stomato(4)) | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2), stomato(4)) | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2), stomato(4)) | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2)) |
| Dice only | MobileSAM | 1/3 (s42: sphero(2)) | 0/3 | 0/3 | 0/3 | 2/3 (s42: sphero(2); s43: sphero(2)) |
| Dice only | EdgeSAM | 0/3 | 1/3 (s43: sphero(2)) | 1/3 (s43: sphero(2)) | 2/3 (s43: sphero(2); s44: sphero(2)) | 2/3 (s43: sphero(2); s44: sphero(2)) |
| Dice only | MobileNet-UNet | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2), stomato(4)) | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2)) | 2/3 (s42: sphero(2), echino(3), stomato(4); s44: sphero(2), stomato(4)) | 2/3 (s42: sphero(2), echino(3), stomato(4); s44: sphero(2)) | 3/3 (s42: sphero(2), echino(3), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2), stomato(4)) |

---

### M108 NEW: 4.2 LoRA Rank and Training Stability

**Insert:**

*[insert figure file artifacts/figures/main/F4_rank_sweep.png]*

**Fig. 4** LoRA rank sweep with the full physics-aware objective: mean Dice, AJI, boundary F1-score, and global IPE against rank. Lines: mean over three seeds; circles: individual seeds; crosses: seeds in which a class was lost.

---

### M109 NEW: 4.2 LoRA Rank and Training Stability

**Insert:**

*[insert figure file artifacts/figures/main/F5_training_curves.png]*

**Fig. 5** Validation Dice during training for the three seeds of MobileSAM (*r* = 8, 32) and EdgeSAM (*r* = 8, 32). Dashed lines: runs that lost a class. Training curves of all ranks: Fig. S2.

---


## 4.3 Parameter-Efficient Adaptation versus Full Fine-Tuning

### M110 REPLACE: 4.3 Parameter-Efficient Adaptation versus Full Fine-Tuning

**Find:** "Parameter-Efficient Adaptation versus Full Fine-Tuning"

**Original:**

> 2.  **Parameter-Efficient Adaptation versus Full Fine-Tuning**

**New text (complete):**

## 4.3 Parameter-Efficient Adaptation versus Full Fine-Tuning

---

### M111 REPLACE: 4.3 Parameter-Efficient Adaptation versus Full Fine-Tuning

**Find:** "To evaluate the effectiveness of parameter-efficient adaptation, LoRA at the ..."

**Original:**

> To evaluate the effectiveness of parameter-efficient adaptation, LoRA at the selected rank (*r* = 8) was compared with full fine-tuning for all three architectures under the same training and evaluation conditions. As summarized in Table 2, LoRA provided more stable adaptation, particularly for the minority RBC morphology classes. For EdgeSAM, full fine-tuning reduced the mean Dice from 0.727 to 0.557 and caused complete collapse of the echinocyte class (Dice: 0.776 to 0.000), while increasing the IPE from 3.63% to 5.08%. Similar minority-class degradation was observed for MobileSAM, whereas full fine-tuning did not resolve the class-collapse problem of MobileNet-UNet.

**New text (complete):**

To evaluate the effectiveness of parameter-efficient adaptation, LoRA was compared with full fine-tuning of all three architectures under the same objective, schedule, and data (Table 3, Fig. S5). Full fine-tuning, which updated 5.7 to 7.6 M parameters and was trained with one seed, lost morphology classes in all three architectures: the spherocyte class for EdgeSAM (mean Dice 0.559, global IPE 8.31%), the spherocyte, echinocyte, and stomatocyte classes for MobileSAM (mean Dice 0.217), and, as with LoRA, the spherocyte and stomatocyte classes for MobileNet-UNet. With the same seed, the LoRA-adapted MobileSAM (*r* = 8) and EdgeSAM (*r* = 32) detected all four classes.

**Why:** Old numbers and class names (echinocyte collapse) replaced.

---

### M112 REPLACE: 4.3 Parameter-Efficient Adaptation versus Full Fine-Tuning

**Find:** "Fig. 4 further illustrates the advantage of LoRA over full ..."

**Original:**

> Fig. 4 further illustrates the advantage of LoRA over full fine-tuning in preserving morphology-specific segmentation performance. Despite using substantially fewer trainable parameters, LoRA maintained more balanced performance across RBC morphologies, whereas unrestricted full fine-tuning resulted in severe degradation or complete loss of minority classes for EdgeSAM and MobileSAM. These results indicate that increasing the number of trainable parameters does not necessarily improve adaptation to QPI phase images. Instead, constraining adaptation to low-rank parameter updates helps retain useful pretrained representations while providing sufficient flexibility for the QPI domain. Among the evaluated configurations, LoRA-adapted EdgeSAM provided the most favorable balance of multi-class segmentation performance, integrated-phase preservation, and parameter efficiency.

**New text (complete):**

These results indicate that increasing the number of trainable parameters does not necessarily improve adaptation to QPI phase images. Constraining adaptation to low-rank updates helps to retain the pretrained representations while providing sufficient flexibility for the QPI domain, which matters most for the minority morphology classes of this small dataset. Because full fine-tuning was trained with a single seed, this comparison is indicative.

**Why:** Removes 'LoRA-adapted EdgeSAM provided the most favorable balance'.

---

### M113 REPLACE: 4.3 Parameter-Efficient Adaptation versus Full Fine-Tuning

**Find:** "TABLE 2 Comparison of LoRA (r = 8) and full fine-tuning across the three ..." (the old table: caption and all 6 table block(s))

**Original caption:**

> **TABLE 2** Comparison of LoRA (*r* = 8) and full fine-tuning across the three architectures. Best result per architecture is shown in bold. Dice~echi~ and Dice~stom~ denote per-class Dice scores for echinocytes and stomatocytes, respectively, and IPE denotes the global Integrated-Phase Error. All models were evaluated on the same validation split using the full physics-aware objective.

**New text (complete):**

**TABLE 3** LoRA versus full fine-tuning (full physics-aware objective). LoRA rows: preferred rank, mean ± SD over three seeds, and the seed-42 run alone; full fine-tuning: single seed-42 run. Per-class Dice: Table S8 and Fig. S5.

| Architecture | LoRA rank | Seeds | Trainable params | Mean Dice | AJI | Boundary F1 | Global IPE (%) | Runs with a lost class |
|---|---|---|---|---|---|---|---|---|
| MobileSAM | 8 | 42, 43, 44 | 0.67 M (10.0%) | 0.810 ± 0.006 | 0.262 ± 0.017 | 0.802 ± 0.011 | 1.32 ± 0.09 | 0/3 |
| MobileSAM | 8 (seed 42 only) | 42 | 0.67 M (10.0%) | 0.810 | 0.243 | 0.792 | 1.42 | none |
| MobileSAM | full fine-tuning | 42 | 6.66 M (100.0%) | 0.217 | 0.448 | 0.808 | 2.16 | sphero(2), echino(3), stomato(4) |
| EdgeSAM | 32 | 42, 43, 44 | 1.47 M (21.2%) | 0.738 ± 0.006 | 0.335 ± 0.020 | 0.642 ± 0.005 | 3.02 ± 0.84 | 0/3 |
| EdgeSAM | 32 (seed 42 only) | 42 | 1.47 M (21.2%) | 0.731 | 0.329 | 0.647 | 2.66 | none |
| EdgeSAM | full fine-tuning | 42 | 5.69 M (100.0%) | 0.559 | 0.278 | 0.635 | 8.31 | sphero(2) |
| MobileNet-UNet | 8 | 42, 43, 44 | 5.58 M (71.8%) | 0.412 ± 0.004 | 0.551 ± 0.053 | 0.814 ± 0.003 | 2.60 ± 0.07 | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2), stomato(4)) |
| MobileNet-UNet | 8 (seed 42 only) | 42 | 5.58 M (71.8%) | 0.412 | 0.603 | 0.816 | 2.52 | sphero(2), stomato(4) |
| MobileNet-UNet | full fine-tuning | 42 | 7.62 M (100.0%) | 0.418 | 0.642 | 0.825 | 2.17 | sphero(2), stomato(4) |

---

### M114 DELETE: 4.3 Parameter-Efficient Adaptation versus Full Fine-Tuning

**Find:** "![image](media/image6.png){width="5.830708661417323in" height="2.4921259842519685in"} Fig. 4 Per-class Dice scores comparing LoRA (r ..."

**Original:**

> ![image](media/image6.png){width="5.830708661417323in" height="2.4921259842519685in"}
>
> **Fig. 4** Per-class Dice scores comparing LoRA (*r* = 8) with full fine-tuning across the three lightweight architectures. Zero-height bars indicate complete failure to detect the corresponding morphology class.

**Delete the whole block.**

**Why:** Old Fig. 4 moves to the SI (Fig. S5); Table 3 carries the numbers (R1-5: keep the main text short).

---


## 4.4 Physics-Aware Loss Ablation

### M115 REPLACE: 4.4 Physics-Aware Loss Ablation

**Find:** "Physics-Aware Loss Ablation"

**Original:**

> 3.  **Physics-Aware Loss Ablation**

**New text (complete):**

## 4.4 Physics-Aware Loss Ablation

---

### M116 REPLACE: 4.4 Physics-Aware Loss Ablation

**Find:** "To evaluate the contribution of each physics-aware constraint, a component-wise ..."

**Original:**

> To evaluate the contribution of each physics-aware constraint, a component-wise ablation was performed across all three architectures at *r* = 8. Four configurations were compared: (i) Multi-Class Dice loss alone, (ii) Dice with PMC, (iii) Dice with PMC and BGA, and (iv) the full objective incorporating IPP. Table 3 summarizes the global segmentation and integrated-phase metrics, while Table 4 provides the corresponding morphology-specific results for EdgeSAM. Note that the EdgeSAM *r* = 8 values in Table 3 (Dice 0.740, IPE 3.04%) differ slightly from the corresponding *r* = 8 entry in Table S-1 (Dice 0.727, IPE 3.63%); both use identical nominal hyperparameters but were trained as independent runs for the rank-sweep and loss-ablation experiments respectively, and the small gap reflects ordinary run-to-run training variation rather than a discrepancy in the underlying pipeline. A larger gap is seen for MobileSAM (Tables 1 and 2: Dice 0.812, IPE 2.57%; Table 3, full objective: Dice 0.607, IPE 7.25%). Although the two runs used identical settings, the loss-ablation full-objective run did not converge: its validation Dice fell from 0.607 at epoch 1 to below 0.42 and remained there, whereas the Table 1 run improved steadily to 0.81. The Table 3 row therefore reflects the epoch-1 checkpoint of a non-converged run and is not a reliable estimate of the effect of IPP on MobileSAM.

**New text (complete):**

Table 4 and Fig. 6 compare the four objectives for the preferred configurations and MobileNet-UNet; the complete ablation at every rank is given in Table S6, Table S7, and Fig. S4. Adding the physics-aware terms lowered the mean global IPE relative to Dice-only training in all 15 architecture-rank combinations; with three seeds, the 95% interval excluded zero in three of them. The reduction was large where Dice-only training was unstable: for EdgeSAM at *r* = 32, the IPE fell from 11.20 ± 6.46% to 3.02 ± 0.84%, and for MobileNet-UNet at *r* = 8 from 15.48 ± 12.73% to 2.60 ± 0.07%. For MobileSAM at *r* = 8, whose Dice-only IPE was already low, the reduction was small (1.53 ± 0.27% to 1.32 ± 0.09%; −0.21 percentage points, −0.75 to +0.33).

**Why:** Removes the aliasing paragraph's premise; all ablation numbers are new (three seeds).

---

### M117 REPLACE: 4.4 Physics-Aware Loss Ablation

**Find:** "TABLE 3 Component-wise ablation of the Physics-Aware Phase Consistency Loss at the selected LoRA ..." (the old table: caption and all 12 table block(s))

**Original caption:**

> **TABLE 3** Component-wise ablation of the Physics-Aware Phase Consistency Loss at the selected LoRA rank (*r* = 8). Best value per column within each architecture is shown in bold. Each row progressively adds a physics-aware constraint, and IPE denotes the global Integrated-Phase Error (%, lower is better). The ablation runs were trained independently; therefore, comparisons are made within this table to account for run-to-run variation.

**New text (complete):**

**TABLE 4** Ablation of the Physics-Aware Phase Consistency Loss. Each objective adds one term to the class-weighted Dice loss; the full objective is the rank-sweep run of the same rank and seeds. Mean ± SD over three seeds; lower part: seed-paired differences with 95% t-intervals.

| Configuration | Objective | Mean Dice | Boundary F1 | Global IPE (%) | Runs with a lost class |
|---|---|---|---|---|---|
| MobileSAM r = 8 | Dice only | 0.811 ± 0.005 | 0.808 ± 0.012 | 1.53 ± 0.27 | 0/3 |
| MobileSAM r = 8 | Dice + PMC | 0.813 ± 0.009 | 0.805 ± 0.002 | 1.60 ± 0.24 | 0/3 |
| MobileSAM r = 8 | Dice + PMC + BGA | 0.813 ± 0.005 | 0.806 ± 0.004 | 1.48 ± 0.11 | 0/3 |
| MobileSAM r = 8 | Dice + PMC + BGA + IPP (full) | 0.810 ± 0.006 | 0.802 ± 0.011 | 1.32 ± 0.09 | 0/3 |
| EdgeSAM r = 32 | Dice only | 0.619 ± 0.109 | 0.519 ± 0.212 | 11.20 ± 6.46 | 2/3 (s43: sphero(2); s44: sphero(2)) |
| EdgeSAM r = 32 | Dice + PMC | 0.680 ± 0.113 | 0.638 ± 0.027 | 7.45 ± 7.17 | 1/3 (s44: sphero(2)) |
| EdgeSAM r = 32 | Dice + PMC + BGA | 0.746 ± 0.006 | 0.643 ± 0.011 | 3.64 ± 0.82 | 0/3 |
| EdgeSAM r = 32 | Dice + PMC + BGA + IPP (full) | 0.738 ± 0.006 | 0.642 ± 0.005 | 3.02 ± 0.84 | 0/3 |
| MobileNet-UNet r = 8 | Dice only | 0.485 ± 0.294 | 0.814 ± 0.054 | 15.48 ± 12.73 | 2/3 (s42: sphero(2), echino(3), stomato(4); s44: sphero(2), stomato(4)) |
| MobileNet-UNet r = 8 | Dice + PMC | 0.414 ± 0.181 | 0.782 ± 0.022 | 19.79 ± 6.71 | 3/3 (s42: sphero(2), echino(3), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2)) |
| MobileNet-UNet r = 8 | Dice + PMC + BGA | 0.544 ± 0.292 | 0.812 ± 0.052 | 13.96 ± 12.46 | 2/3 (s42: sphero(2), echino(3), stomato(4); s44: sphero(2)) |
| MobileNet-UNet r = 8 | Dice + PMC + BGA + IPP (full) | 0.412 ± 0.004 | 0.814 ± 0.003 | 2.60 ± 0.07 | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2), stomato(4)) |

Seed-paired differences of the global IPE (percentage points, mean [95% t-interval]):

| Configuration | Δ IPE (pp), Full − Dice only | Δ IPE (pp), Full − (Dice+PMC+BGA): IPP |
|---|---|---|
| MobileSAM r = 8 | -0.21 [-0.75, +0.33] | -0.16 [-0.41, +0.09] |
| EdgeSAM r = 32 | -8.18 [-23.52, +7.15] | -0.63 [-0.75, -0.51] |
| MobileNet-UNet r = 8 | -12.88 [-44.66, +18.90] | -11.36 [-42.49, +19.77] |

**Why:** Removes 'The ablation runs were trained independently' (the full-objective row is the rank-sweep run, DECISIONS D-09).

---

### M118 DELETE: 4.4 Physics-Aware Loss Ablation

**Find:** "TABLE 4 Per-class IPE and Dice scores for different loss configurations using EdgeSAM (r ..." (the old table: caption and all 10 table block(s))

**Original caption:**

> **TABLE 4** Per-class IPE and Dice scores for different loss configurations using EdgeSAM (*r* = 8). Best IPE and Dice per morphology are shown in bold. The extreme stomatocyte IPE with PMC alone reflects severe class-prediction failure with substantial phase-integral overestimation (Dice = 0.018); global IPE in Table 3 provides the primary measure of integrated-phase preservation.

**Delete the whole block.**

**Why:** Old Table 4 (single-run per-class EdgeSAM ablation, including Dice 0.018 and IPE 1073.58%) is not reproduced; class detail is in Table S6 and Table S8.

---

### M119 REPLACE: 4.4 Physics-Aware Loss Ablation

**Find:** "For EdgeSAM, the Dice-only baseline achieved a mean Dice of ..."

**Original:**

> For EdgeSAM, the Dice-only baseline achieved a mean Dice of 0.757, Boundary F1 of 0.938, and global IPE of 5.22%. Adding PMC alone substantially degraded both boundary fidelity and integrated-phase preservation, whereas the addition of BGA restored the mean Dice to 0.763 and Boundary F1 to 0.929. As illustrated in Fig. S-4, BGA promotes alignment of predicted contours with spatial phase gradients, providing a physical cue for cellular boundary localization. Table 4 further shows that PMC alone severely degraded stomatocyte segmentation, whereas the addition of BGA restored its Dice score from 0.018 to 0.643.

**New text (complete):**

The physics terms also reduced training failures: over the rank sweep, the number of runs that lost a class decreased from 3 of 15 to 1 of 15 for MobileSAM and from 6 of 15 to 3 of 15 for EdgeSAM (Table 2). The contributions of the individual terms could not be separated reliably with three seeds. Adding IPP to Dice + PMC + BGA lowered the mean IPE in 13 of 15 combinations, most clearly for EdgeSAM at *r* = 32 (−0.63 percentage points, −0.75 to −0.51), whereas PMC and BGA alone had no consistent effect (8 and 11 of 15; Table S7). Mean Dice was not systematically changed by the physics terms.

**Why:** Removes the PMC/BGA claims (PMC alone degrades, BGA restores 0.018 to 0.643), which the three-seed results do not reproduce.

---

### M120 REPLACE: 4.4 Physics-Aware Loss Ablation

**Find:** "The complete objective incorporating IPP achieved the lowest global IPE ..."

**Original:**

> The complete objective incorporating IPP achieved the lowest global IPE (3.04%) and the highest Boundary F1 (0.953) for EdgeSAM, as directly compared in Fig. 5. A stronger improvement in integrated-phase preservation was observed for MobileNet-UNet, where IPE decreased from 15.52% with Dice alone to 3.01% with the full objective, while Boundary F1 increased from 0.913 to 0.970. These results demonstrate that explicit integrated-phase preservation provides quantitative information not captured by geometric overlap alone. For MobileSAM, the Dice+PMC+BGA configuration achieved a lower IPE (2.37%) than the full objective (7.25%); however, the full-objective MobileSAM run did not converge (best checkpoint at epoch 1) so this row does not establish an architecture-dependent effect of IPP. To assess whether IPP introduces a phase-dependent bias, cells were stratified into phase quartiles and the false-negative rate within ground-truth cells was compared with and without IPP. Misses concentrated in the lowest-phase quartile for all models (18--33%). For EdgeSAM and MobileNet-UNet, adding IPP did not increase this low-phase bias and reduced it slightly (33.3% to 28.7% and 27.2% to 25.0%, respectively). For MobileSAM, the full-objective run showed a higher lowest-quartile miss rate (18.3% to approximately 30%), but this run did not converge, so no conclusion about IPP is drawn for this architecture. To evaluate robustness to background phase variation, synthetic constant offsets and low-order aberration-like fields (tilt, defocus-like, and astigmatism-like distortions) of increasing magnitude were added to held-out phase maps and re-evaluated. Segmentation accuracy was essentially unaffected by constant offsets (EdgeSAM Dice unchanged, as expected because each input image is standardized before inference, which removes a constant offset) and only modestly reduced under the largest tilt, defocus-like, and astigmatism-like distortions tested (EdgeSAM Dice 0.739 to 0.720, 0.735, and 0.706, respectively). As expected, the uncorrected IPE increased with offset magnitude (EdgeSAM 3.55% to 3.92% at the largest offset tested; the unperturbed baseline of 3.55% differs from the 3.04% in Table 3 because this analysis evaluates at the native 768 × 768 resolution), consistent with IPE being computed directly on the raw, background-uncorrected phase. Overall, the ablation results demonstrate complementary roles of the three physics-aware components: PMC incorporates phase-based foreground information, BGA improves boundary localization using phase gradients, and IPP directly constrains preservation of the integrated quantitative phase within the predicted measurement domain.

**New text (complete):**

Because the IPP term weights each pixel by its phase, it could favor high-phase regions over low-phase cell margins. Misses did concentrate in the low-phase part of the cells: within the annotated cells, the false-negative rate in the lowest phase quartile was 23 to 28%, compared with below 6% in the other quartiles (Table S9). Adding IPP did not consistently increase this deficit: the lowest-quartile false-negative rate changed by −4.0 (−5.3 to −2.8) percentage points for EdgeSAM at *r* = 32 and by +2.4 (−1.2 to +5.9) for MobileSAM at *r* = 8.

**Why:** R2-2 (phase-dependent bias), updated. The old robustness part of this paragraph moves to Section 4.5; the MobileSAM non-convergence statements are removed (MobileSAM now converges in all seeds).

---

### M121 REPLACE: 4.4 Physics-Aware Loss Ablation

**Find:** "![](media/image7.png){width="6.573708442694663in" height="2.4285126859142605in"}"

**Original:**

> ![](media/image7.png){width="6.573708442694663in" height="2.4285126859142605in"}

**New text (complete):**

*[insert figure file artifacts/figures/main/F6_loss_ablation.png]*

---

### M122 REPLACE: 4.4 Physics-Aware Loss Ablation

**Find:** "Fig. 5 Component-wise loss ablation at the selected LoRA rank ..."

**Original:**

> **Fig. 5** Component-wise loss ablation at the selected LoRA rank (*r* = 8), comparing boundary fidelity and integrated-phase preservation. Bars indicate Boundary F1 (left axis), and the dashed line indicates global Integrated-Phase Error (IPE; right axis).

**New text (complete):**

**Fig. 6** Ablation of the physics-aware loss: global IPE (top) and mean Dice (bottom) for the four objectives. Bars: mean over three seeds; circles: seeds; crosses: seeds in which a class was lost. All ranks: Fig. S4.

**Why:** The old figure used two y-axes; the new one shows IPE and Dice in separate rows.

---


## 4.5 Integrated-Phase and Optical-Mass Preservation

### M123 REPLACE: 4.5 Integrated-Phase and Optical-Mass Preservation

**Find:** "Integrated-Phase and Optical-Mass Preservation Analysis"

**Original:**

> 4.  **Integrated-Phase and Optical-Mass Preservation Analysis**

**New text (complete):**

## 4.5 Integrated-Phase and Optical-Mass Preservation

---

### M124 REPLACE: 4.5 Integrated-Phase and Optical-Mass Preservation

**Find:** "The preservation of phase-integrated optical information was evaluated using the ..."

**Original:**

> The preservation of phase-integrated optical information was evaluated using the IPE. As summarized in Table 5, EdgeSAM achieved a global IPE of 3.63%, while MobileSAM and MobileNet-UNet achieved 2.57% and 2.60%, respectively. However, global IPE alone does not fully characterize the reliability of per-cell quantitative measurements. MobileNet-UNet completely failed to detect echinocytes and stomatocytes, resulting in 100% class-specific IPE for both morphologies. MobileSAM achieved the lowest global IPE but exhibited poor instance separation (AJI = 0.249), frequently merging adjacent cells and thereby limiting reliable per-cell measurements. In contrast, EdgeSAM maintained non-degenerate segmentation across all four RBC morphologies while providing adequate instance separation and a low global IPE. These results demonstrate that reliable optical-mass preservation requires integrated-phase fidelity together with accurate morphology coverage and individual-cell separation. Among the evaluated architectures, EdgeSAM therefore provided the most balanced measurement domain for subsequent phase-derived quantitative analysis.

**New text (complete):**

Table 5 summarizes the measurement-related errors. The global IPE was 1.32 ± 0.09% for MobileSAM (*r* = 8) and 3.02 ± 0.84% for EdgeSAM (*r* = 32). For annotated cells that were matched one-to-one to a predicted cell (IoU ≥ 0.5; 66.6% and 66.3% of the cells), the median per-cell IPE was 1.11 ± 0.09% and 1.88 ± 0.24%, and 90% of the matched cells had errors below 8.3% and 16.2%, respectively (Fig. S6). Unmatched cells were mostly merged or missed cells: both SAM-based models produced about two thirds as many cell components as were annotated (0.667 and 0.696), consistent with their AJI. Class-specific IPE (Table S8) is dominated by confusion between morphologies and is therefore much larger than the global IPE.

**Why:** R3-2 (per-cell IPE vs global IPE), R3-4 (why class IPE is larger). Old numbers (3.63%, 2.57%, 2.60%) and the EdgeSAM 'most balanced' statement removed.

---

### M125 REPLACE: 4.5 Integrated-Phase and Optical-Mass Preservation

**Find:** "TABLE 5 Integrated-phase preservation analysis at the selected LoRA rank (r = 8). Best ..." (the old table: caption and all 4 table block(s))

**Original caption:**

> **TABLE 5** Integrated-phase preservation analysis at the selected LoRA rank (*r* = 8). Best value per column is shown in bold. Global and class-specific IPEs are reported for each architecture. Low global IPE alone does not ensure reliable single-cell quantitative analysis when morphology detection or instance separation fails.

**New text (complete):**

**TABLE 5** Integrated-phase and optical-mass preservation (phase-wrapped cells excluded). Per-cell IPE: matched cells (IoU ≥ 0.5). Dry mass: per-image medians, predicted vs annotation masks; LoA: 95% limits of agreement. Mean ± SD over three seeds.

| Configuration | Global IPE (%) | Per-cell \|IPE\| median (%) | Per-cell \|IPE\| P90 (%) | Annotated cells matched (IoU ≥ 0.5, %) | Predicted / annotated cells | Dry-mass MdARE (%) | Dry-mass bias (pg) | LoA low (pg) | LoA high (pg) | Day-0 median dry mass (pg) |
|---|---|---|---|---|---|---|---|---|---|---|
| MobileSAM r = 8 | 1.32 ± 0.09 | 1.11 ± 0.09 | 8.3 ± 1.3 | 66.6 ± 0.6 | 0.667 ± 0.006 | 2.87 ± 0.21 | 0.66 ± 0.31 | -4.71 ± 1.07 | 6.02 ± 1.66 | 28.21 ± 0.07 |
| EdgeSAM r = 32 | 3.02 ± 0.84 | 1.88 ± 0.24 | 16.2 ± 1.7 | 66.3 ± 1.9 | 0.696 ± 0.025 | 3.93 ± 0.52 | 0.46 ± 0.68 | -5.12 ± 1.64 | 6.04 ± 3.00 | 27.88 ± 0.43 |
| EdgeSAM r = 8 | 3.81 ± 1.82 | 2.04 ± 0.65 | 17.5 ± 5.6 | 60.5 ± 2.4 | 0.631 ± 0.014 | 3.63 ± 0.02 | 0.89 ± 0.24 | -6.03 ± 0.39 | 7.80 ± 0.65 | 28.15 ± 0.45 |
| MobileNet-UNet r = 8 | 2.60 ± 0.07 | 1.36 ± 0.03 | 11.1 ± 0.2 | 84.7 ± 1.2 | 0.855 ± 0.015 | 2.48 ± 0.31 | -0.17 ± 0.07 | -3.64 ± 0.69 | 3.30 ± 0.78 | 28.06 ± 0.15 |
| Annotation (reference) |  |  |  |  | 1 |  |  |  |  | 27.83 |

---

### M126 NEW: 4.5 Integrated-Phase and Optical-Mass Preservation

**Insert:**

Dry mass was computed without calibration. The median day-0 dry mass was 27.83 pg from the annotation masks, 0.93 of a typical mean corpuscular hemoglobin of 30 pg, and 28.21 pg and 27.88 pg from the MobileSAM and EdgeSAM masks. Per image, dry mass from the predicted masks agreed with the annotation-derived dry mass within a median absolute relative error of 2.87 ± 0.21% (MobileSAM) and 3.93 ± 0.52% (EdgeSAM), with biases of +0.66 and +0.46 pg and limits of agreement of about −5 to +6 pg (Fig. 7). The Pearson correlation was low (0.24 and 0.38) because the annotation-derived per-image dry mass varied little between images (28.66 ± 1.55 pg); for such a narrow range, the error-based measures describe the agreement better than correlation.

**Why:** Replaces 'r = 0.966, CCC 0.96, MdARE 3.35%' (old pipeline); answers R3-3 (absolute agreement measures, not only Pearson r).

---

### M127 NEW: 4.5 Integrated-Phase and Optical-Mass Preservation

**Insert:**

*[insert figure file artifacts/figures/main/F9_dry_mass_agreement.png]*

**Fig. 7** Dry-mass agreement between predicted and annotation masks (per-image median dry mass, three seeds pooled, phase-wrapped cells excluded). Top: predicted vs annotation-derived values with the identity line; bottom: Bland-Altman plots with bias (solid) and 95% limits of agreement (dashed).

---

### M128 NEW: 4.5 Integrated-Phase and Optical-Mass Preservation

**Insert:**

The global IPE depended on the background referencing of Section 2.2: without it, the IPE of MobileSAM rose from 1.32% to 2.06% (Table S10), whereas estimating the background from the predicted instead of the annotated mask changed it by less than 0.01 percentage points. To test robustness to background phase variations, synthetic tilt, defocus-like, and astigmatism-like fields of up to 0.5 rad were added to the validation phase maps: the mean Dice changed by at most 0.003 and the background-corrected IPE by at most 0.1 percentage points (Table S11, Fig. S9). Constant offsets are removed by the per-image standardization of the network input and are therefore not a test of robustness.

**Why:** R2-1 (background correction and robustness to low-order distortions), updated; was part of the old Section 4.3.

---


## 4.6 Computational Efficiency

### M129 REPLACE: 4.6 Computational Efficiency

**Find:** "Computational Efficiency Analysis"

**Original:**

> 5.  **Computational Efficiency Analysis**

**New text (complete):**

## 4.6 Computational Efficiency

---

### M130 REPLACE: 4.6 Computational Efficiency

**Find:** "Computational efficiency was evaluated on an NVIDIA RTX A5000 using ..."

**Original:**

> Computational efficiency was evaluated on an NVIDIA RTX A5000 using native PyTorch and, where supported, ORT under FP32 and FP16 precision. As summarized in Table 6, EdgeSAM achieved substantial acceleration with ORT, reducing the median latency from 13.36 ms in PyTorch FP32 to 4.33 ms in ONNX FP16 and increasing throughput from 74.82 to 228.33 FPS, while requiring only 16.4 MB of peak allocated VRAM. MobileNet-UNet achieved the highest throughput (376.10 FPS with ONNX FP16) but failed to detect two RBC morphology classes, whereas MobileSAM exhibited higher computational requirements and could not be reliably accelerated using ORT because of unsupported decoder operations. Considering segmentation reliability, integrated-phase preservation, and computational efficiency together, EdgeSAM provided the most balanced overall performance among the evaluated architectures. These measurements characterize GPU-based computational efficiency rather than actual embedded-device deployment. Nevertheless, the low latency and memory requirements of EdgeSAM support its potential for future implementation in resource-constrained holographic imaging systems. The accuracy-throughput trade-off among the three architectures is shown in Fig. S-3.

**New text (complete):**

Inference cost is independent of the LoRA rank after merging, so one measurement per architecture is reported (Table 6). In PyTorch FP32, EdgeSAM needed 10.64 ± 1.45 ms per image (81 to 103 FPS) and 42 MB of GPU memory, MobileSAM 20.35 ± 0.15 ms (49 to 50 FPS) and 358 MB, and MobileNet-UNet 5.66 ± 0.10 ms and 52 MB. ORT reduced the latency of EdgeSAM to 4.11 ± 0.43 ms (229 to 277 FPS) and of MobileNet-UNet to 2.53 ± 0.19 ms, but increased that of MobileSAM to 25.43 ± 0.21 ms, and it used more memory for all models (132 to 1,178 MB). MobileSAM processes 1024-pixel inputs and the other models 256-pixel inputs, so these differences combine architecture and resolution. All ORT models reproduced the PyTorch predictions (Table S14). These measurements characterize a desktop GPU rather than an embedded device (Fig. S11 shows accuracy against latency).

**Why:** Removes ONNX FP16, 228.3 FPS, 16.4 MB and 'MobileSAM could not be accelerated with ORT'.

---

### M131 REPLACE: 4.6 Computational Efficiency

**Find:** "TABLE 6 Computational benchmarking on an NVIDIA RTX A5000 at the selected LoRA rank ..." (the old table: caption and all 10 table block(s))

**Original caption:**

> **TABLE 6** Computational benchmarking on an NVIDIA RTX A5000 at the selected LoRA rank (*r* = 8). Best value per column is shown in bold. Latency is reported as median (*p*~50~) and tail (*p*~99~) latency, and peak VRAM denotes the peak allocated device memory. MobileSAM is reported under native PyTorch only because its cross-attention mask decoder is not fully supported by the ORT CUDA Execution Provider.

**New text (complete):**

**TABLE 6** Computational efficiency on an NVIDIA RTX A5000 (batch 1, strict FP32). P50 and P99: mean ± SD over three seeds; FPS: range of the per-seed values; peak VRAM: largest value over seeds and sessions. Inference cost does not depend on the LoRA rank.

| Architecture | Runtime | Input (px) | Trainable params, preferred rank | P50 (ms) | P99 (ms) | FPS (per-seed range) | Peak VRAM (MB, max) |
|---|---|---|---|---|---|---|---|
| MobileSAM | PyTorch FP32 | 1024 | 0.67 M (r = 8) | 20.35 ± 0.15 | 20.51 ± 0.13 | 49-50 | 358 |
| MobileSAM | ONNX Runtime FP32 | 1024 | 0.67 M (r = 8) | 25.43 ± 0.21 | 25.56 ± 0.18 | 39-40 | 1178 |
| EdgeSAM | PyTorch FP32 | 256 | 1.47 M (r = 32) | 10.64 ± 1.45 | 12.97 ± 1.53 | 81-103 | 42 |
| EdgeSAM | ONNX Runtime FP32 | 256 | 1.47 M (r = 32) | 4.11 ± 0.43 | 5.58 ± 0.31 | 229-277 | 132 |
| MobileNet-UNet | PyTorch FP32 | 256 | 5.58 M (r = 8) | 5.66 ± 0.10 | 8.00 ± 0.63 | 175-180 | 52 |
| MobileNet-UNet | ONNX Runtime FP32 | 256 | 5.58 M (r = 8) | 2.53 ± 0.19 | 3.33 ± 0.79 | 374-431 | 152 |

---


## 4.7 Visualization of Segmentation Results

### M132 REPLACE: 4.7 Visualization of Segmentation Results

**Find:** "Visualization of Segmentation and Morphology Classification Results"

**Original:**

> 6.  **Visualization of Segmentation and Morphology Classification Results**

**New text (complete):**

## 4.7 Visualization of Segmentation Results

---

### M133 REPLACE: 4.7 Visualization of Segmentation Results

**Find:** "Qualitative segmentation results further support the quantitative evaluation. As shown ..."

**Original:**

> Qualitative segmentation results further support the quantitative evaluation. As shown in Fig. 6, the LoRA-adapted EdgeSAM successfully delineated individual RBCs across the four morphology classes, including discocytes, echinocytes, spherocytes, and stomatocytes. The predicted contours generally followed the cellular structures observed in the corresponding quantitative phase maps, including the more complex boundaries of echinocytic cells. MobileNet-UNet recovered the discocytes and spherocytes but missed echinocytes and stomatocytes, consistent with the per-class results in Section 4.1. Importantly, the predicted masks provided individualized cellular regions rather than merged foreground regions, supporting their use as measurement domains for subsequent phase-integrated quantitative analysis. These qualitative results are consistent with the morphology coverage, boundary fidelity, and instance-separation performance reported in Sections 4.1 and 4.4.

**New text (complete):**

Fig. 8 shows representative segmentation results for each morphology class. Consistent with the per-class results, MobileNet-UNet assigns no pixel to the spherocyte or stomatocyte class, so cells of these morphologies are labeled with other classes, whereas the two SAM-based models label all four morphologies. Neighboring cells that touch are sometimes merged into one predicted region by both SAM-based models, more often by MobileSAM, in agreement with their AJI and with the number of predicted cell components (Section 4.5).

**Why:** CHECK AGAINST THE NEW FIGURE: generate Fig. 8 on the server first (make_qualitative_figures.py) and adjust the last sentence to what the figure shows. The first two sentences are certain from the predictions (MobileNet-UNet predicts 0 pixels of mask values 2 and 4).

---

### M134 REPLACE: 4.7 Visualization of Segmentation Results

**Find:** "![](media/image8.png){width="6.532890419947506in" height="1.7261253280839894in"}"

**Original:**

> ![](media/image8.png){width="6.532890419947506in" height="1.7261253280839894in"}

**New text (complete):**

*[insert figure file artifacts/figures/main/FQ1_qualitative_segmentation.png]*

---

### M135 REPLACE: 4.7 Visualization of Segmentation Results

**Find:** "Fig. 6 Qualitative segmentation results of LoRA-adapted MobileNet-UNet, MobileSAM, and ..."

**Original:**

> **Fig. 6** Qualitative segmentation results of LoRA-adapted MobileNet-UNet, MobileSAM, and EdgeSAM (*r* = 8) across different RBC morphologies, including discocytes, echinocytes, spherocytes, and stomatocytes.

**New text (complete):**

**Fig. 8** Qualitative segmentation results of MobileSAM (*r* = 8), EdgeSAM (*r* = 32), and MobileNet-UNet (*r* = 8), seed 42, for the four morphology classes. For each class, the 256 × 256 window with the most annotated pixels of that class is shown with the annotation.

---


## 4.8 Storage-Induced Morphology Changes

### M136 REPLACE: 4.8 Storage-Induced Morphology Changes

**Find:** "Storage-Induced Morphology Degradation"

**Original:**

> 7.  **Storage-Induced Morphology Degradation**

**New text (complete):**

## 4.8 Storage-Induced Morphology Changes

---

### M137 REPLACE: 4.8 Storage-Induced Morphology Changes

**Find:** "To evaluate the applicability of the proposed framework to quantitative ..."

**Original:**

> To evaluate the applicability of the proposed framework to quantitative biological analysis, storage-induced changes in RBC morphology and phase-derived biophysical properties were examined over the 47-day storage period. Geometric and phase-derived quantities were extracted from individual RBCs using the segmentation masks generated by the selected LoRA-adapted EdgeSAM (*r* = 8). Each storage time point consisted of six independent validation patches; therefore, the analysis represents cross-sectional population changes rather than longitudinal tracking of the same individual cells.

**New text (complete):**

To evaluate the applicability of the proposed framework to quantitative biological analysis, storage-induced changes in RBC morphology and phase-derived biophysical properties were examined over the 47-day storage period, using the annotation masks as the reference and the masks predicted by MobileSAM (*r* = 8) and EdgeSAM (*r* = 32). Each storage time point consisted of six independent validation images; therefore, the analysis represents cross-sectional population changes rather than longitudinal tracking of the same cells.

**Why:** The analysis now uses the annotation as reference and both preferred models (three seeds), not a single EdgeSAM r = 8 run.

---

### M138 REPLACE: 4.8 Storage-Induced Morphology Changes

**Find:** "The predicted RBC morphology distributions showed systematic changes over the ..."

**Original:**

> The predicted RBC morphology distributions showed systematic changes over the 47-day storage period. As shown in Fig. 7, discocytes predominated at the initial storage time point (day 0) and gradually decreased with storage duration, whereas echinocytes and spherocytes became increasingly prevalent at later time points. Stomatocytes remained a substantial minority class, rising from 10% at day 0 to a maximum of 26% at day 19 and accounting for 15% at day 47 (Table 7).

**New text (complete):**

The annotation shows the expected storage lesion (Fig. 9, Table 7): discocytes made up 93% of the cells at day 0 and 11% at day 47, while spherocytes increased from 0% to 59% and echinocytes from 2% to 30%; stomatocytes peaked at 15% at day 19. Both preferred models reproduced this composition, with a mean absolute difference per day of 1.7 percentage points for MobileSAM and 2.8 for EdgeSAM.

**Why:** Old composition numbers (stomatocytes 10% to 26%, EdgeSAM only) replaced; class names follow mask values.

---

### M139 REPLACE: 4.8 Storage-Induced Morphology Changes

**Find:** "![](media/image9.png){width="3.65748031496063in" height="2.34251968503937in"}"

**Original:**

> ![](media/image9.png){width="3.65748031496063in" height="2.34251968503937in"}

**New text (complete):**

*[insert figure file artifacts/figures/main/F11_composition.png]*

---

### M140 REPLACE: 4.8 Storage-Induced Morphology Changes

**Find:** "Fig. 7 Predicted RBC morphology composition over the 47-day storage ..."

**Original:**

> **Fig. 7** Predicted RBC morphology composition over the 47-day storage period, pooled across all cells detected at each time point using LoRA-adapted EdgeSAM (*r* = 8).

**New text (complete):**

**Fig. 9** RBC morphology composition over the 47-day storage period: annotation and the masks predicted by MobileSAM (*r* = 8) and EdgeSAM (*r* = 32), mean over three seeds. Numbers in parentheses are mask values.

---

### M141 REPLACE: 4.8 Storage-Induced Morphology Changes

**Find:** "Representative quantitative phase images in Fig. S-5 further illustrate these ..."

**Original:**

> Representative quantitative phase images in Fig. S-5 further illustrate these storage-associated morphological changes, showing a transition from predominantly biconcave discocytes at the initial storage time point to increased occurrence of altered RBC morphologies at later storage stages. The agreement between the predicted morphology distributions and the corresponding QPI appearances demonstrates that the proposed framework can quantify population-level morphological changes associated with prolonged RBC storage.

**New text (complete):**

Representative quantitative phase images in Fig. S12 further illustrate these storage-associated morphological changes, showing a transition from predominantly biconcave discocytes at the initial storage time point to increased occurrence of altered RBC morphologies at later storage stages.

**Why:** Figure reference renumbered; the last sentence of the old paragraph ('demonstrates that the proposed framework can quantify ...') is covered by the composition comparison above.

---


## 4.9 Quantitative Phase-Derived Cellular Analysis Across Storage

### M142 REPLACE: 4.9 Quantitative Phase-Derived Cellular Analysis Across Storage

**Find:** "Quantitative Phase-Derived Cellular Analysis Across Storage"

**Original:**

> 8.  **Quantitative Phase-Derived Cellular Analysis Across Storage**

**New text (complete):**

## 4.9 Quantitative Phase-Derived Cellular Analysis Across Storage

---

### M143 REPLACE: 4.9 Quantitative Phase-Derived Cellular Analysis Across Storage

**Find:** "Quantitative cellular measurements were extracted from the EdgeSAM-predicted masks to ..."

**Original:**

> Quantitative cellular measurements were extracted from the EdgeSAM-predicted masks to evaluate storage-associated changes in RBC morphology and phase-derived biophysical properties. As summarized in Table 7, the projected cell area did not decrease monotonically over the storage period; it instead increased from 63.1 µm² at day 0 to a peak of 83.6 µm² at day 15, before declining to 43.4 µm² by day 47 (Spearman ρ=−0.67, p\<0.001, overall). The day-15 peak coincides with the time point containing the largest number of multi-cell clusters in the annotated data, which may inflate the apparent per-component area at that time point. Circularity also changed over the storage period, consistent with the population-level morphological changes observed in Fig. 7 and Fig. S-5. Phase-derived dry-mass estimates likewise fluctuated rather than remaining stable, showing a significant overall decline with storage duration (Spearman ρ=−0.40, p=0.0009); a partial correlation controlling for projected area was substantially weaker and not statistically significant (partial ρ=−0.15, p=0.23), indicating that the dry-mass decline is largely, though not entirely, explained by the concurrent change in projected area. Fig. 8 shows the relationship between projected area and dry mass, indicating that the substantial change in the two-dimensional cellular footprint was not accompanied by a proportional change in phase-derived dry mass. These results demonstrate that projected morphological changes and phase-derived cellular mass provide complementary quantitative information on storage-associated changes in RBC populations.

**New text (complete):**

With the image as the unit of analysis (Table 8, Fig. 10), the projected area of the annotated cells decreased with storage (Spearman ρ = −0.53, *p* < 0.001; −43.9% from days 0–8 to days 37 and 47), whereas their dry mass did not change (ρ = −0.11, *p* = 0.36; −0.8%; Mann-Whitney *p* = 0.21), and circularity showed no trend. The cells therefore lost projected area without losing dry mass, consistent with the spherocytic transformation, in which membrane is lost while the hemoglobin content is retained [24, 25, 44]. The per-cell mean area was highest at day 15 (Table 7) because of large clustered components, whereas the per-image median area did not peak at that time point (Fig. 10).

**Why:** R1-6 and R1-9: the area peak at day 15 and the dry-mass fluctuation of the old text came from a single run of the superseded pipeline; the corrected annotation shows a monotonic area decrease after day 27 and constant dry mass. Removes ρ = −0.67, ρ = −0.40, partial ρ = −0.15, 63.1/83.6/43.4 µm².

---

### M144 REPLACE: 4.9 Quantitative Phase-Derived Cellular Analysis Across Storage

**Find:** "To evaluate the reliability of the predicted segmentation masks as ..."

**Original:**

> To evaluate the reliability of the predicted segmentation masks as quantitative measurement domains, dry-mass estimates obtained from EdgeSAM-predicted masks were compared with those derived from annotation masks. The two measurements showed strong agreement (*r* = 0.966, CCC=0.96, MdARE=3.35%, Bland-Altman bias=−0.12 pg). These results demonstrate that the proposed physics-aware segmentation framework preserves phase-derived quantitative information while providing reliable measurement boundaries for single-cell quantitative holographic analysis.

**New text (complete):**

The predicted masks reproduced the decrease in area (MobileSAM −40.6%, EdgeSAM −44.3%) but showed a small decrease in dry mass that the annotation does not show (MobileSAM −2.6 ± 0.4%, not significant; EdgeSAM −6.1 ± 0.3%, *p* between 0.008 and 0.05 across seeds). Fig. 10 shows that this bias arises at days 5 to 19, where the predicted per-image dry mass exceeds the annotation-derived value, and is largest at day 15, the time point with the most touching cells. A bias of this kind occurred for all 15 rank-sweep configurations (Table S13). Population-level dry-mass trends from predicted masks should therefore be read against this segmentation bias.

**Why:** The old agreement sentence (r = 0.966 ...) moves, corrected, to Section 4.5. CHECK: 'the time point with the most touching cells' comes from Section 3.1 and the highest annotated cell-mean area at day 15 supports it.

---

### M145 REPLACE: 4.9 Quantitative Phase-Derived Cellular Analysis Across Storage

**Find:** "TABLE 7 Per-cell quantitative morphology and phase-derived dry-mass measurements from EdgeSAM (r = 8) ..." (the old table: caption and all 11 table block(s))

**Original caption:**

> **TABLE 7** Per-cell quantitative morphology and phase-derived dry-mass measurements from EdgeSAM (*r* = 8) predictions over the 47-day storage period (mean ± SD). A total of 1,974 cells from 66 validation images were analyzed. The composition column lists the most frequent morphologies at each time point; classes not listed are omitted, so the values need not sum to 100%.

**New text (complete):**

**TABLE 7** Cell morphology, dry mass, and composition over storage (phase-wrapped cells excluded). Annotation: mean ± SD over cells. Models: per-day cell mean averaged over three seeds ± SD over seeds. Composition order: discocyte (1) / spherocyte (2) / echinocyte (3) / stomatocyte (4).

| Day | Annotated cells | Area (µm²), annotation | Dry mass (pg), annotation | Composition (%), annotation | Area (µm²), MobileSAM r8 | Dry mass (pg), MobileSAM r8 | Area (µm²), EdgeSAM r32 | Dry mass (pg), EdgeSAM r32 |
|---|---|---|---|---|---|---|---|---|
| 0 | 185 | 52.1 ± 21.6 | 31.7 ± 14.3 | 93 / 0 / 2 / 5 | 52.5 ± 0.7 | 33.2 ± 0.5 | 50.5 ± 2.1 | 31.5 ± 0.9 |
| 5 | 211 | 51.2 ± 21.9 | 33.2 ± 15.0 | 91 / 0 / 1 / 8 | 57.1 ± 1.2 | 37.7 ± 0.7 | 58.5 ± 1.7 | 37.3 ± 0.5 |
| 8 | 220 | 51.9 ± 24.4 | 34.9 ± 17.8 | 95 / 0 / 2 / 3 | 55.3 ± 0.5 | 37.3 ± 0.3 | 59.3 ± 4.9 | 39.3 ± 2.7 |
| 12 | 230 | 54.9 ± 31.8 | 36.3 ± 22.3 | 90 / 0 / 3 / 6 | 59.7 ± 1.3 | 39.7 ± 0.8 | 61.6 ± 2.8 | 40.2 ± 1.4 |
| 15 | 206 | 59.3 ± 32.6 | 40.9 ± 25.1 | 93 / 1 / 1 / 4 | 56.8 ± 5.8 | 39.1 ± 3.6 | 60.7 ± 1.1 | 41.6 ± 1.4 |
| 19 | 162 | 49.8 ± 23.7 | 32.2 ± 15.6 | 77 / 2 / 7 / 15 | 52.2 ± 1.6 | 34.7 ± 0.8 | 54.5 ± 3.6 | 35.5 ± 1.7 |
| 23 | 108 | 49.5 ± 20.3 | 30.6 ± 12.4 | 85 / 2 / 10 / 3 | 50.4 ± 1.1 | 31.7 ± 0.4 | 49.8 ± 1.2 | 31.3 ± 0.5 |
| 27 | 93 | 50.8 ± 22.9 | 33.0 ± 13.3 | 74 / 9 / 15 / 2 | 52.9 ± 0.3 | 35.5 ± 0.2 | 49.9 ± 1.5 | 33.2 ± 1.0 |
| 30 | 188 | 46.0 ± 24.0 | 33.5 ± 15.2 | 57 / 16 / 21 / 5 | 47.4 ± 1.3 | 35.4 ± 0.6 | 44.5 ± 0.5 | 32.8 ± 0.3 |
| 37 | 157 | 36.7 ± 16.8 | 31.0 ± 10.4 | 34 / 40 / 23 / 3 | 39.2 ± 0.4 | 32.8 ± 0.4 | 35.5 ± 1.7 | 30.1 ± 1.2 |
| 47 | 228 | 27.1 ± 11.0 | 28.8 ± 7.5 | 11 / 59 / 30 / 0 | 29.5 ± 1.3 | 31.5 ± 0.9 | 28.9 ± 0.6 | 30.1 ± 0.6 |

**TABLE 8** Storage trends with the image as the unit of analysis (per-image medians, 66 images). Early: days 0–8; late: days 37 and 47. Model rows: mean ± SD over seeds of ρ and of the late-vs-early change; *p*-values as the range over seeds.

| Quantity | Masks | Spearman ρ vs day | Late vs early (%) | Partial ρ (dry mass \| area) |
|---|---|---|---|---|
| Projected area | Annotation | -0.53 (p = 4.2e-06) | -43.9 (MW p = 5.3e-06) |  |
| Projected area | MobileSAM r = 8 | -0.56 ± 0.06 (p 1.5e-07 to 2.6e-05) | -40.6 ± 1.3 (MW p 5.3e-06 to 2.1e-05) |  |
| Projected area | EdgeSAM r = 32 | -0.65 ± 0.08 (p 6.8e-13 to 1e-07) | -44.3 ± 1.2 (MW p 5.3e-06 to 6.5e-06) |  |
| Projected area | MobileNet-UNet r = 8 | -0.67 ± 0.03 (p 2.5e-11 to 4.5e-09) | -53.0 ± 0.6 (MW p 5.3e-06 to 5.3e-06) |  |
| Dry mass | Annotation | -0.11 (p = 0.36) | -0.8 (MW p = 0.21) | 0.08 (p = 0.5) |
| Dry mass | MobileSAM r = 8 | -0.17 ± 0.03 (p 0.11 to 0.27) | -2.6 ± 0.4 (MW p 0.094 to 0.12) | 0.04 ± 0.07 (p 0.35 to 0.85) |
| Dry mass | EdgeSAM r = 32 | -0.27 ± 0.04 (p 0.0078 to 0.052) | -6.1 ± 0.3 (MW p 0.013 to 0.049) | 0.08 ± 0.09 (p 0.24 to 0.87) |
| Dry mass | MobileNet-UNet r = 8 | -0.47 ± 0.01 (p 5.5e-05 to 8.2e-05) | -8.2 ± 0.5 (MW p 0.00035 to 0.00056) | -0.10 ± 0.02 (p 0.35 to 0.57) |
| Circularity | Annotation | 0.19 (p = 0.12) | +0.1 (MW p = 0.75) |  |
| Circularity | MobileSAM r = 8 | 0.14 ± 0.08 (p 0.07 to 0.56) | +0.1 ± 0.1 (MW p 0.094 to 0.85) |  |
| Circularity | EdgeSAM r = 32 | -0.15 ± 0.10 (p 0.09 to 0.79) | -1.7 ± 0.7 (MW p 0.0021 to 0.17) |  |
| Circularity | MobileNet-UNet r = 8 | -0.51 ± 0.03 (p 2.1e-06 to 3.5e-05) | -1.7 ± 0.4 (MW p 4.4e-05 to 7.6e-05) |  |

**Why:** Old Table 7 (EdgeSAM r = 8 only, 1,974 cells, old class names) replaced.

---

### M146 REPLACE: 4.9 Quantitative Phase-Derived Cellular Analysis Across Storage

**Find:** "![](media/image10.png){width="6.172784339457568in" height="2.4681681977252845in"}"

**Original:**

> ![](media/image10.png){width="6.172784339457568in" height="2.4681681977252845in"}

**New text (complete):**

*[insert figure file artifacts/figures/main/F10_storage_trends.png]*

---

### M147 REPLACE: 4.9 Quantitative Phase-Derived Cellular Analysis Across Storage

**Find:** "Fig. 8 Phase-derived dry-mass analysis. (a) Per-image median dry-mass estimates ..."

**Original:**

> **Fig. 8** Phase-derived dry-mass analysis. (a) Per-image median dry-mass estimates derived from EdgeSAM (*r* = 8) predictions versus those derived from annotation masks, with storage day indicated by color; the dashed line represents the identity line. (b) Median projected area and phase-derived dry mass normalized to their respective day-0 values over the storage period.

**New text (complete):**

**Fig. 10** Per-image median projected area (left) and dry mass (right) over storage, as the mean of the six images per day: annotation and predicted masks (mean over three seeds; band: range over seeds).

---


## 5. Discussion

### M149 REPLACE: 5. Discussion

**Find:** "The results demonstrate that physics-aware lightweight segmentation can improve the ..."

**Original:**

> The results demonstrate that physics-aware lightweight segmentation can improve the reliability of quantitative holographic cell analysis by treating the predicted cell boundary as a quantitative measurement domain rather than solely as a geometric contour. Conventional segmentation objectives such as Dice and Cross-Entropy primarily optimize spatial agreement between predicted and annotated masks and do not explicitly account for the quantitative phase information enclosed by the predicted boundary. In QPI, however, boundary placement directly determines the region over which the phase signal is integrated for subsequent phase-derived biophysical analysis. The ablation results support this distinction: for EdgeSAM, the full Physics-Aware Phase Consistency Loss reduced the global IPE from 5.22% with Dice-only optimization to 3.04%, while increasing Boundary F1 from 0.938 to 0.953. For MobileNet-UNet, the corresponding IPE decreased from 15.52% to 3.01%. These findings indicate that geometric overlap alone does not fully characterize the suitability of a segmentation mask for quantitative QPI measurements.

**New text (complete):**

Treating the predicted boundary as the measurement domain of the integrated phase changes what a segmentation model is optimized for. With the Physics-Aware Phase Consistency Loss, the global integrated-phase error was lower than with Dice-only training for every architecture and rank examined, and it was 1.3% for MobileSAM and 3.0% for EdgeSAM at the preferred configurations. The gain was largest where geometric training alone was unstable (EdgeSAM and MobileNet-UNet) and small where it was already accurate (MobileSAM), which indicates that the physics terms mainly protect the measurement domain against failure modes that overlap metrics do not penalize strongly. These findings show that geometric overlap alone does not fully characterize the suitability of a segmentation mask for quantitative QPI measurements.

**Why:** Old numbers (5.22 to 3.04%, 0.938 to 0.953, 15.52 to 3.01%) replaced.

---

### M150 REPLACE: 5. Discussion

**Find:** "The three components of the proposed physics-aware objective provide complementary ..."

**Original:**

> The three components of the proposed physics-aware objective provide complementary constraints on the predicted measurement domain. PMC incorporates phase-based foreground information, BGA encourages cellular contours to follow spatial phase transitions, and IPP directly constrains the quantitative phase integrated within the predicted cell region. The component-wise ablation further showed that these constraints must operate together with geometric segmentation objectives. In particular, PMC alone degraded boundary accuracy and minority-class segmentation, whereas the addition of BGA restored boundary localization. Incorporating IPP further reduced global IPE for EdgeSAM and MobileNet-UNet, demonstrating the importance of explicitly considering integrated phase during segmentation optimization. For MobileSAM, the full-objective run did not converge, so the effect of IPP on this architecture could not be established.

**New text (complete):**

The ablation supports the full objective as a whole rather than a ranking of its terms. With three seeds, the full objective gave the lowest and least variable IPE and fewer runs that lost a class; IPP had its clearest effect for EdgeSAM at *r* = 32, and the separate effects of PMC and BGA were not consistent across ranks. Because all three terms act on the class-agnostic foreground, they cannot by themselves prevent confusion between morphologies; this task is carried by the class-weighted Dice term.

**Why:** Removes the 'complementary constraints' and 'PMC alone degraded ..., BGA restored' claims and the MobileSAM non-convergence sentence.

---

### M151 REPLACE: 5. Discussion

**Find:** "Preserving integrated phase is particularly important because segmentation errors propagate ..."

**Original:**

> Preserving integrated phase is particularly important because segmentation errors propagate directly into downstream phase-derived measurements. Projected area and circularity depend on the predicted cellular boundary, while cellular dry mass is estimated from the phase integrated over the segmented region. Consequently, a segmentation mask with high conventional overlap can still introduce quantitative measurement error if its boundary systematically excludes cellular phase information or includes background regions. This measurement-oriented perspective is especially relevant to RBC storage analysis, where changes in cellular morphology and phase-derived biophysical properties provide complementary information on storage-associated cellular alterations \[47, 48\]. DHM and QPI are particularly advantageous in this context because they enable label-free quantitative cellular analysis without the staining requirements of conventional imaging approaches. The strong agreement between dry-mass estimates derived from EdgeSAM predictions and annotation masks (*r* = 0.966, CCC=0.96, MdARE=3.35%, Bland-Altman bias=−0.12 pg) therefore provides direct evidence that the predicted masks can serve as reliable measurement domains for phase-derived quantitative analysis.

**New text (complete):**

Preserving the integrated phase is particularly important because segmentation errors propagate directly into downstream phase-derived measurements. Without any calibration, the dry mass from the predicted masks agreed with the annotation-derived dry mass within a median of 2.9 to 3.9% per image, and the day-0 value (27.8 pg from the annotation) was consistent with a typical mean corpuscular hemoglobin. The low correlation between predicted and annotation-derived per-image dry mass is a consequence of the narrow range of the reference, not of large errors; Bland-Altman limits of about ±6 pg describe the remaining per-image error. DHM and QPI are advantageous in this context because they enable label-free quantitative cellular analysis without the staining required by conventional imaging [47, 48].

**Why:** Removes 'r = 0.966, CCC = 0.96 ... direct evidence'. Keeps refs [47, 48] so they stay cited.

---

### M152 REPLACE: 5. Discussion

**Find:** "The comparison among lightweight architectures and adaptation strategies further demonstrates ..."

**Original:**

> The comparison among lightweight architectures and adaptation strategies further demonstrates that computational efficiency must be considered together with segmentation and measurement reliability. LoRA-adapted EdgeSAM maintained non-zero segmentation performance across all four RBC morphologies while providing high boundary fidelity, adequate instance separation, and low IPE. In contrast, full fine-tuning caused severe minority-class degradation in EdgeSAM and MobileSAM despite updating substantially more parameters. These results suggest that restricting adaptation to low-rank parameter updates can provide a favorable regularization effect for relatively small and imbalanced QPI datasets while substantially reducing the number of trainable parameters. The LoRA rank analysis further showed that *r* = 8 provided the best overall balance among segmentation accuracy, instance separation, integrated-phase preservation, and parameter efficiency.

**New text (complete):**

MobileSAM and EdgeSAM offer different trade-offs. MobileSAM was the more accurate model (higher Dice, boundary F1-score, and class Dice, lower IPE) and was stable at ranks 2 to 16, but it is twice as slow in PyTorch, needs about eight times more GPU memory, and separated touching cells less well. EdgeSAM was faster and lighter, separated cells better, and needed a larger rank (*r* = 32) to train reliably; at smaller ranks, one of three seeds failed to learn the spherocyte class. Full fine-tuning lost minority classes in all three architectures, which suggests that restricting adaptation to low-rank updates regularizes training on relatively small and imbalanced QPI datasets. Reporting every seed was essential: a single seed would have hidden the failure of EdgeSAM at *r* = 8 (seed 44) or of MobileSAM at *r* = 32 (seed 43).

**Why:** Objective comparison (professor's point 2); removes 'r = 8 provided the best overall balance' and the EdgeSAM selection.

---

### M153 REPLACE: 5. Discussion

**Find:** "Computational benchmarking also showed that the selected EdgeSAM configuration is ..."

**Original:**

> Computational benchmarking also showed that the selected EdgeSAM configuration is suitable for efficient QPI analysis. ONNX FP16 inference on an NVIDIA RTX A5000 reduced median latency to 4.33 ms and achieved 228.3 FPS with a peak allocated VRAM of 16.4 MB. MobileNet-UNet achieved higher throughput but failed to detect two RBC morphology classes, whereas MobileSAM exhibited poorer instance separation and limited ONNX compatibility. EdgeSAM therefore provided the most balanced combination of quantitative reliability and computational efficiency among the evaluated architectures. These results characterize performance on a desktop-class GPU rather than actual edge-device deployment; validation on low-power embedded platforms will be required to establish real-world edge performance.

**New text (complete):**

On a desktop GPU, both preferred models run at video rate in PyTorch (about 49 FPS for MobileSAM and 95 FPS for EdgeSAM on average), and EdgeSAM exceeds 200 FPS with ORT. These results characterize performance on a desktop-class GPU rather than actual edge-device deployment; validation on low-power embedded platforms will be required to establish real-world edge performance.

**Why:** Removes ONNX FP16, 4.33 ms, 228.3 FPS, 16.4 MB, 'limited ONNX compatibility'. CHECK: '95 FPS' is the mean of the EdgeSAM PyTorch per-seed FPS (95.1, benchmark file); 49 FPS for MobileSAM.

---

### M154 REPLACE: 5. Discussion

**Find:** "The biological analysis further demonstrates the utility of the proposed ..."

**Original:**

> The biological analysis further demonstrates the utility of the proposed measurement-oriented segmentation framework. Across the 47-day storage period, the predicted RBC populations exhibited systematic morphological changes, accompanied by a non-monotonic change in projected cell area that peaked at day 15 and declined substantially by day 47. Phase-derived dry mass fluctuated with a significant overall decline that was largely, though not entirely, explained by the concurrent area change. Cellular dry mass was estimated from the integrated quantitative phase using the known illumination wavelength, effective pixel pitch, and specific refractive increment, with the phase scale calibrated to the MCH reference (Section 2.6). This enables direct comparison of phase-derived dry-mass measurements across storage time points and between predicted and annotation-derived measurement boundaries.

**New text (complete):**

In the annotated cells, projected area decreased by 44% over 47 days of storage while dry mass remained constant, the signature of the spherocytic transformation in which membrane rather than hemoglobin is lost [24, 25]. This contrast is the kind of information that QPI adds to morphology, and it depends on the preservation of the integrated phase: the predicted masks reproduced the area change and the composition but introduced a dry-mass bias of 2.6 to 6.1% between early and late storage that arose mainly in images with many touching cells. Instance separation, rather than boundary placement, is therefore the main limitation of the present models for population-level optical-mass analysis.

**Why:** Removes the dry-mass decline and 'phase scale calibrated to the MCH reference' (R1-9 consistency).

---

### M155 REPLACE: 5. Discussion

**Find:** "Several limitations remain. The LoRA rank and loss configuration were ..."

**Original:**

> Several limitations remain. The LoRA rank and loss configuration were selected using the available validation set, and an independent held-out test cohort was not available. The framework was evaluated only on RBC QPI data, and its generalizability to other cell types, imaging conditions, and densely crowded cellular environments remains to be established. Because phase unwrapping relies on the phase variation remaining within an unambiguous range, specimens with substantially greater optical thickness than RBCs (where the accumulated phase could exceed the unwrapping range) may require additional validation of the reconstruction before this framework is applied. The full physics-aware objective also reduced instance-separation performance relative to Dice-only optimization for EdgeSAM, indicating a remaining trade-off between integrated-phase preservation and individual-cell separation. For MobileSAM, the full-objective ablation run did not converge, so the effect of IPP on this architecture could not be established and should not be assumed to transfer across architectures. In addition, LoRA adaptation was performed offline, and adaptation to changes in imaging conditions or optical aberrations was not investigated. The dataset used in this study provides only the reconstructed phase channel; the corresponding amplitude images were not available, as the original holograms were not retained by the data provider, so incorporating amplitude information was not evaluated in this work. Amplitude images could potentially provide complementary information in the low-phase and boundary regions where segmentation errors were found to concentrate (Section 4.3), and represent a promising direction for future work. Future work will therefore focus on validation using independent QPI datasets and additional cell types, improving instance separation while maintaining integrated-phase preservation, and evaluating the framework on embedded computing platforms.

**New text (complete):**

Several limitations remain. The validation images were also the evaluation images: no checkpoint was selected on them, but the preferred ranks were chosen after inspecting the rank sweep, and an independent held-out test cohort was not available. Three seeds give wide confidence intervals, and full fine-tuning was run once. Float32 phase maps were available for 182 of the 256 training images, and the annotations were transferred from 8-bit exports by registration. The framework was evaluated only on RBC QPI data, and its generalizability to other cell types, imaging conditions, and densely crowded cellular environments remains to be established. Because the reconstructed phase is limited to an unambiguous range, cells reaching the 2π limit (3.0% of the annotated cells; 12% of echinocytes and 10% of stomatocytes) were excluded from phase-derived results, and specimens with substantially greater optical thickness than RBCs would require phase unwrapping before this framework is applied. In addition, LoRA adaptation was performed offline, and adaptation to changes in imaging conditions or optical aberrations was not investigated. The dataset used in this study provides only the reconstructed phase channel; the corresponding amplitude images were not available, as the original holograms were not retained by the data provider, so incorporating amplitude information was not evaluated in this work. Amplitude images could potentially provide complementary information in the low-phase and boundary regions where segmentation errors were found to concentrate (Section 4.4), and represent a promising direction for future work. Future work will therefore focus on improving instance separation within the physics-aware objective, validation using independent QPI datasets and additional cell types, and evaluation on embedded computing platforms.

**Why:** Keeps the unwrapping (R2-3) and amplitude (R2-5) parts. Removes 'full objective reduced instance separation for EdgeSAM' (not true at r = 32: AJI 0.335 vs 0.329) and the MobileSAM non-convergence sentence; adds the rank-choice and seed disclosures.

---

### M156 REPLACE: 5. Discussion

**Find:** "Overall, the present results establish optical-mass preservation as a measurement-oriented ..."

**Original:**

> Overall, the present results establish optical-mass preservation as a measurement-oriented principle for physics-aware QPI segmentation. In this framework, optical-mass preservation is achieved by constraining the integrated quantitative phase within predicted cellular regions rather than by interpreting QPI as a direct measurement of geometric volume. By combining this physical constraint with lightweight architecture design and parameter-efficient adaptation, the proposed framework enables computationally efficient holographic cell segmentation in which predicted boundaries function as physically meaningful measurement domains for subsequent quantitative cellular analysis.

**New text (complete):**

Overall, the present results support optical-mass preservation as a measurement-oriented principle for physics-aware QPI segmentation. In this framework, optical-mass preservation is achieved by constraining the integrated quantitative phase within predicted cellular regions rather than by interpreting QPI as a direct measurement of geometric volume. By combining this physical constraint with lightweight architecture design and parameter-efficient adaptation, the proposed framework enables computationally efficient holographic cell segmentation in which predicted boundaries function as physically meaningful measurement domains for subsequent quantitative cellular analysis.

**Why:** Only 'establish' -> 'support'.

---


## 6. Conclusion

### M158 REPLACE: 6. Conclusion

**Find:** "This study presented a physics-aware lightweight segmentation framework for quantitative ..."

**Original:**

> This study presented a physics-aware lightweight segmentation framework for quantitative holographic cell analysis, in which predicted cell boundaries serve as quantitative measurement domains. The proposed Physics-Aware Phase Consistency Loss explicitly incorporates quantitative phase information into segmentation optimization, enabling integrated-phase preservation beyond conventional geometric segmentation. The physics-aware objective gave clear gains for EdgeSAM and MobileNet-UNet; for MobileSAM, the full-objective run did not converge, so no benefit could be established. Combined with LoRA-based adaptation, the framework achieved reliable RBC segmentation with computationally efficient inference.

**New text (complete):**

This study presented a physics-aware lightweight segmentation framework for quantitative holographic cell analysis, in which the predicted cell boundary is the measurement domain of the integrated phase. On background-referenced 32-bit phase maps of red blood cells, the Physics-Aware Phase Consistency Loss lowered the integrated-phase error relative to Dice-only training for every architecture and LoRA rank examined and reduced training failures.

**Why:** Removes 'clear gains for EdgeSAM and MobileNet-UNet; for MobileSAM ... did not converge'.

---

### M159 REPLACE: 6. Conclusion

**Find:** "Among the evaluated architectures, LoRA-adapted EdgeSAM provided the most balanced ..."

**Original:**

> Among the evaluated architectures, LoRA-adapted EdgeSAM provided the most balanced performance in segmentation reliability, integrated-phase preservation, and computational efficiency. Application to RBC populations over a 47-day storage period demonstrated storage-associated morphological changes and supported reliable phase-derived dry-mass analysis, with strong agreement between predicted-mask and annotation-derived measurements (*r* = 0.966).

**New text (complete):**

With LoRA, MobileSAM (*r* = 8) gave the most accurate segmentation and the lowest integrated-phase error (1.3%), and EdgeSAM (*r* = 32) the best combination of instance separation and computational cost. Dry mass estimated from the predicted masks without calibration agreed with annotation-derived values within a median of 3 to 4% per image, and the annotation showed that stored cells lose projected area but not dry mass.

**Why:** Removes 'EdgeSAM provided the most balanced performance' and 'r = 0.966'.

---

### M160 REPLACE: 6. Conclusion

**Find:** "Overall, the proposed approach demonstrates that optical-mass preservation can be ..."

**Original:**

> Overall, the proposed approach demonstrates that optical-mass preservation can be incorporated into lightweight QPI segmentation by explicitly preserving the integrated quantitative phase within predicted cellular regions. This measurement-oriented framework provides a basis for computationally efficient quantitative holographic cell analysis, with future work extending validation to independent QPI datasets, additional cell types, and embedded computing platforms.

**New text (complete):**

Overall, the proposed approach shows that optical-mass preservation can be incorporated into lightweight QPI segmentation by explicitly preserving the integrated quantitative phase within predicted cellular regions. Improving the separation of touching cells within the physics-aware objective, and validation on independent QPI datasets, additional cell types, and embedded computing platforms, are the next steps.

---

**M161 KEEP** "Author contributions Usama Ikram implemented the lightweight segmentation framework, LoRA-based adaptation, and Physics-Aware Phase ..."  
*Note:* Check that 'benchmarking experiments' is still accurate.

**M162 KEEP** "Funding This work was supported by the National Research Foundation of Korea (NRF) grant ..."

**M163 KEEP** "Data availability"

**M164 KEEP** "The data generated and/or analyzed during the current study are available from the corresponding ..."  
*Note:* Optional: add 'Code and per-run results are available at <repository URL or DOI>' if the repository will be made public.

**M165 KEEP** "Declarations Conflict of interest The authors declare that they have no competing interests. Ethics ..."

**M166 KEEP** "References [1] D. Gabor, A new microscopic principle. Nature 161, 777-778 (1948). [2] A. ..."  
*Note:* References unchanged. After editing, check that every reference is still cited (e.g. [31], [41], [42]).
