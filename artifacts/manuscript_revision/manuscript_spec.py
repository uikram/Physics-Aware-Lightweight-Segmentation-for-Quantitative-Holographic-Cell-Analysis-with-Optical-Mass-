"""Paragraph-by-paragraph revision of the manuscript (source of EDIT_GUIDE_MANUSCRIPT.md and REVISED_MANUSCRIPT.docx).

Each entry covers one block (or a range of blocks) of the CURRENT manuscript file
artifacts/manuscript_in_prog/revised_manuscript_fv - ui.docx, found by the first words of the block ("find"), in document order.
The builder (artifacts/code/build_revision_docs.py) checks that every block of the current file is covered exactly once.

    action  KEEP     the block stays as it is (the revised document uses the original text)
            REPLACE  the whole block is replaced by "new" (complete, ready-to-paste text)
            DELETE   the block is removed ("new" is empty)
            NEW      "new" is inserted at this position (no original block)
    why     reason; "R1-4" = reviewer 1, comment 4 of the response letter (kept in mind, the letter is edited later)

Placeholders in "new": [[tab:x]], [[fig:x]], [[stab:x]], [[sfig:x]] become table / figure numbers; {{TABLE:x}} inserts a table built
from artifacts/tables; {{FIG:x|caption}} inserts a figure. Numbers in the text are copied from artifacts/tables (the builder lists any
number it cannot find there).
"""

E = lambda find, action, new="", why="", **kw: dict(find=find, action=action, new=new.strip(), why=why, **kw)

ENTRIES = [
    # ------------------------------------------------------------------------------------------------ front matter
    E("Physics-Aware Lightweight Segmentation for Quantitative Holographic Cell Analysis", "KEEP", "# Physics-Aware Lightweight Segmentation for Quantitative Holographic Cell Analysis with Optical Mass Preservation"),
    E(("Usama Ikram", "Corresponding author"), "KEEP"),
    E("Abstract", "KEEP", "**Abstract**"),
    E("Digital holographic microscopy provides label-free", "REPLACE", r"""
Digital holographic microscopy provides label-free quantitative phase images from which cellular dry mass is obtained by integrating
the phase over each cell. Segmentation networks trained with geometric overlap losses do not constrain this integral, so boundary
errors propagate into optical-mass estimates. We propose a physics-aware lightweight segmentation framework in which the predicted cell
boundary is treated as the measurement domain of the integrated phase. A Physics-Aware Phase Consistency Loss adds phase-mask
contrast, boundary-gradient alignment, and integrated-phase preservation terms, computed on background-referenced 32-bit phase maps,
to a class-weighted Dice loss, and low-rank adaptation (LoRA) adapts lightweight foundation-model encoders to single-channel phase
images. On red blood cell phase maps acquired over 47 days of storage, MobileSAM and EdgeSAM were compared at five LoRA ranks with
three random seeds per configuration (180 training runs including the loss ablation). Relative to Dice-only training, the
physics-aware objective lowered the global integrated-phase error in all 15 architecture-rank combinations and reduced the number of
runs that lost a morphology class. MobileSAM (rank 8) reached a mean Dice of 0.810 ± 0.006, a boundary F1-score of 0.802, and a global
integrated-phase error of 1.32 ± 0.09%; EdgeSAM (rank 32) gave better instance separation (aggregated Jaccard index 0.335 vs 0.262)
at about half the latency (10.6 vs 20.4 ms) and one eighth of the GPU memory. Without calibration, per-image dry mass from predicted
masks agreed with annotation-derived values within a median absolute relative error of 2.9% (MobileSAM) and 3.9% (EdgeSAM).
Annotation-derived dry mass showed no storage trend while projected cell area decreased by 44%; the predicted masks reproduced the
area change and the morphological composition but added a small storage-dependent dry-mass bias, which identifies instance
separation as the remaining limitation for optical-mass analysis.
""", why="Your current abstract is already this text; corrected the typo 'aalysis' and the stray 'f[rom' highlight. Numbers: Tables 1, 5, 6, 8."),
    E("Index Terms", "REPLACE", r"""
**Index Terms**---Quantitative Digital Holographic Microscopy, Quantitative Phase Imaging, Physics-Aware Segmentation, Optical Mass
Preservation, Lightweight Deep Learning, Low-Rank Adaptation.
""", why="Optional: 'Low-Rank Adaptation' added because LoRA is now a main axis of the comparison."),

    # ------------------------------------------------------------------------------------------------ 1. Introduction
    E("Introduction", "KEEP", "# 1. Introduction"),
    E("Holography was introduced by Dennis Gabor", "KEEP", why="Added for R1-1 (general background on holography); keep."),
    E("Digital recording followed advances", "KEEP", why="R1-1; keep."),
    E("Applied to microscopy, DH is known as", "KEEP", why="R1-1; keep. First expansion of DHM."),
    E("Quantitative phase imaging (QPI) uses", "KEEP", why="Dry-mass rationale; still correct."),
    E("This is particularly valuable in hematological analysis", "KEEP",
      why="Literature statement (discocytes to echinocytes to spherocytes) agrees with the corrected annotation (spherocytes 0% at day 0, 59% at day 47)."),
    E("Segmentation is regarded as mandatory", "KEEP"),
    E("Lightweight segmentation models are attractive", "KEEP",
      why="Describes the published MobileSAM and EdgeSAM models; how they are used here is stated in Section 2.3."),
    E("Because full-network adaptation of large", "KEEP"),
    E("Despite these advances, an important limitation remains", "KEEP"),
    E("A second limitation concerns domain adaptation", "KEEP"),
    E("This work addresses these limitations", "REPLACE", r"""
This work addresses these limitations with a physics-aware lightweight segmentation framework whose main contribution is a
segmentation objective that preserves the integrated quantitative phase within the predicted cells, so that the predicted boundary
can be used directly for optical-mass estimation. We introduce a Physics-Aware Phase Consistency Loss that incorporates quantitative
phase information into the segmentation objective through three constraints: Phase-Mask Contrast (PMC), a global constraint on
foreground localization based on cell-to-background phase contrast; Boundary-Gradient Alignment (BGA), which encourages predicted
boundaries to follow spatial phase transitions; and Integrated-Phase Preservation (IPP), which explicitly constrains the quantitative
phase integrated within the predicted cellular region. The constraints act on the original 32-bit phase maps in radians after each
map has been referenced to its own background, never on the normalized network input, so that the preserved physical quantity is
altered neither by input normalization nor by the arbitrary zero level of the reconstruction. LoRA then adapts lightweight models
pretrained on natural images to the holographic phase domain with most parameters frozen, which also regularizes training on a
small, imbalanced dataset.
""", why="States the main contribution (professor's point 1) inside the existing paragraph, not as a separate contribution list "
          "(R1-2 objected to such a list). 'Raw, unnormalized phase' is replaced because the phase is now background-referenced (R3-5)."),
    E("Three representative lightweight architectures", "REPLACE", r"""
The image encoders of two lightweight foundation models, MobileSAM [33] and EdgeSAM [34], and a convolutional MobileNet-UNet [32]
baseline are adapted with LoRA and evaluated on quantitative phase images of red blood cells recorded over a 47-day storage period.
Because the stability of parameter-efficient adaptation on a small dataset cannot be judged from a single training run, every
configuration is trained with three random seeds, and the models are compared in terms of segmentation accuracy, integrated-phase and
optical-mass preservation, computational efficiency, and training stability.
""", why="Professor's point 2 (objective comparison of accuracy, efficiency and stability) and point 4 (three seeds)."),

    # ------------------------------------------------------------------------------------------------ 2. Method
    E("Method", "KEEP", "# 2. Method"),
    E("Overall Framework", "KEEP", "## 2.1 Overall Framework"),
    E("The proposed framework treats the predicted cell boundary", "REPLACE", r"""
The proposed framework treats the predicted cell boundary as a quantitative measurement domain rather than as a purely geometric
contour. As illustrated in [[fig:frame]], the pipeline consists of four main stages: background referencing and standardization of
the 32-bit quantitative phase maps, segmentation by a LoRA-adapted lightweight network, optimization with the Physics-Aware Phase
Consistency Loss, and extraction of phase-derived cellular measurements.
""", why="'standardization of the quantitative phase maps' -> 'background referencing and standardization of the 32-bit ...' (preprocessing as implemented)."),
    E("![](media/image1.png)", "KEEP", why="Fig. 1 must be REDRAWN: draw two branches from the phase map (background-referenced phase in rad -> PMC/BGA/IPP; "
                                           "per-image standardized phase -> network); do not draw the SAM prompt encoder or mask decoder (not used)."),
    E("Fig. 1", "REPLACE", r"""
**Fig. 1** Overview of the proposed physics-aware lightweight segmentation framework for quantitative holographic cell analysis.
LoRA-adapted lightweight networks segment cells from the standardized phase image, while the background-referenced phase in radians
enters the Physics-Aware Phase Consistency Loss through the PMC, BGA, and IPP constraints. The resulting segmentation masks serve as
measurement domains that preserve phase-integrated optical information for subsequent quantitative cellular analysis.
""", why="'raw unnormalized phase' -> background-referenced phase; matches the redrawn figure."),
    E("QPI Preprocessing", "KEEP", "## 2.2 QPI Preprocessing"),
    E("The input data consist of single-channel quantitative phase", "REPLACE", r"""
The input data consist of single-channel quantitative phase maps reconstructed as 32-bit floating-point arrays in radians
(768 × 768 pixels). These maps represent the quantitative phase delay associated with the optical path difference introduced by the
cells. Because the measured phase depends jointly on refractive-index contrast and specimen thickness, it is treated as an optical
phase quantity rather than as a direct measurement of geometric cell thickness or volume. Ground-truth semantic masks contain five
classes, with mask values 0 to 4 assigned to background, discocyte, spherocyte, echinocyte, and stomatocyte, respectively. The masks
had been drawn on 8-bit grey-level exports of the phase maps; they were transferred to the 32-bit maps by an affine
enhanced-correlation-coefficient registration of each phase map to its 8-bit export (the estimated transforms consisted of scale and
translation only) with nearest-neighbor resampling of the labels. Pixels of a phase map outside the field of view of its 8-bit export
carry no annotation and are excluded from every loss term and every metric.
""", why="Your current first sentence is kept. Added: map size, class order by mask value (2 = spherocyte, 3 = echinocyte; the old list had the "
          "two swapped), how the masks were carried onto the 32-bit maps, and the unannotated pixels (all as implemented, DECISIONS D-01, D-02)."),
    E("To adapt the single-channel phase maps to segmentation networks", "REPLACE", r"""
The reconstructed phase maps do not share a common zero level: the background phase of the validation maps lay between 0.66 and
2.50 rad. Because an arbitrary constant c adds c·A to the phase integrated over a region of area A, every map was referenced to its
own background before training and evaluation: the median phase of the annotated background pixels more than 5 pixels from any cell
was subtracted, so that the background is 0 rad and the integrated phase of a cell measures the cell only. The same
background-referenced phase is used for the physics-aware loss terms (PMC, BGA, and IPP), for the integrated-phase error, and for the
dry-mass estimation. For the network input only, the referenced map is resized to the input size of each model (256 × 256 pixels for
MobileNet-UNet and EdgeSAM, 1024 × 1024 pixels for MobileSAM) and standardized per image to zero mean and unit variance; the
physics-aware terms receive the background-referenced phase in radians at the same resolution and never the standardized values.
""", why="Background phase correction described as implemented (professor's point 5; R2-1, R3-5: the same correction is applied to the loss, "
          "the IPE and dry mass). Removes 'Spatial resizing and cropping' (there is no cropping). Offsets: Table S10."),
    E("Data augmentation is restricted to transformations", "REPLACE", r"""
Data augmentation is restricted to transformations that preserve the quantitative phase values and cellular morphology. Each training
image is used in its original orientation and in its three lossless 90° rotations. Intensity perturbations, nonlinear contrast
transformations, and phase-value modifications are excluded because they could alter the quantitative optical information that the
proposed framework is designed to preserve.
""", why="Only 90° rotations are used (no flips, no translation); the paragraph is rewritten so it reads correctly."),
    E("Lightweight Segmentation Models", "KEEP", "## 2.3 Lightweight Segmentation Models"),
    E("Three lightweight architectures are evaluated, covering", "REPLACE", r"""
Three lightweight architectures are evaluated, covering convolutional, transformer-based, and hybrid designs. MobileNet-UNet pairs an
ImageNet-pretrained MobileNetV2 encoder with a U-Net-style decoder with skip connections, which retains the fine spatial information
required for accurate cellular boundary delineation [32]. For MobileSAM and EdgeSAM, only the pretrained image encoders are used: the
distilled TinyViT encoder of MobileSAM [33] and the RepViT encoder of EdgeSAM [34]. Because dense five-class segmentation requires no
prompts, each encoder is combined with a compact decoder trained from scratch. In MobileSAM, a learned prompt embedding is added to
five class tokens that attend to the image embedding through one transformer decoder layer, and each class map is the inner product
of its token with the upsampled embedding; in EdgeSAM, four transposed-convolution stages upsample the embedding to the input
resolution, followed by a 1 × 1 classifier. In all three networks, the first convolution is replaced by a single-channel convolution
initialized with the mean of the pretrained RGB filters. EdgeSAM is evaluated as a computationally efficient architecture rather than
as evidence of embedded deployment.
""", why="The old text said MobileSAM and EdgeSAM 'retain the mask-decoding pipeline of the original SAM'; the code uses only their encoders "
          "with task decoders (qpi/models/mobile_sam.py, edge_sam.py). Kept short (R1-3)."),
    E("LoRA-Based Lightweight Adaptation", "KEEP", "## 2.4 LoRA-Based Lightweight Adaptation"),
    E("LoRA is employed to adapt pretrained lightweight", "KEEP"),
    E("$W' = W + (B", "KEEP"),
    E("where *W* is the frozen pretrained weight matrix", "REPLACE", r"""
where *W* is the frozen pretrained weight matrix, *B* and *A* are trainable low-rank matrices, *r* is the rank of the low-dimensional
adaptation space, and *α* is a scaling factor. We set *α* = *r*, so the low-rank update enters with unit scaling at every rank, and *B*
is initialized to zero so that training starts from the pretrained model. To evaluate the balance between adaptation capacity,
parameter efficiency, and training stability, a rank sweep *r* ∈ {2, 4, 8, 16, 32} is performed for every architecture.
""", why="α = r as implemented (configs/experiments.yaml); the ranks are now stated."),
    E("LoRA modules are inserted into architecture-dependent layers", "REPLACE", r"""
LoRA modules are inserted into architecture-dependent layers. In the TinyViT encoder of MobileSAM, they are added to the fused
query-key-value projections of the attention blocks; in the convolutional encoders of EdgeSAM and MobileNet-UNet, they are added to
every non-grouped convolution and linear layer, while depthwise convolutions remain frozen. To address the structural domain shift
from three-channel natural images to single-channel quantitative phase maps, the single-channel input stem and the normalization
parameters are fully unfrozen. The decoder and the output head also remain trainable to generate the five-class RBC segmentation
output, so the fraction of trainable parameters differs among architectures ([[stab:S8]]). For inference, the low-rank matrices are
merged into the frozen weights (*W*′ = *W* + *BA*), so latency and memory do not depend on the rank.
""", why="'query and value projection layers' is not what the code does (qpi/models/lora_utils.py). VERIFY on the server once that the "
          "MobileSAM LoRA layers are the qkv projections (07_REMAINING_WORK Q-D)."),
    E("Physics-Aware Phase Consistency Loss", "KEEP", "## 2.5 Physics-Aware Phase Consistency Loss"),
    E("Conventional segmentation objectives optimize geometric agreement", "KEEP"),
    E("Multi-Class Dice Loss", "KEEP", "### 2.5.1 Multi-Class Dice Loss"),
    E("To mitigate class imbalance among rare morphology", "KEEP"),
    E("$L_{Dice}", "KEEP"),
    E("where $w_{c} \\in", "REPLACE", r"""
where $w_{c} \in \{ 0.5,1.0,1.5,2.0,2.0\}$ correspond to the background and the four morphology classes, respectively, $P_{c}$ and
$G_{c}$ are the predicted probability map and the annotation of class *c*, and pixels without annotation are excluded from all sums.
A Laplace smoothing factor of $+ 1$ prevents division by zero for images in which a class is absent. All loss terms are computed in
32-bit floating point, whereas the network forward pass uses FP16 mixed precision.
""", why="Adds the exclusion of unannotated pixels and the float32 loss computation (as implemented)."),
    E("Phase-Mask Contrast Term", "KEEP", "### 2.5.2 Phase-Mask Contrast Term"),
    E("To ensure the predicted cell region physically maintains", "REPLACE", r"""
To ensure that the predicted cell region maintains higher phase values than the surrounding medium, the contrast term penalizes
predictions in which the mean cell phase does not exceed the mean background phase by at least a predefined margin:
""", why="Wording made consistent with Eq. (3) (the hinge is active when the cell-background contrast is below the margin)."),
    E("$L_{PMC}", "KEEP"),
    E("where $\\mu_{cell}$ and $\\mu_{bg}$", "REPLACE", r"""
where $\mu_{cell}$ and $\mu_{bg}$ denote the mean background-referenced phase inside the predicted cellular foreground and in the
surrounding background, respectively, each weighted by the soft foreground probability, and the margin is 0.1 rad. This term
provides a global physical constraint on foreground localization but does not by itself guarantee accurate cellular boundaries.
""", why="States the phase used and the margin value (configs/base.yaml)."),
    E("Boundary-Gradient Alignment Term", "KEEP", "### 2.5.3 Boundary-Gradient Alignment Term"),
    E("Because probabilistic network outputs", "REPLACE", r"""
Because probabilistic network outputs $\lbrack 0,1\rbrack$ and physical phase measurements exist on incompatible scales, the spatial
gradient magnitudes of the predicted mask ($\nabla M$) and of the background-referenced phase map ($\nabla\Phi$), computed by forward
differences, are first max-normalized per sample to a $\lbrack 0,1\rbrack$ scale. The segmentation boundary is then aligned with
strong phase gradients via an $L_{1}$ penalty:
""", why="'raw phase map' -> background-referenced phase map (R3-5)."),
    E("$L_{BGA}", "KEEP"),
    E("This term geometrically anchors the predicted measurement boundary", "REPLACE", r"""
The $L_{1}$ difference is averaged over annotated pixels. This term anchors the predicted measurement boundary to phase transitions
present in the measured QPI data.
"""),
    E("Integrated-Phase Preservation Term", "KEEP", "### 2.5.4 Integrated-Phase Preservation Term"),
    E("The key quantitative constraint of the proposed framework", "REPLACE", r"""
The key quantitative constraint of the proposed framework is preservation of the phase signal integrated over the segmented cellular
domain. For a background-referenced quantitative phase map *φ~i~*, predicted foreground probability *M~i~*, and ground-truth
foreground mask *G~i~*, the predicted and reference integrated phase quantities are $S_{pred} = \sum_{i}^{}{M_{i}\varphi_{i}},$
$S_{GT} = \sum_{i}^{}{G_{i}\varphi_{i}}$. The IPP loss is then defined as the ground-truth-normalized relative error:
""", why="'raw quantitative phase map' -> background-referenced (R3-5)."),
    E("$L_{IPP}", "KEEP"),
    E("The relative formulation improves numerical stability", "REPLACE", r"""
The relative formulation ($\epsilon = 10^{- 4}$) makes the penalty insensitive to the absolute scale of the phase integral. Because
*φ* is referenced to the background, $S_{GT}$ measures the cells only; without this referencing, a residual offset *c* would add *c*
times the foreground area to both integrals. Because the sums run over the whole foreground of an image, IPP constrains the total
integrated phase of the image; the error of individual cells is evaluated separately (Section 3.3.2). The IPP term directly
constrains the integrated phase quantity used in subsequent optical-mass analysis. Under the conventional specific refractive
increment assumption, this quantity is proportional to cellular dry mass; preserving the integrated phase during segmentation
therefore targets the quantity required for downstream dry-mass estimation.
""", why="R2-1 / R3-5 (offset sensitivity explained), R3-2 (IPP is image-level, per-cell error reported separately). "
          "'directly improves the reliability' softened to 'targets'."),
    E("Final Loss Function", "KEEP", "### 2.5.5 Final Loss Function"),
    E("The final objective combines the biological segmentation task", "KEEP"),
    E("$L_{total}", "KEEP"),
    E("where $\\lambda_{1}$, $\\lambda_{2}$, and $\\lambda_{3}$", "REPLACE", r"""
where $\lambda_{1} = 0.1$, $\lambda_{2} = 0.05$, and $\lambda_{3} = 0.1$ control the relative contributions of the PMC, BGA, and IPP
terms, respectively. All physics-aware terms are evaluated using the binary cellular foreground probability
$M = 1 - p_{background}$ rather than individual morphology classes, and each term is averaged over the images of a mini-batch that
contain at least one annotated cell. This separates the morphological classification objective from the physical requirement that
the total cellular measurement domain be accurately preserved. The contribution of each term is examined by the ablation in
Section 4.4.
""", why="Weights stated; the claim 'As demonstrated by the ablation experiments, the individual constraints are complementary ...' is removed "
          "because the corrected ablation does not separate the terms (Table S7)."),
    E("Quantitative Phase-Derived Cellular Measurements", "KEEP", "## 2.6 Quantitative Phase-Derived Cellular Measurements"),
    E("The principal purpose of the proposed physics-aware segmentation", "KEEP"),
    E("$A = N_{\\Omega}", "KEEP"),
    E("where *N~Ω~* is the number of pixels", "KEEP", why="Pixel pitch 0.1441 µm is the value used by the code; see Section 3.1 note."),
    E("$C = \\frac{4\\pi A}", "KEEP"),
    E("where *P* denotes the perimeter", "KEEP"),
    E("$S_{\\varphi} = \\sum", "KEEP"),
    E("Here, *S~φ~* represents the area-integrated", "REPLACE", r"""
Here, *φ~i~* is the phase of pixel *i* after subtraction of the local background level described below, and *S~φ~* represents the
area-integrated quantitative phase associated with the cell, incorporating the coupled contributions of refractive-index contrast and
optical thickness. It serves as the primary phase-derived optical quantity whose preservation is explicitly enforced by the IPP loss.
Under the conventional assumption that the cellular refractive-index increment is approximately proportional to dry-mass
concentration, cellular dry mass can be estimated from the integrated phase as:
""", why="Defines φ as background-corrected (R3-5)."),
    E("$m = \\frac{\\lambda}", "KEEP"),
    E("where *λ* is the illumination wavelength", "KEEP"),
    E("Per-cell measurements are obtained through connected-component", "REPLACE", r"""
Per-cell measurements are obtained from connected components of the predicted foreground at the native 768 × 768 resolution.
Components with a projected area outside 15–200 µm², which correspond to isolated pixel artifacts or unresolved multi-cell clusters,
and components that touch an unannotated image region are excluded. The cell perimeter is calculated with the Crofton estimator to
reduce sensitivity to pixelation. For dry-mass estimation, the background level is estimated as the median phase of the non-cellular
pixels farther than 5 pixels from the analyzed mask and subtracted from the phase of each cell, so that each measurement uses only the
mask under evaluation (predicted or annotated). Because the phase maps are in radians, Eq. (10) is applied with the physical constants
only; no calibration to a reference hemoglobin content is used. Cells whose phase reaches the 2π upper limit of the reconstructed maps
or shows a wrap discontinuity (phase within 0.3 rad of 2π, or a jump larger than π between neighboring pixels, within 2 pixels of the
cell) are excluded from all phase-derived results; this rule was fixed before the results were computed.
""", why="Removes 'phase scale calibrated to the MCH reference' (no calibration, DECISIONS D-13) and 'No background correction is applied when "
          "computing the segmentation loss or the global IPE' (now applied everywhere, R3-5). Adds the wrapped-cell rule (R2-3)."),

    # ------------------------------------------------------------------------------------------------ 3. Experiments
    E("Experiments", "KEEP", "# 3. Experiments"),
    E("Dataset and Imaging Configuration", "KEEP", "## 3.1 Dataset and Imaging Configuration"),
    E("The experimental evaluation was designed to assess", "REPLACE", r"""
The experimental evaluation was designed to assess four aspects of the proposed framework: (1) multi-class segmentation accuracy
across different RBC morphologies, (2) preservation of quantitative phase information and optical-mass-related measurements,
(3) computational efficiency, and (4) training stability of the lightweight segmentation architectures. The framework was evaluated
using single-channel quantitative phase images acquired with a DHM system during long-term blood storage experiments [26, 44]. Blood
samples (2 mL each) were collected from three healthy donors under a protocol approved by the Institutional Review Board of DGIST
(Approval No. DGIST-180713-BR-012-01), with the same samples imaged repeatedly over the storage period. The quantitative phase maps
were numerically reconstructed from off-axis holograms and represent the phase delay associated with the optical path difference
introduced by the RBCs. Images were acquired at eleven time points over 47 days of storage (days 0, 5, 8, 12, 15, 19, 23, 27, 30, 37,
and 47). Cells were manually annotated and classified by an expert biologist into discocytes, spherocytes, echinocytes, and
stomatocytes using representative example morphologies as classification criteria. Cells were segmented into multi-label masks, and
overlapping cells were separated into individual instances using a distance transform of the mask thresholded at 0.3 to construct
binary marker maps. No cells superposed along the optical axis were identified among the annotated cells: the cells with the highest
phase per pixel relative to the typical cell in the same image were single, optically dense cells rather than superposed pairs.
Laterally touching cells and unresolved multi-cell clusters, which were most frequent at day 15, are discussed in Section 4.9. The
provided split comprises 256 training images and 66 validation images (six per time point). Because 32-bit phase maps were available
for 182 of the training images and for all validation images, the models were trained on 182 images, each with its three 90°
rotations (728 training views), and evaluated on the 66 validation images. These contain 2,049 annotated cells with a projected area
of 15–200 µm², of which 61 (3.0%) met the phase-wrapping criterion of Section 2.6 and were excluded from phase-derived results
([[stab:T1]]).
""", why="Replaces '1,090 images ... 1,024 training' (now 182 training images with 32-bit maps, 728 views, + 66 validation; D-05) and the last "
          "sentence about using the maps 'without further normalization'. Keeps the donors and IRB (R3-1), data source [26, 44] (R1-4) and "
          "the superposed-cell statement (R2-4). Adds 'training stability' as the fourth aspect (professor's point 2)."),
    E("Holograms were recorded with an off-axis transmission", "REPLACE", r"""
Holograms were recorded with an off-axis transmission digital holographic microscope based on a Mach-Zehnder interferometer
[26, 44], as shown in [[fig:setup]]. The light source was a 666 nm laser diode. The sample was imaged with a 40×/0.75 NA microscope
objective onto a CCD sensor (11.3 mm × 7.1 mm, 1920 × 1200 pixels of 5.86 µm × 5.86 µm, of which 1024 × 1024 pixels were used). The
field of view was approximately 140 µm, with an effective pixel size at the sample of 0.1441 µm. The object wave was reconstructed
numerically: the +1 diffraction order was isolated by spatial filtering in the Fourier domain to remove the zero order and twin
image, and the phase was recovered using a Fresnel-approximation numerical diffraction calculation. Background phase aberrations were
corrected using a numerical phase mask fitted to a fringe-free reference region of each hologram. The reconstructed phase maps used in
this study comprise 768 × 768 pixels and are confined to the interval [0, 2π] rad. Relative to the background, the peak phase of the
annotated cells had a median of 2.15 rad (95th percentile 4.16 rad); cells whose phase reached the 2π limit were identified and
excluded from phase-derived results as described in Section 2.6.
""", why="Removes 'Phase unwrapping was applied during reconstruction, and no annotated cell exceeded a phase of 2π' (61 annotated cells, 3.0%, "
          "reach the limit). Answers R2-3 with the measured phase range ([[stab:T1]]). Keeps the optical details (R1-4, R2-6). VERIFY with the "
          "data provider: 1024 × 1024 sensor pixels / 140 µm vs the 768 × 768 maps (07_REMAINING_WORK Q-A); the dry-mass check supports 0.1441 µm."),
    E("![](media/image3.png)", "KEEP"),
    E("Fig. 2", "KEEP"),
    E("Training Configuration", "KEEP", "## 3.2 Training Configuration"),
    E("All networks were optimized using the AdamW optimizer", "REPLACE", r"""
All networks were trained for 50 epochs with the AdamW optimizer and a cosine annealing learning-rate schedule (initial learning rate
5 × 10⁻⁴ for MobileSAM and EdgeSAM and 10⁻³ for MobileNet-UNet; batch size 8 for MobileSAM and 16 for the other models). Mixed
precision (FP16) was used in the forward pass to reduce memory consumption, while all loss terms were computed in 32-bit floating
point. The substantial class imbalance of longitudinal RBC morphology data was addressed by the class-weighted Multi-Class Dice Loss
of Section 2.5.1; mini-batches were drawn by uniform shuffling. Every LoRA configuration (three architectures, five ranks, and four
loss variants; Section 3.5) was trained with three random seeds (42, 43, and 44), giving 180 training runs; full fine-tuning was
trained once per architecture (seed 42). All configurations used the same optimization schedule and data. Because the validation
images also serve as the evaluation images, no checkpoint was selected on them: every reported result is from the model after the
final epoch.
""", why="Removes the WeightedRandomSampler (not used, DECISIONS D-04). Adds learning rates, batch sizes, seeds, run count and the "
          "final-epoch checkpoint (professor's point 4)."),
    E("Evaluation Metrics", "KEEP", "## 3.3 Evaluation Metrics"),
    E("The proposed framework was evaluated using three complementary", "REPLACE", r"""
The proposed framework was evaluated using four categories of metrics: segmentation performance, quantitative phase preservation,
computational efficiency, and training stability.
"""),
    E("Segmentation Performance", "KEEP", "### 3.3.1 Segmentation Performance and Training Stability"),
    E("Spatial agreement between predicted and ground-truth masks", "REPLACE", r"""
Spatial agreement between predicted and ground-truth masks was evaluated using the Dice coefficient and Intersection-over-Union (IoU)
at the native 768 × 768 resolution, after bilinear upsampling of the network output. For each morphology class, Dice and IoU were
computed from all validation pixels pooled and then macro-averaged over the four foreground classes. A morphology class that was not
predicted by the model received a score of zero rather than being excluded from the average, so that the dominant background class
cannot inflate the aggregate performance and failures in minority classes are reflected directly in the reported scores.
Instance-level separation in densely populated regions was evaluated with the Aggregated Jaccard Index (AJI) on the connected
components of the binary foreground. Training stability was assessed by repeating every configuration with three seeds: a run in
which any morphology class reached a Dice below 0.05 is reported as having lost that class. Results are given as mean ± standard
deviation over the three seeds, and every seed is reported in the Supplementary Information.
""", why="States pooling, native grid, and the lost-class definition (professor's point 4). Heading renamed to include training stability."),
    E("Quantitative Phase Preservation", "KEEP", "### 3.3.2 Quantitative Phase Preservation"),
    E("The physical fidelity of the predicted measurement boundaries", "REPLACE", r"""
The physical fidelity of the predicted measurement boundaries was evaluated using the integrated-phase error (IPE). This metric
quantifies the relative difference between the phase-integrated optical quantities obtained from the predicted and ground-truth
cellular regions on the background-referenced phase, consistent with the integrated-phase definition in Eq. (9). For a predicted
integrated phase $S_{pred}$ and the corresponding ground-truth value $S_{GT}$, IPE is defined as
""", why="IPE expanded at its first use (R1-11); background-referenced phase (R3-5)."),
    E("$IPE(\\%)", "KEEP"),
    E("IPE was evaluated both globally and for individual RBC", "REPLACE", r"""
IPE was evaluated both globally and for individual RBC morphology classes. For each image, $S_{pred}$ and $S_{GT}$ are accumulated by
summing the background-referenced phase over all annotated pixels in the relevant region (the full foreground for the global
formulation, or the corresponding class mask for class-specific IPE), and Eq. (11) is evaluated per image; the reported global IPE is
the mean of these per-image values, and class-specific IPE is the mean over the images that contain that class. Because Eq. (11)
operates on $\left| S_{pred} - S_{GT} \right|$, IPE is an absolute (unsigned) error, so over- and underestimation in different images
cannot cancel in the reported mean; within one image, however, the phase is summed over all cells before the absolute value is taken.
A morphology class with zero detections yields $S_{pred} = 0$ for that class, which is not excluded but is reflected directly in
Eq. (11) as an IPE of 100%. To assess the error of individual cells, a per-cell IPE was also computed by matching each ground-truth
cell (projected area 15–200 µm²) one-to-one with a predicted cell (IoU ≥ 0.5) and evaluating the relative phase-integral error for that
cell, using phase after subtraction of a least-squares background plane; unmatched and phase-wrapped cells were excluded from this
per-cell statistic, and the fraction of matched cells is reported. A lower IPE indicates that the predicted segmentation boundary
more accurately preserves the phase-integrated optical information associated with the cell. Boundary accuracy was additionally
evaluated using the boundary F1-score with a 2-pixel tolerance, pooled over all images, excluding boundary pixels adjacent to
unannotated regions. Together, the boundary F1-score and IPE measure geometric boundary accuracy and optical-mass preservation,
respectively.
""", why="Keeps the aggregation explanation (R3-4) and the per-cell IPE (R3-2); the old EdgeSAM r = 8 numbers (3.04%, 11.57%, 73.7%) move to "
          "Section 4.5 with the corrected values."),
    E("Computational Efficiency", "KEEP", "### 3.3.3 Computational Efficiency"),
    E("Computational efficiency was characterized using median", "KEEP"),
    E("Computational Benchmarking Protocol", "KEEP", "## 3.4 Computational Benchmarking Protocol"),
    E("Computational benchmarking was performed on an NVIDIA RTX A5000", "REPLACE", r"""
Computational benchmarking was performed on an NVIDIA RTX A5000 GPU to characterize the relative inference efficiency of the evaluated
lightweight architectures, independently of training. Each model was evaluated in strict 32-bit floating point (TF32 disabled) with
batch size 1 at its native input size, both in native PyTorch and through ONNX Runtime (ORT) with the CUDA Execution Provider. For each
runtime, 50 unrecorded warm-up iterations were followed by 500 synchronized forward passes on real validation phase maps kept on the
GPU; for ORT, three independent sessions were timed. The procedure was repeated with three seeds. We report the median (P50) and
99th-percentile (P99) latency, the throughput (FPS = 1000/P50), and the peak GPU memory, measured as the device memory in use after the
timed loop minus that before the model was loaded.
""", why="FP16 removed (not part of the corrected benchmark, manuscript/benchmarking_protocol.md); memory definition changed (the old 16.4 MB "
          "was not ORT memory)."),
    E("The purpose of this experiment was to establish the relative", "REPLACE", r"""
The purpose of this experiment was to establish the relative computational efficiency of the candidate architectures under a
controlled GPU environment rather than to demonstrate actual embedded deployment. Because LoRA is merged into the weights before
inference, latency and memory are independent of the rank and were measured once per architecture. MobileSAM is timed at its
1024-pixel input and the other models at 256 pixels, so latency differences between architectures combine architecture and input
size. Before any ORT result was reported, the exported models were compared with the PyTorch models on all validation images under
acceptance criteria fixed in advance (argmax agreement ≥ 99.9%, change in mean Dice ≤ 0.001, and mean per-image dry-mass difference
≤ 0.1%); all three models passed ([[stab:S9]]).
""", why="Removes 'MobileSAM was excluded from ONNX benchmarking ...' (MobileSAM is exported and passes the check)."),
    E("Ablation Studies", "KEEP", "## 3.5 Ablation Studies"),
    E("Three groups of ablation experiments were designed", "REPLACE", r"""
Three groups of experiments were performed for all three architectures, each configuration with three random seeds. First, a LoRA
rank sweep (*r* = 2, 4, 8, 16, and 32) characterized the relation among adaptation capacity, segmentation accuracy,
integrated-phase preservation, and training stability; the complete sweep is given in the Supplementary Information. From this
sweep, MobileSAM at *r* = 8 and EdgeSAM at *r* = 32 are presented as the preferred configurations, and all architectures are
additionally compared at the same rank (*r* = 8). Second, LoRA was compared with full fine-tuning of each network under the same
objective and schedule. Third, the Physics-Aware Phase Consistency Loss was ablated by progressively adding the PMC, BGA, and IPP terms
to the Dice loss at every rank (60 configurations); the full objective is the rank-sweep run of the same rank and seed, so every
comparison within a rank is paired by seed. The preferred ranks were chosen after inspection of the rank sweep on the evaluation
images, because no separate selection set was available (Section 5).
""", why="Professor's point 3 (preferred configurations + same-rank comparison + full sweep in the SI). Discloses how the ranks were chosen."),
    E("Statistical Analysis", "KEEP", "## 3.6 Statistical Analysis"),
    E("Statistical analyses were performed to evaluate storage-associated", "REPLACE", r"""
Statistical analyses were performed to evaluate storage-associated changes in RBC morphology and phase-derived quantitative
measurements. Because cells segmented from the same QPI share common acquisition conditions, individual cells were not treated as
statistically independent observations. The image was therefore used as the unit of analysis, with each image represented by the
median of its detected cells. Cell-level distributions were reported as descriptive summaries. Monotonic associations between
quantitative measurements and storage duration were evaluated using Spearman rank correlation. Differences between the early
discocyte-dominated storage period (days 0–8, 18 images) and the terminal storage period (days 37 and 47, 12 images) were evaluated
using a two-sided Mann-Whitney test. For phase-integrated quantities that co-vary with projected area, partial Spearman correlation
controlling for area was used to determine whether an independent association with storage duration remained. Agreement between
dry-mass estimates derived from predicted and manually annotated boundaries was assessed on per-image median dry mass by the median
absolute relative error (MdARE: the median, across images, of the absolute difference between predicted-mask and annotation-mask dry
mass divided by the annotation-mask dry mass), the Bland-Altman bias (mean predicted-minus-annotation difference) with 95% limits of
agreement [46], Pearson correlation, and Lin's concordance correlation coefficient (CCC) [45]. Because correlation coefficients depend
on the spread of the reference values, agreement was judged primarily by MdARE, bias, and the limits of agreement. Differences between
model configurations were computed per seed (paired by seed) and are reported with 95% t-intervals; with three seeds these intervals
are wide and are used descriptively rather than as significance tests.
""", why="Keeps MdARE (R3-3); adds early/late definitions, limits of agreement, and the seed statistics (professor's point 4)."),

    # ------------------------------------------------------------------------------------------------ 4. Results
    E("Results", "KEEP", "# 4. Results"),
    E("The proposed lightweight, physics-aware segmentation framework was evaluated", "REPLACE", r"""
The framework was evaluated on segmentation accuracy, training stability, integrated-phase and optical-mass preservation, and
computational efficiency. Unless stated otherwise, values are means ± standard deviation over three seeds of the final-epoch models,
computed on the 66 validation images at their native resolution. MobileSAM at *r* = 8 and EdgeSAM at *r* = 32 are presented as the
preferred configurations and MobileNet-UNet at *r* = 8 as a convolutional baseline, and all architectures are also compared at the
same rank (*r* = 8).
"""),
    E("Segmentation Accuracy Comparison", "REPLACE", "## 4.1 Segmentation Accuracy"),
    E("Three lightweight architectures---EdgeSAM, MobileSAM, and MobileNet-UNet---were first", "REPLACE", r"""
[[tab:seg]] lists the preferred configurations and the comparison at the same rank. MobileSAM (*r* = 8) reached the highest mean Dice
(0.810 ± 0.006) and boundary F1-score (0.802 ± 0.011) and the highest Dice for every morphology class, including echinocytes
(0.760 ± 0.006) and stomatocytes (0.708 ± 0.020) ([[fig:F3]]). EdgeSAM (*r* = 32) reached a mean Dice of 0.738 ± 0.006 and separated
individual cells better than MobileSAM (AJI 0.335 ± 0.020 vs 0.262 ± 0.017).
""", why="Every number of the old paragraph is outdated; EdgeSAM is no longer 'selected' (professor's point 2: objective comparison)."),
    E(("TABLE 1", "EdgeSAM          8      0.727"), "REPLACE", r"""
**TABLE 1** Segmentation performance of the preferred LoRA configurations (MobileSAM *r* = 8, EdgeSAM *r* = 32) and of all architectures at
the same rank (*r* = 8), full physics-aware objective. Mean ± SD over three seeds (42, 43, 44); final-epoch models; native 768 × 768
grid. "Runs with a lost class": seeds in which a class had Dice < 0.05 (sphero(2) = spherocyte, stomato(4) = stomatocyte). Per-class
Dice: [[fig:F3]] and [[stab:S1]].

{{TABLE:seg}}
""", why="Old Table 1 (single run, old class names) is replaced; generated from artifacts/tables/main/T2_preferred and T3_same_rank."),
    E("![image](media/image5.png)", "REPLACE", "{{FIG:F3}}"),
    E("Fig. 3 Per-class", "REPLACE", r"""
**Fig. 3** Per-class Dice of MobileSAM (*r* = 8), EdgeSAM (*r* = 32 and *r* = 8), and MobileNet-UNet (*r* = 8). Bars: mean over three seeds;
circles: individual seeds; crosses: seeds in which the class was lost (Dice < 0.05). Numbers in parentheses are mask values.
"""),
    E("The per-class Dice scores shown in Fig. 3 further reveal", "REPLACE", r"""
MobileNet-UNet reached a high boundary F1-score (0.814) and the highest AJI (0.551), but it did not predict a single spherocyte or
stomatocyte pixel in any seed, so its mean Dice was 0.412. Boundary F1 and AJI are computed on the binary foreground, which
MobileNet-UNet delineates well; they are blind to the class assigned to each cell. Aggregate boundary and instance metrics alone
therefore do not reveal the loss of whole morphology classes, and per-class results must be read together with them. For quantitative
holographic analysis, complete morphology coverage matters because the predicted region of each class defines where its phase is
integrated.
""", why="R1-8 (why MobileNet-UNet has high values despite missing classes). Class names corrected: it loses spherocytes (2) and stomatocytes (4), "
          "not echinocytes."),
    E("Among the evaluated architectures, EdgeSAM was the only model", "REPLACE", r"""
At the same rank (*r* = 8), MobileSAM exceeded EdgeSAM in boundary F1-score (paired difference +0.194, 95% interval +0.145 to +0.243).
The differences in mean Dice (+0.164, −0.133 to +0.461) and in global IPE (−2.49 percentage points, −7.15 to +2.17) were not
separable from seed variation, because one of the three EdgeSAM seeds lost the spherocyte class at *r* = 8 (Section 4.2).
""", why="Removes 'EdgeSAM was the only model ...' (not supported). Same-rank comparison (professor's point 3); values from Table 1 source (T3)."),
    E("The aggregate comparison across criteria is shown in Fig. S-2", "DELETE",
      why="The radar chart (old Fig. S-2) is removed (it hid lost classes); the EdgeSAM selection statements are no longer supported."),
    E("LoRA rank was selected for EdgeSAM from a sweep", "DELETE", why="Moved and rewritten in the new Section 4.2 (rank 8 is not the best EdgeSAM rank)."),
    E(None, "NEW", "## 4.2 LoRA Rank and Training Stability", why="New subsection for the rank sweep and training instability (professor's points 3 and 4)."),
    E(None, "NEW", r"""
Across the rank sweep ([[fig:F4]], [[stab:S1]]), MobileSAM was stable from *r* = 2 to *r* = 16 (mean Dice 0.808 to 0.813, seed SD at most
0.006). At *r* = 32, one seed lost the spherocyte and stomatocyte classes; its validation Dice dropped after the first epoch and did not
recover ([[fig:F5]]). EdgeSAM improved with rank in AJI and boundary F1-score and was stable at *r* = 4 and *r* = 32, but at *r* = 2, 8,
and 16 the same seed (44) lost the spherocyte class ([[tab:stab]]). EdgeSAM converged late, with a step increase of the validation Dice
between about epochs 15 and 25; in the unstable runs the Dice stayed near 0.5 and this step did not occur ([[fig:F5]]). Between EdgeSAM
*r* = 32 and *r* = 8, the mean Dice difference was +0.091 (−0.196 to +0.379) and the AJI difference +0.086 (+0.026 to +0.146)
([[stab:S3]]). MobileNet-UNet lost the spherocyte class in all 15 runs, which is a systematic failure rather than instability.
"""),
    E(None, "NEW", r"""
Increasing the rank therefore did not improve MobileSAM, whereas EdgeSAM benefited from the larger adaptation capacity; we present
MobileSAM at *r* = 8 and EdgeSAM at *r* = 32 for this reason, without claiming an optimal rank. LoRA is merged into the weights for
inference, so the rank changes the number of trainable parameters ([[stab:S8]]) but not the inference cost.
"""),
    E(None, "NEW", r"""
**TABLE 2** Training stability: number of seeds (of three) in which a morphology class was lost (Dice < 0.05), with the seed and the lost
classes, for every architecture and LoRA rank, with the full physics-aware objective and with the Dice loss only.

{{TABLE:stab}}
"""),
    E(None, "NEW", r"""
{{FIG:F4}}

**Fig. 4** LoRA rank sweep with the full physics-aware objective: mean Dice, AJI, boundary F1-score, and global IPE against rank. Lines:
mean over three seeds; circles: individual seeds; crosses: seeds in which a class was lost.
"""),
    E(None, "NEW", r"""
{{FIG:F5}}

**Fig. 5** Validation Dice during training for the three seeds of MobileSAM (*r* = 8, 32) and EdgeSAM (*r* = 8, 32). Dashed lines: runs that
lost a class. Training curves of all ranks: [[sfig:FS7]].
"""),
    E("Parameter-Efficient Adaptation versus Full Fine-Tuning", "REPLACE", "## 4.3 Parameter-Efficient Adaptation versus Full Fine-Tuning"),
    E("To evaluate the effectiveness of parameter-efficient adaptation, LoRA", "REPLACE", r"""
To evaluate the effectiveness of parameter-efficient adaptation, LoRA was compared with full fine-tuning of all three architectures
under the same objective, schedule, and data ([[tab:ft]], [[sfig:F7]]). Full fine-tuning, which updated 5.7 to 7.6 M parameters and was
trained with one seed, lost morphology classes in all three architectures: the spherocyte class for EdgeSAM (mean Dice 0.559, global
IPE 8.31%), the spherocyte, echinocyte, and stomatocyte classes for MobileSAM (mean Dice 0.217), and, as with LoRA, the spherocyte and
stomatocyte classes for MobileNet-UNet. With the same seed, the LoRA-adapted MobileSAM (*r* = 8) and EdgeSAM (*r* = 32) detected all
four classes.
""", why="Old numbers and class names (echinocyte collapse) replaced."),
    E("Fig. 4 further illustrates the advantage of LoRA over full fine-tuning", "REPLACE", r"""
These results indicate that increasing the number of trainable parameters does not necessarily improve adaptation to QPI phase images.
Constraining adaptation to low-rank updates helps to retain the pretrained representations while providing sufficient flexibility for
the QPI domain, which matters most for the minority morphology classes of this small dataset. Because full fine-tuning was trained with
a single seed, this comparison is indicative.
""", why="Removes 'LoRA-adapted EdgeSAM provided the most favorable balance'."),
    E(("TABLE 2** Comparison of LoRA", "Full Fine-Tune   7.62M"), "REPLACE", r"""
**TABLE 3** LoRA versus full fine-tuning (full physics-aware objective). LoRA rows: preferred rank, mean ± SD over three seeds, and the
seed-42 run alone; full fine-tuning: single seed-42 run. Per-class Dice: [[stab:S7]] and [[sfig:F7]].

{{TABLE:ft}}
"""),
    E(("![image](media/image6.png)", "Fig. 4** Per-class Dice scores comparing LoRA"), "DELETE",
      why="Old Fig. 4 moves to the SI ([[sfig:F7]]); Table 3 carries the numbers (R1-5: keep the main text short)."),
    E("Physics-Aware Loss Ablation", "REPLACE", "## 4.4 Physics-Aware Loss Ablation"),
    E("To evaluate the contribution of each physics-aware constraint", "REPLACE", r"""
[[tab:abl]] and [[fig:F6]] compare the four objectives for the preferred configurations and MobileNet-UNet; the complete ablation at every
rank is given in [[stab:S4]], [[stab:S5]], and [[sfig:FS2]]. Adding the physics-aware terms lowered the mean global IPE relative to
Dice-only training in all 15 architecture-rank combinations; with three seeds, the 95% interval excluded zero in three of them. The
reduction was large where Dice-only training was unstable: for EdgeSAM at *r* = 32, the IPE fell from 11.20 ± 6.46% to 3.02 ± 0.84%, and
for MobileNet-UNet at *r* = 8 from 15.48 ± 12.73% to 2.60 ± 0.07%. For MobileSAM at *r* = 8, whose Dice-only IPE was already low, the
reduction was small (1.53 ± 0.27% to 1.32 ± 0.09%; −0.21 percentage points, −0.75 to +0.33).
""", why="Removes the aliasing paragraph's premise; all ablation numbers are new (three seeds)."),
    E(("TABLE 3** Component-wise ablation", "Full $\\mathcal{L}$ ($\\mathbf{+ \\,}\\mathcal{L}_{\\text{IPP}}$)   0.607"), "REPLACE", r"""
**TABLE 4** Ablation of the Physics-Aware Phase Consistency Loss. Each objective adds one term to the class-weighted Dice loss; the full
objective is the rank-sweep run of the same rank and seeds. Mean ± SD over three seeds; lower part: seed-paired differences with 95%
t-intervals.

{{TABLE:abl}}
""", why="Removes 'The ablation runs were trained independently' (the full-objective row is the rank-sweep run, DECISIONS D-09)."),
    E(("TABLE 4** Per-class IPE and Dice", "Full $\\mathcal{L}$                                                         0.900"), "DELETE",
      why="Old Table 4 (single-run per-class EdgeSAM ablation, including Dice 0.018 and IPE 1073.58%) is not reproduced; class detail is in "
          "[[stab:S4]] and [[stab:S7]]."),
    E("For EdgeSAM, the Dice-only baseline achieved a mean Dice of 0.757", "REPLACE", r"""
The physics terms also reduced training failures: over the rank sweep, the number of runs that lost a class decreased from 3 of 15 to
1 of 15 for MobileSAM and from 6 of 15 to 3 of 15 for EdgeSAM ([[tab:stab]]). The contributions of the individual terms could not be
separated reliably with three seeds. Adding IPP to Dice + PMC + BGA lowered the mean IPE in 13 of 15 combinations, most clearly for
EdgeSAM at *r* = 32 (−0.63 percentage points, −0.75 to −0.51), whereas PMC and BGA alone had no consistent effect (8 and 11 of 15;
[[stab:S5]]). Mean Dice was not systematically changed by the physics terms.
""", why="Removes the PMC/BGA claims (PMC alone degrades, BGA restores 0.018 to 0.643), which the three-seed results do not reproduce."),
    E("The complete objective incorporating IPP achieved the lowest global IPE", "REPLACE", r"""
Because the IPP term weights each pixel by its phase, it could favor high-phase regions over low-phase cell margins. Misses did
concentrate in the low-phase part of the cells: within the annotated cells, the false-negative rate in the lowest phase quartile was
23 to 28%, compared with below 6% in the other quartiles ([[stab:S12]]). Adding IPP did not consistently increase this deficit: the
lowest-quartile false-negative rate changed by −4.0 (−5.3 to −2.8) percentage points for EdgeSAM at *r* = 32 and by +2.4 (−1.2 to +5.9)
for MobileSAM at *r* = 8.
""", why="R2-2 (phase-dependent bias), updated. The old robustness part of this paragraph moves to Section 4.5; the MobileSAM "
          "non-convergence statements are removed (MobileSAM now converges in all seeds)."),
    E("![](media/image7.png)", "REPLACE", "{{FIG:F6}}"),
    E("Fig. 5** Component-wise loss ablation", "REPLACE", r"""
**Fig. 6** Ablation of the physics-aware loss: global IPE (top) and mean Dice (bottom) for the four objectives. Bars: mean over three seeds;
circles: seeds; crosses: seeds in which a class was lost. All ranks: [[sfig:FS2]].
""", why="The old figure used two y-axes; the new one shows IPE and Dice in separate rows."),
    E("Integrated-Phase and Optical-Mass Preservation Analysis", "REPLACE", "## 4.5 Integrated-Phase and Optical-Mass Preservation"),
    E("The preservation of phase-integrated optical information was evaluated using the IPE", "REPLACE", r"""
[[tab:mass]] summarizes the measurement-related errors. The global IPE was 1.32 ± 0.09% for MobileSAM (*r* = 8) and 3.02 ± 0.84% for
EdgeSAM (*r* = 32). For annotated cells that were matched one-to-one to a predicted cell (IoU ≥ 0.5; 66.6% and 66.3% of the cells), the
median per-cell IPE was 1.11 ± 0.09% and 1.88 ± 0.24%, and 90% of the matched cells had errors below 8.3% and 16.2%, respectively
([[sfig:FS3]]). Unmatched cells were mostly merged or missed cells: both SAM-based models produced about two thirds as many cell
components as were annotated (0.667 and 0.696), consistent with their AJI. Class-specific IPE ([[stab:S7]]) is dominated by confusion
between morphologies and is therefore much larger than the global IPE.
""", why="R3-2 (per-cell IPE vs global IPE), R3-4 (why class IPE is larger). Old numbers (3.63%, 2.57%, 2.60%) and the EdgeSAM 'most balanced' "
          "statement removed."),
    E(("TABLE 5** Integrated-phase preservation analysis", "EdgeSAM          8                     3.63"), "REPLACE", r"""
**TABLE 5** Integrated-phase and optical-mass preservation (phase-wrapped cells excluded). Per-cell IPE: matched cells (IoU ≥ 0.5).
Dry mass: per-image medians, predicted vs annotation masks; LoA: 95% limits of agreement. Mean ± SD over three seeds.

{{TABLE:mass}}
"""),
    E(None, "NEW", r"""
Dry mass was computed without calibration. The median day-0 dry mass was 27.83 pg from the annotation masks, 0.93 of a typical mean
corpuscular hemoglobin of 30 pg, and 28.21 pg and 27.88 pg from the MobileSAM and EdgeSAM masks. Per image, dry mass from the predicted
masks agreed with the annotation-derived dry mass within a median absolute relative error of 2.87 ± 0.21% (MobileSAM) and
3.93 ± 0.52% (EdgeSAM), with biases of +0.66 and +0.46 pg and limits of agreement of about −5 to +6 pg ([[fig:F9]]). The Pearson
correlation was low (0.24 and 0.38) because the annotation-derived per-image dry mass varied little between images
(28.66 ± 1.55 pg); for such a narrow range, the error-based measures describe the agreement better than correlation.
""", why="Replaces 'r = 0.966, CCC 0.96, MdARE 3.35%' (old pipeline); answers R3-3 (absolute agreement measures, not only Pearson r)."),
    E(None, "NEW", r"""
{{FIG:F9}}

**Fig. 7** Dry-mass agreement between predicted and annotation masks (per-image median dry mass, three seeds pooled, phase-wrapped cells
excluded). Top: predicted vs annotation-derived values with the identity line; bottom: Bland-Altman plots with bias (solid) and 95%
limits of agreement (dashed).
"""),
    E(None, "NEW", r"""
The global IPE depended on the background referencing of Section 2.2: without it, the IPE of MobileSAM rose from 1.32% to 2.06%
([[stab:S14]]), whereas estimating the background from the predicted instead of the annotated mask changed it by less than 0.01
percentage points. To test robustness to background phase variations, synthetic tilt, defocus-like, and astigmatism-like fields of up
to 0.5 rad were added to the validation phase maps: the mean Dice changed by at most 0.003 and the background-corrected IPE by at most
0.1 percentage points ([[stab:S10]], [[sfig:FS4]]). Constant offsets are removed by the per-image standardization of the network input
and are therefore not a test of robustness.
""", why="R2-1 (background correction and robustness to low-order distortions), updated; was part of the old Section 4.3."),
    E("Computational Efficiency Analysis", "REPLACE", "## 4.6 Computational Efficiency"),
    E("Computational efficiency was evaluated on an NVIDIA RTX A5000 using native PyTorch", "REPLACE", r"""
Inference cost is independent of the LoRA rank after merging, so one measurement per architecture is reported ([[tab:bench]]). In
PyTorch FP32, EdgeSAM needed 10.64 ± 1.45 ms per image (81 to 103 FPS) and 42 MB of GPU memory, MobileSAM 20.35 ± 0.15 ms (49 to 50 FPS)
and 358 MB, and MobileNet-UNet 5.66 ± 0.10 ms and 52 MB. ORT reduced the latency of EdgeSAM to 4.11 ± 0.43 ms (229 to 277 FPS) and of
MobileNet-UNet to 2.53 ± 0.19 ms, but increased that of MobileSAM to 25.43 ± 0.21 ms, and it used more memory for all models (132 to
1,178 MB). MobileSAM processes 1024-pixel inputs and the other models 256-pixel inputs, so these differences combine architecture and
resolution. All ORT models reproduced the PyTorch predictions ([[stab:S9]]). These measurements characterize a desktop GPU rather than
an embedded device ([[sfig:F8]] shows accuracy against latency).
""", why="Removes ONNX FP16, 228.3 FPS, 16.4 MB and 'MobileSAM could not be accelerated with ORT'."),
    E(("TABLE 6** Computational benchmarking", "PyTorch   FP16        17.33"), "REPLACE", r"""
**TABLE 6** Computational efficiency on an NVIDIA RTX A5000 (batch 1, strict FP32). P50 and P99: mean ± SD over three seeds; FPS: range of
the per-seed values; peak VRAM: largest value over seeds and sessions. Inference cost does not depend on the LoRA rank.

{{TABLE:bench}}
"""),
    E("Visualization of Segmentation and Morphology Classification Results", "REPLACE", "## 4.7 Visualization of Segmentation Results"),
    E("Qualitative segmentation results further support the quantitative evaluation", "REPLACE", r"""
[[fig:FQ1]] shows representative segmentation results for each morphology class. Consistent with the per-class results, MobileNet-UNet
assigns no pixel to the spherocyte or stomatocyte class, so cells of these morphologies are labeled with other classes, whereas the
two SAM-based models label all four morphologies. Neighboring cells that touch are sometimes merged into one predicted region by both
SAM-based models, more often by MobileSAM, in agreement with their AJI and with the number of predicted cell components
(Section 4.5).
""", why="CHECK AGAINST THE NEW FIGURE: generate Fig. 8 on the server first (make_qualitative_figures.py) and adjust the last sentence to what "
          "the figure shows. The first two sentences are certain from the predictions (MobileNet-UNet predicts 0 pixels of mask values 2 and 4)."),
    E("![](media/image8.png)", "REPLACE", "{{FIG:FQ1}}"),
    E("Fig. 6** Qualitative segmentation", "REPLACE", r"""
**Fig. 8** Qualitative segmentation results of MobileSAM (*r* = 8), EdgeSAM (*r* = 32), and MobileNet-UNet (*r* = 8), seed 42, for the four
morphology classes. For each class, the 256 × 256 window with the most annotated pixels of that class is shown with the annotation.
"""),
    E("Storage-Induced Morphology Degradation", "REPLACE", "## 4.8 Storage-Induced Morphology Changes"),
    E("To evaluate the applicability of the proposed framework to quantitative biological analysis", "REPLACE", r"""
To evaluate the applicability of the proposed framework to quantitative biological analysis, storage-induced changes in RBC morphology
and phase-derived biophysical properties were examined over the 47-day storage period, using the annotation masks as the reference and
the masks predicted by MobileSAM (*r* = 8) and EdgeSAM (*r* = 32). Each storage time point consisted of six independent validation
images; therefore, the analysis represents cross-sectional population changes rather than longitudinal tracking of the same cells.
""", why="The analysis now uses the annotation as reference and both preferred models (three seeds), not a single EdgeSAM r = 8 run."),
    E("The predicted RBC morphology distributions showed systematic changes", "REPLACE", r"""
The annotation shows the expected storage lesion ([[fig:F11]], [[tab:day]]): discocytes made up 93% of the cells at day 0 and 11% at day
47, while spherocytes increased from 0% to 59% and echinocytes from 2% to 30%; stomatocytes peaked at 15% at day 19. Both preferred
models reproduced this composition, with a mean absolute difference per day of 1.7 percentage points for MobileSAM and 2.8 for EdgeSAM.
""", why="Old composition numbers (stomatocytes 10% to 26%, EdgeSAM only) replaced; class names follow mask values."),
    E("![](media/image9.png)", "REPLACE", "{{FIG:F11}}"),
    E("Fig. 7** Predicted RBC morphology composition", "REPLACE", r"""
**Fig. 9** RBC morphology composition over the 47-day storage period: annotation and the masks predicted by MobileSAM (*r* = 8) and EdgeSAM
(*r* = 32), mean over three seeds. Numbers in parentheses are mask values.
"""),
    E("Representative quantitative phase images in Fig. S-5", "REPLACE", r"""
Representative quantitative phase images in [[sfig:FQS2]] further illustrate these storage-associated morphological changes, showing a
transition from predominantly biconcave discocytes at the initial storage time point to increased occurrence of altered RBC
morphologies at later storage stages.
""", why="Figure reference renumbered; the last sentence of the old paragraph ('demonstrates that the proposed framework can quantify ...') is "
          "covered by the composition comparison above."),
    E("Quantitative Phase-Derived Cellular Analysis Across Storage", "REPLACE", "## 4.9 Quantitative Phase-Derived Cellular Analysis Across Storage"),
    E("Quantitative cellular measurements were extracted from the EdgeSAM-predicted masks", "REPLACE", r"""
With the image as the unit of analysis ([[tab:stat]], [[fig:F10]]), the projected area of the annotated cells decreased with storage
(Spearman ρ = −0.53, *p* < 0.001; −43.9% from days 0–8 to days 37 and 47), whereas their dry mass did not change (ρ = −0.11, *p* = 0.36;
−0.8%; Mann-Whitney *p* = 0.21), and circularity showed no trend. The cells therefore lost projected area without losing dry mass,
consistent with the spherocytic transformation, in which membrane is lost while the hemoglobin content is retained [24, 25, 44]. The
per-cell mean area was highest at day 15 ([[tab:day]]) because of large clustered components, whereas the per-image median area did not
peak at that time point ([[fig:F10]]).
""", why="R1-6 and R1-9: the area peak at day 15 and the dry-mass fluctuation of the old text came from a single run of the superseded pipeline; "
          "the corrected annotation shows a monotonic area decrease after day 27 and constant dry mass. Removes ρ = −0.67, ρ = −0.40, partial "
          "ρ = −0.15, 63.1/83.6/43.4 µm²."),
    E("To evaluate the reliability of the predicted segmentation masks as quantitative measurement domains", "REPLACE", r"""
The predicted masks reproduced the decrease in area (MobileSAM −40.6%, EdgeSAM −44.3%) but showed a small decrease in dry mass that the
annotation does not show (MobileSAM −2.6 ± 0.4%, not significant; EdgeSAM −6.1 ± 0.3%, *p* between 0.008 and 0.05 across seeds).
[[fig:F10]] shows that this bias arises at days 5 to 19, where the predicted per-image dry mass exceeds the annotation-derived value, and
is largest at day 15, the time point with the most touching cells. A bias of this kind occurred for all 15 rank-sweep configurations
([[stab:S13]]). Population-level dry-mass trends from predicted masks should therefore be read against this segmentation bias.
""", why="The old agreement sentence (r = 0.966 ...) moves, corrected, to Section 4.5. CHECK: 'the time point with the most touching cells' "
          "comes from Section 3.1 and the highest annotated cell-mean area at day 15 supports it."),
    E(("TABLE 7** Per-cell quantitative morphology", "47    250"), "REPLACE", r"""
**TABLE 7** Cell morphology, dry mass, and composition over storage (phase-wrapped cells excluded). Annotation: mean ± SD over cells. Models:
per-day cell mean averaged over three seeds ± SD over seeds. Composition order: discocyte (1) / spherocyte (2) / echinocyte (3) /
stomatocyte (4).

{{TABLE:day}}

**TABLE 8** Storage trends with the image as the unit of analysis (per-image medians, 66 images). Early: days 0–8; late: days 37 and 47.
Model rows: mean ± SD over seeds of ρ and of the late-vs-early change; *p*-values as the range over seeds.

{{TABLE:stat}}
""", why="Old Table 7 (EdgeSAM r = 8 only, 1,974 cells, old class names) replaced."),
    E("![](media/image10.png)", "REPLACE", "{{FIG:F10}}"),
    E("Fig. 8** Phase-derived dry-mass analysis", "REPLACE", r"""
**Fig. 10** Per-image median projected area (left) and dry mass (right) over storage, as the mean of the six images per day: annotation and
predicted masks (mean over three seeds; band: range over seeds).
"""),

    # ------------------------------------------------------------------------------------------------ 5. Discussion
    E("Discussion", "KEEP", "# 5. Discussion"),
    E("The results demonstrate that physics-aware lightweight segmentation can improve", "REPLACE", r"""
Treating the predicted boundary as the measurement domain of the integrated phase changes what a segmentation model is optimized for.
With the Physics-Aware Phase Consistency Loss, the global integrated-phase error was lower than with Dice-only training for every
architecture and rank examined, and it was 1.3% for MobileSAM and 3.0% for EdgeSAM at the preferred configurations. The gain was
largest where geometric training alone was unstable (EdgeSAM and MobileNet-UNet) and small where it was already accurate (MobileSAM),
which indicates that the physics terms mainly protect the measurement domain against failure modes that overlap metrics do not
penalize strongly. These findings show that geometric overlap alone does not fully characterize the suitability of a segmentation mask
for quantitative QPI measurements.
""", why="Old numbers (5.22 to 3.04%, 0.938 to 0.953, 15.52 to 3.01%) replaced."),
    E("The three components of the proposed physics-aware objective provide", "REPLACE", r"""
The ablation supports the full objective as a whole rather than a ranking of its terms. With three seeds, the full objective gave the
lowest and least variable IPE and fewer runs that lost a class; IPP had its clearest effect for EdgeSAM at *r* = 32, and the separate
effects of PMC and BGA were not consistent across ranks. Because all three terms act on the class-agnostic foreground, they cannot by
themselves prevent confusion between morphologies; this task is carried by the class-weighted Dice term.
""", why="Removes the 'complementary constraints' and 'PMC alone degraded ..., BGA restored' claims and the MobileSAM non-convergence sentence."),
    E("Preserving integrated phase is particularly important because segmentation errors", "REPLACE", r"""
Preserving the integrated phase is particularly important because segmentation errors propagate directly into downstream
phase-derived measurements. Without any calibration, the dry mass from the predicted masks agreed with the annotation-derived dry mass
within a median of 2.9 to 3.9% per image, and the day-0 value (27.8 pg from the annotation) was consistent with a typical mean
corpuscular hemoglobin. The low correlation between predicted and annotation-derived per-image dry mass is a consequence of the narrow
range of the reference, not of large errors; Bland-Altman limits of about ±6 pg describe the remaining per-image error. DHM and QPI are
advantageous in this context because they enable label-free quantitative cellular analysis without the staining required by
conventional imaging [47, 48].
""", why="Removes 'r = 0.966, CCC = 0.96 ... direct evidence'. Keeps refs [47, 48] so they stay cited."),
    E("The comparison among lightweight architectures and adaptation strategies", "REPLACE", r"""
MobileSAM and EdgeSAM offer different trade-offs. MobileSAM was the more accurate model (higher Dice, boundary F1-score, and class Dice,
lower IPE) and was stable at ranks 2 to 16, but it is twice as slow in PyTorch, needs about eight times more GPU memory, and separated
touching cells less well. EdgeSAM was faster and lighter, separated cells better, and needed a larger rank (*r* = 32) to train reliably;
at smaller ranks, one of three seeds failed to learn the spherocyte class. Full fine-tuning lost minority classes in all three
architectures, which suggests that restricting adaptation to low-rank updates regularizes training on relatively small and imbalanced
QPI datasets. Reporting every seed was essential: a single seed would have hidden the failure of EdgeSAM at *r* = 8 (seed 44) or of
MobileSAM at *r* = 32 (seed 43).
""", why="Objective comparison (professor's point 2); removes 'r = 8 provided the best overall balance' and the EdgeSAM selection."),
    E("Computational benchmarking also showed that the selected EdgeSAM configuration", "REPLACE", r"""
On a desktop GPU, both preferred models run at video rate in PyTorch (about 49 FPS for MobileSAM and 95 FPS for EdgeSAM on average), and
EdgeSAM exceeds 200 FPS with ORT. These results characterize performance on a desktop-class GPU rather than actual edge-device
deployment; validation on low-power embedded platforms will be required to establish real-world edge performance.
""", why="Removes ONNX FP16, 4.33 ms, 228.3 FPS, 16.4 MB, 'limited ONNX compatibility'. CHECK: '95 FPS' is the mean of the EdgeSAM PyTorch "
          "per-seed FPS (95.1, benchmark file); 49 FPS for MobileSAM."),
    E("The biological analysis further demonstrates the utility", "REPLACE", r"""
In the annotated cells, projected area decreased by 44% over 47 days of storage while dry mass remained constant, the signature of the
spherocytic transformation in which membrane rather than hemoglobin is lost [24, 25]. This contrast is the kind of information that
QPI adds to morphology, and it depends on the preservation of the integrated phase: the predicted masks reproduced the area change and
the composition but introduced a dry-mass bias of 2.6 to 6.1% between early and late storage that arose mainly in images with many
touching cells. Instance separation, rather than boundary placement, is therefore the main limitation of the present models for
population-level optical-mass analysis.
""", why="Removes the dry-mass decline and 'phase scale calibrated to the MCH reference' (R1-9 consistency)."),
    E("Several limitations remain.", "REPLACE", r"""
Several limitations remain. The validation images were also the evaluation images: no checkpoint was selected on them, but the
preferred ranks were chosen after inspecting the rank sweep, and an independent held-out test cohort was not available. Three seeds give
wide confidence intervals, and full fine-tuning was run once. Float32 phase maps were available for 182 of the 256 training images, and
the annotations were transferred from 8-bit exports by registration. The framework was evaluated only on RBC QPI data, and its
generalizability to other cell types, imaging conditions, and densely crowded cellular environments remains to be established. Because
the reconstructed phase is limited to an unambiguous range, cells reaching the 2π limit (3.0% of the annotated cells; 12% of
echinocytes and 10% of stomatocytes) were excluded from phase-derived results, and specimens with substantially greater optical
thickness than RBCs would require phase unwrapping before this framework is applied. In addition, LoRA adaptation was performed
offline, and adaptation to changes in imaging conditions or optical aberrations was not investigated. The dataset used in this study
provides only the reconstructed phase channel; the corresponding amplitude images were not available, as the original holograms were
not retained by the data provider, so incorporating amplitude information was not evaluated in this work. Amplitude images could
potentially provide complementary information in the low-phase and boundary regions where segmentation errors were found to
concentrate (Section 4.4), and represent a promising direction for future work. Future work will therefore focus on improving instance
separation within the physics-aware objective, validation using independent QPI datasets and additional cell types, and evaluation on
embedded computing platforms.
""", why="Keeps the unwrapping (R2-3) and amplitude (R2-5) parts. Removes 'full objective reduced instance separation for EdgeSAM' (not true "
          "at r = 32: AJI 0.335 vs 0.329) and the MobileSAM non-convergence sentence; adds the rank-choice and seed disclosures."),
    E("Overall, the present results establish optical-mass preservation", "REPLACE", r"""
Overall, the present results support optical-mass preservation as a measurement-oriented principle for physics-aware QPI segmentation.
In this framework, optical-mass preservation is achieved by constraining the integrated quantitative phase within predicted cellular
regions rather than by interpreting QPI as a direct measurement of geometric volume. By combining this physical constraint with
lightweight architecture design and parameter-efficient adaptation, the proposed framework enables computationally efficient
holographic cell segmentation in which predicted boundaries function as physically meaningful measurement domains for subsequent
quantitative cellular analysis.
""", why="Only 'establish' -> 'support'."),

    # ------------------------------------------------------------------------------------------------ 6. Conclusion
    E("Conclusion", "KEEP", "# 6. Conclusion"),
    E("This study presented a physics-aware lightweight segmentation framework", "REPLACE", r"""
This study presented a physics-aware lightweight segmentation framework for quantitative holographic cell analysis, in which the
predicted cell boundary is the measurement domain of the integrated phase. On background-referenced 32-bit phase maps of red blood
cells, the Physics-Aware Phase Consistency Loss lowered the integrated-phase error relative to Dice-only training for every
architecture and LoRA rank examined and reduced training failures.
""", why="Removes 'clear gains for EdgeSAM and MobileNet-UNet; for MobileSAM ... did not converge'."),
    E("Among the evaluated architectures, LoRA-adapted EdgeSAM provided the most balanced", "REPLACE", r"""
With LoRA, MobileSAM (*r* = 8) gave the most accurate segmentation and the lowest integrated-phase error (1.3%), and EdgeSAM (*r* = 32)
the best combination of instance separation and computational cost. Dry mass estimated from the predicted masks without calibration
agreed with annotation-derived values within a median of 3 to 4% per image, and the annotation showed that stored cells lose projected
area but not dry mass.
""", why="Removes 'EdgeSAM provided the most balanced performance' and 'r = 0.966'."),
    E("Overall, the proposed approach demonstrates that optical-mass preservation", "REPLACE", r"""
Overall, the proposed approach shows that optical-mass preservation can be incorporated into lightweight QPI segmentation by explicitly
preserving the integrated quantitative phase within predicted cellular regions. Improving the separation of touching cells within the
physics-aware objective, and validation on independent QPI datasets, additional cell types, and embedded computing platforms, are the
next steps.
"""),

    # ------------------------------------------------------------------------------------------------ back matter
    E(("Author contributions", "Usama Ikram implemented"), "KEEP", why="Check that 'benchmarking experiments' is still accurate."),
    E(("Funding", "This work was supported by the National Research Foundation"), "KEEP"),
    E("Data availability", "KEEP"),
    E("The data generated and/or analyzed during the current study", "KEEP",
      why="Optional: add 'Code and per-run results are available at <repository URL or DOI>' if the repository will be made public."),
    E(("Declarations", "Not applicable."), "KEEP"),
    E(("References", "[48] M. Bardyn"), "KEEP",
      why="References unchanged. After editing, check that every reference is still cited (e.g. [31], [41], [42])."),
]

# ------------------------------------------------------------------------------------------------ numbering of tables and figures
MAIN_TABLES = {"seg": 1, "stab": 2, "ft": 3, "abl": 4, "mass": 5, "bench": 6, "day": 7, "stat": 8}
MAIN_FIGS = {"frame": 1, "setup": 2, "F3": 3, "F4": 4, "F5": 5, "F6": 6, "F9": 7, "FQ1": 8, "F11": 9, "F10": 10}
SI_TABLES = {"T1": 1, "S1": 2, "S2": 3, "S3": 4, "S8": 5, "S4": 6, "S5": 7, "S7": 8, "S12": 9, "S14": 10, "S10": 11, "S11": 12,
             "S13": 13, "S9": 14}
SI_FIGS = {"FS1": 1, "FS7": 2, "FS5": 3, "FS2": 4, "F7": 5, "FS3": 6, "FS6": 7, "FQS1": 8, "FS4": 9, "FS8": 10, "F8": 11, "FQS2": 12}
