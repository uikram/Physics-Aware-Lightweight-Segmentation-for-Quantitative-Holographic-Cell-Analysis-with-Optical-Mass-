# 06. Methods exactly as implemented (with draft text for Sections 2 and 3)

Every statement below was read from the code (file and decision cited) or from the outputs. The draft paragraphs are written to be
pasted into the manuscript and then edited for style. Items marked **[verify]** could not be confirmed from the repository alone.

## 1. Data and preprocessing (replaces Sec. 2.2 and the data part of Sec. 3.1)

| Step | Implementation | Code / decision |
|---|---|---|
| Input data | float32 quantitative phase maps in radians, 768 x 768 px, effective pixel pitch 0.1441 um (field of view about 111 um) **[verify with the data provider: Sec. 3.1 says 1024 x 1024 sensor pixels and about 140 um; the dry-mass check (0.93 x MCH) and the day-0 cell area (median 46.8 um2) are consistent with 0.1441 um on the 768 grid]** | `configs/base.yaml` physics, `qpi/data/prepare.py` |
| Annotations | five-class masks drawn on the earlier 8-bit exports; carried onto the float grid by an ECC affine registration of the float map to the 8-bit image (measured transforms: scale and translation only), nearest-neighbour warping | `qpi/data/registration.py`, D-01, D-16 |
| Registration gate | ECC ≥ 0.98 and Pearson ≥ 0.90; two borderline images accepted by a documented geometric check (peak shift 0 px, Spearman ≥ 0.95 and ≥ Pearson + 0.02) | `prepare.py: registration_gate`, D-16 |
| Unannotated pixels | float-grid pixels outside the 8-bit field of view get label 255 and are excluded from every loss term and metric | D-02, `losses.py`, `metrics.py` |
| Background referencing | per image, the median phase of annotated background pixels farther than 5 px from any cell is subtracted once (validation offsets 0.66 to 2.50 rad); units stay radians | `prepare.py`, D-03 |
| Network input | bilinear resize (anti-aliased when downsampling) to the model input size (EdgeSAM, MobileNet-UNet 256 px; MobileSAM 1024 px), then per-image z-score | `qpi/data/dataset.py` |
| Physics-loss input | the background-referenced phase in radians resized to the same model resolution; never the z-scored input | `dataset.py: phase_raw`, D-11 |
| Masks at model size | `nearest-exact` resize (pixel-centre sampling, no half-pixel shift) | `dataset.py` |
| Augmentation | the three lossless 90 degree rotations of each training image (k = 1, 2, 3; identical to the provided rotated files); no flips, translations or intensity changes | `dataset.py`, D-06 |
| Phase wrapping | a cell is "wrapped" if, within 2 px, the phase as provided is within 0.3 rad of 2π or two neighbouring pixels differ by more than π; 61 of 2,049 annotated cells (3.0%); wrapped cells are kept for segmentation and excluded from dry mass and per-cell phase results; rule fixed before the corrected results existed | `qpi/analysis/wrapping.py`, D-21 |
| Split | provided train / validation split; leakage test on stems and rotation-invariant content hashes | `scripts/verify_dataset.py`, D-07 |

**Draft text (Sec. 2.2, QPI preprocessing).**
The input data are single-channel quantitative phase maps reconstructed as 32-bit floating-point arrays in radians (768 × 768 pixels,
effective object-plane pixel pitch 0.1441 µm). The expert annotations had been drawn on 8-bit grey-level exports of these maps; they
were transferred to the floating-point grid by an affine enhanced-correlation-coefficient registration of each phase map to its
8-bit export (the estimated transforms consisted of scale and translation only) with nearest-neighbour resampling of the label map.
Pixels of the phase map that lie outside the field of view of the 8-bit export carry no annotation; they were excluded from every
loss term and every metric. The reconstructed maps do not share a common zero level (the background of the validation maps lay
between 0.66 and 2.50 rad), and an arbitrary constant c adds c × A to the phase integrated over a region of area A. Every map was
therefore referenced to its own background before training and evaluation: the median phase of annotated background pixels more
than 5 pixels from any cell was subtracted, so that the background is 0 rad and the integrated phase of a cell measures the cell
only. For the network input, the referenced map was resized to the model input size (256 × 256 pixels for MobileNet-UNet and
EdgeSAM, 1024 × 1024 pixels for MobileSAM) and standardised per image (zero mean, unit variance). The physics-aware loss terms
received the referenced phase in radians at the same resolution, never the standardised input. The training set was augmented only
with the three lossless 90° rotations of each image; no flipping, translation or intensity transformation was applied, because such
operations either duplicate the rotations or change the quantitative phase values.

## 2. Models and LoRA (replaces Sec. 2.3 and 2.4)

| Item | Implementation | Code |
|---|---|---|
| MobileNet-UNet | ImageNet MobileNetV2 encoder, U-Net decoder with skip connections, 256 px input | `qpi/models/mobilenet_unet.py` |
| MobileSAM | pretrained TinyViT image encoder of MobileSAM (1024 px input); task decoder trained from scratch: a learned prompt embedding added to five class tokens, one transformer decoder layer (8 heads) attending to the image embedding, class logits as the dot product of each token with the 4x bilinearly upsampled embedding, resized to the input | `qpi/models/mobile_sam.py` |
| EdgeSAM | pretrained RepViT image encoder of EdgeSAM (`edge_sam_3x.pth`, vendored code, 256 px input); task decoder trained from scratch: four transposed-convolution stages (x2 each, batch norm, ReLU6) and a 1x1 classifier | `qpi/models/edge_sam.py`, D-14 |
| SAM prompt encoder / mask decoder | **not used** (neither the original prompt encoder nor the original mask decoder) | as above |
| Single-channel stem | first convolution replaced by a 1-channel convolution initialised with the mean of the RGB filters; trainable | model files, `lora_utils.py` |
| LoRA placement | MobileSAM: linear layers of the encoder attention whose names contain `qkv` / `attn` (the fused query-key-value projections); EdgeSAM and MobileNet-UNet: every non-grouped convolution and linear layer of the encoder (depthwise convolutions excluded) | `lora_utils.py: _should_inject` **[verify the MobileSAM module list by printing the replaced names once]** |
| LoRA scaling | α = r (scaling α/r = 1), dropout 0, B initialised to zero | `configs/experiments.yaml`, `lora_utils.py` |
| Also trainable with LoRA | decoder and output head, all normalisation layers, the single-channel stem | `lora_utils.py` |
| Deployment | LoRA merged into the weights (W' = W + BA), so inference cost does not depend on r | `lora_utils.py: merge`, `qpi/benchmark.py` |

**Draft text (Sec. 2.3).**
Three lightweight networks were adapted. MobileNet-UNet combines an ImageNet-pretrained MobileNetV2 encoder with a U-Net decoder with
skip connections [32]. For MobileSAM and EdgeSAM only the pretrained image encoders were used, the distilled TinyViT encoder of
MobileSAM [33] and the RepViT encoder of EdgeSAM [34]; prompt-based mask decoding is not required for dense five-class segmentation,
so each encoder was combined with a compact task decoder trained from scratch. For MobileSAM, a learned prompt embedding is added to
five class tokens that attend to the image embedding through one transformer decoder layer, and each class map is the inner product
of its token with the bilinearly upsampled embedding. For EdgeSAM, four transposed-convolution stages upsample the embedding to the
input resolution, followed by a 1 × 1 classifier. In all three networks the first convolution was replaced by a single-channel
convolution initialised with the mean of the pretrained RGB filters.

**Draft text (Sec. 2.4, after Eq. 1).**
We used α = r, so the low-rank update enters with unit scaling at every rank. In MobileSAM, LoRA was applied to the fused
query-key-value projections of the attention blocks of the TinyViT encoder; in EdgeSAM and MobileNet-UNet, to every non-grouped
convolution and linear layer of the encoder (depthwise convolutions were left frozen). In all models the single-channel stem, the
normalisation layers, the decoder and the output head were trainable, so the fraction of trainable parameters differs between
architectures (Table S8). For inference the low-rank matrices are merged into the frozen weights, so latency and memory do not
depend on the rank.

## 3. Physics-aware loss (Sec. 2.5): what must change

The equations (2) to (6) are implemented as written, with these facts to state:

* All terms are computed in float32 outside the FP16 autocast region; the forward pass uses FP16 mixed precision (`trainer.py`).
* φ in Eqs. (3) to (5) is the **background-referenced phase in radians** at the model resolution (not "raw, unnormalised" in the
  sense of "as provided": replace "raw" by "background-referenced, unstandardised").
* M is the soft foreground probability 1 − p(background); G the annotated foreground; unannotated pixels are excluded from every
  sum and mean.
* PMC: hinge with margin 0.1 rad. BGA: gradients by forward differences, each map max-normalised per sample, L1 averaged over valid
  pixels whose 3 x 3 neighbourhood is annotated. IPP: |S_pred − S_GT| / (|S_GT| + 1e−4).
* Each physics term is averaged over the images of the batch that contain at least one annotated cell.
* λ1 = 0.1 (PMC), λ2 = 0.05 (BGA), λ3 = 0.1 (IPP); Dice class weights 0.5, 1.0, 1.5, 2.0, 2.0; Laplace smoothing 1.
* Remove: "As demonstrated by the ablation experiments, the individual constraints are complementary" (not supported, 02 B-15).

## 4. Training (replaces Sec. 3.2)

| Item | Value | Source |
|---|---|---|
| Optimiser, schedule | AdamW, cosine annealing over 50 epochs | `trainer.py`, `base.yaml` |
| Learning rate, batch size | EdgeSAM 5e−4, 16; MobileSAM 5e−4, 8; MobileNet-UNet 1e−3, 16 | `experiments.yaml` |
| Precision | FP16 autocast for the forward pass, losses in float32, gradient scaling | `trainer.py` |
| Sampler | plain shuffling (no WeightedRandomSampler) | D-04 |
| Checkpoint for every reported number | the last epoch (epoch 50); no checkpoint is selected on the evaluation images | D-08 |
| Seeds | 42, 43, 44 for every LoRA configuration; full fine-tuning seed 42 only | D-22, D-24 |
| Determinism | deterministic algorithms requested; bitwise equality across GPUs not guaranteed | README |
| Hardware | NVIDIA RTX A5000, PyTorch 2.0.1, CUDA 11.7 | `run_metadata.json` |

**Draft text (Sec. 3.2).**
All networks were trained for 50 epochs with AdamW and cosine annealing (learning rate 5 × 10⁻⁴ for the SAM-based models and
10⁻³ for MobileNet-UNet; batch size 16, or 8 for MobileSAM), with FP16 mixed precision in the forward pass and all loss terms in
float32. Class imbalance was handled by the class-weighted Dice loss of Eq. (2); mini-batches were drawn by plain shuffling. Every
LoRA configuration (three architectures × five ranks r ∈ {2, 4, 8, 16, 32} × four loss variants) was trained with three random
seeds (42, 43, 44), 180 runs in total; full fine-tuning was trained once per architecture (seed 42). No checkpoint selection was
performed: every reported number is from the final epoch, because the validation images are also the evaluation images.

## 5. Evaluation metrics (Sec. 3.3): what to state

* Metrics at the native 768 x 768 grid: logits bilinearly upsampled, then arg-max (`evaluate.py`, D-12).
* Dice and IoU per class pooled over all validation pixels (not averaged per image), macro-averaged over the four foreground classes;
  a class present in the annotation and never predicted scores 0. (A class absent from both would be undefined; it does not occur.)
* AJI on connected components of the binary foreground; boundary F1 pooled, 2-px tolerance, boundary pixels near unannotated
  regions dropped.
* Global IPE: per image on the background-referenced phase, then the mean over images (Eq. 11 is unchanged).
* Per-cell IPE: annotated cells 15 to 200 um2 not cut by the annotated field of view, matched one-to-one to predicted components
  with IoU ≥ 0.5; phase after subtraction of a least-squares background plane; wrapped cells excluded.
* **Lost class** (training instability): a class with Dice < 0.05 in a run; reported per seed.
* Statistics over seeds: mean ± sample SD (n = 3); paired differences with 95% t-intervals, described as descriptive.

## 6. Dry mass (Sec. 2.6): exact definition

m = λ / (2πα) · Σ_{i∈Ω} (φ_i − φ_bg) · Δx², with λ = 0.666 µm, α = 0.2 mL/g, Δx = 0.1441 µm, φ in radians. φ_bg is the median
phase of the annotated, non-cellular pixels farther than 5 px from the mask being analysed (so the predicted-mask dry mass uses a
background estimated from the prediction). Cells are connected components of the (predicted or annotated) foreground with area
15 to 200 µm² that do not touch an unannotated region; perimeter by the Crofton estimator; circularity 4πA/P².
**No calibration** to a reference MCH is applied; MCH = 30 pg is used only as a plausibility check (annotation day-0 median
27.83 pg = 0.93 × MCH). Agreement statistics use the per-image median dry mass (66 images): MdARE, Bland-Altman bias and limits of
agreement; Pearson r and Lin's CCC are reported but are not informative when the reference varies little.

**Draft text (replaces the last two sentences of the dry-mass paragraph in Sec. 2.6).**
The phase maps are in radians, so Eq. (10) was applied with the physical constants only; no calibration to a reference
haemoglobin content was used. As a plausibility check, the median day-0 dry mass derived from the annotation masks was 27.8 pg,
0.93 of the typical mean corpuscular haemoglobin of 30 pg. Cells whose phase reached the 2π limit of the reconstruction or showed a
wrap discontinuity (3.0% of annotated cells) were excluded from all phase-derived results.

## 7. Benchmark (Sec. 3.4): exact protocol

Protocol v5 (`manuscript/benchmarking_protocol.md`): batch 1, each model at its native input size, forward pass only, strict FP32
in PyTorch and ONNX Runtime (CUDA provider, IO binding, TF32 off), 50 warm-up and 500 timed synchronised passes, three seeds each
in its own process, three fresh ONNX sessions per seed; P50, P99, FPS = 1000/P50, Peak VRAM = device memory after the timed loop
minus before model placement. Equivalence gate before reporting any ONNX number (arg-max agreement ≥ 99.9%, |ΔDice| ≤ 0.001,
dry-mass error mean ≤ 0.1%, P99 ≤ 0.5%); all three models passed. FP16 was examined and not included.
