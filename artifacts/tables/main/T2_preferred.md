# T2_preferred: Segmentation at the preferred LoRA configurations (full physics-aware objective)

| Architecture | LoRA rank | Trainable params | Mean Dice | Mean IoU | AJI | Boundary F1 | Global IPE (%) | Dice discocyte (1) | Dice spherocyte (2) | Dice echinocyte (3) | Dice stomatocyte (4) | Runs with a lost class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MobileSAM | 8 | 0.67 M (10.0%) | 0.810 ± 0.006 | 0.690 ± 0.008 | 0.262 ± 0.017 | 0.802 ± 0.011 | 1.32 ± 0.09 | 0.923 ± 0.003 | 0.851 ± 0.004 | 0.760 ± 0.006 | 0.708 ± 0.020 | 0/3 |
| EdgeSAM | 32 | 1.47 M (21.2%) | 0.738 ± 0.006 | 0.598 ± 0.007 | 0.335 ± 0.020 | 0.642 ± 0.005 | 3.02 ± 0.84 | 0.895 ± 0.001 | 0.789 ± 0.005 | 0.655 ± 0.010 | 0.614 ± 0.019 | 0/3 |
| MobileNet-UNet | 8 | 5.58 M (71.8%) | 0.412 ± 0.004 | 0.353 ± 0.006 | 0.551 ± 0.053 | 0.814 ± 0.003 | 2.60 ± 0.07 | 0.891 ± 0.002 | 0.000 ± 0.000 | 0.756 ± 0.015 | 0.000 ± 0.000 | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2), stomato(4)) |

Notes:

* MobileSAM r = 8 and EdgeSAM r = 32 are the presentation choice of the revision (D-27, D-30); no selection rule is claimed. MobileNet-UNet (r = 8) is the convolutional baseline.
* Mean ± SD over seeds 42, 43, 44; last-epoch checkpoint; metrics at the native 768 × 768 grid.
* Lost class: per-class Dice < 0.05. Class names carry their mask value: sphero(2) = spherocyte, echino(3) = echinocyte, stomato(4) = stomatocyte.
