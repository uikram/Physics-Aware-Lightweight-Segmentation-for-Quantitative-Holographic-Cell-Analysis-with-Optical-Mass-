# T6_lora_vs_full_ft: LoRA versus full fine-tuning (full physics-aware objective)

| Architecture | LoRA rank | Trainable params | Mean Dice | Mean IoU | AJI | Boundary F1 | Global IPE (%) | Dice discocyte (1) | Dice spherocyte (2) | Dice echinocyte (3) | Dice stomatocyte (4) | Runs with a lost class | Seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MobileSAM | 8 | 0.67 M (10.0%) | 0.810 ± 0.006 | 0.690 ± 0.008 | 0.262 ± 0.017 | 0.802 ± 0.011 | 1.32 ± 0.09 | 0.923 ± 0.003 | 0.851 ± 0.004 | 0.760 ± 0.006 | 0.708 ± 0.020 | 0/3 | 42, 43, 44 |
| MobileSAM | 8 (seed 42 only) | 0.67 M (10.0%) | 0.810 | 0.689 | 0.243 | 0.792 | 1.42 | 0.921 | 0.855 | 0.763 | 0.700 | none | 42 |
| MobileSAM | full fine-tuning | 6.66 M (100.0%) | 0.217 | 0.192 | 0.448 | 0.808 | 2.16 | 0.869 | 0.000 | 0.000 | 0.000 | sphero(2), echino(3), stomato(4) | 42 |
| EdgeSAM | 8 | 0.53 M (8.9%) | 0.647 ± 0.114 | 0.514 ± 0.095 | 0.249 ± 0.043 | 0.608 ± 0.009 | 3.81 ± 1.82 | 0.882 ± 0.012 | 0.513 ± 0.444 | 0.628 ± 0.006 | 0.563 ± 0.019 | 1/3 (s44: sphero(2)) | 42, 43, 44 |
| EdgeSAM | 32 | 1.47 M (21.2%) | 0.738 ± 0.006 | 0.598 ± 0.007 | 0.335 ± 0.020 | 0.642 ± 0.005 | 3.02 ± 0.84 | 0.895 ± 0.001 | 0.789 ± 0.005 | 0.655 ± 0.010 | 0.614 ± 0.019 | 0/3 | 42, 43, 44 |
| EdgeSAM | 8 (seed 42 only) | 0.53 M (8.9%) | 0.706 | 0.562 | 0.224 | 0.615 | 3.13 | 0.891 | 0.765 | 0.621 | 0.545 | none | 42 |
| EdgeSAM | 32 (seed 42 only) | 1.47 M (21.2%) | 0.731 | 0.590 | 0.329 | 0.647 | 2.66 | 0.896 | 0.784 | 0.652 | 0.593 | none | 42 |
| EdgeSAM | full fine-tuning | 5.69 M (100.0%) | 0.559 | 0.455 | 0.278 | 0.635 | 8.31 | 0.892 | 0.000 | 0.681 | 0.665 | sphero(2) | 42 |
| MobileNet-UNet | 8 | 5.58 M (71.8%) | 0.412 ± 0.004 | 0.353 ± 0.006 | 0.551 ± 0.053 | 0.814 ± 0.003 | 2.60 ± 0.07 | 0.891 ± 0.002 | 0.000 ± 0.000 | 0.756 ± 0.015 | 0.000 ± 0.000 | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2), stomato(4)) | 42, 43, 44 |
| MobileNet-UNet | 8 (seed 42 only) | 5.58 M (71.8%) | 0.412 | 0.353 | 0.603 | 0.816 | 2.52 | 0.891 | 0.000 | 0.757 | 0.000 | sphero(2), stomato(4) | 42 |
| MobileNet-UNet | full fine-tuning | 7.62 M (100.0%) | 0.418 | 0.361 | 0.642 | 0.825 | 2.17 | 0.893 | 0.000 | 0.778 | 0.000 | sphero(2), stomato(4) | 42 |

Notes:

* Full fine-tuning was trained once (seed 42, D-24); its row is a single run (no SD). Compare it with the 'seed 42 only' LoRA rows (same seed) and read the 3-seed rows for the LoRA seed spread.
* Full fine-tuning lost spherocytes in EdgeSAM, all three minority classes in MobileSAM, and spherocytes and stomatocytes in MobileNet-UNet.
