# T4_training_stability: Training stability: runs that lost a morphology class, per architecture and LoRA rank

| Objective | Architecture | r = 2 | r = 4 | r = 8 | r = 16 | r = 32 |
|---|---|---|---|---|---|---|
| Full physics-aware | MobileSAM | 0/3 | 0/3 | 0/3 | 0/3 | 1/3 (s43: sphero(2), stomato(4)) |
| Full physics-aware | EdgeSAM | 1/3 (s44: sphero(2)) | 0/3 | 1/3 (s44: sphero(2)) | 1/3 (s44: sphero(2)) | 0/3 |
| Full physics-aware | MobileNet-UNet | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2), stomato(4)) | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2), stomato(4)) | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2), stomato(4)) | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2), stomato(4)) | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2)) |
| Dice only | MobileSAM | 1/3 (s42: sphero(2)) | 0/3 | 0/3 | 0/3 | 2/3 (s42: sphero(2); s43: sphero(2)) |
| Dice only | EdgeSAM | 0/3 | 1/3 (s43: sphero(2)) | 1/3 (s43: sphero(2)) | 2/3 (s43: sphero(2); s44: sphero(2)) | 2/3 (s43: sphero(2); s44: sphero(2)) |
| Dice only | MobileNet-UNet | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2), stomato(4)) | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2)) | 2/3 (s42: sphero(2), echino(3), stomato(4); s44: sphero(2), stomato(4)) | 2/3 (s42: sphero(2), echino(3), stomato(4); s44: sphero(2)) | 3/3 (s42: sphero(2), echino(3), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2), stomato(4)) |

**Seed SD of mean Dice (full objective)**

| Architecture | r = 2 | r = 4 | r = 8 | r = 16 | r = 32 |
|---|---|---|---|---|---|
| MobileSAM | 0.003 | 0.004 | 0.006 | 0.002 | 0.234 |
| EdgeSAM | 0.108 | 0.009 | 0.114 | 0.114 | 0.006 |
| MobileNet-UNet | 0.004 | 0.002 | 0.004 | 0.002 | 0.108 |

Notes:

* k/3 = number of the three seeds in which at least one class had Dice < 0.05; the seed and class are named.
* MobileNet-UNet loses spherocytes in every run (and stomatocytes in 14 of 15 full-objective runs); this is a systematic failure, not instability.
* Training curves: Fig. 5 (artifacts/figures/main).
