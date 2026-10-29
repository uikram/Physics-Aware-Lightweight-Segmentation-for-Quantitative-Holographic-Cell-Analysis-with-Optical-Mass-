# T5_loss_ablation: Component-wise ablation of the physics-aware loss at the preferred and same-rank configurations

| Configuration | Objective | Mean Dice | Boundary F1 | AJI | Global IPE (%) | Runs with a lost class |
|---|---|---|---|---|---|---|
| MobileSAM r = 8 | Dice only | 0.811 ± 0.005 | 0.808 ± 0.012 | 0.248 ± 0.022 | 1.53 ± 0.27 | 0/3 |
| MobileSAM r = 8 | Dice + PMC | 0.813 ± 0.009 | 0.805 ± 0.002 | 0.242 ± 0.035 | 1.60 ± 0.24 | 0/3 |
| MobileSAM r = 8 | Dice + PMC + BGA | 0.813 ± 0.005 | 0.806 ± 0.004 | 0.259 ± 0.034 | 1.48 ± 0.11 | 0/3 |
| MobileSAM r = 8 | Dice + PMC + BGA + IPP (full) | 0.810 ± 0.006 | 0.802 ± 0.011 | 0.262 ± 0.017 | 1.32 ± 0.09 | 0/3 |
| EdgeSAM r = 32 | Dice only | 0.619 ± 0.109 | 0.519 ± 0.212 | 0.329 ± 0.015 | 11.20 ± 6.46 | 2/3 (s43: sphero(2); s44: sphero(2)) |
| EdgeSAM r = 32 | Dice + PMC | 0.680 ± 0.113 | 0.638 ± 0.027 | 0.388 ± 0.059 | 7.45 ± 7.17 | 1/3 (s44: sphero(2)) |
| EdgeSAM r = 32 | Dice + PMC + BGA | 0.746 ± 0.006 | 0.643 ± 0.011 | 0.406 ± 0.034 | 3.64 ± 0.82 | 0/3 |
| EdgeSAM r = 32 | Dice + PMC + BGA + IPP (full) | 0.738 ± 0.006 | 0.642 ± 0.005 | 0.335 ± 0.020 | 3.02 ± 0.84 | 0/3 |
| EdgeSAM r = 8 | Dice only | 0.624 ± 0.095 | 0.496 ± 0.209 | 0.334 ± 0.052 | 10.82 ± 5.39 | 1/3 (s43: sphero(2)) |
| EdgeSAM r = 8 | Dice + PMC | 0.672 ± 0.109 | 0.621 ± 0.031 | 0.386 ± 0.033 | 7.97 ± 6.80 | 1/3 (s44: sphero(2)) |
| EdgeSAM r = 8 | Dice + PMC + BGA | 0.735 ± 0.010 | 0.628 ± 0.013 | 0.390 ± 0.046 | 4.25 ± 1.07 | 0/3 |
| EdgeSAM r = 8 | Dice + PMC + BGA + IPP (full) | 0.647 ± 0.114 | 0.608 ± 0.009 | 0.249 ± 0.043 | 3.81 ± 1.82 | 1/3 (s44: sphero(2)) |
| MobileNet-UNet r = 8 | Dice only | 0.485 ± 0.294 | 0.814 ± 0.054 | 0.623 ± 0.111 | 15.48 ± 12.73 | 2/3 (s42: sphero(2), echino(3), stomato(4); s44: sphero(2), stomato(4)) |
| MobileNet-UNet r = 8 | Dice + PMC | 0.414 ± 0.181 | 0.782 ± 0.022 | 0.585 ± 0.071 | 19.79 ± 6.71 | 3/3 (s42: sphero(2), echino(3), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2)) |
| MobileNet-UNet r = 8 | Dice + PMC + BGA | 0.544 ± 0.292 | 0.812 ± 0.052 | 0.602 ± 0.125 | 13.96 ± 12.46 | 2/3 (s42: sphero(2), echino(3), stomato(4); s44: sphero(2)) |
| MobileNet-UNet r = 8 | Dice + PMC + BGA + IPP (full) | 0.412 ± 0.004 | 0.814 ± 0.003 | 0.551 ± 0.053 | 2.60 ± 0.07 | 3/3 (s42: sphero(2), stomato(4); s43: sphero(2), stomato(4); s44: sphero(2), stomato(4)) |

**Seed-paired differences [95% t-interval, n = 3]**

| Configuration | Δ IPE (pp), Full − Dice only | Δ Dice, Full − Dice only | Δ IPE (pp), Full − (Dice+PMC+BGA): IPP | Δ Dice, Full − (Dice+PMC+BGA): IPP |
|---|---|---|---|---|
| MobileSAM r = 8 | -0.21 [-0.75, +0.33] | -0.000 [-0.017, +0.016] | -0.16 [-0.41, +0.09] | -0.003 [-0.030, +0.024] |
| EdgeSAM r = 32 | -8.18 [-23.52, +7.15] | +0.119 [-0.166, +0.404] | -0.63 [-0.75, -0.51] | -0.008 [-0.039, +0.023] |
| EdgeSAM r = 8 | -7.01 [-19.68, +5.66] | +0.023 [-0.306, +0.352] | -0.44 [-4.77, +3.90] | -0.089 [-0.357, +0.179] |
| MobileNet-UNet r = 8 | -12.88 [-44.66, +18.90] | -0.073 [-0.796, +0.650] | -11.36 [-42.49, +19.77] | -0.132 [-0.853, +0.588] |

Notes:

* The full-objective row of each block is the rank-sweep run of that rank (same configuration and seed; it is not trained twice, D-09). The ablation runs are separate trainings with the same seeds.
* The physics terms act on the binary foreground probability and on the background-referenced phase in radians.
* All ranks: Table S4/S5.
