# Multi-seed results

Scope: stages 1 to 3.

Missing or unfinished runs: 0

## MS-1  LoRA rank sweep, EdgeSAM (full physics-aware loss)
Purpose: is rank 8 a defensible choice? Mean ± SD over seeds (42 from outputs/full; others from outputs/multiseed). Last-epoch checkpoint. 'Runs with a missing class' = runs in which at least one class has Dice < 0.05.

| Configuration | n | Mean Dice | AJI | BF1 | Global IPE (%) | Dice sphero | Dice stom | Runs with a missing class |
|---|---|---|---|---|---|---|---|---|
| r=2 | 3 | 0.625 ± 0.108 | 0.249 ± 0.017 | 0.598 ± 0.020 | 4.45 ± 2.71 | 0.497 ± 0.431 | 0.506 ± 0.013 | 1/3 |
| r=4 | 3 | 0.703 ± 0.009 | 0.252 ± 0.021 | 0.615 ± 0.012 | 3.48 ± 0.91 | 0.766 ± 0.013 | 0.544 ± 0.025 | 0/3 |
| r=8 | 3 | 0.647 ± 0.114 | 0.249 ± 0.043 | 0.608 ± 0.009 | 3.81 ± 1.82 | 0.513 ± 0.444 | 0.563 ± 0.019 | 1/3 |
| r=16 | 3 | 0.665 ± 0.114 | 0.277 ± 0.022 | 0.623 ± 0.024 | 4.66 ± 3.53 | 0.521 ± 0.451 | 0.591 ± 0.027 | 1/3 |
| r=32 | 3 | 0.738 ± 0.006 | 0.335 ± 0.020 | 0.642 ± 0.005 | 3.02 ± 0.84 | 0.789 ± 0.005 | 0.614 ± 0.019 | 0/3 |

Paired differences to r=8 (rank r minus rank 8, same seed; mean [95% t-interval] (seeds in which r is higher)):

| Rank | Mean Dice | AJI | BF1 | Global IPE (pp) |
|---|---|---|---|---|
| r=2 | -0.021 [-0.037, -0.005] (0/3 seeds higher) | -0.001 [-0.090, +0.089] (2/3 seeds higher) | -0.010 [-0.044, +0.024] (1/3 seeds higher) | +0.64 [-1.84, +3.13] (2/3 seeds higher) |
| r=4 | +0.057 [-0.204, +0.318] (2/3 seeds higher) | +0.003 [-0.057, +0.063] (2/3 seeds higher) | +0.007 [-0.020, +0.034] (2/3 seeds higher) | -0.34 [-2.64, +1.97] (1/3 seeds higher) |
| r=16 | +0.019 [+0.013, +0.024] (3/3 seeds higher) | +0.028 [-0.032, +0.087] (3/3 seeds higher) | +0.016 [-0.022, +0.053] (2/3 seeds higher) | +0.85 [-3.47, +5.18] (1/3 seeds higher) |
| r=32 | +0.091 [-0.196, +0.379] (3/3 seeds higher) | +0.086 [+0.026, +0.146] (3/3 seeds higher) | +0.034 [+0.022, +0.046] (3/3 seeds higher) | -0.80 [-3.22, +1.62] (0/3 seeds higher) |

Rule (fixed in qpi/analysis/multiseed.py before the runs): per metric, the best rank by mean is found; ranks whose paired difference to the best has a 95% interval containing 0 are 'not distinguishable from the best'.

* mean_dice: best by mean r=32; not distinguishable from best: [2, 8, 16, 32]; selected r=8 IS among them.
* aji: best by mean r=32; not distinguishable from best: [32]; selected r=8 is NOT among them.
* bf1: best by mean r=32; not distinguishable from best: [16, 32]; selected r=8 is NOT among them.
* ipe_global_pct: best by mean r=32; not distinguishable from best: [2, 8, 16, 32]; selected r=8 IS among them.

## MS-1  LoRA rank sweep, MobileSAM (full physics-aware loss)
Purpose: is rank 8 a defensible choice? Mean ± SD over seeds (42 from outputs/full; others from outputs/multiseed). Last-epoch checkpoint. 'Runs with a missing class' = runs in which at least one class has Dice < 0.05.

| Configuration | n | Mean Dice | AJI | BF1 | Global IPE (%) | Dice sphero | Dice stom | Runs with a missing class |
|---|---|---|---|---|---|---|---|---|
| r=2 | 3 | 0.808 ± 0.003 | 0.264 ± 0.013 | 0.801 ± 0.006 | 1.30 ± 0.09 | 0.848 ± 0.002 | 0.694 ± 0.013 | 0/3 |
| r=4 | 3 | 0.811 ± 0.004 | 0.268 ± 0.037 | 0.803 ± 0.002 | 1.34 ± 0.07 | 0.855 ± 0.005 | 0.691 ± 0.021 | 0/3 |
| r=8 | 3 | 0.810 ± 0.006 | 0.262 ± 0.017 | 0.802 ± 0.011 | 1.32 ± 0.09 | 0.851 ± 0.004 | 0.708 ± 0.020 | 0/3 |
| r=16 | 3 | 0.813 ± 0.002 | 0.279 ± 0.054 | 0.801 ± 0.013 | 1.29 ± 0.18 | 0.855 ± 0.002 | 0.696 ± 0.009 | 0/3 |
| r=32 | 3 | 0.687 ± 0.234 | 0.283 ± 0.020 | 0.794 ± 0.023 | 1.76 ± 0.53 | 0.574 ± 0.497 | 0.485 ± 0.420 | 1/3 |

Paired differences to r=8 (rank r minus rank 8, same seed; mean [95% t-interval] (seeds in which r is higher)):

| Rank | Mean Dice | AJI | BF1 | Global IPE (pp) |
|---|---|---|---|---|
| r=2 | -0.002 [-0.012, +0.007] (1/3 seeds higher) | +0.002 [-0.039, +0.044] (2/3 seeds higher) | -0.000 [-0.022, +0.022] (1/3 seeds higher) | -0.02 [-0.23, +0.19] (1/3 seeds higher) |
| r=4 | +0.000 [-0.024, +0.024] (1/3 seeds higher) | +0.006 [-0.044, +0.057] (2/3 seeds higher) | +0.002 [-0.022, +0.025] (2/3 seeds higher) | +0.03 [-0.13, +0.18] (2/3 seeds higher) |
| r=16 | +0.002 [-0.015, +0.019] (2/3 seeds higher) | +0.017 [-0.086, +0.120] (2/3 seeds higher) | -0.001 [-0.031, +0.030] (1/3 seeds higher) | -0.03 [-0.46, +0.40] (1/3 seeds higher) |
| r=32 | -0.123 [-0.691, +0.445] (2/3 seeds higher) | +0.022 [-0.005, +0.048] (3/3 seeds higher) | -0.008 [-0.061, +0.045] (2/3 seeds higher) | +0.44 [-0.91, +1.78] (3/3 seeds higher) |

Rule (fixed in qpi/analysis/multiseed.py before the runs): per metric, the best rank by mean is found; ranks whose paired difference to the best has a 95% interval containing 0 are 'not distinguishable from the best'.

* mean_dice: best by mean r=16; not distinguishable from best: [2, 4, 8, 16, 32]; selected r=8 IS among them.
* aji: best by mean r=32; not distinguishable from best: [2, 4, 8, 16, 32]; selected r=8 IS among them.
* bf1: best by mean r=4; not distinguishable from best: [2, 4, 8, 16, 32]; selected r=8 IS among them.
* ipe_global_pct: best by mean r=16; not distinguishable from best: [2, 4, 8, 16, 32]; selected r=8 IS among them.

## MS-1  LoRA rank sweep, MobileNet-UNet (full physics-aware loss)
Purpose: is rank 8 a defensible choice? Mean ± SD over seeds (42 from outputs/full; others from outputs/multiseed). Last-epoch checkpoint. 'Runs with a missing class' = runs in which at least one class has Dice < 0.05.

| Configuration | n | Mean Dice | AJI | BF1 | Global IPE (%) | Dice sphero | Dice stom | Runs with a missing class |
|---|---|---|---|---|---|---|---|---|
| r=2 | 3 | 0.410 ± 0.004 | 0.534 ± 0.029 | 0.816 ± 0.006 | 2.61 ± 0.23 | 0.000 ± 0.000 | 0.000 ± 0.000 | 3/3 |
| r=4 | 3 | 0.406 ± 0.002 | 0.560 ± 0.035 | 0.805 ± 0.006 | 2.50 ± 0.19 | 0.000 ± 0.000 | 0.000 ± 0.000 | 3/3 |
| r=8 | 3 | 0.412 ± 0.004 | 0.551 ± 0.053 | 0.814 ± 0.003 | 2.60 ± 0.07 | 0.000 ± 0.000 | 0.000 ± 0.000 | 3/3 |
| r=16 | 3 | 0.414 ± 0.002 | 0.505 ± 0.012 | 0.805 ± 0.009 | 3.09 ± 0.65 | 0.000 ± 0.000 | 0.000 ± 0.000 | 3/3 |
| r=32 | 3 | 0.476 ± 0.108 | 0.541 ± 0.049 | 0.824 ± 0.014 | 5.83 ± 5.75 | 0.000 ± 0.000 | 0.236 ± 0.408 | 3/3 |

Paired differences to r=8 (rank r minus rank 8, same seed; mean [95% t-interval] (seeds in which r is higher)):

| Rank | Mean Dice | AJI | BF1 | Global IPE (pp) |
|---|---|---|---|---|
| r=2 | -0.001 [-0.019, +0.016] (1/3 seeds higher) | -0.016 [-0.190, +0.157] (1/3 seeds higher) | +0.002 [-0.020, +0.024] (2/3 seeds higher) | +0.00 [-0.65, +0.65] (2/3 seeds higher) |
| r=4 | -0.005 [-0.016, +0.005] (0/3 seeds higher) | +0.010 [-0.170, +0.189] (1/3 seeds higher) | -0.009 [-0.023, +0.005] (0/3 seeds higher) | -0.10 [-0.48, +0.27] (1/3 seeds higher) |
| r=16 | +0.002 [-0.005, +0.009] (2/3 seeds higher) | -0.046 [-0.185, +0.094] (1/3 seeds higher) | -0.008 [-0.027, +0.011] (0/3 seeds higher) | +0.49 [-0.94, +1.92] (2/3 seeds higher) |
| r=32 | +0.065 [-0.214, +0.343] (2/3 seeds higher) | -0.010 [-0.211, +0.191] (2/3 seeds higher) | +0.010 [-0.015, +0.035] (2/3 seeds higher) | +3.23 [-10.99, +17.45] (2/3 seeds higher) |

Rule (fixed in qpi/analysis/multiseed.py before the runs): per metric, the best rank by mean is found; ranks whose paired difference to the best has a 95% interval containing 0 are 'not distinguishable from the best'.

* mean_dice: best by mean r=32; not distinguishable from best: [2, 4, 8, 16, 32]; selected r=8 IS among them.
* aji: best by mean r=4; not distinguishable from best: [4, 8, 16, 32]; selected r=8 IS among them.
* bf1: best by mean r=32; not distinguishable from best: [2, 4, 8, 16, 32]; selected r=8 IS among them.
* ipe_global_pct: best by mean r=4; not distinguishable from best: [2, 4, 8, 16, 32]; selected r=8 IS among them.

## MS-2  Architectures at rank 8 (full physics-aware loss)
Purpose: which architecture is best, and is the gap larger than the run-to-run spread?

| Configuration | n | Mean Dice | AJI | BF1 | Global IPE (%) | Dice sphero | Dice stom | Runs with a missing class |
|---|---|---|---|---|---|---|---|---|
| MobileNet-UNet | 3 | 0.412 ± 0.004 | 0.551 ± 0.053 | 0.814 ± 0.003 | 2.60 ± 0.07 | 0.000 ± 0.000 | 0.000 ± 0.000 | 3/3 |
| MobileSAM | 3 | 0.810 ± 0.006 | 0.262 ± 0.017 | 0.802 ± 0.011 | 1.32 ± 0.09 | 0.851 ± 0.004 | 0.708 ± 0.020 | 0/3 |
| EdgeSAM | 3 | 0.647 ± 0.114 | 0.249 ± 0.043 | 0.608 ± 0.009 | 3.81 ± 1.82 | 0.513 ± 0.444 | 0.563 ± 0.019 | 1/3 |

Paired differences (first minus second, same seed):

| Comparison | Mean Dice | AJI | BF1 | Global IPE (pp) |
|---|---|---|---|---|
| MobileSAM minus EdgeSAM | +0.164 [-0.133, +0.461] (3/3 seeds higher) | +0.012 [-0.088, +0.113] (2/3 seeds higher) | +0.194 [+0.145, +0.243] (3/3 seeds higher) | -2.49 [-7.15, +2.17] (0/3 seeds higher) |
| MobileNet-UNet minus EdgeSAM | -0.235 [-0.508, +0.038] (0/3 seeds higher) | +0.301 [+0.133, +0.469] (3/3 seeds higher) | +0.206 [+0.181, +0.231] (3/3 seeds higher) | -1.21 [-5.70, +3.28] (1/3 seeds higher) |

## MS-4  Physics-aware loss ablation (rank 8)
Purpose: which effects of PMC, BGA and IPP are reproducible across seeds? 'Full' = the rank-8 run of the sweep.

### EdgeSAM

| Configuration | n | Mean Dice | AJI | BF1 | Global IPE (%) | Dice sphero | Dice stom | Runs with a missing class |
|---|---|---|---|---|---|---|---|---|
| Dice only | 3 | 0.624 ± 0.095 | 0.334 ± 0.052 | 0.496 ± 0.209 | 10.82 ± 5.39 | 0.343 ± 0.373 | 0.592 ± 0.006 | 1/3 |
| + PMC | 3 | 0.672 ± 0.109 | 0.386 ± 0.033 | 0.621 ± 0.031 | 7.97 ± 6.80 | 0.524 ± 0.454 | 0.601 ± 0.016 | 1/3 |
| + PMC + BGA | 3 | 0.735 ± 0.010 | 0.390 ± 0.046 | 0.628 ± 0.013 | 4.25 ± 1.07 | 0.783 ± 0.013 | 0.612 ± 0.020 | 0/3 |
| + PMC + BGA + IPP (full) | 3 | 0.647 ± 0.114 | 0.249 ± 0.043 | 0.608 ± 0.009 | 3.81 ± 1.82 | 0.513 ± 0.444 | 0.563 ± 0.019 | 1/3 |

Paired differences, full objective minus Dice only: Dice +0.023 [-0.306, +0.352] (1/3 seeds higher); BF1 +0.112 [-0.413, +0.637] (2/3 seeds higher); AJI -0.084 [-0.176, +0.007] (0/3 seeds higher); IPE (pp) -7.01 [-19.68, +5.66] (0/3 seeds higher)

### MobileSAM

| Configuration | n | Mean Dice | AJI | BF1 | Global IPE (%) | Dice sphero | Dice stom | Runs with a missing class |
|---|---|---|---|---|---|---|---|---|
| Dice only | 3 | 0.811 ± 0.005 | 0.248 ± 0.022 | 0.808 ± 0.012 | 1.53 ± 0.27 | 0.849 ± 0.005 | 0.713 ± 0.007 | 0/3 |
| + PMC | 3 | 0.813 ± 0.009 | 0.242 ± 0.035 | 0.805 ± 0.002 | 1.60 ± 0.24 | 0.852 ± 0.007 | 0.701 ± 0.027 | 0/3 |
| + PMC + BGA | 3 | 0.813 ± 0.005 | 0.259 ± 0.034 | 0.806 ± 0.004 | 1.48 ± 0.11 | 0.854 ± 0.004 | 0.710 ± 0.018 | 0/3 |
| + PMC + BGA + IPP (full) | 3 | 0.810 ± 0.006 | 0.262 ± 0.017 | 0.802 ± 0.011 | 1.32 ± 0.09 | 0.851 ± 0.004 | 0.708 ± 0.020 | 0/3 |

Paired differences, full objective minus Dice only: Dice -0.000 [-0.017, +0.016] (2/3 seeds higher); BF1 -0.006 [-0.014, +0.001] (0/3 seeds higher); AJI +0.014 [-0.004, +0.032] (3/3 seeds higher); IPE (pp) -0.21 [-0.75, +0.33] (1/3 seeds higher)

### MobileNet-UNet

| Configuration | n | Mean Dice | AJI | BF1 | Global IPE (%) | Dice sphero | Dice stom | Runs with a missing class |
|---|---|---|---|---|---|---|---|---|
| Dice only | 3 | 0.485 ± 0.294 | 0.623 ± 0.111 | 0.814 ± 0.054 | 15.48 ± 12.73 | 0.282 ± 0.489 | 0.233 ± 0.404 | 2/3 |
| + PMC | 3 | 0.414 ± 0.181 | 0.585 ± 0.071 | 0.782 ± 0.022 | 19.79 ± 6.71 | 0.000 ± 0.000 | 0.229 ± 0.397 | 3/3 |
| + PMC + BGA | 3 | 0.544 ± 0.292 | 0.602 ± 0.125 | 0.812 ± 0.052 | 13.96 ± 12.46 | 0.284 ± 0.491 | 0.470 ± 0.407 | 2/3 |
| + PMC + BGA + IPP (full) | 3 | 0.412 ± 0.004 | 0.551 ± 0.053 | 0.814 ± 0.003 | 2.60 ± 0.07 | 0.000 ± 0.000 | 0.000 ± 0.000 | 3/3 |

Paired differences, full objective minus Dice only: Dice -0.073 [-0.796, +0.650] (1/3 seeds higher); BF1 -0.000 [-0.143, +0.143] (2/3 seeds higher); AJI -0.073 [-0.478, +0.332] (1/3 seeds higher); IPE (pp) -12.88 [-44.66, +18.90] (1/3 seeds higher)

## MS-6  Physics-aware loss ablation by LoRA rank
Purpose: does the effect of PMC, BGA and IPP depend on the LoRA rank? Every rank has the seeds of the multi-seed plan (42, 43, 44); differences are paired by seed. 'Full' = the rank-sweep run of that rank.

### EdgeSAM

| Rank, configuration | n | Mean Dice | AJI | BF1 | Global IPE (%) | Dice sphero | Dice stom | Runs with a missing class |
|---|---|---|---|---|---|---|---|---|
| r=2, Dice only | 3 | 0.576 ± 0.013 | 0.311 ± 0.034 | 0.535 ± 0.063 | 14.42 ± 1.81 | 0.186 ± 0.036 | 0.573 ± 0.009 | 0/3 |
| r=2, + PMC | 3 | 0.710 ± 0.005 | 0.374 ± 0.061 | 0.606 ± 0.027 | 5.26 ± 1.37 | 0.767 ± 0.013 | 0.557 ± 0.019 | 0/3 |
| r=2, + PMC + BGA | 3 | 0.717 ± 0.006 | 0.385 ± 0.055 | 0.610 ± 0.021 | 4.88 ± 1.59 | 0.777 ± 0.015 | 0.569 ± 0.020 | 0/3 |
| r=2, full | 3 | 0.625 ± 0.108 | 0.249 ± 0.017 | 0.598 ± 0.020 | 4.45 ± 2.71 | 0.497 ± 0.431 | 0.506 ± 0.013 | 1/3 |
| r=4, Dice only | 3 | 0.610 ± 0.085 | 0.321 ± 0.057 | 0.476 ± 0.210 | 10.96 ± 5.30 | 0.331 ± 0.360 | 0.571 ± 0.030 | 1/3 |
| r=4, + PMC | 3 | 0.727 ± 0.009 | 0.363 ± 0.032 | 0.619 ± 0.021 | 4.68 ± 1.20 | 0.772 ± 0.019 | 0.588 ± 0.006 | 0/3 |
| r=4, + PMC + BGA | 3 | 0.723 ± 0.011 | 0.392 ± 0.039 | 0.614 ± 0.025 | 4.98 ± 1.57 | 0.770 ± 0.008 | 0.593 ± 0.024 | 0/3 |
| r=4, full | 3 | 0.703 ± 0.009 | 0.252 ± 0.021 | 0.615 ± 0.012 | 3.48 ± 0.91 | 0.766 ± 0.013 | 0.544 ± 0.025 | 0/3 |
| r=8, Dice only | 3 | 0.624 ± 0.095 | 0.334 ± 0.052 | 0.496 ± 0.209 | 10.82 ± 5.39 | 0.343 ± 0.373 | 0.592 ± 0.006 | 1/3 |
| r=8, + PMC | 3 | 0.672 ± 0.109 | 0.386 ± 0.033 | 0.621 ± 0.031 | 7.97 ± 6.80 | 0.524 ± 0.454 | 0.601 ± 0.016 | 1/3 |
| r=8, + PMC + BGA | 3 | 0.735 ± 0.010 | 0.390 ± 0.046 | 0.628 ± 0.013 | 4.25 ± 1.07 | 0.783 ± 0.013 | 0.612 ± 0.020 | 0/3 |
| r=8, full | 3 | 0.647 ± 0.114 | 0.249 ± 0.043 | 0.608 ± 0.009 | 3.81 ± 1.82 | 0.513 ± 0.444 | 0.563 ± 0.019 | 1/3 |
| r=16, Dice only | 3 | 0.621 ± 0.103 | 0.315 ± 0.047 | 0.507 ± 0.221 | 11.05 ± 5.98 | 0.273 ± 0.434 | 0.636 ± 0.028 | 2/3 |
| r=16, + PMC | 3 | 0.741 ± 0.010 | 0.398 ± 0.077 | 0.643 ± 0.009 | 3.71 ± 0.71 | 0.783 ± 0.001 | 0.604 ± 0.029 | 0/3 |
| r=16, + PMC + BGA | 3 | 0.742 ± 0.006 | 0.339 ± 0.024 | 0.644 ± 0.009 | 3.43 ± 0.81 | 0.778 ± 0.011 | 0.617 ± 0.005 | 0/3 |
| r=16, full | 3 | 0.665 ± 0.114 | 0.277 ± 0.022 | 0.623 ± 0.024 | 4.66 ± 3.53 | 0.521 ± 0.451 | 0.591 ± 0.027 | 1/3 |
| r=32, Dice only | 3 | 0.619 ± 0.109 | 0.329 ± 0.015 | 0.519 ± 0.212 | 11.20 ± 6.46 | 0.272 ± 0.439 | 0.630 ± 0.013 | 2/3 |
| r=32, + PMC | 3 | 0.680 ± 0.113 | 0.388 ± 0.059 | 0.638 ± 0.027 | 7.45 ± 7.17 | 0.522 ± 0.452 | 0.625 ± 0.007 | 1/3 |
| r=32, + PMC + BGA | 3 | 0.746 ± 0.006 | 0.406 ± 0.034 | 0.643 ± 0.011 | 3.64 ± 0.82 | 0.781 ± 0.005 | 0.627 ± 0.024 | 0/3 |
| r=32, full | 3 | 0.738 ± 0.006 | 0.335 ± 0.020 | 0.642 ± 0.005 | 3.02 ± 0.84 | 0.789 ± 0.005 | 0.614 ± 0.019 | 0/3 |

Paired differences within a rank (seed-matched; Dice, global IPE in percentage points):

| Rank | Full minus Dice only | Full minus (+ PMC + BGA) = effect of IPP |
|---|---|---|
| r=2 | Dice +0.049 [-0.245, +0.344] (2/3 seeds higher); IPE -9.97 [-18.58, -1.35] (0/3 seeds higher) | Dice -0.092 [-0.346, +0.163] (0/3 seeds higher); IPE -0.43 [-4.29, +3.43] (1/3 seeds higher) |
| r=4 | Dice +0.093 [-0.107, +0.293] (3/3 seeds higher); IPE -7.48 [-19.64, +4.67] (0/3 seeds higher) | Dice -0.020 [-0.028, -0.012] (0/3 seeds higher); IPE -1.50 [-4.04, +1.03] (0/3 seeds higher) |
| r=8 | Dice +0.023 [-0.306, +0.352] (1/3 seeds higher); IPE -7.01 [-19.68, +5.66] (0/3 seeds higher) | Dice -0.089 [-0.357, +0.179] (0/3 seeds higher); IPE -0.44 [-4.77, +3.90] (2/3 seeds higher) |
| r=16 | Dice +0.045 [-0.229, +0.319] (1/3 seeds higher); IPE -6.39 [-19.07, +6.29] (0/3 seeds higher) | Dice -0.077 [-0.353, +0.199] (1/3 seeds higher); IPE +1.23 [-5.74, +8.21] (2/3 seeds higher) |
| r=32 | Dice +0.119 [-0.166, +0.404] (2/3 seeds higher); IPE -8.18 [-23.52, +7.15] (0/3 seeds higher) | Dice -0.008 [-0.039, +0.023] (1/3 seeds higher); IPE -0.63 [-0.75, -0.51] (0/3 seeds higher) |

### MobileSAM

| Rank, configuration | n | Mean Dice | AJI | BF1 | Global IPE (%) | Dice sphero | Dice stom | Runs with a missing class |
|---|---|---|---|---|---|---|---|---|
| r=2, Dice only | 3 | 0.741 ± 0.123 | 0.219 ± 0.036 | 0.779 ± 0.035 | 5.35 ± 6.63 | 0.565 ± 0.489 | 0.710 ± 0.008 | 1/3 |
| r=2, + PMC | 3 | 0.805 ± 0.007 | 0.228 ± 0.026 | 0.800 ± 0.003 | 1.55 ± 0.12 | 0.846 ± 0.006 | 0.697 ± 0.027 | 0/3 |
| r=2, + PMC + BGA | 3 | 0.740 ± 0.125 | 0.237 ± 0.026 | 0.776 ± 0.036 | 5.32 ± 6.59 | 0.572 ± 0.495 | 0.698 ± 0.006 | 1/3 |
| r=2, full | 3 | 0.808 ± 0.003 | 0.264 ± 0.013 | 0.801 ± 0.006 | 1.30 ± 0.09 | 0.848 ± 0.002 | 0.694 ± 0.013 | 0/3 |
| r=4, Dice only | 3 | 0.811 ± 0.002 | 0.268 ± 0.025 | 0.801 ± 0.007 | 1.44 ± 0.15 | 0.849 ± 0.006 | 0.713 ± 0.008 | 0/3 |
| r=4, + PMC | 3 | 0.806 ± 0.006 | 0.236 ± 0.025 | 0.799 ± 0.005 | 1.50 ± 0.13 | 0.850 ± 0.009 | 0.695 ± 0.018 | 0/3 |
| r=4, + PMC + BGA | 3 | 0.741 ± 0.122 | 0.247 ± 0.037 | 0.783 ± 0.036 | 5.23 ± 6.58 | 0.564 ± 0.488 | 0.711 ± 0.004 | 1/3 |
| r=4, full | 3 | 0.811 ± 0.004 | 0.268 ± 0.037 | 0.803 ± 0.002 | 1.34 ± 0.07 | 0.855 ± 0.005 | 0.691 ± 0.021 | 0/3 |
| r=8, Dice only | 3 | 0.811 ± 0.005 | 0.248 ± 0.022 | 0.808 ± 0.012 | 1.53 ± 0.27 | 0.849 ± 0.005 | 0.713 ± 0.007 | 0/3 |
| r=8, + PMC | 3 | 0.813 ± 0.009 | 0.242 ± 0.035 | 0.805 ± 0.002 | 1.60 ± 0.24 | 0.852 ± 0.007 | 0.701 ± 0.027 | 0/3 |
| r=8, + PMC + BGA | 3 | 0.813 ± 0.005 | 0.259 ± 0.034 | 0.806 ± 0.004 | 1.48 ± 0.11 | 0.854 ± 0.004 | 0.710 ± 0.018 | 0/3 |
| r=8, full | 3 | 0.810 ± 0.006 | 0.262 ± 0.017 | 0.802 ± 0.011 | 1.32 ± 0.09 | 0.851 ± 0.004 | 0.708 ± 0.020 | 0/3 |
| r=16, Dice only | 3 | 0.813 ± 0.002 | 0.226 ± 0.008 | 0.802 ± 0.006 | 1.54 ± 0.12 | 0.852 ± 0.001 | 0.711 ± 0.009 | 0/3 |
| r=16, + PMC | 3 | 0.813 ± 0.007 | 0.244 ± 0.035 | 0.811 ± 0.006 | 1.58 ± 0.24 | 0.848 ± 0.012 | 0.714 ± 0.010 | 0/3 |
| r=16, + PMC + BGA | 3 | 0.817 ± 0.003 | 0.237 ± 0.015 | 0.807 ± 0.008 | 1.55 ± 0.12 | 0.853 ± 0.004 | 0.719 ± 0.013 | 0/3 |
| r=16, full | 3 | 0.813 ± 0.002 | 0.279 ± 0.054 | 0.801 ± 0.013 | 1.29 ± 0.18 | 0.855 ± 0.002 | 0.696 ± 0.009 | 0/3 |
| r=32, Dice only | 3 | 0.674 ± 0.126 | 0.230 ± 0.037 | 0.770 ± 0.043 | 9.23 ± 6.63 | 0.284 ± 0.492 | 0.722 ± 0.008 | 2/3 |
| r=32, + PMC | 3 | 0.817 ± 0.004 | 0.265 ± 0.017 | 0.814 ± 0.007 | 1.58 ± 0.20 | 0.859 ± 0.006 | 0.713 ± 0.024 | 0/3 |
| r=32, + PMC + BGA | 3 | 0.818 ± 0.007 | 0.266 ± 0.019 | 0.814 ± 0.003 | 1.46 ± 0.16 | 0.858 ± 0.006 | 0.714 ± 0.011 | 0/3 |
| r=32, full | 3 | 0.687 ± 0.234 | 0.283 ± 0.020 | 0.794 ± 0.023 | 1.76 ± 0.53 | 0.574 ± 0.497 | 0.485 ± 0.420 | 1/3 |

Paired differences within a rank (seed-matched; Dice, global IPE in percentage points):

| Rank | Full minus Dice only | Full minus (+ PMC + BGA) = effect of IPP |
|---|---|---|
| r=2 | Dice +0.067 [-0.233, +0.368] (1/3 seeds higher); IPE -4.06 [-20.36, +12.24] (0/3 seeds higher) | Dice +0.068 [-0.238, +0.373] (1/3 seeds higher); IPE -4.02 [-20.23, +12.19] (0/3 seeds higher) |
| r=4 | Dice -0.001 [-0.016, +0.014] (1/3 seeds higher); IPE -0.10 [-0.30, +0.11] (0/3 seeds higher) | Dice +0.070 [-0.227, +0.367] (2/3 seeds higher); IPE -3.89 [-20.06, +12.28] (1/3 seeds higher) |
| r=8 | Dice -0.000 [-0.017, +0.016] (2/3 seeds higher); IPE -0.21 [-0.75, +0.33] (1/3 seeds higher) | Dice -0.003 [-0.030, +0.024] (1/3 seeds higher); IPE -0.16 [-0.41, +0.09] (0/3 seeds higher) |
| r=16 | Dice -0.000 [-0.005, +0.004] (1/3 seeds higher); IPE -0.26 [-0.55, +0.04] (0/3 seeds higher) | Dice -0.004 [-0.010, +0.002] (0/3 seeds higher); IPE -0.26 [-0.67, +0.14] (0/3 seeds higher) |
| r=32 | Dice +0.014 [-0.481, +0.509] (2/3 seeds higher); IPE -7.48 [-23.20, +8.24] (0/3 seeds higher) | Dice -0.130 [-0.728, +0.468] (2/3 seeds higher); IPE +0.29 [-1.32, +1.91] (1/3 seeds higher) |

### MobileNet-UNet

| Rank, configuration | n | Mean Dice | AJI | BF1 | Global IPE (%) | Dice sphero | Dice stom | Runs with a missing class |
|---|---|---|---|---|---|---|---|---|
| r=2, Dice only | 3 | 0.416 ± 0.004 | 0.579 ± 0.027 | 0.790 ± 0.004 | 17.98 ± 0.40 | 0.000 ± 0.000 | 0.000 ± 0.000 | 3/3 |
| r=2, + PMC | 3 | 0.413 ± 0.181 | 0.570 ± 0.050 | 0.781 ± 0.030 | 19.88 ± 6.25 | 0.000 ± 0.000 | 0.233 ± 0.403 | 3/3 |
| r=2, + PMC + BGA | 3 | 0.549 ± 0.112 | 0.647 ± 0.027 | 0.803 ± 0.015 | 12.40 ± 6.34 | 0.287 ± 0.497 | 0.232 ± 0.402 | 3/3 |
| r=2, full | 3 | 0.410 ± 0.004 | 0.534 ± 0.029 | 0.816 ± 0.006 | 2.61 ± 0.23 | 0.000 ± 0.000 | 0.000 ± 0.000 | 3/3 |
| r=4, Dice only | 3 | 0.482 ± 0.109 | 0.629 ± 0.017 | 0.799 ± 0.014 | 16.52 ± 2.83 | 0.000 ± 0.000 | 0.240 ± 0.417 | 3/3 |
| r=4, + PMC | 3 | 0.417 ± 0.185 | 0.595 ± 0.075 | 0.781 ± 0.032 | 19.39 ± 6.20 | 0.000 ± 0.000 | 0.240 ± 0.416 | 3/3 |
| r=4, + PMC + BGA | 3 | 0.490 ± 0.224 | 0.560 ± 0.060 | 0.790 ± 0.043 | 14.70 ± 10.39 | 0.286 ± 0.495 | 0.242 ± 0.419 | 3/3 |
| r=4, full | 3 | 0.406 ± 0.002 | 0.560 ± 0.035 | 0.805 ± 0.006 | 2.50 ± 0.19 | 0.000 ± 0.000 | 0.000 ± 0.000 | 3/3 |
| r=8, Dice only | 3 | 0.485 ± 0.294 | 0.623 ± 0.111 | 0.814 ± 0.054 | 15.48 ± 12.73 | 0.282 ± 0.489 | 0.233 ± 0.404 | 2/3 |
| r=8, + PMC | 3 | 0.414 ± 0.181 | 0.585 ± 0.071 | 0.782 ± 0.022 | 19.79 ± 6.71 | 0.000 ± 0.000 | 0.229 ± 0.397 | 3/3 |
| r=8, + PMC + BGA | 3 | 0.544 ± 0.292 | 0.602 ± 0.125 | 0.812 ± 0.052 | 13.96 ± 12.46 | 0.284 ± 0.491 | 0.470 ± 0.407 | 2/3 |
| r=8, full | 3 | 0.412 ± 0.004 | 0.551 ± 0.053 | 0.814 ± 0.003 | 2.60 ± 0.07 | 0.000 ± 0.000 | 0.000 ± 0.000 | 3/3 |
| r=16, Dice only | 3 | 0.550 ± 0.297 | 0.619 ± 0.089 | 0.815 ± 0.054 | 14.18 ± 12.75 | 0.283 ± 0.491 | 0.491 ± 0.425 | 2/3 |
| r=16, + PMC | 3 | 0.413 ± 0.176 | 0.558 ± 0.098 | 0.781 ± 0.018 | 20.13 ± 5.77 | 0.000 ± 0.000 | 0.230 ± 0.399 | 3/3 |
| r=16, + PMC + BGA | 3 | 0.612 ± 0.197 | 0.642 ± 0.033 | 0.816 ± 0.047 | 11.50 ± 8.96 | 0.284 ± 0.492 | 0.481 ± 0.417 | 2/3 |
| r=16, full | 3 | 0.414 ± 0.002 | 0.505 ± 0.012 | 0.805 ± 0.009 | 3.09 ± 0.65 | 0.000 ± 0.000 | 0.000 ± 0.000 | 3/3 |
| r=32, Dice only | 3 | 0.359 ± 0.110 | 0.592 ± 0.023 | 0.783 ± 0.020 | 20.83 ± 5.37 | 0.000 ± 0.000 | 0.000 ± 0.000 | 3/3 |
| r=32, + PMC | 3 | 0.486 ± 0.102 | 0.599 ± 0.033 | 0.800 ± 0.013 | 15.96 ± 2.29 | 0.000 ± 0.000 | 0.243 ± 0.421 | 3/3 |
| r=32, + PMC + BGA | 3 | 0.418 ± 0.185 | 0.568 ± 0.078 | 0.788 ± 0.027 | 19.54 ± 6.58 | 0.000 ± 0.000 | 0.244 ± 0.422 | 3/3 |
| r=32, full | 3 | 0.476 ± 0.108 | 0.541 ± 0.049 | 0.824 ± 0.014 | 5.83 ± 5.75 | 0.000 ± 0.000 | 0.236 ± 0.408 | 3/3 |

Paired differences within a rank (seed-matched; Dice, global IPE in percentage points):

| Rank | Full minus Dice only | Full minus (+ PMC + BGA) = effect of IPP |
|---|---|---|
| r=2 | Dice -0.006 [-0.024, +0.013] (1/3 seeds higher); IPE -15.37 [-16.63, -14.12] (0/3 seeds higher) | Dice -0.139 [-0.410, +0.132] (0/3 seeds higher); IPE -9.80 [-25.71, +6.12] (0/3 seeds higher) |
| r=4 | Dice -0.075 [-0.349, +0.198] (0/3 seeds higher); IPE -14.02 [-20.74, -7.31] (0/3 seeds higher) | Dice -0.083 [-0.644, +0.478] (1/3 seeds higher); IPE -12.20 [-38.39, +13.99] (0/3 seeds higher) |
| r=8 | Dice -0.073 [-0.796, +0.650] (1/3 seeds higher); IPE -12.88 [-44.66, +18.90] (1/3 seeds higher) | Dice -0.132 [-0.853, +0.588] (1/3 seeds higher); IPE -11.36 [-42.49, +19.77] (1/3 seeds higher) |
| r=16 | Dice -0.136 [-0.871, +0.598] (1/3 seeds higher); IPE -11.09 [-44.34, +22.17] (1/3 seeds higher) | Dice -0.198 [-0.691, +0.295] (0/3 seeds higher); IPE -8.40 [-30.79, +13.98] (1/3 seeds higher) |
| r=32 | Dice +0.117 [-0.156, +0.390] (2/3 seeds higher); IPE -15.00 [-40.27, +10.28] (0/3 seeds higher) | Dice +0.058 [-0.208, +0.324] (1/3 seeds higher); IPE -13.71 [-42.50, +15.08] (0/3 seeds higher) |

