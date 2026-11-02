# S12_error_strata: Where segmentation errors occur: boundary distance and phase quartile

| Configuration | pixel error, band 0-2 px | pixel error, band 2-4 px | pixel error, band 4-8 px | FN rate, phase Q Q1 | FN rate, phase Q Q2 | FN rate, phase Q Q3 | FN rate, phase Q Q4 |
|---|---|---|---|---|---|---|---|
| MobileSAM r = 8 | 28.9 ± 0.4 | 5.4 ± 0.4 | 0.5 ± 0.1 | 23.2 ± 0.8 | 0.6 ± 0.1 | 0.2 ± 0.0 | 0.2 ± 0.1 |
| EdgeSAM r = 32 | 35.2 ± 0.1 | 13.8 ± 0.2 | 3.5 ± 0.3 | 28.0 ± 2.5 | 5.6 ± 0.7 | 3.5 ± 0.6 | 2.3 ± 0.7 |
| EdgeSAM r = 8 | 36.4 ± 0.1 | 15.7 ± 0.3 | 4.3 ± 0.4 | 26.0 ± 5.1 | 5.5 ± 1.4 | 3.6 ± 1.5 | 2.8 ± 1.7 |
| MobileNet-UNet r = 8 | 28.3 ± 0.2 | 5.4 ± 0.1 | 0.9 ± 0.1 | 25.7 ± 1.1 | 2.1 ± 0.1 | 0.9 ± 0.1 | 1.3 ± 0.2 |

**Does IPP increase misses of low-phase pixels? False-negative rate in the lowest phase quartile, with and without IPP**

| Configuration | FN rate Q1, Dice+PMC+BGA (%) | FN rate Q1, full (+IPP) (%) | Δ (full − without IPP), pp |
|---|---|---|---|
| MobileSAM r = 8 | 20.8 ± 1.2 | 23.2 ± 0.8 | +2.4 [-1.2, +5.9] |
| EdgeSAM r = 32 | 32.1 ± 2.3 | 28.0 ± 2.5 | -4.0 [-5.3, -2.8] |
| EdgeSAM r = 8 | 33.0 ± 3.2 | 26.0 ± 5.1 | -7.0 [-20.9, +6.9] |
| MobileNet-UNet r = 8 | 26.8 ± 5.3 | 25.7 ± 1.1 | -1.1 [-16.8, +14.5] |

Notes:

* Pixel error rate (%) in distance bands from the annotated boundary; false-negative rate (%) inside annotated cells by per-image quartile of background-corrected phase (Q1 = lowest phase). Errors concentrate at the boundary and in the lowest-phase quartile.
