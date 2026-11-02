# S14_reference_sensitivity: Global IPE under different background references

| Configuration | Background from annotation (%) | Background from prediction (%) | No reference, phase as provided (%) | Residual background, max |rad| |
|---|---|---|---|---|
| MobileSAM r = 8 | 1.32 ± 0.09 | 1.32 ± 0.09 | 2.06 ± 0.16 | 0.0112 ± 0.0026 |
| EdgeSAM r = 32 | 3.02 ± 0.84 | 3.02 ± 0.85 | 3.53 ± 0.89 | 0.0128 ± 0.0076 |
| EdgeSAM r = 8 | 3.81 ± 1.82 | 3.81 ± 1.82 | 4.82 ± 1.48 | 0.0093 ± 0.0051 |
| MobileNet-UNet r = 8 | 2.60 ± 0.07 | 2.60 ± 0.07 | 3.83 ± 0.16 | 0.0123 ± 0.0031 |

**Phase offsets removed by the background referencing (rad)**

```
{
 "date_named": {
  "n_images": 66,
  "min_rad": 0.661676,
  "median_rad": 1.499129,
  "mean_rad": 1.5116169696969695,
  "max_rad": 2.502313
 }
}
```

Notes:

* The as-provided maps do not share a zero level (validation background 0.66 to 2.50 rad, median 1.50 rad; the training X_* maps are near zero-mean, D-03); integrated phase depends on that constant, so every map is referenced to its own background median before training and evaluation. The choice of reference (annotation vs prediction) changes the IPE by less than 0.01 pp.
