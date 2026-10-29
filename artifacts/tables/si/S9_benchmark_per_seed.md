# S9_benchmark_per_seed: Benchmark per seed and ONNX Runtime equivalence gate

| Architecture | Runtime | Seed | P50 (ms) | P99 (ms) | FPS | Peak VRAM (MB) |
|---|---|---|---|---|---|---|
| EdgeSAM | onnx | 42 | 3.61 | 5.22 | 277.0 | 132 |
| EdgeSAM | onnx | 43 | 4.37 | 5.74 | 229.1 | 100 |
| EdgeSAM | onnx | 44 | 4.35 | 5.77 | 229.9 | 124 |
| EdgeSAM | pytorch | 42 | 9.88 | 12.82 | 101.2 | 42 |
| EdgeSAM | pytorch | 43 | 12.31 | 14.57 | 81.3 | 42 |
| EdgeSAM | pytorch | 44 | 9.72 | 11.53 | 102.8 | 42 |
| MobileSAM | onnx | 42 | 25.19 | 25.36 | 39.7 | 1178 |
| MobileSAM | onnx | 43 | 25.54 | 25.66 | 39.2 | 1122 |
| MobileSAM | onnx | 44 | 25.57 | 25.67 | 39.1 | 1178 |
| MobileSAM | pytorch | 42 | 20.19 | 20.39 | 49.5 | 358 |
| MobileSAM | pytorch | 43 | 20.47 | 20.64 | 48.8 | 358 |
| MobileSAM | pytorch | 44 | 20.39 | 20.49 | 49.0 | 358 |
| MobileNet-UNet | onnx | 42 | 2.32 | 2.50 | 431.0 | 144 |
| MobileNet-UNet | onnx | 43 | 2.60 | 3.41 | 384.9 | 146 |
| MobileNet-UNet | onnx | 44 | 2.68 | 4.07 | 373.7 | 152 |
| MobileNet-UNet | pytorch | 42 | 5.71 | 7.30 | 175.3 | 52 |
| MobileNet-UNet | pytorch | 43 | 5.72 | 8.20 | 174.7 | 52 |
| MobileNet-UNet | pytorch | 44 | 5.54 | 8.49 | 180.4 | 52 |

**ONNX Runtime FP32 vs PyTorch FP32 (gate fixed before the first trial)**

| Architecture | Arg-max agreement, 16 maps (%) | Max |Δ logit| | Arg-max agreement, 66 images (%) | |Δ mean Dice| | Dry-mass rel. error mean / P99 | Verdict |
|---|---|---|---|---|---|---|
| EdgeSAM | 99.9999 | 2.0e-04 | 100.0000 | 5.7e-07 | 4.5e-07 / 7.5e-06 | accepted |
| MobileSAM | 100.0000 | 3.2e-04 | 100.0000 | 1.1e-07 | 5.2e-08 / 1.5e-06 | accepted |
| MobileNet-UNet | 100.0000 | 2.0e-04 | 100.0000 | 2.3e-08 | 1.3e-08 / 2.9e-07 | accepted |

Notes:

* GPU: NVIDIA RTX A5000. Acceptance: arg-max agreement ≥ 99.9%, |Δ mean Dice| ≤ 0.001, dry-mass relative error mean ≤ 0.1% and P99 ≤ 0.5%.
* The timed and gated checkpoint is the seed-42 rank-8 best.pt of each architecture (D-26); timing does not depend on the weights or on the rank after merging.
