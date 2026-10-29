# T8_efficiency: Computational efficiency (NVIDIA RTX A5000, batch 1, strict FP32, 3 seeds)

| Architecture | Runtime | Input (px) | Trainable params, preferred rank | Trainable params, r = 8 | P50 (ms) | P99 (ms) | FPS (per-seed range) | Peak VRAM (MB, max) |
|---|---|---|---|---|---|---|---|---|
| MobileSAM | PyTorch FP32 | 1024 | 0.67 M (r = 8) | 0.67 M | 20.35 ± 0.15 | 20.51 ± 0.13 | 49-50 | 358 |
| MobileSAM | ONNX Runtime FP32 | 1024 | 0.67 M (r = 8) | 0.67 M | 25.43 ± 0.21 | 25.56 ± 0.18 | 39-40 | 1178 |
| EdgeSAM | PyTorch FP32 | 256 | 1.47 M (r = 32) | 0.53 M | 10.64 ± 1.45 | 12.97 ± 1.53 | 81-103 | 42 |
| EdgeSAM | ONNX Runtime FP32 | 256 | 1.47 M (r = 32) | 0.53 M | 4.11 ± 0.43 | 5.58 ± 0.31 | 229-277 | 132 |
| MobileNet-UNet | PyTorch FP32 | 256 | 5.58 M (r = 8) | 5.58 M | 5.66 ± 0.10 | 8.00 ± 0.63 | 175-180 | 52 |
| MobileNet-UNet | ONNX Runtime FP32 | 256 | 5.58 M (r = 8) | 5.58 M | 2.53 ± 0.19 | 3.33 ± 0.79 | 374-431 | 152 |

Notes:

* LoRA is merged into the weights before timing, so latency and memory do not depend on the rank: one measurement per architecture covers every rank (trainable parameters differ, inference cost does not).
* MobileSAM is timed at 1024 px and the other two at 256 px (each model's native input), so the latency gap mixes architecture and input resolution.
* P50/P99: mean ± SD over three seeds (500 timed passes each after 50 warm-up; ONNX: 3 fresh sessions per seed). FPS = 1000/P50. Peak VRAM: device memory after the timed loop minus before model placement; largest value over seeds and sessions. Protocol: manuscript/benchmarking_protocol.md (v5).
* ONNX Runtime FP32 passed the equivalence gate for all three models (Table S9).
