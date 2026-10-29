# Benchmarking protocol (v5)

Inference benchmark of the three segmentation models, PyTorch FP32 against ONNX Runtime FP32. This file describes what is measured, how, and how the numbers are summarised. It contains no results; the results are produced by `ops/run_benchmark.sh` and written to `outputs/<mode>/benchmark/` and `manuscript/outputs/`. The decision record is `DECISIONS.md`, D-26.

## 1. Question and scope

How fast is each trained model at inference in the two deployment runtimes, how much GPU memory does it need, and is the ONNX Runtime model the same function as the PyTorch model.

In scope: six combinations, three models by two runtimes, all in FP32.

| Model | Input size (px) | Runtimes |
|---|---|---|
| EdgeSAM | 256 | PyTorch FP32, ONNX Runtime FP32 |
| MobileSAM | 1024 | PyTorch FP32, ONNX Runtime FP32 |
| MobileNet-UNet | 256 | PyTorch FP32, ONNX Runtime FP32 |

Out of scope and not run: FP16, autocast and any mixed precision, TF32 as a result, CPU inference, TensorRT, CUDA graphs, ONNX Runtime algorithm tuning, end-to-end pipeline time (pre- and post-processing, cell analysis), random-weight timing, batch sizes other than 1, other GPUs.

## 2. What is timed

- The trained reference model: `best.pt` of the selected LoRA rank (8), loaded with `strict=True`, LoRA merged into the base weights (deployment form). After merging, the network has the same shapes and cost for every rank, so latency and memory do not depend on the rank and are measured once. The checkpoint path, epoch and a 16-character SHA-256 are written to every row. A missing checkpoint stops the run.
- Batch size 1.
- Each model at its own deployment input size (table above). MobileSAM runs at 1024 px and the other two at 256 px, so a latency difference between architectures mixes architecture and resolution.
- The forward pass only. No data loading, no pre-processing, no post-processing, no resize of the logits, no accuracy computation inside any timed loop.
- Inputs are 16 distinct real validation phase maps (chosen by a seed-dependent permutation of the validation set), kept on the GPU and cycled through. PyTorch receives GPU tensors. ONNX Runtime receives GPU-resident `OrtValue` inputs through IO binding and writes its output to a GPU buffer, so neither runtime is charged for host-device copies.

## 3. Hardware and software

Recorded in `benchmark.json` (`host_details`, `environment`) for every run. For the server run: NVIDIA RTX A5000, Python 3.10, PyTorch 2.0.1 (CUDA 11.7), ONNX Runtime 1.16.3 (CUDA execution provider), onnx 1.16.2. Physical GPU 1 by default (`GPU=1`); a GPU outside the default policy lifts the policy for that launch (D-23) and the GPU is recorded. Other jobs must not run on the GPU during the benchmark; the GPU clock, temperature and utilisation are read from `nvidia-smi` before and after each timed loop and stored in the per-seed pass files.

## 4. FP32 means FP32 in both runtimes

On Ampere GPUs, PyTorch convolutions use TF32 by default and ONNX Runtime's CUDA provider uses TF32 for convolutions and matrix products, so an unmanaged "FP32" comparison compares different arithmetic. The benchmark therefore runs in strict FP32:

- PyTorch: `cudnn.allow_tf32 = False`, `cuda.matmul.allow_tf32 = False`, `float32_matmul_precision = "highest"`.
- ONNX Runtime: `NVIDIA_TF32_OVERRIDE=0` in the environment of every process before any CUDA library loads (exported by `ops/run_benchmark.sh`, also set at the top of the scripts).
- `TF32=1 ./ops/run_benchmark.sh` switches TF32 on in both runtimes; the effective state is written to `benchmark.json` (`protocol.tf32`). The strict setting is the one used for reported results.
- `cudnn.benchmark` is off and the training-time determinism switches are off, so PyTorch uses its library defaults.

## 5. Procedure

One run of `scripts/run_benchmark.py` executes, for each seed in 42, 43, 44, one pass in its own process, one pass after the other on the same GPU (no concurrent passes, so no CPU or GPU contention between them). The order of the architectures is rotated between passes.

Inside a pass, before any timing, the CUDA libraries (cuDNN and cuBLAS kernels, the ONNX Runtime CUDA provider) are loaded once, so the first model does not pay for them in its memory figure.

For each architecture and runtime:

1. Build the model with the seed fixed (`seed_everything(seed, deterministic=False)`), load `best.pt`, merge LoRA, set `eval()`.
2. Draw the 16 inputs for the seed and move them to the GPU.
3. Time the loop: 50 unrecorded warm-up passes, then 500 timed passes. Every timed pass is `synchronize`, `perf_counter`, forward, `synchronize`, `perf_counter`. Latency is the difference in ms.

PyTorch: `torch.inference_mode()`, plain `model(x)`.

ONNX Runtime:
- The PyTorch model is exported once per pass with `torch.onnx.export` (opset 17, constant folding). Two export-only patches apply while exporting: `Tensor.unflatten` and `scaled_dot_product_attention` are replaced by mathematically identical operations the exporter can trace; `nn.MultiheadAttention` modules are exported through their ordinary (non-fast-path) graph. PyTorch timings use the stock kernels.
- Before the sessions are timed, the PyTorch outputs of 4 inputs are stored for the correctness check in section 8.
- Three fresh `InferenceSession`s are created on the one exported file, one after the other. Options: `ORT_ENABLE_ALL` graph optimisation, CUDA provider with `arena_extend_strategy = kSameAsRequested` (the memory pool grows by what is requested, not by powers of two), CPU provider listed only as fallback, default cuDNN convolution algorithm search. The session must report the CUDA provider or the run stops.
- Each session gets its own 50 warm-up and 500 timed passes. A fresh session is used because the time of a session depends on the convolution algorithms cuDNN picks when it is created; one session would hide that variation.

If any export or session fails, the run stops. No combination is skipped silently.

## 6. Metrics

All four are per model and runtime, computed per seed first.

| Metric | Definition |
|---|---|
| P50 (ms) | 50th percentile of the timed per-inference latencies of the seed. PyTorch: 500 runs. ONNX Runtime: the 3 sessions of the seed pooled, 1500 runs. |
| P99 (ms) | 99th percentile of the same latencies. |
| FPS | 1000 / P50 of the same seed. A latency-based rate for batch 1, not a throughput measured with a larger batch or a pipeline. |
| Peak VRAM (MB) | Used GPU memory read at the device (`cudaMemGetInfo`) after the timed loop, minus the used memory read before the model was placed on the GPU, after garbage collection and `empty_cache`. The CUDA context and the libraries loaded before the baseline are excluded. ONNX Runtime: the largest value over the 3 sessions of the seed. |

Notes on Peak VRAM. It is one measure for both runtimes, so the two columns are comparable. It is the high-water footprint on the assumption that neither the PyTorch caching allocator nor the ONNX Runtime arena returns memory to the driver while the loop runs; this is how both behave by default, but it is not read from a peak counter. It includes the weights, activations, workspace, the 16 GPU-resident inputs and any library kernel loaded on first use. The PyTorch allocator counters (`max_memory_allocated`) are not used because they do not see ONNX Runtime's memory.

## 7. Aggregation over seeds

- Percentiles are taken inside one seed. Latency samples of different seeds are not pooled.
- The summary row is the mean and the sample standard deviation (n - 1) over the three per-seed values of P50, P99, FPS and Peak VRAM. The summary FPS is the mean of the per-seed FPS, not 1000 / mean P50.
- With three seeds the SD describes the spread between the seeds; it is not a confidence interval.
- In the tables P50 and P99 are shown as mean ± SD. FPS is shown as the smallest and largest per-seed value (one number when all seeds agree); FPS is 1000 / P50, so an SD of FPS would repeat the P50 spread. Peak VRAM is shown as one value, the largest over the three seeds and, for ONNX Runtime, over all fresh sessions (a conservative memory requirement; it is identical across seeds for PyTorch). All per-seed and per-session memory values stay in the JSON and CSV. The JSON and CSV keep the means and SDs of all four metrics.
- The per-seed values are reported next to the summary (`benchmark_per_seed.csv`, per-seed table in the tex report). The three ONNX Runtime session medians of each seed are stored in `benchmark.json` (`per_seed[].session_p50_ms`).
- A seed changes the 16 input maps and the process; it does not change the trained weights, which are the same checkpoint in all three passes.

## 8. Correctness and quality gate (outside every timed loop)

Speed is only reported for a model that computes the same function in both runtimes.

1. In the benchmark, for each seed, the ONNX Runtime outputs on 4 inputs are compared with the stored PyTorch outputs: arg-max agreement and the largest absolute logit difference (`argmax_agreement_pct`, `max_abs_logit_diff` in the ONNX rows).
2. `scripts/check_equivalence.py` (`qpi/equivalence.py`), run once after the benchmark with the same checkpoints, strict FP32:
   - Logits of both models on the 16 validation maps (seed 42 selection): arg-max agreement, maximum and mean absolute logit difference, non-finite values, and whether every disagreeing pixel is a near-tie (margin of the two top PyTorch logits within twice the largest logit difference).
   - Segmentation quality on every validation image (rotation 0; 66 images in the full run): mean Dice against the annotation for both models, computed by the evaluation code of the paper, and the per-image total dry mass of the predicted cells, computed by the analysis code, of the ONNX model relative to the PyTorch model.
3. Acceptance gate, fixed before the first trial and unchanged: argmax agreement >= 99.9 %, |change in mean Dice| <= 0.001, per-image dry-mass relative error mean <= 0.1 % and 99th percentile <= 0.5 %. A rejected model fails the run (exit status 1) and `scripts/verify_outputs.py`.

With 66 images, the 99th percentile of the dry-mass error is set by the worst image. The smallest reference dry mass and the worst image are stored, so a near-zero denominator would be visible.

## 9. Outputs

In `outputs/<mode>/benchmark/` (replaced at the start of every run):

| File | Content |
|---|---|
| `benchmark.json` | protocol and definitions, environment and host, summary rows, per-seed rows, trainable parameters by rank |
| `benchmark.csv` | summary: P50, P99, FPS, Peak VRAM, each with SD over seeds |
| `benchmark_per_seed.csv` | the same four metrics for each seed |
| `benchmark_parameters_by_rank.csv` | trainable-parameter counts per architecture and rank (structure only) |
| `passes/pass_seed<seed>.json` | raw per-run latencies, GPU state, environment for each seed |
| `equivalence.json` | section 8 results and the gate verdict |

Then `ops/make_manuscript_outputs.sh` rebuilds `manuscript/outputs/results.json`, the figures and `multiseed_results.tex` from these files; no benchmark number is typed by hand. `scripts/verify_outputs.py` checks protocol version 5, exactly six combinations for each seed, finite positive P50, P99, FPS and Peak VRAM, FPS = 1000 / P50, P99 >= P50, that `results.json` equals `benchmark.json`, that `equivalence.json` covers every architecture with the validation image count, and that no model was rejected.

## 10. How to reproduce

```
GPU=1 SEEDS="42 43 44" ./ops/run_benchmark.sh
```

This runs the benchmark, the equivalence check, the regeneration of the manuscript outputs and the verification. Tests: `python -m pytest tests -q`.

## 11. What may and may not be claimed

May be reported: per model and runtime the mean and SD over three seeds of P50, P99, FPS and Peak VRAM on this GPU, with the per-seed values; the ONNX Runtime to PyTorch ratio for the same model; that ONNX FP32 passed the gate (or which model failed it).

Must not be claimed:
- Results for other GPUs, other drivers or other library versions, for CPU, for FP16 or mixed precision, or for a full pipeline. FP16 was examined in a one-seed trial and removed from the benchmark; it is not part of this protocol.
- Any cause for a runtime difference that was not tested. In particular, why ONNX Runtime is slower than PyTorch for MobileSAM, why its memory footprint is larger, and why EdgeSAM ONNX Runtime times vary between sessions (a cuDNN algorithm-selection effect is a hypothesis) are open.
- A latency ranking of architectures as an architecture effect: MobileSAM is timed at 1024 px and the others at 256 px.
- Accuracy from the timing runs. The benchmark does not compute accuracy; accuracy and dry-mass agreement come from the separate gate in section 8, which compares ONNX with PyTorch and does not re-validate the models against the annotation beyond the Dice change.

## 12. Known limitations

- One GPU model, one machine, one software stack.
- Some models do not keep one speed level. EdgeSAM in PyTorch and in ONNX Runtime, and to a smaller degree MobileNet-UNet, are seen at two levels (for example EdgeSAM ONNX Runtime about 3.5 ms in one session and about 4.4 ms in another session of the same process, same file, same options). The cause has not been tested. The P50 mean ± SD of such a model describes this variation, not a single stable speed. The ranking between runtimes was the same in every seed; the size of the EdgeSAM speed-up was not.
- Three seeds and, for ONNX Runtime, three sessions per seed. If sessions of the same model fall at different levels, the SD over seeds reflects that variation, and the individual session medians should be read from `benchmark.json`.
- Peak VRAM is a device-level reading with the high-water assumption of section 6, and it includes the 16 GPU-resident inputs (about 4 MB each for MobileSAM at 1024 px).
- Latency is measured with a synchronised loop on batch 1; it includes the CPU time to launch the kernels. Small models can be limited by that launch time rather than by the GPU, so their latency depends on the host CPU.
- The trained checkpoint is the one-seed single run (seed 42) at rank 8; the benchmark does not time the checkpoints of the multi-seed runs.
- `configs/base.yaml` and `configs/smoke.yaml` still contain `benchmark.precisions` and `benchmark.onnx_skip`; the code does not read them.

## 13. Earlier protocols

Earlier benchmark protocols (random weights, FP16, pooling of latency samples across seeds, torch-allocator memory for ONNX Runtime) are superseded. Their numbers are not comparable and are not used.
