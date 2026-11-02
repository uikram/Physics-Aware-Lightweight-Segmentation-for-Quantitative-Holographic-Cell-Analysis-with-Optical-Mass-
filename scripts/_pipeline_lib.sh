# Shared helpers for run_full_experiment.sh and run_smoke_test.sh (sourced, not executed).
# shellcheck shell=bash

PIPELINE_START=$(date +%s)

# Always use the Python of the ACTIVE conda environment (a non-interactive `bash script.sh` can otherwise pick up the
# base Anaconda python). PYTHON_BIN_DIR can be set to force a specific interpreter directory.
if [ -n "${PYTHON_BIN_DIR:-}" ]; then
  export PATH="$PYTHON_BIN_DIR:$PATH"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
  export PATH="$CONDA_PREFIX/bin:$PATH"
fi

log()  { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die()  { printf '\n[FATAL] %s\n' "$*" >&2; exit 1; }
on_exit() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '\n[FATAL] pipeline stopped (exit %s) in step: %s\n' "$rc" "${CURRENT_STEP:-unknown}" >&2
  fi
}
trap on_exit EXIT

step() {                       # step "<name>" cmd args...
  CURRENT_STEP="$1"; shift
  local t0; t0=$(date +%s)
  log ">>> ${CURRENT_STEP}"
  "$@"
  log "<<< ${CURRENT_STEP} ($(( $(date +%s) - t0 )) s)"
}

require_gpu_policy() {
  # Default: only physical GPUs 2 and 3 may be used (visible indices 0 and 1).
  # QPI_ALLOW_ALL_GPUS=1 (set by ./ops/start_multiseed.sh when GPUS is not 2,3; D-23) allows the explicit list in CUDA_VISIBLE_DEVICES.
  if [ "${QPI_ALLOW_ALL_GPUS:-0}" = 1 ]; then
    [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || die "QPI_ALLOW_ALL_GPUS=1 needs an explicit CUDA_VISIBLE_DEVICES list, e.g. 0,1,2,3,4,5,6"
    log "GPU policy lifted for this launch (D-23): physical GPUs ${CUDA_VISIBLE_DEVICES}"
    python - <<'PY'
import os, sys, torch
if not torch.cuda.is_available():
    sys.exit("CUDA is not available")
ids = [t for t in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if t.strip()]
if torch.cuda.device_count() != len(ids):
    sys.exit("expected %d visible GPUs, found %d" % (len(ids), torch.cuda.device_count()))
for i, phys in enumerate(ids):
    p = torch.cuda.get_device_properties(i)
    print("  visible cuda:%d -> physical GPU %s: %s, %.1f GiB" % (i, phys, p.name, p.total_memory / 2**30))
PY
    return 0
  fi
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "2,3" ]; then
    die "CUDA_VISIBLE_DEVICES is '${CUDA_VISIBLE_DEVICES}'. This pipeline must run with CUDA_VISIBLE_DEVICES=2,3 (never GPUs 0 or 1). Unset it or set it to 2,3."
  fi
  export CUDA_VISIBLE_DEVICES=2,3
  python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("CUDA is not available: cannot run on GPUs 2,3")
n = torch.cuda.device_count()
if n != 2:
    sys.exit(f"expected exactly 2 visible GPUs under CUDA_VISIBLE_DEVICES=2,3, found {n}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  visible cuda:{i} -> physical GPU {2 + i}: {p.name}, {p.total_memory / 2**30:.1f} GiB")
PY
}

check_python_env() {
  python - <<'PY'
import importlib, sys
need = ["torch", "torchvision", "timm", "numpy", "scipy", "pandas", "skimage", "cv2", "tifffile", "yaml",
        "matplotlib", "onnx", "onnxruntime", "mobile_sam"]
bad = []
for m in need:
    try:
        importlib.import_module(m)
    except Exception as e:                       # noqa: BLE001
        bad.append(f"{m}: {e}")
if bad:
    sys.exit("missing/broken python packages:\n  " + "\n  ".join(bad) + "\nInstall with: pip install -r requirements.txt")
import torch, onnxruntime as ort
print(f"  interpreter {sys.executable}")
print(f"  python {sys.version.split()[0]}  torch {torch.__version__}  onnxruntime providers {ort.get_available_providers()}")
PY
}

check_weights() {
  for f in weights/edge_sam_3x.pth weights/mobile_sam.pt; do
    [ -s "$f" ] || die "missing pretrained weights: $f (see README, 'Weights')"
  done
  python - <<'PY'
# ImageNet weights of MobileNetV2 (torchvision) are fetched into cache/ on first use; fail early if that is impossible.
import os, sys
from pathlib import Path
os.environ.setdefault("TORCH_HOME", str(Path("cache").resolve()))
import torchvision
try:
    torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
except Exception as e:                           # noqa: BLE001
    sys.exit("cannot obtain MobileNetV2 ImageNet weights (no network?): %s\n"
             "Copy mobilenet_v2-b0353104.pth to cache/hub/checkpoints/ and retry." % e)
print("  MobileNetV2 ImageNet weights available")
PY
}

pipeline_footer() {
  log "pipeline finished successfully in $(( $(date +%s) - PIPELINE_START )) s"
}
