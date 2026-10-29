"""Device policy and run-environment metadata."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import torch


def _visible_physical_ids() -> Optional[List[int]]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    ids = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        if not tok.lstrip("-").isdigit():
            return None  # UUID style; cannot be checked
        ids.append(int(tok))
    return ids


def enforce_gpu_policy(allowed_physical: Iterable[int]) -> None:
    """Refuse to use any GPU outside ``allowed_physical``.

    Physical ids are read from ``CUDA_VISIBLE_DEVICES``. After remapping, the
    process only ever sees indices 0..N-1, so code must address GPUs by those
    *visible* indices (see :func:`resolve_device`). ``QPI_ALLOW_ALL_GPUS=1`` lifts the
    restriction for one launch (an explicit list in CUDA_VISIBLE_DEVICES is still required;
    D-23). When CUDA is not available
    (smoke tests on a laptop) the policy is not applicable.
    """
    allowed = set(allowed_physical)
    if not torch.cuda.is_available():
        return
    vis = _visible_physical_ids()
    if vis is None:
        raise RuntimeError(
            "CUDA is available but CUDA_VISIBLE_DEVICES is not set to explicit ids. "
            f"Export CUDA_VISIBLE_DEVICES={','.join(map(str, sorted(allowed)))} before running."
        )
    if os.environ.get("QPI_ALLOW_ALL_GPUS") == "1":
        # Explicit opt-in for one launch (DECISIONS.md D-23): any physical GPU listed in CUDA_VISIBLE_DEVICES may be used.
        # The list is recorded in run_metadata.json (cuda_visible_devices); the default policy is unchanged.
        return
    bad = [i for i in vis if i not in allowed]
    if bad:
        raise RuntimeError(f"Physical GPU(s) {bad} are not allowed; allowed set is {sorted(allowed)}.")


def resolve_device(gpu: Optional[int]) -> torch.device:
    """Map a *visible* GPU index to a torch device (cpu when CUDA is unavailable)."""
    if gpu is None or not torch.cuda.is_available():
        return torch.device("cpu")
    n = torch.cuda.device_count()
    if not 0 <= gpu < n:
        raise RuntimeError(f"Visible GPU index {gpu} out of range; {n} device(s) visible "
                           f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}).")
    return torch.device(f"cuda:{gpu}")


def _git_revision() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def collect_environment(device: Optional[torch.device] = None) -> Dict:
    import importlib.metadata as md

    def ver(pkg: str) -> Optional[str]:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_revision": _git_revision(),
        "packages": {p: ver(p) for p in (
            "numpy", "scipy", "pandas", "scikit-image", "opencv-python-headless", "tifffile",
            "torchvision", "timm", "PyYAML", "matplotlib", "onnx", "onnxruntime", "onnxruntime-gpu",
            "edge_sam", "mobile_sam")},
    }
    if device is not None and device.type == "cuda":
        idx = device.index or 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_total_memory_mb"] = round(torch.cuda.get_device_properties(idx).total_memory / 2**20)
    return info
