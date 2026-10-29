"""PyTorch dataset over the processed (float-frame) data.

Each item is one *view* of a base image: ``np.rot90(base, -k)`` for k in the split's rotation list
(k = 0 is the original). This reproduces the offline rotation augmentation of the original 1,024-image
training set (files ``rotation<stem>_<k>``) exactly and losslessly.

Returned tensors
    phase      (1, S, S)  network input: per-image z-scored phase at the model resolution S
    phase_raw  (1, S, S)  phase in radians at resolution S (used by the physics terms; never normalised)
    mask       (S, S)     long, 0..4, ``ignore_index`` outside the annotated field of view
    phase_native (1, N, N), mask_native (N, N)   only when ``return_native`` (evaluation, N = 768)
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from qpi.utils.seed import worker_init_fn


def read_manifest(processed_root: Path, split: str) -> List[Dict]:
    path = processed_root / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/prepare_dataset.py first")
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == split and r["registration_ok"] == "True"]
    if not rows:
        raise RuntimeError(f"no usable {split} rows in {path}")
    return sorted(rows, key=lambda r: r["stem"])


class QPIDataset(Dataset):
    def __init__(self, processed_root: Path, split: str, image_size: int, rotations: Sequence[int],
                 ignore_index: int = 255, max_samples: Optional[int] = None, return_native: bool = False):
        self.root = Path(processed_root)
        self.split = split
        self.image_size = int(image_size)
        self.ignore_index = ignore_index
        self.return_native = return_native
        rows = read_manifest(self.root, split)
        if max_samples is not None:
            rows = rows[: int(max_samples)]
        self.rows = rows
        self.items = [(i, int(k)) for i in range(len(rows)) for k in rotations]
        for _, k in self.items:
            if k not in (0, 1, 2, 3):
                raise ValueError(f"rotation index must be in 0..3, got {k}")

    def __len__(self) -> int:
        return len(self.items)

    def sample_id(self, idx: int) -> str:
        i, k = self.items[idx]
        return self.rows[i]["stem"] if k == 0 else f"rotation{self.rows[i]['stem']}_{k}"

    def __getitem__(self, idx: int) -> Dict:
        i, k = self.items[idx]
        row = self.rows[i]
        phase = tifffile.imread(self.root / self.split / "phase" / f"{row['stem']}.tif").astype(np.float32)
        mask = tifffile.imread(self.root / self.split / "mask" / f"{row['stem']}.tif").astype(np.uint8)
        if k:
            phase = np.ascontiguousarray(np.rot90(phase, -k))
            mask = np.ascontiguousarray(np.rot90(mask, -k))
        ph = torch.from_numpy(phase)[None]
        mk = torch.from_numpy(mask.astype(np.int64))
        out = {"stem": row["stem"], "sample_id": self.sample_id(idx), "rotation": k,
               "storage_day": int(row["storage_day"])}
        if self.return_native:
            out["phase_native"] = ph
            out["mask_native"] = mk
        S = self.image_size
        if S != ph.shape[-1]:
            ph_s = F.interpolate(ph[None], size=(S, S), mode="bilinear", align_corners=False,
                                 antialias=S < ph.shape[-1])[0]
            # "nearest-exact" samples pixel centres (the same half-pixel convention as the bilinear phase resize);
            # plain "nearest" takes the top-left pixel of each block and shifts the mask by ~(scale-1)/2 native pixels
            mk_s = F.interpolate(mk[None, None].float(), size=(S, S), mode="nearest-exact")[0, 0].long()
        else:
            ph_s, mk_s = ph, mk
        out["phase_raw"] = ph_s
        out["phase"] = (ph_s - ph_s.mean()) / ph_s.std().clamp(min=1e-8)
        out["mask"] = mk_s
        return out

    def dominant_classes(self) -> List[int]:
        """Dominant foreground class (1..4: most annotated pixels) of the base image behind every item; 0 if none."""
        cache: Dict[int, int] = {}
        out: List[int] = []
        for i, _ in self.items:
            if i not in cache:
                m = tifffile.imread(self.root / self.split / "mask" / f"{self.rows[i]['stem']}.tif")
                cnt = np.bincount(m[(m >= 1) & (m <= 4)].astype(np.int64), minlength=5)[1:5]
                cache[i] = int(cnt.argmax()) + 1 if cnt.sum() > 0 else 0
            out.append(cache[i])
        return out

    def dominant_class_weights(self) -> Tuple["torch.Tensor", Dict]:
        """Per-item sampling weights = 1 / (number of items whose base image has the same dominant class).

        Multi-class images have no single label, so the image label is its dominant foreground class (see DECISIONS.md D-18).
        Rotated views share the label of their base image. Returns (weights, summary).
        """
        labels = np.array(self.dominant_classes())
        counts = np.bincount(labels, minlength=5).astype(np.float64)
        w = 1.0 / np.maximum(counts[labels], 1.0)
        eff = np.bincount(labels, weights=w, minlength=5)
        return torch.as_tensor(w, dtype=torch.double), {
            "label_definition": "dominant foreground class of the base image (most annotated pixels among classes 1-4)",
            "items_per_label": {str(c): int(counts[c]) for c in range(5)},
            "expected_sampling_share_per_label": {str(c): float(eff[c] / eff.sum()) for c in range(5)}}


def build_datasets(cfg: Dict, image_size: int, return_native_val: bool = False):
    d, root = cfg["data"], cfg["paths_abs"]["processed_root"]
    train = QPIDataset(root, "train", image_size, d["train_rotations"], d["ignore_index"], d["max_train_samples"])
    val = QPIDataset(root, "val", image_size, d["val_rotations"], d["ignore_index"], d["max_val_samples"],
                     return_native=return_native_val)
    return train, val


def build_loaders(cfg: Dict, run_cfg: Dict, seed: int):
    """Plain shuffled training loader (seeded generator) and an ordered validation loader."""
    train, val = build_datasets(cfg, run_cfg["image_size"])
    g = torch.Generator()
    g.manual_seed(seed)
    nw = cfg["data"]["num_workers"]
    common = dict(num_workers=nw, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn,
                  persistent_workers=nw > 0)
    drop_last = cfg["training"]["drop_last"] and len(train) >= run_cfg["batch_size"]
    sampler_name = run_cfg["training"].get("sampler", "shuffle")
    if sampler_name == "shuffle":
        train_loader = DataLoader(train, batch_size=run_cfg["batch_size"], shuffle=True, generator=g,
                                  drop_last=drop_last, **common)
        train_loader.sampler_info = {"sampler": "shuffle"}
    elif sampler_name == "weighted_dominant_class":
        w, info = train.dominant_class_weights()
        sampler = WeightedRandomSampler(w, num_samples=len(train), replacement=True, generator=g)
        train_loader = DataLoader(train, batch_size=run_cfg["batch_size"], sampler=sampler, shuffle=False,
                                  drop_last=drop_last, **common)
        train_loader.sampler_info = {"sampler": sampler_name, **info}
    else:
        raise ValueError(f"unknown sampler {sampler_name!r}")
    val_loader = DataLoader(val, batch_size=run_cfg["batch_size"], shuffle=False, **common)
    return train_loader, val_loader
