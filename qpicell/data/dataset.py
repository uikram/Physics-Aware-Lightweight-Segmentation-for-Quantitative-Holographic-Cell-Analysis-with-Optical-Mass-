"""PyTorch dataset over data/prepared.

One item = one view of a base image: np.rot90(image, -k) for k in the split's rotations (lossless augmentation that
reproduces the original 4 x 256 training set). Tensors of an item:
  input         (1, S, S) float32  network input: the referenced phase at the model size S, z-scored per image
  phase         (1, S, S) float32  referenced phase in RADIANS at size S (physics-aware losses; never normalised)
  mask          (S, S)   int64     labels 0..4, 255 outside the annotated field of view
  cells, territories, contacts     (C, C) cell maps for the cell-level loss, C = min(S, cell_map_size)
  phase_native, mask_native, cells_native, wrapped_native   (768 x 768) when return_native (evaluation)
Resizing: phase bilinear (antialiased when shrinking); labels 'nearest-exact' (pixel-centre convention).
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from qpicell.data.cells import contact_lines, territories
from qpicell.physics import assert_phase
from qpicell.seeding import worker_init_fn


def read_manifest(prepared: Path, split: str) -> List[Dict]:
    with open(Path(prepared) / "manifest.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == split and r["registration_ok"] == "True"]
    if not rows:
        raise RuntimeError(f"no usable {split} images in {prepared}/manifest.csv")
    return sorted(rows, key=lambda r: r["stem"])


def tuning_fold(stem: str, modulus: int = 5) -> int:
    """Deterministic fold of a base training image (all rotations share it)."""
    return int(hashlib.sha1(stem.encode()).hexdigest(), 16) % modulus


def _resize_labels(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x[None, None].float(), size=(size, size), mode="nearest-exact")[0, 0].long()


class PhaseDataset(Dataset):
    def __init__(self, cfg: Dict, split: str, image_size: int, rotations: Sequence[int], return_native: bool = False,
                 stems: Optional[Sequence[str]] = None):
        self.root = Path(cfg["paths_abs"]["prepared"])
        self.split, self.S, self.return_native = split, int(image_size), return_native
        self.ignore = int(cfg["data"]["ignore_index"])
        self.C = min(self.S, int(cfg["cells"]["cell_map_size"]))
        if self.S % self.C:
            raise ValueError(f"model size {self.S} is not a multiple of the cell-map size {self.C}")
        self.radius_native = float(cfg["cells"]["territory_radius_native_px"])
        rows = read_manifest(self.root, split)
        if stems is not None:
            keep = set(stems); rows = [r for r in rows if r["stem"] in keep]
        self.rows = rows
        self.items = [(i, int(k)) for i in range(len(rows)) for k in rotations]

    def __len__(self) -> int:
        return len(self.items)

    def sample_id(self, idx: int) -> str:
        i, k = self.items[idx]
        return self.rows[i]["stem"] if k == 0 else f"{self.rows[i]['stem']}_rot{k}"

    def _read(self, sub: str, stem: str) -> np.ndarray:
        return tifffile.imread(self.root / self.split / sub / f"{stem}.tif")

    def __getitem__(self, idx: int) -> Dict:
        i, k = self.items[idx]
        row = self.rows[i]; stem = row["stem"]
        arrs = {n: self._read(n, stem) for n in ("phase", "mask", "cells", "wrapped")}
        assert_phase(arrs["phase"], f"{self.split}/{stem}")              # float32 radians, never 8-bit
        if k:
            arrs = {n: np.ascontiguousarray(np.rot90(a, -k)) for n, a in arrs.items()}
        ph = torch.from_numpy(arrs["phase"])[None]
        mk = torch.from_numpy(arrs["mask"].astype(np.int64))
        cl = torch.from_numpy(arrs["cells"].astype(np.int64))
        N = ph.shape[-1]
        out = {"stem": stem, "sample_id": self.sample_id(idx), "rotation": k, "storage_day": int(row["storage_day"])}
        if self.return_native:
            out.update(phase_native=ph, mask_native=mk, cells_native=cl, wrapped_native=torch.from_numpy(arrs["wrapped"].astype(bool)))
        S, C = self.S, self.C
        if S != N:
            ph_s = F.interpolate(ph[None], size=(S, S), mode="bilinear", align_corners=False, antialias=S < N)[0]
            mk_s = _resize_labels(mk, S)
        else:
            ph_s, mk_s = ph, mk
        out["phase"] = ph_s
        out["input"] = (ph_s - ph_s.mean()) / ph_s.std().clamp(min=1e-8)
        out["mask"] = mk_s
        cl_c = _resize_labels(cl, C) if C != N else cl
        mk_c = _resize_labels(mk, C) if C != N else mk
        cl_c = cl_c * (mk_c != self.ignore).long()
        out["cells"] = cl_c
        out["territories"] = torch.from_numpy(territories(cl_c.numpy(), self.radius_native * C / N)).long() * (mk_c != self.ignore).long()
        out["contacts"] = torch.from_numpy(contact_lines(cl_c.numpy())).float()
        return out


def build_loaders(cfg: Dict, arch: str, seed: int):
    """Training loader (train stems, 4 rotations, shuffled) and validation loader (fold-0 stems of the TRAINING split,
    rotation 0). The test images (validation split, 66) are never loaded here."""
    from qpicell.physics import split_stems
    sp = split_stems(cfg)
    m = cfg["models"][arch]; d = cfg["data"]
    train = PhaseDataset(cfg, "train", m["image_size"], d["train_rotations"], stems=sp["train"])
    val = PhaseDataset(cfg, "train", m["image_size"], d["eval_rotations"], return_native=True, stems=sp["validation"])
    if len(train.rows) != cfg["split"]["expected"]["train"] or len(val.rows) != cfg["split"]["expected"]["validation"]:
        raise RuntimeError(f"split sizes train {len(train.rows)} / validation {len(val.rows)} differ from configs/config.yaml -> split.expected")
    g = torch.Generator(); g.manual_seed(seed)
    nw = int(d["num_workers"])
    common = dict(num_workers=nw, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn, persistent_workers=nw > 0)
    drop = bool(cfg["training"]["drop_last"]) and len(train) >= m["batch_size"]
    train_loader = DataLoader(train, batch_size=m["batch_size"], shuffle=True, generator=g, drop_last=drop, **common)
    val_loader = DataLoader(val, batch_size=m["batch_size"], shuffle=False, **common)
    return train_loader, val_loader
