"""Training loop: AdamW, cosine schedule over all epochs, fp16 autocast for the forward pass only, every loss in
float32, gradient scaling. After every epoch: validation at the model resolution (macro Dice for the curves and for
best.pt), checkpoints/last.pt; checkpoints/best.pt when the validation Dice improves. Validation = fold 0 of the
TRAINING split (never the test images); 'last' is the primary checkpoint. A non-finite loss stops the
run and marks it diverged."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from qpicell.evaluation.segmentation import SegmentationScores
from qpicell.losses.objective import COMPONENTS, Objective


class Trainer:
    def __init__(self, model, cfg: Dict, arch: str, arm: str, device, out_dir: Path, train_loader, val_loader,
                 max_steps: Optional[int] = None, log=print, learning_rate: Optional[float] = None):
        self.model, self.cfg, self.device, self.out = model.to(device), cfg, device, Path(out_dir)
        self.train_loader, self.val_loader, self.max_steps, self.log = train_loader, val_loader, max_steps, log
        self.epochs = int(cfg["training"]["epochs"])
        params = [p for p in model.parameters() if p.requires_grad]
        self.lr = float(learning_rate if learning_rate is not None else cfg["models"][arch]["learning_rate"])
        self.opt = torch.optim.AdamW(params, lr=self.lr)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.epochs)
        self.loss = Objective(cfg, arm).to(device)
        self.amp = cfg["training"]["mixed_precision"] == "fp16" and device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        (self.out / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.history: List[Dict] = []
        self.best = {"epoch": 0, "val_dice": -1.0}

    def _to(self, b):
        d = self.device
        return (b["input"].to(d, non_blocking=True), b["mask"].to(d, non_blocking=True), b["phase"].to(d, non_blocking=True),
                b["cells"].to(d, non_blocking=True), b["territories"].to(d, non_blocking=True), b["contacts"].to(d, non_blocking=True))

    def _train_epoch(self, epoch: int) -> Optional[Dict]:
        self.model.train()
        sums = {k: 0.0 for k in COMPONENTS + ("loss",)}; n = 0
        t_data = t_comp = 0.0; t0 = time.time(); mark = t0
        for step, batch in enumerate(self.train_loader):
            if self.max_steps is not None and step >= self.max_steps:
                break
            got = time.time(); t_data += got - mark
            x, y, ph, cl, te, co = self._to(batch)
            self.opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                logits = self.model(x)
            loss, comps = self.loss(logits.float(), y, ph, cl, te, co)
            if not torch.isfinite(loss):
                self.log(f"non-finite loss at epoch {epoch}, step {step}: run marked diverged")
                return None
            self.scaler.scale(loss).backward(); self.scaler.step(self.opt); self.scaler.update()
            n += 1; sums["loss"] += float(loss.detach())
            for k in COMPONENTS:
                sums[k] += float(comps[k])
            mark = time.time(); t_comp += mark - got
        if n == 0:
            raise RuntimeError("the training loader produced no batch")
        out = {f"train_{k}": v / n for k, v in sums.items()}
        out.update(time_train_s=time.time() - t0, time_data_wait_s=t_data, time_compute_s=t_comp)
        return out

    @torch.no_grad()
    def _val_epoch(self) -> Dict:
        self.model.eval()
        sc = SegmentationScores(self.cfg)
        tot, nb = 0.0, 0
        for batch in self.val_loader:
            x, y, ph, cl, te, co = self._to(batch)
            logits = self.model(x.float()).float()
            tot += float(self.loss(logits, y, ph, cl, te, co)[0]); nb += 1
            pred = logits.argmax(1).cpu().numpy()
            for i in range(pred.shape[0]):
                sc.add(pred[i], y[i].cpu().numpy(), ph[i, 0].cpu().numpy())
        m = sc.summary()
        return {"val_loss": tot / max(nb, 1), "val_dice": m["mean_dice"], "val_aji": m["aji"], "val_ipe_global": m["ipe_global"]}

    def fit(self) -> Dict:
        t0 = time.time(); status = "trained"
        for epoch in range(1, self.epochs + 1):
            tr = self._train_epoch(epoch)
            if tr is None:
                status = "diverged"; break
            va = self._val_epoch()
            self.sched.step()
            row = {"epoch": epoch, "lr": self.opt.param_groups[0]["lr"], **tr, **va}
            self.history.append(row)
            self._write_history()
            torch.save({"epoch": epoch, "model_state": self.model.state_dict()}, self.out / "checkpoints" / "last.pt")
            if va["val_dice"] > self.best["val_dice"]:
                self.best = {"epoch": epoch, "val_dice": va["val_dice"]}
                torch.save({"epoch": epoch, "model_state": self.model.state_dict(), "val_dice": va["val_dice"]},
                           self.out / "checkpoints" / "best.pt")
            el = time.time() - t0; left = el / epoch * (self.epochs - epoch)
            self.log(f"epoch {epoch:3d}/{self.epochs}  loss {tr['train_loss']:.4f}  val Dice {va['val_dice']:.4f}  "
                     f"val IPE {va['val_ipe_global']:.4f}  best {self.best['val_dice']:.4f}@{self.best['epoch']}  "
                     f"({tr['time_train_s']:.0f} s/epoch, left ~{left / 60:.0f} min)")
        return {"status": status, "epochs_completed": len(self.history), "train_seconds": time.time() - t0, "best_epoch": self.best}

    def _write_history(self) -> None:
        tmp = self.out / "history.csv.tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.history[0].keys())); w.writeheader(); w.writerows(self.history)
        tmp.replace(self.out / "history.csv")
