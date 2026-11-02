"""Training loop: AdamW + cosine annealing, FP16 autocast (forward only), physics-aware loss in float32."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from qpi.evaluation.metrics import SegmentationMetrics
from qpi.training.losses import build_loss


class Trainer:
    def __init__(self, model: torch.nn.Module, run_cfg: Dict, cfg: Dict, device: torch.device, run_dir: Path,
                 train_loader, val_loader, max_steps_per_epoch: Optional[int] = None):
        self.model, self.run_cfg, self.device, self.run_dir = model.to(device), run_cfg, device, Path(run_dir)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.max_steps = max_steps_per_epoch
        self.epochs = int(run_cfg["training"]["epochs"])
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("no trainable parameters")
        self.optimizer = torch.optim.AdamW(params, lr=run_cfg["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.criterion = build_loss(run_cfg).to(device)
        use_amp = run_cfg["training"]["mixed_precision"] == "fp16" and device.type == "cuda"
        self.use_amp = use_amp
        try:                                    # torch >= 2.3
            self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except (AttributeError, TypeError):     # older torch (e.g. 2.0.1)
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = cfg["data"]["class_names"]
        self.ignore = run_cfg["ignore_index"]
        self.cfg = cfg
        self.history: List[Dict] = []
        self.best = {"epoch": 0, "mean_dice": -1.0}
        self.status = "running"

    # -------------------------------------------------------------------------------------------
    def fit(self) -> Dict:
        log_every = int(self.run_cfg["training"]["log_every_steps"])
        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            tr = self._train_epoch(epoch, log_every)
            if tr is None:
                self.status = "diverged"
                break
            val = self._val_epoch()
            self.scheduler.step()
            row = {"epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"], **tr, **val}
            self.history.append(row)
            self._write_history()
            torch.save({"epoch": epoch, "model_state": self.model.state_dict()}, self.ckpt_dir / "last.pt")
            if val["val_mean_dice"] > self.best["mean_dice"]:
                self.best = {"epoch": epoch, "mean_dice": val["val_mean_dice"]}
                torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                            "mean_dice": val["val_mean_dice"]}, self.ckpt_dir / "best.pt")
            print(f"[{self.run_cfg['run_id']}] epoch {epoch}/{self.epochs} loss={tr['train_loss']:.4f} "
                  f"val_dice={val['val_mean_dice']:.4f} val_ipe={val['val_ipe_global']:.4f}", flush=True)
        else:
            self.status = "trained"
        return {"status": self.status, "epochs_completed": len(self.history), "train_seconds": time.time() - t0,
                "best_by_val": self.best}

    def _write_history(self) -> None:
        keys = list(self.history[0].keys())
        tmp = self.run_dir / "history.csv.tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(self.history)
        tmp.replace(self.run_dir / "history.csv")

    def _train_epoch(self, epoch: int, log_every: int) -> Optional[Dict]:
        self.model.train()
        sums = {"L_dice": 0.0, "L_pmc": 0.0, "L_bga": 0.0, "L_pv": 0.0, "train_loss": 0.0}
        n = 0
        for step, batch in enumerate(tqdm(self.train_loader, desc=f"train {epoch}", leave=False)):
            if self.max_steps is not None and step >= self.max_steps:
                break
            x = batch["phase"].to(self.device, non_blocking=True)
            y = batch["mask"].to(self.device, non_blocking=True)
            ph = batch["phase_raw"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(x)
            loss, comps = self.criterion(logits.float(), y, ph, return_components=True)
            if not torch.isfinite(loss):
                print(f"[{self.run_cfg['run_id']}] non-finite loss at epoch {epoch} step {step}; run marked diverged",
                      flush=True)
                return None
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            n += 1
            sums["train_loss"] += float(loss.detach())
            for k in ("L_dice", "L_pmc", "L_bga", "L_pv"):
                sums[k] += float(comps[k])
        if n == 0:
            raise RuntimeError("training loader produced no batches (dataset smaller than batch size with drop_last?)")
        return {k: v / n for k, v in sums.items()}

    @torch.no_grad()
    def _val_epoch(self) -> Dict:
        """Validation at the model resolution (cheap, used for curves and optional best-checkpoint selection)."""
        self.model.eval()
        metrics = SegmentationMetrics(self.run_cfg["num_classes"], self.class_names,
                                      self.cfg["evaluation"]["boundary_tolerance_px"], self.ignore)
        total, nb = 0.0, 0
        for batch in self.val_loader:
            x = batch["phase"].to(self.device).float()
            y = batch["mask"].to(self.device)
            ph = batch["phase_raw"].to(self.device).float()
            logits = self.model(x).float()
            total += float(self.criterion(logits, y, ph))
            nb += 1
            pred = logits.argmax(1).cpu().numpy()
            for i in range(pred.shape[0]):
                metrics.update(pred[i], y[i].cpu().numpy(), ph[i, 0].cpu().numpy(),
                               batch["sample_id"][i], int(batch["storage_day"][i]))
        m = metrics.compute()
        return {"val_loss": total / max(nb, 1), "val_mean_dice": m["mean_dice"], "val_aji": m["aji"],
                "val_bf1": m["bf1"], "val_ipe_global": m["ipe_global"]}
