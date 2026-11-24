"""Run identities, the experiment matrix and the output folders.

condition  'r8' / 'r16' / 'r32' / 'r64'  LoRA of that rank (alpha = rank)      | 'fullft'  full fine-tuning (no LoRA)
run id     <arch>_<condition>_<loss>_s<seed>            e.g. edge_sam_r32_dice_pmc_bga_cipp_s42, mobile_sam_fullft_dice_s43
folders    training_results/<Arch>/<condition>/<loss>/seed<seed>/     training outputs (checkpoints, logs, history)
           results/<Arch>/<condition>/<loss>/seed<seed>/              evaluation JSON (one file per checkpoint)
           with <Arch> = MobileSAM, EdgeSAM, MobileNet_UNet
Matrix: 3 architectures x (4 ranks + full fine-tuning) x 7 losses x 3 seeds = 315 runs (252 LoRA + 63 full fine-tuning).
Order (queue): seed 42 for every run, then seed 43, then seed 44; inside a seed the slowest runs first.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ARCH_ORDER = ("mobile_sam", "edge_sam", "mobilenet_unet")
COST = {"mobile_sam": 3, "edge_sam": 2, "mobilenet_unet": 1}


@dataclass(frozen=True)
class Run:
    arch: str
    condition: str          # r8 / r16 / r32 / r64 / fullft
    arm: str
    seed: int

    @property
    def rank(self) -> Optional[int]:
        return None if self.condition == "fullft" else int(self.condition[1:])

    @property
    def full_finetune(self) -> bool:
        return self.condition == "fullft"

    @property
    def run_id(self) -> str:
        return f"{self.arch}_{self.condition}_{self.arm}_s{self.seed}"

    def to_dict(self) -> Dict:
        return {**asdict(self), "rank": self.rank, "full_finetune": self.full_finetune, "run_id": self.run_id}


def conditions(cfg: Dict) -> List[str]:
    c = [f"r{int(r)}" for r in cfg["conditions"]["lora_ranks"]]
    return c + (["fullft"] if cfg["conditions"]["full_finetune"]["enabled"] else [])


def parse(run_id: str, cfg: Dict) -> Run:
    for arch in cfg["models"]:
        if run_id.startswith(arch + "_"):
            cond, rest = run_id[len(arch) + 1:].split("_", 1)
            arm, seed = rest.rsplit("_s", 1)
            if cond in conditions(cfg) and arm in cfg["arms"]:
                return Run(arch, cond, arm, int(seed))
    raise ValueError(f"not a run id: {run_id}")


def grid(cfg: Dict, archs: Optional[Sequence[str]] = None, conds: Optional[Sequence[str]] = None,
         arms: Optional[Sequence[str]] = None, seeds: Optional[Sequence[int]] = None) -> List[Run]:
    """The experiment matrix in queue order: seed-major, then slowest architecture / largest condition first."""
    archs = list(archs or cfg["models"].keys()); conds = list(conds or conditions(cfg))
    arms = list(arms or cfg["arms"].keys()); seeds = list(seeds or cfg["training"]["seeds"])
    runs = [Run(a, c, m, s) for s in seeds for a in archs for c in conds for m in arms]
    order_c = {c: i for i, c in enumerate(conditions(cfg)[::-1])}          # fullft first, then r64 ... r8
    order_m = {m: i for i, m in enumerate(cfg["arms"])}
    return sorted(runs, key=lambda r: (seeds.index(r.seed), -COST.get(r.arch, 0), order_c.get(r.condition, 99), order_m[r.arm]))


def display(cfg: Dict, arch: str) -> str:
    return cfg["models"][arch]["display"]


def training_dir(cfg: Dict, run: Run) -> Path:
    return Path(cfg["paths_abs"]["training_results"]) / display(cfg, run.arch) / run.condition / run.arm / f"seed{run.seed}"


def results_dir(cfg: Dict, run: Run) -> Path:
    return Path(cfg["paths_abs"]["results"]) / display(cfg, run.arch) / run.condition / run.arm / f"seed{run.seed}"


def learning_rate(cfg: Dict, run: Run) -> float:
    return float(cfg["conditions"]["full_finetune"]["learning_rate"] if run.full_finetune else cfg["models"][run.arch]["learning_rate"])
