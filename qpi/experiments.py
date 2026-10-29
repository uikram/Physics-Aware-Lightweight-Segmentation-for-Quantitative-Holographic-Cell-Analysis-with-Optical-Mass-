"""Experiment grid: the single place that defines which training runs exist.

Run identifiers
    <arch>_lora_r<r>          LoRA rank sweep (full physics-aware loss, alpha = r)
    <arch>_full_finetune      all parameters trainable (full physics-aware loss)
    <arch>_abl_<name>         loss ablation at the selected rank (dice_only, pmc, pmc_bga)
    <arch>_abl_full           alias of <arch>_lora_r<selected_rank> (same configuration, trained once)

Multi-seed runs (``configs/multiseed.yaml``, output root ``outputs/multiseed``; see EXPERIMENTS.md)
    <base run id>__s<seed>          the same configuration as <base run id> trained with another seed
    <base run id>_wsamp__s<seed>    same, with the weighted sampler (``weighted_dominant_class``)
    <arch>_abl_<name>_r<r>__s<seed> loss ablation at a rank other than the selected one (block type ``ablation_by_rank``)
The seed-42 member of every configuration of the main grid is the run of the same name in ``outputs/full`` (never
retrained). Ablations at other ranks are not in the main grid, so their seed-42 runs are trained in ``outputs/multiseed``.
Every block carries a ``stage`` (1, 2, 3). ``--max-stage N`` (run_queue.py, aggregate_multiseed.py) restricts to stages <= N.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    group: str                   # rank_sweep | full_finetune | ablation | ablation_rank | sampler
    arch: str
    lora_r: Optional[int]        # None -> full fine-tuning
    loss_type: str               # physics_aware | dice_only
    lambda_pmc: float = 0.0
    lambda_bga: float = 0.0
    lambda_pv: float = 0.0
    alias_of: Optional[str] = None
    # ---- multi-seed bookkeeping (all None/"" in the main single-seed grid) ----
    seed: Optional[int] = None         # None -> cfg["seed"]
    sampler: Optional[str] = None      # None -> cfg["training"]["sampler"]
    base_id: Optional[str] = None      # run id of the same configuration in the main grid
    block: str = ""                    # experiment block name from configs/multiseed.yaml
    priority: int = 5                  # queue order: lower first
    stage: int = 1                     # multi-seed stage (1 = needed for the paper, 2 and 3 = extensions)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SAMPLERS = ("shuffle", "weighted_dominant_class")


def build_multiseed_grid(cfg: Dict[str, Any]) -> List[RunSpec]:
    """Runs of ``cfg['multiseed']['blocks']``.

    A block is ``{name, seeds, [stage], [priority], [sampler]}`` plus exactly one of
      * ``runs``: run ids of the main grid, each trained with extra seeds (aliases are not allowed: an alias is the same
        training as its target);
      * ``ablation_by_rank``: ``{archs, ranks, losses}``: the loss ablation at ranks other than the selected one, one run per
        (architecture, rank, loss, seed); ``losses`` are names of ``ablation_losses`` except ``full`` (the full objective at
        each rank is the rank-sweep run ``<arch>_lora_r<r>``).
    Every (run, seed) pair may occur only once overall.
    """
    ms = cfg["multiseed"]
    base_cfg = {k: v for k, v in cfg.items() if k != "multiseed"}
    base = {s.run_id: s for s in build_grid(base_cfg)}
    sel = int(cfg["lora"]["selected_rank"])
    specs: List[RunSpec] = []
    seen: Dict[str, str] = {}

    def add(spec: RunSpec, blk_name: str) -> None:
        if spec.run_id in seen:
            raise ValueError(f"{spec.run_id} occurs in blocks {seen[spec.run_id]!r} and {blk_name!r}")
        seen[spec.run_id] = blk_name
        specs.append(spec)

    for blk in ms["blocks"]:
        sampler = blk.get("sampler")
        if sampler is not None and sampler not in SAMPLERS:
            raise ValueError(f"block {blk['name']}: unknown sampler {sampler!r}")
        suffix = "_wsamp" if sampler and sampler != "shuffle" else ""
        meta = dict(block=blk["name"], priority=int(blk.get("priority", 5)), stage=int(blk.get("stage", 1)))
        if ("runs" in blk) == ("ablation_by_rank" in blk):
            raise ValueError(f"block {blk['name']}: give exactly one of 'runs' and 'ablation_by_rank'")
        for rid in blk.get("runs", []):
            if rid not in base:
                raise ValueError(f"block {blk['name']}: {rid} is not a run of the main grid")
            if base[rid].alias_of is not None:
                raise ValueError(f"block {blk['name']}: {rid} is an alias of {base[rid].alias_of}; list the target instead")
            for sd in blk["seeds"]:
                group = "sampler" if suffix else base[rid].group
                add(replace(base[rid], run_id=f"{rid}{suffix}__s{int(sd)}", group=group, seed=int(sd), sampler=sampler,
                            base_id=rid, **meta), blk["name"])
        ab = blk.get("ablation_by_rank")
        if ab:
            for arch in ab["archs"]:
                if arch not in cfg["architectures"]:
                    raise ValueError(f"block {blk['name']}: unknown architecture {arch!r}")
                for r in ab["ranks"]:
                    if int(r) == sel:
                        raise ValueError(f"block {blk['name']}: rank {r} is the selected rank; its ablation is in the main grid")
                    if int(r) not in cfg["lora"]["ranks"]:
                        raise ValueError(f"block {blk['name']}: rank {r} is not in lora.ranks")
                    for name in ab["losses"]:
                        if name == "full" or name not in cfg["ablation_losses"]:
                            raise ValueError(f"block {blk['name']}: {name!r} is not an ablation loss without IPP")
                        ls = cfg["ablation_losses"][name]
                        rid = f"{arch}_abl_{name}_r{int(r)}"
                        for sd in blk["seeds"]:
                            add(RunSpec(f"{rid}{suffix}__s{int(sd)}", "ablation_rank", arch, int(r), ls["loss_type"],
                                        ls.get("lambda_pmc", 0.0), ls.get("lambda_bga", 0.0), ls.get("lambda_pv", 0.0),
                                        seed=int(sd), sampler=sampler, base_id=rid, **meta), blk["name"])
    if not specs:
        raise ValueError("multiseed grid is empty")
    return specs


def up_to_stage(specs: List[RunSpec], max_stage: Optional[int]) -> List[RunSpec]:
    """Specs of stages <= max_stage (all of them if max_stage is None)."""
    return list(specs) if max_stage is None else [s for s in specs if s.stage <= int(max_stage)]


def build_grid(cfg: Dict[str, Any]) -> List[RunSpec]:
    """Return every RunSpec (including aliases) for the configuration."""
    if cfg.get("multiseed"):
        return build_multiseed_grid(cfg)
    smoke = cfg.get("smoke")
    archs = list(cfg["architectures"])
    ranks = list(cfg["lora"]["ranks"])
    groups = dict(cfg["groups"])
    if smoke:
        archs = [a for a in archs if a in smoke["architectures"]]
        ranks = list(smoke["ranks"])
        groups = dict(smoke["groups"])
    sel = int(cfg["lora"]["selected_rank"])
    if sel not in ranks:
        ranks = sorted(set(ranks) | {sel})
    full = cfg["ablation_losses"]["full"]
    specs: List[RunSpec] = []
    for arch in archs:
        if groups.get("rank_sweep", True):
            for r in ranks:
                specs.append(RunSpec(f"{arch}_lora_r{r}", "rank_sweep", arch, r, full["loss_type"],
                                     full["lambda_pmc"], full["lambda_bga"], full["lambda_pv"]))
        if groups.get("full_finetune", True):
            specs.append(RunSpec(f"{arch}_full_finetune", "full_finetune", arch, None, full["loss_type"],
                                 full["lambda_pmc"], full["lambda_bga"], full["lambda_pv"]))
        if groups.get("ablation", True):
            for name, ab in cfg["ablation_losses"].items():
                rid = f"{arch}_abl_{name}"
                if name == "full":
                    specs.append(RunSpec(rid, "ablation", arch, sel, ab["loss_type"], ab["lambda_pmc"],
                                         ab["lambda_bga"], ab["lambda_pv"], alias_of=f"{arch}_lora_r{sel}"))
                else:
                    specs.append(RunSpec(rid, "ablation", arch, sel, ab["loss_type"], ab.get("lambda_pmc", 0.0),
                                         ab.get("lambda_bga", 0.0), ab.get("lambda_pv", 0.0)))
    ids = [s.run_id for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate run ids in grid")
    by_id = {s.run_id: s for s in specs}
    for s in specs:
        if s.alias_of is not None:
            t = by_id.get(s.alias_of)
            if t is None or (t.loss_type, t.lambda_pmc, t.lambda_bga, t.lambda_pv, t.lora_r) != \
                    (s.loss_type, s.lambda_pmc, s.lambda_bga, s.lambda_pv, s.lora_r):
                raise ValueError(f"alias {s.run_id} does not match its target {s.alias_of}")
    return specs


def trainable_specs(specs: List[RunSpec]) -> List[RunSpec]:
    """Runs that need their own training job (aliases excluded)."""
    return [s for s in specs if s.alias_of is None]


def resolve(specs: List[RunSpec], run_id: str) -> RunSpec:
    by_id = {s.run_id: s for s in specs}
    s = by_id[run_id]
    return by_id[s.alias_of] if s.alias_of else s


def run_dir(cfg: Dict[str, Any], run_id: str) -> Path:
    return Path(cfg["paths_abs"]["output_root"]) / "runs" / run_id


def run_config(cfg: Dict[str, Any], spec: RunSpec) -> Dict[str, Any]:
    """Self-contained configuration of one training run (also written next to its outputs)."""
    a = cfg["architectures"][spec.arch]
    training = dict(cfg["training"])
    if spec.sampler is not None:
        training["sampler"] = spec.sampler
    return {
        "run_id": spec.run_id,
        "group": spec.group,
        "arch": spec.arch,
        "image_size": a["image_size"],
        "batch_size": a["batch_size"],
        "learning_rate": a["learning_rate"],
        "pretrained_weights": a["pretrained_weights"],
        "lora": None if spec.lora_r is None else {
            "r": spec.lora_r, "alpha": float(spec.lora_r), "dropout": cfg["lora"]["dropout"],
            "insertion_strategy": a["insertion_strategy"]},
        "loss": {
            "type": spec.loss_type,
            "lambda_pmc": spec.lambda_pmc, "lambda_bga": spec.lambda_bga, "lambda_pv": spec.lambda_pv,
            "pmc_margin": cfg["loss"]["pmc_margin"],
            "class_weights": cfg["loss"]["class_weights"],
            "pv_epsilon": cfg["loss"]["pv_epsilon"],
        },
        "training": training,
        "seed": cfg["seed"] if spec.seed is None else spec.seed,
        "num_classes": cfg["data"]["num_classes"],
        "ignore_index": cfg["data"]["ignore_index"],
    }
