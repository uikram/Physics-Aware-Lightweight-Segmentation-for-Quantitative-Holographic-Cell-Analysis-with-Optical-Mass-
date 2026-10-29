"""Build the publication versions of the manuscript and the SI from the final-review Word files.

    python artifacts/code/build_final_docs.py

Input (not modified): artifacts/manuscript_in_prog/Claudes_Update/{revised_manuscript_fv - ui - final review.docx,
                                                                    supplementry_info_fv - ui - final review.docx}
Output: artifacts/manuscript_final/
    revised_manuscript_final.docx          every changed or new run highlighted in yellow (existing yellow kept)
    supplementary_information_final.docx   rebuilt SI, all content of this revision highlighted in yellow
    *_review_copy.docx                     the same, but this round's changes in bright green (earlier yellow kept) so the
                                           changes of this revision can be told apart from the previous one
    change_log.md                          every edit (old -> new) and the source of every number written

Paragraphs are addressed by their index in the final-review file AND checked against the start of their text, so a changed
input file stops the build instead of editing the wrong paragraph. Numbers come from manuscript_facts.py (outputs/ only).
"""
from __future__ import annotations

import copy
import re
import string
import sys
from pathlib import Path
from typing import Dict, List

import docx
import numpy as np
import pandas as pd
from docx.oxml.ns import qn

import docx_tools as T
from common import ART, CLASSES, COLLAPSE_DICE, LOSSES, RANKS, SEEDS, cells_gt, cells_pred, runs_frame
from manuscript_facts import CFG, facts

SRC = ART / "manuscript_in_prog" / "Claudes_Update"
MS_IN = SRC / "revised_manuscript_fv - ui - final review.docx"
SI_IN = SRC / "supplementry_info_fv - ui - final review.docx"
OUTD = ART / "manuscript_final"
FIG = ART / "figures" / "manuscript"
PH = FIG / "placeholders"
NAME = {"mobile_sam": "MobileSAM", "edge_sam": "EdgeSAM", "mobilenet_unet": "MobileNet-UNet"}
LOSS_LAB = {"dice_only": "Dice", "pmc": "+ PMC", "pmc_bga": "+ PMC + BGA", "full": "+ PMC + BGA + IPP (full)"}
MV = {c: i + 1 for i, c in enumerate(CLASSES)}

F = facts()
LOG: List[str] = []
USED: Dict[str, str] = {}
RF = runs_frame()
LORA = RF[RF.loss != "full_finetune"]


# ------------------------------------------------------------------------------------------------ facts in text
class _Fmt(string.Formatter):
    def get_value(self, key, args, kwargs):
        if key not in F:
            raise KeyError(f"unknown fact {key}")
        USED[key] = F[key].source
        return F[key].text


def fill(text: str, used: Dict[str, str]) -> str:
    before = dict(USED)
    out = _Fmt().format(text)
    for k in USED:
        if k not in before:
            used[k] = USED[k]
    for k in re.findall(r"\{(\w+)\}", text):
        used[k] = F[k].source
    return out


def runs(a, r, loss="full"):
    return LORA[(LORA.arch == a) & (LORA["rank"] == r) & (LORA.loss == loss)].sort_values("seed")


def pm(s, nd=3):
    return f"{s.mean():.{nd}f} ± {s.std(ddof=1):.{nd}f}"


def lost_cell(d, with_classes=True):
    parts = []
    for _, row in d.iterrows():
        lost = [str(MV[c]) for c in CLASSES if not row[f"dice_{c}"] >= COLLAPSE_DICE]
        if lost:
            parts.append(f"seed {row.seed}: {', '.join(lost)}" if with_classes else str(row.seed))
    n = len(parts)
    return f"{n}/{len(d)}" + (f" ({'; '.join(parts)})" if parts else "")


# ------------------------------------------------------------------------------------------------ run-level substitution
def _flat(p):
    """[(element, text)] for the direct children that carry text; equations count as one opaque character."""
    items = []
    for ch in p:
        if ch.tag == qn("w:r"):
            ts = ch.findall(qn("w:t"))
            if len(ts) > 1:
                ts[0].text = "".join(t.text or "" for t in ts)
                for t in ts[1:]:
                    ch.remove(t)
            items.append((ch, ts[0].text or "" if ts else ""))
        elif ch.tag.endswith("}oMath") or ch.tag.endswith("}oMathPara"):
            items.append((ch, "\x00"))
    return items


def _split(run, k):
    t = run.find(qn("w:t"))
    second = copy.deepcopy(run)
    t.text, second.find(qn("w:t")).text = t.text[:k], t.text[k:]
    for x in (t, second.find(qn("w:t"))):
        x.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    run.addnext(second)
    return second


def sub(p, old: str, new: str) -> None:
    items = _flat(p)
    full = "".join(t for _, t in items)
    i = full.find(old)
    if i < 0:
        raise ValueError(f"text not found: {old[:80]!r}")
    j = i + len(old)
    pos = 0
    for el, t in list(items):
        if el.tag != qn("w:r"):
            pos += len(t)
            continue
        if pos < i < pos + len(t):
            _split(el, i - pos)
            return sub(p, old, new)
        if pos < j < pos + len(t):
            _split(el, j - pos)
            return sub(p, old, new)
        pos += len(t)
    pos, inside, first = 0, [], None
    for el, t in items:
        if pos >= i and pos + len(t) <= j and len(t) > 0:
            inside.append(el)
        pos += len(t)
    if any(el.tag != qn("w:r") for el in inside):
        raise ValueError(f"substitution would cross an equation: {old[:60]!r}")
    rpr = T.base_rpr(p) if not inside or inside[0].find(qn("w:rPr")) is None else copy.deepcopy(inside[0].find(qn("w:rPr")))
    for tag in ("w:highlight", "w:i", "w:iCs", "w:vertAlign"):
        for e in rpr.findall(qn(tag)):
            rpr.remove(e)
    new_runs = T.runs_from_markup(new, rpr)
    anchor = inside[0]
    for r in new_runs:
        anchor.addprevious(r)
    for el in inside:
        p.remove(el)


NOTATION = [(re.compile(r"\b(EdgeSAM|MobileSAM|MobileNet-UNet) at r = (\d+)"), r"\1 (r=\2)"),
            (re.compile(r"\b(EdgeSAM|MobileSAM|MobileNet-UNet) \(r = (\d+)\)"), r"\1 (r=\2)"),
            (re.compile(r"\b(EdgeSAM|MobileSAM|MobileNet-UNet) at rank (\d+)"), r"\1 (r=\2)"),
            (re.compile(r"\b(EdgeSAM|MobileSAM|MobileNet-UNet) (?:LoRA )?r(\d+)\b"), r"\1 (r=\2)"),
            (re.compile(r"\bLoRA \(r = (\d+)\)"), r"LoRA (r=\1)"),
            (re.compile(r"\bat r = (\d+)(?![\d,]| to| and)"), r"at r=\1")]


def notation_pass(p, where: str) -> None:
    for _ in range(40):
        full = "".join(t for _, t in _flat(p))
        for rx, rep in NOTATION:
            m = rx.search(full)
            if m:
                new = m.expand(rep)
                sub(p, m.group(0), new)
                LOG.append(f"| {where} | notation | `{m.group(0)}` | `{new}` | n/a |")
                break
        else:
            return


# ------------------------------------------------------------------------------------------------ manuscript
def caption(template, label, text, used):
    return T.clone_par(template, fill(text, used), prefix=((label, True), (" ", False)))


def table1():
    rows = []
    for k in ("u8", "m8", "e8", "e32"):
        a, r = CFG[k]
        d = runs(a, r)
        rows.append([NAME[a], str(r), f"{d.trainable_params.iloc[0] / 1e6:.2f} M ({100 * d.trainable_params.iloc[0] / d.total_params.iloc[0]:.1f}%)",
                     pm(d.mean_dice), pm(d.aji), pm(d.bf1), pm(d.ipe_global_pct, 2), lost_cell(d)])
    hdr = ["Architecture", "LoRA rank", "Trainable params", "Mean Dice", "AJI", "Boundary F1", "Global IPE (%)",
           "Seeds with a lost class"]
    return T.build_table(hdr, rows, [14, 7, 13, 12, 12, 12, 12, 18], shade_rows=[3])


def table2():
    rows = []
    for a in ("mobilenet_unet", "mobile_sam", "edge_sam"):
        l42 = runs(a, 8).set_index("seed").loc[42]
        ft = RF[(RF.arch == a) & (RF.loss == "full_finetune")].iloc[0]
        for lab, row in (("LoRA (r=8)", l42), ("Full fine-tuning", ft)):
            lost = [str(MV[c]) for c in CLASSES if not row[f"dice_{c}"] >= COLLAPSE_DICE]
            rows.append([NAME[a], lab, f"{row.trainable_params / 1e6:.2f} M ({100 * row.trainable_params / row.total_params:.1f}%)",
                         f"{row.mean_dice:.3f}", f"{row.aji:.3f}", f"{row.bf1:.3f}", f"{row.ipe_global_pct:.2f}",
                         ", ".join(lost) if lost else "none"])
    hdr = ["Architecture", "Adaptation", "Trainable params", "Mean Dice", "AJI", "Boundary F1", "Global IPE (%)", "Lost classes"]
    return T.build_table(hdr, rows, [15, 15, 15, 11, 10, 11, 11, 12])


def table3():
    rows, shade = [], []
    for k in ("e32", "m8", "e8", "u8"):
        a, r = CFG[k]
        for loss in LOSSES:
            d = runs(a, r, loss)
            rows.append([f"{NAME[a]} (r={r})" if loss == "dice_only" else "", LOSS_LAB[loss], pm(d.mean_dice), pm(d.bf1),
                         pm(d.aji), pm(d.ipe_global_pct, 2), lost_cell(d, with_classes=False)])
            if loss == "full":
                shade.append(len(rows) - 1)
    hdr = ["Configuration", "Loss", "Mean Dice", "Boundary F1", "AJI", "Global IPE (%)", "Seeds with a lost class"]
    return T.build_table(hdr, rows, [17, 19, 13, 13, 12, 13, 13], shade_rows=shade)


def table4():
    from common import compiled
    rows_ = {(r["architecture"], r["backend"]): r for r in compiled("benchmark")["rows"]}
    rows = []
    for a, size in (("edge_sam", 256), ("mobilenet_unet", 256), ("mobile_sam", 1024)):
        for b, lab in (("pytorch", "PyTorch"), ("onnx", "ONNX Runtime")):
            r = rows_[(a, b)]
            rows.append([NAME[a] if b == "pytorch" else "", lab, f"{size} × {size}",
                         f"{r['latency_p50_ms']['mean']:.2f} ± {r['latency_p50_ms']['sd']:.2f}",
                         f"{r['latency_p99_ms']['mean']:.2f} ± {r['latency_p99_ms']['sd']:.2f}",
                         f"{r['fps']['min']:.1f}–{r['fps']['max']:.1f}", f"{r['peak_vram_mb_max']:.0f}"])
    hdr = ["Model (r=8)", "Runtime", "Input (px)", "P50 latency (ms)", "P99 latency (ms)", "Throughput (frames/s)",
           "GPU memory (MB)"]
    return T.build_table(hdr, rows, [17, 15, 12, 14, 14, 15, 13])


def table5():
    def f(k):
        USED[k] = F[k].source
        return F[k].text
    rows = [["Annotation (reference)", "–", "–", "–", "–", f("ann_area_chg"), f("ann_area_p"), f("ann_dm_chg"), f("ann_dm_p"), "–"]]
    for k, lab in (("e32", "EdgeSAM (r=32)"), ("m8", "MobileSAM (r=8)")):
        rows.append([lab, f(f"{k}_mdare"), f(f"{k}_bias"), f(f"{k}_r"), f(f"{k}_ccc"), f(f"{k}_area_chg"), f(f"{k}_area_p"),
                     f(f"{k}_dm_chg"), f(f"{k}_dm_p"), f(f"{k}_bias_p")])
    hdr = ["Segmentation", "MdARE (%)", "Bias (pg)", "Pearson r", "CCC", "Δ area (%)", "p, area", "Δ dry mass (%)",
           "p, dry mass", "p, bias vs day"]
    return T.build_table(hdr, rows, [14, 9, 10, 9, 9, 10, 10, 10, 9, 10])


MS_TEXT = {
    6: ("Digital holographic", "text", "Digital holographic microscopy yields label-free phase images from which cellular dry mass can be derived, but segmentation networks trained with overlap losses do not constrain the phase integrated inside the predicted boundary. We propose a physics-aware lightweight segmentation framework in which phase-mask contrast, boundary-gradient alignment, and integrated-phase preservation terms, computed on background-referenced phase in radians, are added to a class-weighted Dice loss, and Low-Rank Adaptation (LoRA) adapts pretrained encoders to 32-bit phase maps of red blood cells. MobileNet-UNet, MobileSAM, and EdgeSAM were trained at five LoRA ranks with four loss configurations and three seeds ({n_lora_runs} runs). EdgeSAM (r=32), the primary configuration, retained all four morphology classes in every seed (Dice {e32_dice}, integrated-phase error (IPE) {e32_ipe}%); MobileSAM (r=8) reached a higher Dice ({m8_dice}) with weaker instance separation; MobileNet-UNet lost the spherocyte class in all 15 full-objective runs and the stomatocyte class in 14. The full objective lowered the IPE relative to Dice alone in {abl_seed_FltD} of 45 seed-paired comparisons, but the 95% intervals excluded zero in only {abl_ci_FltD} of 15 architecture–rank configurations. With merged rank-8 weights, the EdgeSAM forward pass took {bm_e_onnx_p50} ms (ONNX Runtime, FP32). Over 47 days of storage, projected cell area decreased by about 44% in both the annotation and the predictions, whereas a decrease of dry mass was not statistically supported under date clustering. Predicted per-image dry mass agreed with the annotation within a median absolute relative error of {e32_mdare}%, with weak correlation (r = {e32_r}). Absolute dry mass was not independently validated, and no held-out test cohort was available."),
    20: ("Three representative", "text", "Three representative lightweight architectures, MobileNet-UNet [32], MobileSAM [33], and EdgeSAM [34], are evaluated on quantitative phase images of red blood cells recorded over a 47-day storage period. Each architecture is trained at five LoRA ranks with four loss configurations and three training seeds, so that segmentation accuracy, integrated-phase preservation, training stability, and computational efficiency can be compared across convolutional, transformer-based, and hybrid designs, and the predicted boundaries are then used for phase-derived measurements across the storage period."),
    25: ("Fig. 1", "caption:Fig. 1", "Overview of the physics-aware lightweight segmentation framework. A LoRA-adapted encoder and a trainable decoder segment the standardized phase map into five classes; the PMC, BGA, and IPP terms act on the background-referenced phase in radians; the predicted masks are the measurement domains for projected area, circularity, integrated phase, and dry mass."),
    111: ("Three lightweight architectures", "text", "The three architectures were first compared at the same LoRA rank (r=8), with EdgeSAM additionally at r=32 (Table 1, Fig. 3; mean ± SD over three training seeds). No single aggregate metric was sufficient to compare them. MobileSAM (r=8) achieved the highest mean Dice ({m8_dice}) and the lowest global IPE ({m8_ipe}%) and retained all four annotated classes in every seed, but its AJI ({m8_aji}) was lower than that of EdgeSAM (r=32) ({e32_aji}); its boundary F1 ({m8_bf1}) was similar to that of MobileNet-UNet ({u8_bf1})."),
    112: ("TABLE 1", "caption:TABLE 1", "Segmentation accuracy and integrated-phase error at the same LoRA rank (r=8) and for EdgeSAM (r=32), full objective. Mean ± SD over training seeds 42, 43, and 44."),
    116: ("Fig. 3", "caption:Fig. 3", "Per-class Dice of the configurations of Table 1. Bars: mean of three seeds; dots: individual seeds; crosses: a class lost in a seed (Dice < 0.05). Mask values: 1 discocyte, 2 spherocyte, 3 echinocyte, 4 stomatocyte."),
    117: ("The per-class Dice", "text", "The per-class Dice scores (Fig. 3) show where the configurations differ. MobileNet-UNet lost spherocytes and stomatocytes (mask values 2 and 4; Dice 0.000) in all three seeds, despite the highest AJI of the compared configurations ({u8_aji}) and a boundary F1 of {u8_bf1}. MobileSAM (r=8) retained all four classes (Dice {m8_dice_disc}, {m8_dice_sphe}, {m8_dice_echi}, and {m8_dice_stom} for mask values 1 to 4). EdgeSAM (r=8) lost spherocytes in one of three seeds (seed {e8_lost_seed}; mean Dice {e8_dice}), whereas EdgeSAM (r=32) retained every class in every seed (Dice {e32_dice_disc}, {e32_dice_sphe}, {e32_dice_echi}, and {e32_dice_stom}). Aggregate boundary and instance metrics can therefore hide the complete loss of a minority class, and such losses depended on the training seed. Because the predicted boundary is the domain over which the phase is integrated, a configuration that misses a morphology class cannot provide class-specific phase-derived measurements, whatever its aggregate scores."),
    118: ("Among the compared", "text", "With {m8_params} trainable parameters, MobileSAM (r=8) gave the highest mean Dice and no lost class. EdgeSAM needed {e8_params} trainable parameters at r=8 but was less stable across seeds; at r=32 ({e32_params}) it retained every class in every seed and had a higher AJI than MobileSAM (r=8), at a lower Dice and boundary F1. The seed-paired difference between EdgeSAM (r=32) and EdgeSAM (r=8) was {e32_minus_e8_dice} in Dice, not separable from zero, and {e32_minus_e8_aji} in AJI."),
    119: ("The aggregate comparison", "text", "No configuration was best on all criteria. EdgeSAM (r=32) is used as the primary configuration for the loss-ablation and storage analyses because it combined complete class retention in every seed with the highest AJI of the SAM-based configurations, and MobileSAM (r=8) is used as the sensitivity comparison because it had the highest Dice and the lowest IPE; this is a presentation choice, not the result of a selection rule applied to the validation data. The relation between accuracy and forward-pass throughput at rank 8 is shown in Fig. S-3."),
    120: ("The effect of LoRA rank", "text", "The complete rank sweep (r = 2, 4, 8, 16, and 32) is given for all three architectures and every seed in Table S-1 and Figs. S-1 and S-2. The dependence on rank was not monotonic. For EdgeSAM, r=32 gave the highest Dice, AJI, and boundary F1 and the lowest IPE of the five ranks, and spherocytes were lost in seed 44 at {e_lost_ranks}. MobileSAM gave a Dice between {m_dice_r2_16} at r = 2 to 16, but at r=32 one seed ({m32_lost_seed}) collapsed in the second epoch and lost two classes. MobileNet-UNet lost at least one class in every seed at every rank."),
    122: ("To evaluate the effectiveness", "text", "LoRA at r=8 was compared with full fine-tuning of each architecture under the same objective, learning rate, and validation split (Table 2, Fig. 4). Full fine-tuning was run once (seed 42), so Table 2 compares it with the seed-42 LoRA run. For EdgeSAM, full fine-tuning gave a mean Dice of {ft_e_dice} (LoRA: {l42_e_dice}), lost spherocytes, and raised the global IPE from {l42_e_ipe}% to {ft_e_ipe}%. For MobileSAM, full fine-tuning retained only discocytes (mean Dice {ft_m_dice}; LoRA: {l42_m_dice}). For MobileNet-UNet, whose LoRA configuration already trains 71.8% of the parameters, full fine-tuning did not resolve the loss of spherocytes and stomatocytes (mean Dice {ft_u_dice}; LoRA: {l42_u_dice})."),
    123: ("Fig. 4 compares", "text", "In this comparison, training all parameters did not improve adaptation to the phase images, and restricting adaptation to low-rank updates is consistent with retaining useful pretrained representations. Because full fine-tuning was run for one seed, the comparison cannot separate the adaptation method from the training seed; for EdgeSAM in particular, LoRA (r=8) also lost spherocytes in one of its three seeds."),
    124: ("TABLE 2", "caption:TABLE 2", "LoRA (r=8) versus full fine-tuning, full objective, training seed 42."),
    128: ("Fig. 4", "caption:Fig. 4", "Per-class Dice of LoRA (r=8; bars: mean of three seeds, dots: individual seeds) and full fine-tuning (hatched; seed 42). Crosses mark a lost class (Dice < 0.05)."),
    130: ("To evaluate the contribution", "text", "The contribution of each loss term was assessed by adding PMC, BGA, and IPP in turn to the Dice loss (Table 3, Fig. 5; all ranks in Table S-2). Over the full rank sweep, the mean global IPE of the full objective was lower than that of Dice alone in {abl_cfg_FltD} of the 15 architecture–rank configurations, but the seed-paired 95% interval excluded zero in only {abl_ci_FltD} of them. For individual seeds, the full objective gave a lower IPE than Dice alone in {abl_seed_FltD} of 45 comparisons (EdgeSAM {abl_e_seed_FltD}, MobileSAM {abl_m_seed_FltD}, MobileNet-UNet {abl_u_seed_FltD} of 15) and a lower IPE than Dice with PMC and BGA in {abl_seed_FltB} (EdgeSAM {abl_e_seed_FltB}, MobileSAM {abl_m_seed_FltB}, MobileNet-UNet {abl_u_seed_FltB}). These counts do not separate the effect of the objective from training instability: a class was lost with Dice alone in {abl_e_lost_D} of 15 EdgeSAM runs, {abl_m_lost_D} of 15 MobileSAM runs, and {abl_u_lost_D} of 15 MobileNet-UNet runs, and with the full objective in {abl_e_lost_F}, {abl_m_lost_F}, and {abl_u_lost_F} runs, respectively. The full objective thus reduced class loss for EdgeSAM and MobileSAM but not for MobileNet-UNet, which lost classes in more runs with the full objective than with Dice alone."),
    131: ("TABLE 3", "caption:TABLE 3", "Component-wise ablation of the physics-aware loss (mean ± SD over three seeds). Each row adds one term to the row above; shaded rows are the full objective."),
    136: ("For EdgeSAM at r = 8", "text", "For EdgeSAM (r=8), Dice alone gave a mean Dice of {e8D_dice} and a global IPE of {e8D_ipe}%, with spherocytes lost in seed {e8D_lostseeds}. Adding PMC gave {e8P_dice} and {e8P_ipe}%, again with one lost class (seed {e8P_lostseeds}), whereas adding BGA gave {e8B_dice}, a boundary F1 of {e8B_bf1}, an IPE of {e8B_ipe}%, and no lost class; the mean spherocyte Dice rose from {e8D_sph} with Dice alone to {e8P_sph} with PMC and {e8B_sph} with PMC and BGA (Table S-3). An example of predicted contours on the phase-gradient magnitude is shown in Fig. S-4. At r=32, Dice alone lost spherocytes in two of three seeds (seeds {e32D_lostseeds}; Dice {e32D_dice}, IPE {e32D_ipe}%) and Dice with PMC in one (seed {e32P_lostseeds}; {e32P_dice}, {e32P_ipe}%), whereas Dice with PMC and BGA ({e32B_dice}) and the full objective ({e32F_dice}) lost no class."),
    137: ("The complete objective", "text", "Adding IPP gave the lowest mean global IPE for EdgeSAM at both ranks ({e8F_ipe}% at r=8 and {e32F_ipe}% at r=32; Fig. 5), without a higher boundary F1 than Dice with PMC and BGA. Relative to Dice with PMC and BGA, the seed-paired change in IPE was {e32_FmB_ipe} percentage points at r=32, with a Dice change of {e32_FmB_dice}, whereas at r=8 it was {e8_FmB_ipe} percentage points and not separable from zero. For MobileSAM (r=8), the four loss configurations gave mean Dice values within {m8_abl_dice_range} of each other and global IPEs between {m8_abl_ipe_range}, and the difference between the full objective and Dice alone was {m8_FmD_ipe} percentage points. For MobileNet-UNet (r=8), the mean IPE fell from {u8D_ipe}% with Dice alone to {u8F_ipe}% with the full objective, but classes were lost in two or three seeds in every configuration and the paired difference ({u8_FmD_ipe} percentage points) was not separable from zero. For EdgeSAM, the ablation is therefore consistent with complementary terms: PMC lowered the IPE but did not prevent class loss, BGA removed the lost classes, and IPP gave the lowest IPE. For MobileSAM and MobileNet-UNet, no effect of an individual term could be separated from seed-to-seed variation with three seeds."),
    139: ("Fig. 5", "caption:Fig. 5", "Component-wise loss ablation for EdgeSAM (r=32), MobileSAM (r=8), and MobileNet-UNet (r=8): mean Dice (top) and global IPE (bottom). Each step adds one term to the previous configuration. Bars: mean of three seeds; dots: individual seeds; crosses: seeds in which a class was lost."),
    141: ("The preservation of phase-integrated", "text", "The global IPE was {m8_ipe}% for MobileSAM (r=8), {u8_ipe}% for MobileNet-UNet (r=8), and {e8_ipe}% and {e32_ipe}% for EdgeSAM at r=8 and r=32 (Table 1). Global IPE alone does not characterize per-cell measurement quality: MobileNet-UNet lost spherocytes and stomatocytes in every seed, which gives a class-specific IPE of 100% for both (Table S-3), and EdgeSAM (r=8) varied more between seeds (SD of the global IPE {e8_ipe_sd} percentage points). When annotated cells were matched one-to-one to predicted components at IoU ≥ 0.5, {e32F_matched}% of the annotated cells were recovered by EdgeSAM (r=32) and {m8F_matched}% by MobileSAM (r=8); for the matched cells (wrapped cells excluded), the median absolute per-cell IPE was {e32F_pc_med}% and {m8F_pc_med}%, and the 90th percentile {e32F_pc_p90}% and {m8F_pc_p90}%, respectively. Integrated-phase fidelity must therefore be read together with class coverage, instance separation, and seed-to-seed stability, and no configuration was best on all of these criteria."),
    145: ("Computational efficiency was evaluated", "text", "Forward-pass latency and GPU memory were measured for the rank-8 models with merged LoRA weights in strict FP32 on an NVIDIA RTX A5000 at batch size 1, with native PyTorch and ONNX Runtime (Table 4; Section 3.4). ONNX Runtime reduced the median latency of EdgeSAM from {bm_e_pytorch_p50} ms to {bm_e_onnx_p50} ms ({bm_e_onnx_fps} frames per second across the three timing passes) and that of MobileNet-UNet from {bm_u_pytorch_p50} ms to {bm_u_onnx_p50} ms, whereas MobileSAM, which operates on 1024 × 1024 px inputs, was slower with ONNX Runtime ({bm_m_onnx_p50} ms) than with PyTorch ({bm_m_pytorch_p50} ms). The GPU-memory footprint was larger with ONNX Runtime than with PyTorch for all three architectures. MobileNet-UNet had the lowest latency but lost two classes in every seed. EdgeSAM (r=32), the configuration used in the storage analysis, was not timed; because merging yields the same network at every rank, its forward-pass cost is expected to equal that of the rank-8 model, but this was not measured. The timings cover the forward pass only, on a desktop-class GPU, and do not establish end-to-end or embedded performance."),
    146: ("TABLE 6", "caption:TABLE 4", "Forward-pass latency, throughput, and GPU memory of the rank-8 models (LoRA merged) in strict FP32 on an NVIDIA RTX A5000 at batch size 1. Latency: mean ± SD over three timing passes."),
    149: ("Qualitative results are shown", "text", "Qualitative results are shown in Fig. 6 for one 256 × 256 px window per annotated class, selected from the annotation alone by a fixed rule: the validation image with the most annotated pixels of that class and, within that image, the window with the most pixels of that class on a 32 px grid. The windows are therefore not the maximum over all images and positions; they are examples, not a statistical sample. All four predictions in the figure come from training seed 42. In these windows, MobileNet-UNet assigned the cells annotated as spherocytes and stomatocytes to other classes, consistent with its Dice of zero for these classes (Fig. 3), whereas the SAM-based configurations assigned most annotated spherocytes, echinocytes, and stomatocytes to the annotated class, with most errors on stomatocytes; where cells touch, the predicted foreground of all models tended to form joined regions. These observations were quantified over all 66 validation images in a separate pixel-level analysis, for the training seeds given in parentheses, which are not all those of the figure. MobileNet-UNet (r=8; seed 42) predicted 68% of the annotated spherocyte pixels and 82% of the annotated stomatocyte pixels as discocyte. Of the annotated stomatocyte pixels, 68% were predicted as stomatocyte by MobileSAM (r=8), 52% by EdgeSAM (r=8), and 56% by EdgeSAM (r=32) (seed 43), and 22% to 30% as discocyte. An annotated cell (15 to 200 µm², not touching an unannotated region) was counted as bridged when one predicted component covered at least 30% of its pixels and of the pixels of another annotated cell: 35.1–36.9% of the annotated cells were bridged for MobileSAM (r=8; seeds 43 and 44), 33.4–41.8% for EdgeSAM (r=8; seeds 43 and 44), 30.5–35.3% for EdgeSAM (r=32; seeds 42 to 44), and 13.2% for MobileNet-UNet (r=8; seed 42); no statistical test was made. The bridging rates of the three SAM-based configurations overlap, so they do not show that one of them separates adjacent cells better than another, although EdgeSAM (r=32) had the highest AJI of the three ({e32_aji}, against {m8_aji} for MobileSAM (r=8) and {e8_aji} for EdgeSAM (r=8); Table 1). Limited instance separation is a possible source of per-cell measurement error, but its effect on the dry-mass estimates was not quantified separately."),
    151: ("Fig. 6", "caption:Fig. 6", "Qualitative segmentation in four 256 × 256 px validation windows, one per annotated class (rows; mask values 1 to 4; selection rule in Section 4.6). Columns: phase, annotation, and the predictions of MobileNet-UNet (r=8), MobileSAM (r=8), EdgeSAM (r=8), and EdgeSAM (r=32), all from training seed 42. Grey: outside the annotated field of view."),
    153: ("To evaluate the applicability", "text", "Storage-associated changes in RBC morphology and phase-derived properties were examined over the 47-day storage period with the masks predicted by EdgeSAM (r=32) for each of the three training seeds, and with MobileSAM (r=8) as the sensitivity comparison. Each storage day is represented by six validation images from the same samples, which are not independent observations (Section 3.6); the analysis therefore describes cross-sectional population changes, not the tracking of individual cells."),
    154: ("Class composition was computed", "text", "Class composition was computed for each storage day as the share of cells assigned to each class, pooling the six images of the day; cells are 4-connected foreground components of 15 to 200 µm² that do not touch an unannotated region and are not phase-wrapped (Section 2.6). For the predictions, the shares were computed for each seed and averaged (Fig. 7; per-seed values in Table S-7). Discocytes predominated at day 0 ({sh_ann_disc_0} in the annotation, {sh_e32_disc_0} in the prediction) and fell to {sh_ann_disc_47} and {sh_e32_disc_47} by day 47, when spherocytes ({sh_ann_sphe_47} and {sh_e32_sphe_47}) and echinocytes ({sh_ann_echi_47} and {sh_e32_echi_47}) predominated. Stomatocytes remained a minority class, with an annotated share of {sh_ann_stom_0} at day 0, {sh_ann_stom_19} at day 19 ({ann_stom_19_n} of {ann_cells_19} annotated cells), and {sh_ann_stom_47} at day 47, against {sh_e32_stom_0}, {sh_e32_stom_19} (seed range {sh_e32_stom_19_range}), and {sh_e32_stom_47} for EdgeSAM (r=32) and {sh_e8_stom} for EdgeSAM (r=8)."),
    156: ("Fig. 7", "caption:Fig. 7", "Class composition over the 47-day storage period: (a) annotation; (b) EdgeSAM (r=32), mean of seeds 42, 43, and 44. Cells of the six images of each day are pooled ({ann_cells_day_range} annotated cells per day); phase-wrapped cells are excluded."),
    158: ("Representative quantitative phase", "text", "The predicted composition reproduced the direction and approximate timing of the annotated shift, but not its exact values: averaged over the three seeds, predicted and annotated shares differed by {e32_comp_mean} percentage points on average over the 11 days and four classes, with a maximum of {e32_comp_max} percentage points ({e32_comp_max_at}, where few predicted components passed the cell filters), and by up to {e32_comp_seedmax} percentage points for an individual seed. For MobileSAM (r=8), the mean difference was {m8_comp_mean} and the maximum {m8_comp_max} percentage points. Because predicted components are not cell counts, the predicted composition describes the components that passed the cell filters. Representative phase images over the storage period are shown in Fig. S-5."),
    160: ("Quantitative cellular measurements", "text", "After exclusion of wrapped cells, the annotation contains {ann_cells} cells in the 66 validation images, and the EdgeSAM (r=32) predictions contain {e32_kept_tot} kept components for seeds 42, 43, and 44 (Table S-4). Projected cell area decreased with storage (Fig. 8b, Table 5): between early (days 0 to 8) and late storage (days 37 to 47), the median area changed by {ann_area_chg}% in the annotation and by {e32_area_chg}% in the predictions. With the 11 acquisition dates as clusters, the permutation p-value of this trend was {ann_area_p} for the annotation and {e32_area_p} for seeds 42, 43, and 44. The 95% date-bootstrap intervals of the Spearman correlation excluded zero for the three seeds ({e32_area_boot}) but not for the annotation ({ann_area_boot}), so for the annotation the decrease is borderline under date clustering. The median dry mass changed by {ann_dm_chg}% in the annotation and by {e32_dm_chg}% in the predictions, but a decrease was not statistically supported under date clustering (p = {ann_dm_p} for the annotation and {e32_dm_p} for the three seeds; every bootstrap interval included zero; Table S-6). Treating the 66 images as independent observations would give p = {e32_dm_pnaive} for the three seeds, but this ignores that the six images of a date come from the same samples and is therefore not used. The absence of a supported trend is not evidence that dry mass was constant, because a test with 11 clusters has limited power. The change in projected area was thus not accompanied by a statistically supported change in phase-derived dry mass (Fig. 8b, c)."),
    161: ("To evaluate the reliability", "text", "Dry mass from the predicted masks was compared with that from the annotation masks using the per-image median dry mass (66 images per seed). For EdgeSAM (r=32), the median absolute relative error (MdARE) was {e32_mdare}% ({e32_mdare_pct_seeds}% for seeds 42, 43, and 44), the Bland-Altman bias {e32_bias} pg with 95% limits of agreement from {e32_loa} pg, the Pearson correlation r = {e32_r} ({e32_pearson_r_seeds}), and the concordance correlation coefficient {e32_ccc} (Fig. 8a, Table 5). For MobileSAM (r=8), MdARE was {m8_mdare}%, r = {m8_r}, and CCC {m8_ccc}. The correlations are weak and depend on the seed, partly because the annotation-derived per-image medians span a narrow range ({ann_img_dm_range} pg). MdARE is a median, not an upper bound: half of the images had larger relative errors, and the limits of agreement show errors of several pg for individual images. To test for a storage-dependent bias, the per-image difference between predicted and annotation-derived median dry mass was related to storage day with the same date-clustered test. For EdgeSAM (r=32), this correlation was weakly negative (Spearman ρ = {e32_bias_rho} for the three seeds) but not statistically supported (p = {e32_bias_p}; every bootstrap interval included zero), and for MobileSAM (r=8) no association was found (p = {m8_bias_p}). The larger early-to-late change in the predicted than in the annotated dry mass therefore cannot be attributed to a storage-dependent bias with these data. Both segmentations are evaluated on the same phase maps, each with its own background estimate, so these values describe agreement between segmentations and not the accuracy of absolute dry mass (Section 2.6)."),
    162: ("The predicted values in Table 7", "text", "The predicted components that pass the cell filters are not cell counts: touching cells can merge into one component and fragments are discarded. On days 8, 12, and 15, the number of kept EdgeSAM (r=32) components was only {e32_kept_8_12_15} of the annotated cell count (mean over seeds; open markers in Fig. 8b, c), against {e32_kept_other} on the other days; MobileSAM (r=8) showed the same pattern ({m8_kept_8_12_15}). On day 15, including wrapped components and averaged over the three seeds, the six images yielded 61.3 kept components for {ann_cells_15_all} annotated cells, with 116 fragments smaller than 15 µm², 22 components larger than 200 µm², and 46 components of admissible size touching the unannotated region, while the predicted foreground area was 0.98 of the annotated area (Fig. S-6). The low counts on these days are consistent with merging and fragmentation in crowded images and do not indicate a reduction in cell number; predicted mean area and dry mass on these days are selection-biased and are not interpreted biologically. With MobileSAM (r=8), the early-to-late change was {m8_area_chg}% for area (p = {m8_area_p}; the bootstrap interval excluded zero for {m8_area_boot_excl0} of the 3 seeds) and {m8_dm_chg}% for dry mass (p = {m8_dm_p}) (Table 5, Table S-5)."),
    163: ("TABLE 7", "caption:TABLE 5", "Dry-mass agreement with the annotation and storage trends in the 66 validation images (11 acquisition dates). Mean ± SD over seeds 42, 43, and 44; p-values per seed."),
    165: ("a. Cell-level SD", "note", "MdARE, bias, Pearson r, and CCC compare the per-image median dry mass of prediction and annotation. Δ: change of the median of the per-image medians between early (days 0 to 8, 18 images) and late storage (days 37 to 47, 12 images). p: two-sided permutation test of the image-level Spearman correlation with storage day, permuting the 11 acquisition dates as whole blocks (exact test over all 11! permutations); bias vs day: the same test on the per-image difference prediction − annotation. Wrapped cells excluded. Bootstrap intervals and per-day values: Tables S-4 to S-6. Absolute dry mass was not independently validated."),
    172: ("Fig. 8", "caption:Fig. 8", "Dry mass and projected area over storage. (a) Per-image median dry mass of EdgeSAM (r=32) against the annotation (markers: seeds; colour: storage day; dashed: identity). (b) Projected area and (c) dry mass: per-day median of the per-image medians divided by the day-0 value for EdgeSAM (r=32) and MobileSAM (r=8) (line: mean of seeds; band: seed range) and for the annotation (dashed). Open markers: days on which kept components were fewer than 0.6 of the annotated cells."),
}

MS_SUB = {
    19: [("Raw, unnormalized phase values are supplied directly to this objective so that the preserved physical quantity is not altered by input normalization.",
          "The objective is evaluated on the background-referenced phase in radians rather than on the standardized network input, so that the preserved physical quantity is not altered by input normalization.")],
    42: [("the counts for all ranks are given in the Supplementary Information", "the counts for all ranks are given in Table S-1")],
    79: [("Cells flagged as phase-wrapped (Section 2.2) are excluded from Table 7, Fig. 8 and the agreement and storage-trend statistics (Section 3.6); the component-count and component-fate analysis in Fig. S-6 includes wrapped components.",
          "Cells flagged as phase-wrapped (Section 2.2) are excluded from the composition, agreement, and storage-trend analyses (Figs. 7 and 8, Table 5, and Tables S-4 to S-6); the component-fate analysis of Fig. S-6 includes them.")],
    97: [("For the EdgeSAM configuration at r = 8, 60.9 ± 2.4% of the ground-truth cells were matched (mean ± SD over three seeds), with a per-cell median absolute IPE of 2.08 ± 0.65% and a 90th percentile of 17.3 ± 5.6%. Excluding the cells flagged as phase-wrapped (61 of 2,049 annotated cells, 3.0%; Section 2.2), these values are 2.04 ± 0.65% and 17.5 ± 5.6%.",
          "Per-cell results are reported in Section 4.4 without the cells flagged as phase-wrapped ({ann_wrapped} of {ann_cells_all} annotated cells, 3.0%; Section 2.2).")],
    105: [("In the main text, MobileSAM at r = 8 and EdgeSAM at r = 32 are discussed as the preferred configurations, together with the same-rank comparison at r = 8; this is a presentation choice and not the output of a formal selection rule.",
           "In the main text, EdgeSAM (r=32) is the primary configuration and MobileSAM (r=8) the sensitivity comparison, together with the same-rank comparison of the three architectures at r=8; this is a presentation choice (Section 4.1), not the output of a formal selection rule, and the complete rank sweep with every seed is given in Section S-I of the Supplementary Information."),
          ("Third, a component-wise ablation of the Physics-Aware Phase Consistency Loss was performed across all three lightweight architectures.",
           "Third, a component-wise ablation of the Physics-Aware Phase Consistency Loss was performed for all three lightweight architectures at all five ranks.")],
    107: [("For EdgeSAM r32, each agreement and trend statistic was computed separately for each of the three training seeds (42, 43, 44) and is reported as mean ± SD (n − 1) over the three per-seed values; p-values are listed per seed and are not averaged; nothing is pooled over seeds.",
           "For EdgeSAM (r=32) and MobileSAM (r=8), each agreement and trend statistic was computed separately for each training seed (42, 43, 44) and is reported as mean ± SD (n − 1) over the three per-seed values; p-values are listed per seed and are not averaged, and cells are not pooled over seeds. To test whether the prediction error depended on storage time, the same date-clustered test was applied to the per-image difference between predicted and annotation-derived median dry mass.")],
    180: [("Because phase unwrapping relies on the phase variation remaining within an unambiguous range, specimens with substantially greater optical thickness than RBCs (where the accumulated phase could exceed the unwrapping range) may require additional validation of the reconstruction before this framework is applied.",
           "No phase unwrapping was applied, and {ann_wrapped} of {ann_cells_all} annotated cells (3.0%) were flagged as possibly wrapped and excluded from the per-cell analyses; specimens with a larger optical thickness than RBCs would require reliable unwrapping before this framework is applied."),
          ("Future work will therefore focus", "Touching cells can merge into one predicted component, so limited instance separation is a possible source of per-cell error whose effect on dry mass was not isolated. Future work will therefore focus")],
}

MS_SUB.update({
    107: MS_SUB[107] + [("(10,000 permutations)", "(exact test over all 11! = 39,916,800 permutations)"), ("2,000 bootstrap resamples", "20,000 bootstrap resamples")],
    175: [("raised the mean Dice of spherocytes (Table 4)", "raised the mean Dice of spherocytes (Table S-3)")],
    179: [("area decrease.  Cellular", "area decrease. Cellular"), ("not independently validated.  This enables", "not independently validated. This enables")],
    183: [("EdgeSAM at rank 32 (the primary configuration) and MobileSAM at rank 8 (the sensitivity comparison) retained",
           "EdgeSAM (r=32), the primary configuration, and MobileSAM (r=8), the sensitivity comparison, retained")],
})

MS_APPEND = {
    49: " In the implementation, the weighted sum over classes is divided by Σ~c~ w~c~ + 10^−4^.",
    56: " The gradient magnitude is computed as (∂~x~^2^ + ∂~y~^2^ + 10^−4^)^1/2^ from forward differences before the max-normalization.",
    68: " In all experiments, λ~1~ = 0.1, λ~2~ = 0.05, and λ~3~ = 0.1.",
    88: " In total, {n_lora_runs} LoRA runs (three architectures × five ranks × four loss configurations × three seeds) and {n_ft_runs} full fine-tuning runs were trained.",
    102: " Because merging yields the same network at every rank, the forward-pass cost does not depend on the rank in principle; latency and memory were nevertheless measured only for the rank-8 checkpoints, and EdgeSAM (r=32) was not timed.",
    176: " MdARE describes the typical image and is not an upper bound on the error of an individual image.",
}

MS_INSERT = {
    137: "To check whether IPP introduces a phase-dependent bias, the false-negative rate of ground-truth foreground pixels was computed within per-image quartiles of the background-referenced phase, with and without IPP (Dice with PMC and BGA versus the full objective). Missed pixels concentrated in the lowest-phase quartile for EdgeSAM (r=32: {e32B_fnrq1}% without and {e32F_fnrq1}% with IPP; r=8: {e8B_fnrq1}% and {e8F_fnrq1}%; at most {fnr_e_q24max}% in the other quartiles of any seed), for MobileSAM (r=8: {m8B_fnrq1}% and {m8F_fnrq1}%; at most {fnr_m_q24max}% elsewhere), and for MobileNet-UNet (r=8: {u8B_fnrq1}% and {u8F_fnrq1}%), so IPP did not increase the low-phase bias of EdgeSAM or MobileNet-UNet and increased it slightly for MobileSAM. Robustness to background variation was tested by adding constant offsets and low-order tilt, defocus-like, and astigmatism-like fields to the validation phase maps. For the four configurations of Table 1, the mean Dice changed by at most {rob_max_ddice_mean}, as expected for constant offsets because each network input is standardized. The IPE computed on the perturbed phase without re-referencing increased with the offset (EdgeSAM (r=32): {e32_rob_ipe0}% without and {e32_rob_ipe06}% at the largest offset; EdgeSAM (r=8): {e8_rob_ipe0}% and {e8_rob_ipe06}%).",
}

MS_DELETE = [134, 135, 142, 143, 166, 167, 168, 169, 170]
MS_IMAGES = {115: FIG / "Fig3_per_class_dice.png", 127: FIG / "Fig4_lora_vs_full_finetuning.png",
             138: FIG / "Fig5_loss_ablation.png", 150: PH / "Fig6_qualitative.png", 155: FIG / "Fig7_composition.png",
             171: FIG / "Fig8_dry_mass_and_area.png", 24: FIG / "Fig1_framework.png"}
MS_TABLES = {113: table1, 125: table2, 132: table3, 147: table4, 164: table5}
MS_NOTES = {113: "A class is lost in a seed when its Dice is below 0.05; the mask values of the lost classes are given after the seed (2 spherocyte, 4 stomatocyte). Per-class Dice: Fig. 3; class-specific IPE: Table S-3.",
            125: "Lost classes by mask value (Dice < 0.05). Three-seed LoRA results: Table 1.",
            132: "Seeds in which at least one class was lost (Dice < 0.05) are given in parentheses. All ranks: Table S-2; class-specific IPE and Dice: Table S-3.",
            147: "Throughput: 1000/P50 of each timing pass, range over the three passes. GPU memory: baseline-subtracted device-level reading after the timed loop, maximum over the passes (not a continuously sampled peak). Timing covers the model forward pass with preprocessed, GPU-resident inputs (Section 3.4)."}


def extra_facts():
    m = [runs("mobile_sam", r).mean_dice.mean() for r in (2, 4, 8, 16)]
    from manuscript_facts import Fact
    F["m_dice_r2_16"] = Fact(f"{min(m):.3f} and {max(m):.3f}", "cell_analysis_multiseed.json: mobile_sam|lora_r2..r16 mean_dice mean")
    ranks = [r for r in RANKS if (runs("edge_sam", r).set_index("seed").loc[44, "n_lost"] > 0)]
    F["e_lost_ranks"] = Fact("r = " + ", ".join(str(r) for r in ranks[:-1]) + f", and {ranks[-1]}",
                             "cell_analysis_multiseed.json: edge_sam|lora_r* per-seed class Dice, seed 44")
    d = [runs("mobile_sam", 8, l).mean_dice.mean() for l in LOSSES]
    i = [runs("mobile_sam", 8, l).ipe_global_pct.mean() for l in LOSSES]
    F["m8_abl_dice_range"] = Fact(f"{max(d) - min(d):.3f}", "cell_analysis_multiseed.json: mobile_sam ablation r8 mean_dice (max - min)")
    F["m8_abl_ipe_range"] = Fact(f"{min(i):.2f}% and {max(i):.2f}%", "cell_analysis_multiseed.json: mobile_sam ablation r8 ipe_global_pct")


def build_ms(hl: str, out: Path) -> None:
    T.HL["color"] = hl
    d = docx.Document(MS_IN)
    body = list(d.element.body.iterchildren())
    tpl_note = body[165]
    tpl_cap = body[112]
    for i, (prefix, kind, text) in MS_TEXT.items():
        p = body[i]
        old = T.para_text(p)
        if not old.startswith(prefix):
            raise RuntimeError(f"paragraph {i}: expected {prefix!r}, found {old[:40]!r}")
        used: Dict[str, str] = {}
        new = fill(text, used)
        if kind == "text":
            T.set_text(p, new)
        elif kind.startswith("caption:"):
            T.set_text(p, new, prefix=((kind.split(":")[1], True), (" ", False)))
        elif kind == "note":
            T.set_text(p, new)
        LOG.append(f"| ms ¶{i} | {kind} | {old} | {new} | {', '.join(sorted(used))} |")
    for i, text in MS_INSERT.items():
        used = {}
        new = fill(text, used)
        body[i].addnext(T.clone_par(body[i], new))
        LOG.append(f"| ms after ¶{i} | new paragraph | – | {new} | {', '.join(sorted(used))} |")
    for i, pairs in MS_SUB.items():
        for old, new in pairs:
            used = {}
            sub(body[i], old, fill(new, used))
            LOG.append(f"| ms ¶{i} | replace | {old} | {fill(new, used)} | {', '.join(sorted(used))} |")
    for i, text in MS_APPEND.items():
        used = {}
        new = fill(text, used)
        p = body[i]
        for r in T.runs_from_markup(new, T.base_rpr(p)):
            p.append(r)
        LOG.append(f"| ms ¶{i} | append | – | {new.strip()} | {', '.join(sorted(used))} |")
    for i, path in MS_IMAGES.items():
        T.replace_image(d, body[i], path, width_in=6.9 if i != 115 else 6.5)
        LOG.append(f"| ms ¶{i} | figure | old embedded image | {path.relative_to(ART.parent)} | {'make_data_figures.py (placeholder)' if 'placeholders' in str(path) else 'make_manuscript_figures.py / patch_static_figures.py'} |")
    for i, fn in MS_TABLES.items():
        new = fn()
        body[i].addprevious(new)
        if i in MS_NOTES:
            note = T.clone_par(tpl_note, MS_NOTES[i])
            new.addnext(note)
        body[i].getparent().remove(body[i])
        LOG.append(f"| ms table at ¶{i} | table | old table | rebuilt from outputs ({fn.__name__}) | {fn.__doc__ or 'runs_frame() / compiled benchmark / manuscript_facts'} |")
    for i in MS_DELETE:
        LOG.append(f"| ms ¶{i} | delete | {T.para_text(body[i])[:160] if body[i].tag == qn('w:p') else 'table'} | – | moved to SI or superseded |")
        body[i].getparent().remove(body[i])
    # (r=8) notation in every remaining paragraph not rewritten above
    for i, el in enumerate(body):
        if el.tag == qn("w:p") and el.getparent() is not None and i not in MS_TEXT:
            notation_pass(el, f"ms ¶{i}")
    d.save(out)


# ------------------------------------------------------------------------------------------------ SI
def si_tables():
    def s1():
        rows, shade = [], []
        for a in ("edge_sam", "mobile_sam", "mobilenet_unet"):
            for r in RANKS:
                d = runs(a, r)
                rows.append([NAME[a] if r == 2 else "", str(r),
                             f"{d.trainable_params.iloc[0] / 1e6:.2f} M ({100 * d.trainable_params.iloc[0] / d.total_params.iloc[0]:.1f}%)",
                             pm(d.mean_dice), pm(d.aji), pm(d.bf1), pm(d.ipe_global_pct, 2), lost_cell(d)])
                if (a, r) in (("edge_sam", 32), ("mobile_sam", 8)):
                    shade.append(len(rows) - 1)
        return T.build_table(["Architecture", "LoRA rank", "Trainable params", "Mean Dice", "AJI", "Boundary F1",
                              "Global IPE (%)", "Seeds with a lost class (mask values)"], rows,
                             [13, 7, 13, 12, 12, 12, 12, 19], style="si", shade_rows=shade)

    def s2():
        rows = []
        for a in ("edge_sam", "mobile_sam", "mobilenet_unet"):
            for r in RANKS:
                ds = [runs(a, r, l) for l in LOSSES]
                rows.append([NAME[a] if r == 2 else "", str(r),
                             " / ".join(f"{d.mean_dice.mean():.3f}" for d in ds),
                             " / ".join(f"{d.ipe_global_pct.mean():.2f}" for d in ds),
                             " / ".join(str(int((d.n_lost > 0).sum())) for d in ds)])
        return T.build_table(["Architecture", "LoRA rank", "Mean Dice (D / P / B / F)", "Global IPE, % (D / P / B / F)",
                              "Seeds with a lost class (D / P / B / F)"], rows, [15, 9, 27, 27, 22], style="si")

    def s3():
        rows = []
        items = [("u8", "full"), ("m8", "full"), ("e8", "full"), ("e32", "full")] + \
                [(k, l) for k in ("e8", "e32") for l in ("dice_only", "pmc", "pmc_bga")]
        for k, l in items:
            a, r = CFG[k]
            d = runs(a, r, l)
            rows.append([f"{NAME[a]} (r={r})", LOSS_LAB[l]] + [f"{d[f'ipe_{c}_pct'].mean():.1f}" for c in CLASSES]
                        + [f"{d[f'dice_{c}'].mean():.3f}" for c in CLASSES])
        hdr = ["Configuration", "Loss"] + [f"IPE (%), {MV[c]}" for c in CLASSES] + [f"Dice, {MV[c]}" for c in CLASSES]
        return T.build_table(hdr, rows, [15, 15] + [8.75] * 8, style="si")

    def storage(arch, rank):
        gt = cells_gt()
        g = gt.groupby("storage_day")
        per = [cells_pred(arch, rank, "full", s).groupby("storage_day") for s in SEEDS]
        rows = []
        for day in sorted(gt.storage_day.unique()):
            gd = g.get_group(day)
            ps = [p.get_group(day) for p in per]
            n = [len(x) for x in ps]
            ratio = np.mean(n) / len(gd)
            am = [x.area_um2.mean() for x in ps]
            dm = [x.dry_mass_pg.mean() for x in ps]
            rows.append([str(day), str(len(gd)), f"{gd.area_um2.mean():.1f} ({gd.area_um2.std(ddof=1):.1f})",
                         f"{gd.dry_mass_pg.mean():.1f} ({gd.dry_mass_pg.std(ddof=1):.1f})", " / ".join(map(str, n)),
                         f"{ratio:.2f}" + (" †" if ratio < 0.6 else ""),
                         f"{np.mean(am):.1f} [{min(am):.1f}–{max(am):.1f}]", f"{np.mean(dm):.1f} [{min(dm):.1f}–{max(dm):.1f}]"])
        return T.build_table(["Day", "Annotated cells", "Area, annotation (µm²)", "Dry mass, annotation (pg)",
                              "Kept components (seeds 42 / 43 / 44)", "Kept / annotated", "Area, prediction (µm²)",
                              "Dry mass, prediction (pg)"], rows, [6, 9, 14, 14, 15, 10, 16, 16], style="si")

    def s6():
        import json
        st = json.loads((ART / "audit" / "claims_check.json").read_text())["storage"]
        rows = []
        lab = {"area_um2": "Area", "dry_mass_pg": "Dry mass", "bias_diff_pg": "Dry-mass difference (prediction − annotation)"}

        def f(x, nd=2):
            return (f"{x:.3f}" if abs(x) < 0.01 else f"{x:.{nd}f}").replace("-", "−")

        def pv(p):
            return "< 0.0001" if p < 0.0001 else (f"{p:.4f}" if p < 0.001 else (f"{p:.3f}" if p < 0.1 else f"{p:.2f}"))
        for q in ("area_um2", "dry_mass_pg"):
            v = st["annotation"][q]
            rows.append(["Annotation", "–", lab[q], f(v["rho"]), pv(v["p_naive"]), pv(v["p_perm_dates"]), f"{f(v['boot95'][0])} to {f(v['boot95'][1])}",
                         f"{f(v['lodo_range'][0])} to {f(v['lodo_range'][1])}"])
        for tag, name in (("edge_sam_r32", "EdgeSAM (r=32)"), ("mobile_sam_r8", "MobileSAM (r=8)")):
            for q in ("area_um2", "dry_mass_pg", "bias_diff_pg"):
                for s in SEEDS:
                    v = st[f"{tag}_s{s}"][q]
                    rows.append([name, str(s), lab[q], f(v["rho"]), pv(v["p_naive"]), pv(v["p_perm_dates"]),
                                 f"{f(v['boot95'][0])} to {f(v['boot95'][1])}", f"{f(v['lodo_range'][0])} to {f(v['lodo_range'][1])}"])
        return T.build_table(["Segmentation", "Seed", "Quantity (per-image median)", "Spearman ρ", "p (66 images as independent)",
                              "p (date permutation)", "95% date-bootstrap interval", "Leave-one-date-out range"], rows,
                             [14, 5, 21, 8, 12, 12, 14, 14], style="si")
    def s7():
        gtn = cells_gt().groupby("storage_day").size()
        rows = []
        for arch, rank in (("edge_sam", 32), ("mobile_sam", 8)):
            for sd in SEEDS:
                p = cells_pred(arch, rank, "full", sd)
                for k, (day, g) in enumerate(p.groupby("storage_day")):
                    rows.append([f"{NAME[arch]} (r={rank})" if (sd == SEEDS[0] and k == 0) else "",
                                 str(sd) if k == 0 else "", str(day), str(len(g)), f"{len(g) / gtn[day]:.2f}",
                                 f"{g.area_um2.mean():.1f} ({g.area_um2.std(ddof=1):.1f})",
                                 f"{g.dry_mass_pg.mean():.1f} ({g.dry_mass_pg.std(ddof=1):.1f})",
                                 f"{g.circularity.mean():.3f}"]
                                + [f"{100 * (g.class_name == c).mean():.1f}" for c in CLASSES])
        return T.build_table(["Configuration", "Seed", "Day", "Kept", "Kept / ann.", "Area (µm²)",
                              "Dry mass (pg)", "Circularity", "Class 1 (%)", "Class 2 (%)", "Class 3 (%)", "Class 4 (%)"],
                             rows, [12, 5, 5, 6, 8, 12, 12, 10, 7.5, 7.5, 7.5, 7.5], style="si")
    return s1, s2, s3, (lambda: storage("edge_sam", 32)), (lambda: storage("mobile_sam", 8)), s6, s7


SI_CONTENT = [
    ("h", "S-I. LoRA Rank Sweep for All Architectures"),
    ("p", "Table S-1 lists the segmentation and integrated-phase results of the full objective for every architecture and LoRA rank (r = 2, 4, 8, 16, and 32), and Fig. S-1 shows every training seed. The main text uses EdgeSAM (r=32) as the primary configuration and MobileSAM (r=8) as the sensitivity comparison (shaded rows); this choice is descriptive and not the result of a selection rule. The dependence on rank was not monotonic for any architecture. Training was unstable in some seeds (Fig. S-2): EdgeSAM lost the spherocyte class in seed 44 at {e_lost_ranks}, MobileSAM (r=32) collapsed in seed {m32_lost_seed} in the second epoch and lost two classes, and MobileNet-UNet lost at least one class in every seed at every rank. Latency and memory were measured for the rank-8 models only (Table 4 of the main text)."),
    ("tcap", "TABLE S-1", "LoRA rank sweep, full objective. Mean ± SD over seeds 42, 43, and 44; shaded: configurations used in the main text."),
    ("table", 0),
    ("fig", FIG / "FigS1_rank_sweep_seeds.png", 6.3),
    ("fcap", "Fig. S-1", "Mean Dice (top) and global IPE (bottom) against LoRA rank, full objective. Dots: individual seeds; crosses: seeds in which a class was lost (Dice < 0.05); line: mean of three seeds."),
    ("fig", FIG / "FigS2_validation_curves.png", 6.3),
    ("fcap", "Fig. S-2", "Validation mean Dice during training of EdgeSAM and MobileSAM at r=8 and r=32, one line per seed; seeds that ended with a lost class are marked in the legend."),
    ("h", "S-II. Loss Ablation at All LoRA Ranks"),
    ("p", "Table S-2 gives the component-wise ablation (D: Dice only; P: Dice with PMC; B: Dice with PMC and BGA; F: full objective with IPP) for every architecture and rank. Over the 15 architecture–rank configurations, the mean global IPE of the full objective was lower than that of Dice alone in {abl_cfg_FltD} and lower than that of Dice with PMC and BGA in {abl_cfg_FltB}; the seed-paired 95% interval excluded zero in {abl_ci_FltD} and {abl_ci_FltB} configurations, respectively. Over the 45 seed-paired comparisons, the full objective had the lower IPE in {abl_seed_FltD} against Dice alone and in {abl_seed_FltB} against Dice with PMC and BGA. A class was lost in {abl_e_lost_D}, {abl_m_lost_D}, and {abl_u_lost_D} of the 15 Dice-only runs of EdgeSAM, MobileSAM, and MobileNet-UNet, and in {abl_e_lost_F}, {abl_m_lost_F}, and {abl_u_lost_F} of the 15 full-objective runs. For EdgeSAM, the AJI of the full objective was higher than that of Dice with PMC and BGA in {abl_e_ajiFgtB} of 15 seed-paired comparisons, and its Dice was higher than with Dice alone in {abl_e_diceFgtD} of 15."),
    ("tcap", "TABLE S-2", "Loss ablation at all LoRA ranks (means over seeds 42, 43, and 44)."),
    ("table", 1),
    ("h", "S-III. Class-Specific Integrated-Phase Error and Dice"),
    ("p", "Table S-3 lists the class-specific IPE and Dice of the configurations of Table 1 and of the EdgeSAM loss ablation. A class that is not predicted in an image gives an IPE of 100% for that image, so the class-specific IPE is dominated by misclassification between morphologies; the global IPE over the whole foreground is the measure of integrated-phase preservation used in the main text."),
    ("tcap", "TABLE S-3", "Class-specific IPE (%) and Dice by mask value (1 discocyte, 2 spherocyte, 3 echinocyte, 4 stomatocyte), means over seeds 42, 43, and 44, including seeds in which a class was lost."),
    ("table", 2),
    ("h", "S-IV. Segmentation Accuracy versus Inference Throughput"),
    ("p", "Fig. S-3 relates the mean Dice of the rank-8 models to their FP32 forward-pass throughput (Table 4 of the main text); both quantities refer to rank 8. Dice was evaluated with the PyTorch checkpoints and is plotted at the same value for both runtimes."),
    ("fig", FIG / "FigS3_accuracy_vs_throughput.png", 4.6),
    ("fcap", "Fig. S-3", "Mean Dice (mean ± SD over three seeds) against FP32 forward-pass throughput (mean of three timing passes) of the rank-8 models. Circles: PyTorch; squares: ONNX Runtime; class losses are noted next to each model."),
    ("h", "S-V. Phase-Gradient Alignment of Predicted Contours"),
    ("p", "Fig. S-4 shows the predicted cell contours of EdgeSAM (r=32) on the phase-gradient magnitude for one validation image, as a qualitative illustration of the boundary cue used by the BGA term. Its quantitative effect is assessed by the loss ablation (Table 3 of the main text and Table S-2)."),
    ("fig", PH / "FigS4_gradient_alignment.png", 6.3),
    ("fcap", "Fig. S-4", "Phase map (rad), phase-gradient magnitude, and contours of EdgeSAM (r=32) (training seed 42) on the phase-gradient magnitude, for the full 768 × 768 px field (110.7 µm) of the validation image with the most annotated foreground. Scale bars: 20 µm."),
    ("h", "S-VI. Representative Phase Images over the Storage Period"),
    ("p", "Fig. S-5 shows one validation phase map from each of five storage days, with the EdgeSAM (r=32) contours."),
    ("fig", PH / "FigS5_storage_images.png", 6.3),
    ("fcap", "Fig. S-5", "Phase maps from storage days 0, 8, 19, 27, and 47 (first validation image of each day); bottom row: EdgeSAM (r=32) contours (seed 42)."),
    ("h", "S-VII. Storage-Period Measurements per Day"),
    ("p", "Tables S-4 and S-5 give the per-day measurements behind Fig. 8 for EdgeSAM (r=32) and MobileSAM (r=8). Cells are 4-connected foreground components of 15 to 200 µm² that do not touch an unannotated region; phase-wrapped cells are excluded. Annotation columns give the mean and, in parentheses, the cell-level SD; prediction columns give the mean over seeds of the per-day cell mean and, in brackets, the smallest and largest seed value. Predicted components are not cell counts. On days marked †, fewer than 0.6 kept components were found per annotated cell, and the predicted means describe only the components that passed the filters. Fig. S-6 shows the kept-component ratio and the fate of all predicted components per day."),
    ("tcap", "TABLE S-4", "Per-day measurements: annotation and EdgeSAM (r=32), seeds 42, 43, and 44."),
    ("table", 3),
    ("tcap", "TABLE S-5", "Per-day measurements: annotation and MobileSAM (r=8), seeds 42, 43, and 44."),
    ("table", 4),
    ("fig", PH / "FigS6_component_fate.png", 6.3),
    ("fcap", "Fig. S-6", "Predicted components of EdgeSAM (r=32) per storage day: (a) kept components divided by annotated cells (markers: seeds); (b) fate of the predicted components inside the annotated field (mean of three seeds, wrapped components included), with the annotated cells kept by the same filters."),
    ("h", "S-VIII. Date-Clustered Storage Statistics"),
    ("p", "Table S-6 gives, for each segmentation and training seed, the image-level Spearman correlation with storage day, the permutation p-value with the 11 acquisition dates permuted as whole blocks (exact test over all 11! = 39,916,800 permutations, so the p-values carry no Monte Carlo error), the 95% date-bootstrap interval (20,000 resamples), and the range of the correlation when each date is left out, for projected area, dry mass, and the per-image difference between predicted and annotation-derived dry mass. With 11 clusters and an undocumented donor structure, these tests are approximate and exploratory. For comparison, the table also gives the p-value of the image-level Spearman test that treats the 66 images as independent. That test is anti-conservative here, because the six images of a date share the same samples: for the EdgeSAM (r=32) dry mass it gives p = {e32_dm_pnaive} for seeds 42, 43, and 44, whereas the date-clustered test gives p = {e32_dm_p}. Conclusions in the main text are based on the date-clustered test."),
    ("tcap", "TABLE S-6", "Date-clustered storage statistics of the per-image medians (wrapped cells excluded)."),
    ("table", 5),
    ("h", "S-IX. Per-Seed Storage Measurements"),
    ("p", "Table S-7 lists, for EdgeSAM (r=32) and MobileSAM (r=8), every training seed and storage day: the number of kept components, their ratio to the annotated cells, the mean and cell-level SD of projected area and dry mass, the mean circularity, and the share of components assigned to each class (mask values 1 to 4: discocyte, spherocyte, echinocyte, stomatocyte). Cells are not pooled across seeds; phase-wrapped cells are excluded. The annotation values for each day are given in Tables S-4 and S-5."),
    ("tcap", "TABLE S-7", "Per-seed, per-day measurements of the kept predicted components (six images per day). Kept: number of kept components; Kept / ann.: ratio to the annotated cells of the day; area and dry mass: mean (cell-level SD); Class 1 to 4: share of components by mask value."),
    ("table", 6),
]


def build_si(hl: str, out: Path) -> None:
    T.HL["color"] = hl
    d = docx.Document(SI_IN)
    body = list(d.element.body.iterchildren())
    tpl_h, tpl_p, tpl_tcap, tpl_fig, tpl_fcap = body[8], body[9], body[10], body[12], body[13]
    sect = body[-1]
    old_heads = [T.para_text(e) for e in body[8:-1] if e.tag == qn("w:p") and re.match(r"S-[IVX]+\.", T.para_text(e))]
    for e in body[8:-1]:
        e.getparent().remove(e)
    LOG.append(f"| SI | rebuild | sections {'; '.join(old_heads)} | sections {'; '.join(c[1] for c in SI_CONTENT if c[0] == 'h')} | see entries below |")
    tables = si_tables()
    for item in SI_CONTENT:
        kind = item[0]
        used: Dict[str, str] = {}
        if kind == "h":
            el = T.clone_par(tpl_h, item[1])
            for r in el.findall(qn("w:r")):
                T._set(r.find(qn("w:rPr")), "w:b")
        elif kind == "p":
            el = T.clone_par(tpl_p, fill(item[1], used))
        elif kind in ("tcap", "fcap"):
            el = T.clone_par(tpl_tcap if kind == "tcap" else tpl_fcap, fill(item[2], used), prefix=((item[1], True), (" ", False)))
        elif kind == "table":
            el = tables[item[1]]()
        elif kind == "fig":
            el = T.picture_par(d, tpl_fig, item[1], item[2])
        sect.addprevious(el)
        if kind != "fig" and kind != "table":
            LOG.append(f"| SI | {kind} | – | {T.para_text(el)} | {', '.join(sorted(used))} |")
        elif kind == "fig":
            LOG.append(f"| SI | figure | – | {item[1].relative_to(ART.parent)} | {'make_data_figures.py (placeholder)' if 'placeholders' in str(item[1]) else 'make_manuscript_figures.py'} |")
        else:
            LOG.append(f"| SI | table | – | Table from si_tables()[{item[1]}] | runs_frame(), cells_gt/cells_pred CSVs, {('claims_check.json' if item[1] == 5 else 'compiled cell_analysis')} |")
    d.save(out)


def supplementary_data(out: Path) -> None:
    rows = []
    for arch, rank in (("edge_sam", 32), ("mobile_sam", 8)):
        gtn = cells_gt().groupby("storage_day").size()
        for s in SEEDS:
            p = cells_pred(arch, rank, "full", s)
            for day, g in p.groupby("storage_day"):
                r = {"configuration": f"{NAME[arch]} (r={rank})", "seed": s, "storage_day": day, "kept_components": len(g),
                     "images": g.stem.nunique(), "kept_over_annotated": len(g) / gtn[day]}
                for q in ("area_um2", "dry_mass_pg", "circularity"):
                    r[f"{q}_mean"], r[f"{q}_sd"] = g[q].mean(), g[q].std(ddof=1)
                for c in CLASSES:
                    r[f"share_{MV[c]}_{c}_pct"] = 100 * (g.class_name == c).mean()
                rows.append(r)
    ann = []
    for day, g in cells_gt().groupby("storage_day"):
        r = {"storage_day": day, "cells": len(g), "images": g.stem.nunique()}
        for q in ("area_um2", "dry_mass_pg", "circularity"):
            r[f"{q}_mean"], r[f"{q}_sd"] = g[q].mean(), g[q].std(ddof=1)
        for c in CLASSES:
            r[f"share_{MV[c]}_{c}_pct"] = 100 * (g.class_name == c).mean()
        ann.append(r)
    with pd.ExcelWriter(out) as xw:
        pd.DataFrame(rows).round(4).to_excel(xw, sheet_name="predictions_per_seed_day", index=False)
        pd.DataFrame(ann).round(4).to_excel(xw, sheet_name="annotation_per_day", index=False)
        pd.DataFrame([{"note": "Wrapped cells excluded. Cells: 4-connected foreground components, 15-200 um2, not touching an "
                               "unannotated region. Prediction sheets: one row per segmentation, training seed and storage day; "
                               "means and SDs are over the cells of the six images of that day."}]
                     ).to_excel(xw, sheet_name="readme", index=False)


UNVERIFIED = """
## Numbers kept from the final-review version that cannot be recomputed in this repository

These need `data/processed` (phase maps and masks), which is not in the repository. They were left unchanged and are
recomputed by `artifacts/code/make_data_figures.py` (writes `artifacts/audit/server_numbers.json`). Compare and update
them after running that script on the machine with the data.

| Location | Statement | Recomputed by |
|---|---|---|
| ms ¶29 (Section 2.2) | unannotated part of the grid, "14% of the grid on average" | server_numbers.json `unannotated_fraction_mean_pct` (validation only; 14% may refer to all images) |
| ms ¶30 (Section 2.2) | training-map offsets "−0.24 to −0.02 rad" for the 145 X_* maps | dataset manifest (`phase_offset_rad`), not in the repository |
| ms ¶82 (Section 3.1) | day-15 foreground fraction 0.48 (0.40 to 0.54) | `day15_foreground_fraction` |
| ms ¶149 (Section 4.6) | 68% / 82% of spherocyte / stomatocyte pixels predicted as discocyte (MobileNet-UNet); 68% / 52% / 56% stomatocyte pixels correct and 22–30% as discocyte (SAM models); bridged-cell rates 35.1–36.9%, 33.4–41.8%, 30.5–35.3%, 13.2% | `bridging_and_confusion` (now for every seed) |
| ms ¶162 (Section 4.8) | day 15: 61.3 kept components, 116 fragments < 15 µm², 22 components > 200 µm², 46 touching, predicted/annotated foreground area 0.98 | `day15_fate_edge_sam_r32_mean_of_seeds`, `day15_pred_over_ann_foreground_area` |

Note: counting the components of the saved day-15 predictions over the whole 768 × 768 grid (not only the annotated field)
gives more fragments (mean 435) and more large components (mean 38); the pipeline counts components inside the annotated
field only (`qpi/analysis/common.py: components`), so the two counts are not comparable and the whole-grid count does not
contradict the values above.
"""


def write_log(path: Path) -> None:
    lines = ["# Change log: final-review manuscript and SI → publication version", "",
             "Built by `artifacts/code/build_final_docs.py` from the Word files in `artifacts/manuscript_in_prog/Claudes_Update/`.",
             "Every number written comes from `outputs/` (read only) through `artifacts/code/manuscript_facts.py`; the source of each",
             "key is listed in the second table. Paragraph numbers (¶) are body-element indices of the final-review manuscript.",
             "Highlighting: every changed or new run is yellow in the final files; in the `_review_copy` files the changes of this",
             "revision are green and the earlier yellow is kept.", "",
             "## Edits", "", "| Where | Type | Old | New | Number keys |", "|---|---|---|---|---|"]
    lines += [l.replace("\n", " ").replace("\x00", "[equation]") for l in LOG]
    lines += ["", "## Sources of every number key used", "", "| Key | Value | Source |", "|---|---|---|"]
    for k in sorted(USED):
        lines.append(f"| `{k}` | {F[k].text} | {F[k].source} |")
    lines += [UNVERIFIED]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    OUTD.mkdir(exist_ok=True)
    extra_facts()
    build_ms("yellow", OUTD / "revised_manuscript_final.docx")
    build_si("yellow", OUTD / "supplementary_information_final.docx")
    log_final = list(LOG)
    LOG.clear()
    build_ms("green", OUTD / "revised_manuscript_final_review_copy.docx")
    build_si("green", OUTD / "supplementary_information_final_review_copy.docx")
    LOG[:] = log_final
    write_log(OUTD / "change_log.md")
    print("wrote", *sorted(p.name for p in OUTD.iterdir()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
