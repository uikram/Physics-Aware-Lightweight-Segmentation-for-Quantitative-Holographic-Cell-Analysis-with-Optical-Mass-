"""Architectures: fingerprint of the trainable parameters (identical to the submitted models), LoRA rank changes only
the adapters on a fixed set of layers, full fine-tuning trains everything, LoRA merge equivalence, checkpoint
save / load."""
import pytest
import torch

from qpicell.models import build
from qpicell.models.lora import count_parameters, merge, parameter_report

TRAINABLE = {("edge_sam", 8): 534205, ("edge_sam", 32): 1471189, ("edge_sam", 64): 2720501,
             ("mobile_sam", 8): 673184, ("mobile_sam", 32): 851360,
             ("mobilenet_unet", 8): 5584173, ("mobilenet_unet", 32): 6035781}
ARCHS = ["edge_sam", "mobile_sam", "mobilenet_unet"]


@pytest.mark.parametrize("arch,rank", sorted(TRAINABLE))
def test_trainable_parameters(cfg, arch, rank):
    assert count_parameters(build(cfg, arch, rank, pretrained=False))["trainable"] == TRAINABLE[(arch, rank)]


@pytest.mark.parametrize("arch", ARCHS)
def test_rank_changes_only_the_adapters(cfg, arch):
    reps = {r: parameter_report(build(cfg, arch, r, pretrained=False), r) for r in cfg["conditions"]["lora_ranks"]}
    names = {tuple(v["lora_layer_names"]) for v in reps.values()}
    assert len(names) == 1 and reps[8]["lora_layers"] > 0                          # same adapted layers at every rank
    per = {v["lora_parameters"] / r for r, v in reps.items()}
    assert len({round(x, 6) for x in per}) == 1                                     # adapter parameters linear in the rank
    assert len({v["trainable_non_lora"] for v in reps.values()}) == 1               # nothing else depends on the rank
    assert len({v["total"] - v["lora_parameters"] for v in reps.values()}) == 1      # frozen base unchanged


@pytest.mark.parametrize("arch", ARCHS)
def test_full_finetune_trains_everything(cfg, arch):
    m = build(cfg, arch, None, pretrained=False); rep = parameter_report(m, None)
    assert rep["trainable"] == rep["total"] and rep["lora_layers"] == 0 and rep["lora_parameters"] == 0


@pytest.mark.parametrize("arch", ["edge_sam", "mobilenet_unet"])
def test_lora_merge_keeps_output(cfg, arch):
    torch.manual_seed(0)
    m = build(cfg, arch, 8, pretrained=False).eval()
    for n, p in m.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.02)
    S = cfg["models"][arch]["image_size"]; x = torch.randn(1, 1, S, S)
    with torch.no_grad():
        a = m(x); b = merge(m)(x)
    assert torch.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("arch,rank", [("edge_sam", 16), ("mobilenet_unet", None)])
def test_checkpoint_roundtrip(cfg, arch, rank, tmp_path):
    torch.manual_seed(0)
    m = build(cfg, arch, rank, pretrained=False).eval()
    for p in m.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=0.01)
    ck = tmp_path / "last.pt"
    torch.save({"epoch": 3, "model_state": m.state_dict()}, ck)
    n = build(cfg, arch, rank, pretrained=False).eval()
    st = torch.load(ck, map_location="cpu", weights_only=True); n.load_state_dict(st["model_state"], strict=True)
    S = cfg["models"][arch]["image_size"]; x = torch.randn(1, 1, S, S)
    with torch.no_grad():
        assert torch.equal(m(x), n(x)) and st["epoch"] == 3
