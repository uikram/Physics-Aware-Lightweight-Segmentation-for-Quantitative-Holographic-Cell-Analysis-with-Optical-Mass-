"""Loss terms: arm definitions, cIPP behaviour, block-averaging invariance, float32."""
import pytest
import torch
import torch.nn.functional as F

from qpicell.losses.cell_ipp import CellIPP
from qpicell.losses.objective import Objective


def _scene(S=64):
    """Two touching square cells on a flat background; phase 1 rad inside the cells."""
    cells = torch.zeros(1, S, S, dtype=torch.long)
    cells[0, 20:40, 10:30] = 1; cells[0, 20:40, 30:50] = 2
    terr = cells.clone(); terr[0, 17:43, 7:30][terr[0, 17:43, 7:30] == 0] = 1; terr[0, 17:43, 30:53][terr[0, 17:43, 30:53] == 0] = 2
    contacts = torch.zeros(1, S, S); contacts[0, 20:40, 29:31] = 1
    phi = (cells > 0).float() * 1.0
    target = (cells > 0).long()
    return cells, terr, contacts, phi, target


def test_arms(cfg):
    assert list(cfg["arms"]) == ["dice", "dice_pmc", "dice_pmc_bga", "dice_pmc_bga_ipp", "dice_cipp", "dice_pmc_cipp", "dice_pmc_bga_cipp"]
    for name, a in cfg["arms"].items():
        assert not (a["ipp"] and a["cipp"]), name
    assert cfg["loss"]["lambda_cipp"] == 0.1 and cfg["loss"]["lambda_ipp"] == 0.1


def test_ipp_and_cipp_never_combined(cfg):
    c = {**cfg, "arms": {**cfg["arms"], "bad": {"pmc": False, "bga": False, "ipp": True, "cipp": True}}}
    with pytest.raises(ValueError):
        Objective(c, "bad")


def test_cipp_perfect_and_failures():
    cells, terr, contacts, phi, target = _scene()
    L = CellIPP()
    valid = torch.ones_like(phi)
    perfect = L((cells > 0).float(), phi, valid, cells, terr, contacts)
    assert perfect["cipp_phase"] < 1e-6 and perfect["cipp_far"] == 0
    missed = (cells == 1).float()                                    # cell 2 missed
    assert L(missed, phi, valid, cells, terr, contacts)["cipp_phase"] > 0.4
    merged = (cells > 0).float()                                     # contact line predicted as foreground
    assert perfect["cipp_separation"] == pytest.approx(merged.mul(contacts).sum().item() / contacts.sum().item())
    split = merged.clone(); split[0, 20:40, 29:31] = 0
    assert L(split, phi, valid, cells, terr, contacts)["cipp_separation"] == 0
    extra = merged.clone(); extra[0, 50:60, 50:60] = 1; phi2 = phi.clone(); phi2[0, 50:60, 50:60] = 1
    assert L(extra, phi2, valid, cells, terr, contacts)["cipp_far"] > 0


def test_cipp_block_average_invariance():
    """Loss maps 4x larger than the cell maps (MobileSAM) give the same value as at the cell-map size."""
    cells, terr, contacts, phi, _ = _scene(64)
    torch.manual_seed(0)
    p = torch.rand(1, 64, 64)
    small = CellIPP()(p, phi, torch.ones_like(p), cells, terr, contacts)
    up = lambda x: F.interpolate(x[:, None], scale_factor=4, mode="nearest")[:, 0]
    big = CellIPP()(up(p), up(phi), torch.ones(1, 256, 256), cells, terr, contacts)
    for k in small:
        assert big[k].item() == pytest.approx(small[k].item(), rel=1e-5, abs=1e-7), k


def test_objective_all_arms_float32_and_gradients(cfg):
    cells, terr, contacts, phi, target = _scene()
    logits = torch.randn(1, cfg["data"]["num_classes"], 64, 64, dtype=torch.float16, requires_grad=True)
    for arm in cfg["arms"]:
        total, comp = Objective(cfg, arm)(logits, target, phi[:, None], cells, terr, contacts)
        assert total.dtype == torch.float32 and torch.isfinite(total)
        g, = torch.autograd.grad(total, logits)
        assert torch.isfinite(g).all()
        a = cfg["arms"][arm]
        assert (comp["cipp"] != 0) == a["cipp"] and (comp["ipp"] != 0) == a["ipp"]


@pytest.mark.parametrize("arch,rank", [("edge_sam", 8), ("edge_sam", None), ("mobilenet_unet", 8), ("mobilenet_unet", None),
                                       ("mobile_sam", 8)])
def test_every_loss_differentiable_through_each_architecture(cfg, arch, rank):
    """Forward + backward of the real architecture with every loss configuration (synthetic 2-cell scene at the model
    size; cell maps at 256 as in training): finite loss and finite gradients on the trainable parameters."""
    from qpicell.models import build
    torch.manual_seed(0)
    S = cfg["models"][arch]["image_size"]; C = min(S, cfg["cells"]["cell_map_size"]); f = S // C
    cells, terr, contacts, phi, target = _scene(C)
    up = lambda x: F.interpolate(x[:, None].float(), scale_factor=f, mode="nearest")[:, 0]
    target_S = up(target).long(); phi_S = up(phi).float()[:, None]
    model = build(cfg, arch, rank, pretrained=False).train()
    x = (phi_S - phi_S.mean()) / phi_S.std()
    logits = model(x)
    params = [p for p in model.parameters() if p.requires_grad]
    for arm in cfg["arms"]:
        total, comp = Objective(cfg, arm)(logits, target_S, phi_S, cells, terr, contacts)
        assert torch.isfinite(total), arm
        grads = torch.autograd.grad(total, params, retain_graph=True, allow_unused=True)
        assert any(g is not None and g.abs().sum() > 0 for g in grads), arm
        assert all(g is None or torch.isfinite(g).all() for g in grads), arm
        assert comp["cipp_separation"] == 0                       # eta = 0: no separation term


def test_objective_rejects_non_float32_phase(cfg):
    cells, terr, contacts, phi, target = _scene()
    logits = torch.randn(1, 5, 64, 64)
    with pytest.raises(TypeError):
        Objective(cfg, "dice_pmc")(logits, target, phi[:, None].double(), cells, terr, contacts)
