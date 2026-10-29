"""After the LoRA merge the deployed network has the same structure at every rank (so latency and memory cannot depend on the rank)."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from qpi.models.lora_utils import inject_lora_into_model, merge_lora_weights  # noqa: E402
except ModuleNotFoundError:        # qpi/models/__init__ needs the vendored model code; lora_utils itself only needs torch
    import importlib.util
    _spec = importlib.util.spec_from_file_location("lora_utils_alone", Path(__file__).resolve().parents[1] / "qpi" / "models" / "lora_utils.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    inject_lora_into_model, merge_lora_weights = _mod.inject_lora_into_model, _mod.merge_lora_weights


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.encoder = nn.ModuleDict({"stage1": nn.Conv2d(1, 16, 3, padding=1), "stage2": nn.Conv2d(16, 16, 3, padding=1),
                                      "blocks": nn.Linear(16, 16)})
        self.head = nn.Conv2d(16, 5, 1)

    def forward(self, x):
        x = self.encoder["stage2"](torch.relu(self.encoder["stage1"](x)))
        x = self.encoder["blocks"](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.head(x)


def _merged(r):
    m = Toy()
    inject_lora_into_model(m, r=r, lora_alpha=float(r), strategy="encoder_only")
    for p in m.parameters():                                   # make the LoRA update non-zero
        if p.requires_grad:
            nn.init.normal_(p, std=0.02)
    return merge_lora_weights(m).eval()


def test_merged_toy_network_is_identical_in_structure_for_r8_and_r32():
    a, b = _merged(8), _merged(32)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    assert all(sa[k].shape == sb[k].shape for k in sa)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    assert not any("lora" in k for k in sa)
    x = torch.randn(1, 1, 16, 16)
    assert a(x).shape == b(x).shape


@pytest.mark.parametrize("arch", ["edge_sam", "mobile_sam", "mobilenet_unet"])
def test_merged_real_architectures_do_not_depend_on_the_rank(arch):
    from qpi.config import load_config
    from qpi.experiments import build_grid, resolve, run_config
    try:
        from qpi.models import build_model
        cfg = load_config()
        shapes = {}
        for r in (8, 32):
            rc = run_config(cfg, resolve(build_grid(cfg), f"{arch}_lora_r{r}"))
            model = build_model(rc, None, pretrained=False)
            model.merge_lora() if hasattr(model, "merge_lora") else merge_lora_weights(model)
            shapes[r] = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    except (ImportError, ModuleNotFoundError) as exc:           # vendored model code or optional package not installed here
        pytest.skip(f"model code not importable here: {exc}")
    assert shapes[8] == shapes[32]
    assert not any("lora" in k for k in shapes[32])
