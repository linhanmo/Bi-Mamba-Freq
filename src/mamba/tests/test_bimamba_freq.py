import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
MAMBA_SRC_DIR = SRC_DIR / "mamba"
for p in (SRC_DIR, MAMBA_SRC_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from models import (
    BiMambaFreqConfig,
    BiMambaFreqModel,
    bimamba_freq_multitask_loss,
)


def test_bimamba_freq_forward_and_interpret_shapes():
    torch.manual_seed(123)
    config = BiMambaFreqConfig(
        input_dim=6,
        d_model=12,
        n_layers=2,
        num_classes=1,
        low_rank=4,
        pscan=False,
    )
    model = BiMambaFreqModel(config)
    x = torch.randn(3, 10, 6)
    outputs = model(x, return_interpret=True)

    assert outputs["logits"].shape == (3, 1)
    assert outputs["classification"].shape == (3, 1)
    assert outputs["volatility"].shape == (3, 1)
    assert outputs["features"].shape == (3, 12)
    assert outputs["consistency_loss"].ndim == 0

    interpret = outputs["interpret"]
    assert interpret["delta_fused"].shape == (3, 2, 10, config.d_inner)
    assert interpret["delta_low"].shape == (3, 2, 10, config.d_inner)
    assert interpret["delta_mid"].shape == (3, 2, 10, config.d_inner)
    assert interpret["delta_high"].shape == (3, 2, 10, config.d_inner)
    assert interpret["h_fwd"].shape == (3, 2, 10, config.d_inner)
    assert interpret["h_bwd"].shape == (3, 2, 10, config.d_inner)
    assert interpret["h_bi"].shape == (3, 2, 10, config.d_inner)
    assert interpret["y_freq"].shape == (3, 2, 10, config.d_inner)
    assert interpret["gates"].shape == (3, 2, 10, 3, config.d_inner)
    assert interpret["consistency_map"].shape == (3, 2, 10)


def test_bimamba_freq_multitask_loss_is_finite():
    torch.manual_seed(456)
    config = BiMambaFreqConfig(
        input_dim=6,
        d_model=8,
        n_layers=1,
        num_classes=1,
        low_rank=4,
        pscan=False,
    )
    model = BiMambaFreqModel(config)
    x = torch.randn(2, 8, 6)
    outputs = model(x)
    cls_target = torch.randint(0, 2, (2, 1)).float()
    vol_target = torch.rand(2, 1)
    losses = bimamba_freq_multitask_loss(outputs, cls_target, vol_target, alpha=1.0, beta=0.5, gamma=0.1)

    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["cls_loss"])
    assert torch.isfinite(losses["vol_loss"])
    assert torch.isfinite(losses["consistency_loss"])
    assert losses["loss"].item() >= 0.0
