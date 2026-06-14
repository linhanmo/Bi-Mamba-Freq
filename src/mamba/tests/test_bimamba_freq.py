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
)
from training.distillation import direction_alignment_loss, freq_alignment_loss


def test_bimamba_freq_forward_and_interpret_shapes():
    torch.manual_seed(123)
    config = BiMambaFreqConfig(
        input_dim=6,
        d_model=12,
        n_layers=2,
        num_classes=3,
        low_rank=4,
        bidirectional_merge="concat",
        pscan=False,
    )
    model = BiMambaFreqModel(config)
    x = torch.randn(3, 10, 6)
    outputs = model(x, return_interpret=True)

    assert outputs["logits"].shape == (3, 3)
    assert outputs["volatility"].shape == (3, 1)
    assert outputs["features"].shape == (3, 24)

    interpret = outputs["interpret"]
    assert interpret["delta_fwd"].shape == (3, 2, 10, config.d_inner)
    assert interpret["delta_bwd"].shape == (3, 2, 10, config.d_inner)
    assert interpret["bi_divergence"].shape == (3, 2, 10)
    assert interpret["h_fwd"].shape == (3, 2, 10, config.d_model)
    assert interpret["h_bwd"].shape == (3, 2, 10, config.d_model)
    assert interpret["group_states_fwd"]["high"].shape[0:3] == (3, 2, 10)


def test_bimamba_freq_distill_losses_are_finite():
    torch.manual_seed(456)
    student = torch.randn(2, 8, 16)
    teacher = torch.randn(2, 8, 16)
    loss_dir = direction_alignment_loss(student, teacher)
    loss_freq = freq_alignment_loss(student, teacher, levels=2)
    assert torch.isfinite(loss_dir)
    assert torch.isfinite(loss_freq)
    assert loss_dir.item() >= 0.0
    assert loss_freq.item() >= 0.0
