from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .bimamba_freq import BiMambaFreqConfig, BiMambaFreqResidualBlock, LowRankLinear


@dataclass
class BiMambaFreqForecastConfig:
    input_dim: int
    seq_len: int
    pred_len: int
    d_model: int = 128
    n_layers: int = 4
    d_state: int = 16
    low_rank: int = 32
    freq_group_dt: Tuple[float, ...] = (0.1, 1.0, 10.0)
    dropout: float = 0.1
    target_dim: Optional[int] = None
    pool: str = "mean"
    pscan: bool = True


class BiMambaFreqForecastModel(nn.Module):
    def __init__(self, config: BiMambaFreqForecastConfig):
        super().__init__()
        self.forecast_config = config
        self.target_dim = config.target_dim or config.input_dim
        backbone_cfg = BiMambaFreqConfig(
            input_dim=config.input_dim,
            d_model=config.d_model,
            n_layers=config.n_layers,
            num_classes=1,
            d_state=config.d_state,
            low_rank=config.low_rank,
            freq_group_dt=config.freq_group_dt,
            dropout=config.dropout,
            pool=config.pool,
            pscan=config.pscan,
        )
        self.backbone_config = backbone_cfg
        self.W_down_shared = LowRankLinear(
            backbone_cfg.input_dim, backbone_cfg.d_model, rank=backbone_cfg.input_rank, bias=backbone_cfg.bias
        )
        self.W_up_shared = LowRankLinear(
            backbone_cfg.d_inner, backbone_cfg.d_model, rank=backbone_cfg.model_rank, bias=backbone_cfg.bias
        )
        self.layers = nn.ModuleList(
            [BiMambaFreqResidualBlock(backbone_cfg, shared_output_proj=self.W_up_shared) for _ in range(backbone_cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(backbone_cfg.d_model)
        self.temporal_head = nn.Linear(config.seq_len, config.pred_len)
        self.feature_head = nn.Linear(backbone_cfg.d_model, self.target_dim)

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> Dict[str, torch.Tensor]:
        hidden = self.W_down_shared(x)
        consistency_terms = []
        layer_consistency = []
        for layer in self.layers:
            if return_aux:
                hidden, _, consistency_loss, aux = layer(hidden, return_aux=True)
                layer_consistency.append(aux["consistency_loss"])
            else:
                hidden, _, consistency_loss = layer(hidden, return_aux=False)
            consistency_terms.append(consistency_loss)

        hidden = self.final_norm(hidden)
        temporal = self.temporal_head(hidden.transpose(1, 2)).transpose(1, 2)
        prediction = self.feature_head(temporal)
        consistency_loss = torch.stack(consistency_terms).mean()
        outputs: Dict[str, torch.Tensor] = {
            "prediction": prediction,
            "consistency_loss": consistency_loss,
        }
        if return_aux:
            outputs["hidden"] = hidden
            outputs["layer_consistency"] = torch.stack(layer_consistency) if layer_consistency else torch.empty(0)
        return outputs
