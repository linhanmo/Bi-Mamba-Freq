import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mambapy.mamba import RMSNorm
    from mambapy.pscan import pscan
except ImportError:
    mamba_src = Path(__file__).resolve().parents[1] / "mamba"
    if str(mamba_src) not in sys.path:
        sys.path.append(str(mamba_src))
    from mambapy.mamba import RMSNorm
    from mambapy.pscan import pscan


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.U = nn.Linear(in_features, rank, bias=False)
        self.V = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.V(self.U(x))


class DeltaProjector(nn.Module):
    def __init__(
        self,
        d_inner: int,
        dt_rank: int,
        scale: float,
        dt_min: float,
        dt_max: float,
        dt_init: str,
        dt_scale: float,
        dt_init_floor: float,
        rms_norm_eps: float,
        inner_layernorms: bool,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.dt_init_floor = dt_init_floor
        self.in_proj = nn.Linear(d_inner, dt_rank, bias=False)
        self.out_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.dt_norm = RMSNorm(dt_rank, rms_norm_eps, False) if inner_layernorms else None

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.out_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.out_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        with torch.no_grad():
            dt = torch.exp(
                torch.rand(d_inner, dtype=self.out_proj.bias.dtype)
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            self.out_proj.bias.copy_(_inverse_softplus(dt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = self.in_proj(x)
        if self.dt_norm is not None:
            dt = self.dt_norm(dt)
        delta = self.out_proj(dt)
        return F.softplus(delta).clamp(min=self.dt_init_floor) * self.scale


@dataclass
class BiMambaFreqConfig:
    input_dim: int
    d_model: int
    n_layers: int
    num_classes: int = 1

    dt_rank: Union[int, str] = "auto"
    d_state: int = 16
    expand_factor: int = 2
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02
    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False
    pscan: bool = True

    freq_groups: int = 3
    freq_group_dt: Tuple[float, ...] = (0.1, 1.0, 10.0)
    low_rank: Optional[int] = None
    bidirectional_merge: str = "sum"
    project_concat_for_residual: bool = False
    pool: str = "mean"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        self.d_inner = self.expand_factor * self.d_model
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        if self.low_rank is None:
            self.low_rank = max(1, self.d_model // 4)
        if self.freq_groups != 3:
            raise ValueError("Bi-Delta-FD Mamba expects freq_groups == 3 for low/mid/high branches")
        if len(self.freq_group_dt) != self.freq_groups:
            raise ValueError("len(freq_group_dt) must equal freq_groups")
        if self.bidirectional_merge != "sum":
            raise ValueError("Bi-Delta-FD Mamba requires bidirectional_merge='sum'")
        if self.num_classes != 1:
            raise ValueError("The multitask head is binary and requires num_classes == 1")
        self.bi_output_dim = self.d_model
        self.group_names = ["low", "mid", "high"]
        self.input_rank = max(1, min(self.low_rank, self.input_dim, self.d_model))
        self.model_rank = max(1, min(self.low_rank, self.d_model))


class BiMambaFreq(nn.Module):
    def __init__(self, config: BiMambaFreqConfig):
        super().__init__()
        self.config = config
        self.in_proj = LowRankLinear(config.d_model, 2 * config.d_inner, rank=config.model_rank, bias=config.bias)
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )
        self.bc_proj = nn.Linear(config.d_inner, 2 * config.d_state, bias=False)
        self.fused_delta_proj = DeltaProjector(
            d_inner=config.d_inner,
            dt_rank=config.dt_rank,
            scale=1.0,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init=config.dt_init,
            dt_scale=config.dt_scale,
            dt_init_floor=config.dt_init_floor,
            rms_norm_eps=config.rms_norm_eps,
            inner_layernorms=config.inner_layernorms,
        )
        self.freq_delta_proj = nn.ModuleDict(
            {
                name: DeltaProjector(
                    d_inner=config.d_inner,
                    dt_rank=config.dt_rank,
                    scale=scale,
                    dt_min=config.dt_min,
                    dt_max=config.dt_max,
                    dt_init=config.dt_init,
                    dt_scale=config.dt_scale,
                    dt_init_floor=config.dt_init_floor,
                    rms_norm_eps=config.rms_norm_eps,
                    inner_layernorms=config.inner_layernorms,
                )
                for name, scale in zip(config.group_names, config.freq_group_dt)
            }
        )
        self.freq_gate = nn.Linear(config.d_inner * config.freq_groups, config.d_inner * config.freq_groups, bias=True)
        self.B_layernorm = RMSNorm(config.d_state, config.rms_norm_eps, False) if config.inner_layernorms else None
        self.C_layernorm = RMSNorm(config.d_state, config.rms_norm_eps, False) if config.inner_layernorms else None

        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

    def _project_xz(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, length, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)
        x_branch = self.conv1d(x_branch.transpose(1, 2))[:, :, :length].transpose(1, 2)
        x_branch = F.silu(x_branch)
        z_branch = F.silu(z_branch)
        return x_branch, z_branch

    def _project_bc(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C = self.bc_proj(x).chunk(2, dim=-1)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return B, C

    def _scan(
        self, x: torch.Tensor, delta: torch.Tensor, B: torch.Tensor, C: torch.Tensor, return_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        A = -torch.exp(self.A_log).to(dtype=x.dtype)
        D = self.D.to(dtype=x.dtype)
        if self.config.pscan:
            return self.selective_scan(x, delta, A, B, C, D, return_states=return_states)
        return self.selective_scan_seq(x, delta, A, B, C, D, return_states=return_states)

    def _bidirectional_ssm(
        self, x: torch.Tensor, return_aux: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x_fwd, z_fwd = self._project_xz(x)
        delta_fused = self.fused_delta_proj(x_fwd)
        B_fused, C_fused = self._project_bc(x_fwd)
        ssm_fwd, states_fwd = self._scan(x_fwd, delta_fused, B_fused, C_fused, return_states=return_aux)
        h_fwd = ssm_fwd * z_fwd

        x_bwd_input = torch.flip(x, dims=[1])
        x_bwd, z_bwd = self._project_xz(x_bwd_input)
        delta_bwd = torch.flip(delta_fused, dims=[1])
        B_bwd, C_bwd = self._project_bc(x_bwd)
        ssm_bwd_rev, states_bwd_rev = self._scan(x_bwd, delta_bwd, B_bwd, C_bwd, return_states=return_aux)
        h_bwd = torch.flip(ssm_bwd_rev * z_bwd, dims=[1])
        h_bi = h_fwd + h_bwd

        aux = {
            "delta_fused": delta_fused,
            "h_fwd": h_fwd,
            "h_bwd": h_bwd,
            "h_bi": h_bi,
            "x_fwd": x_fwd,
            "z_fwd": z_fwd,
        }
        if return_aux:
            aux["states_fwd"] = states_fwd
            aux["states_bwd"] = torch.flip(states_bwd_rev, dims=[1])
        return h_bi, h_fwd, h_bwd, aux

    def _frequency_ssm(self, h_bi: torch.Tensor, return_aux: bool) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B_freq, C_freq = self._project_bc(h_bi)
        branch_outputs: List[torch.Tensor] = []
        branch_states: Dict[str, torch.Tensor] = {}
        aux: Dict[str, torch.Tensor] = {}

        for name in self.config.group_names:
            delta = self.freq_delta_proj[name](h_bi)
            y_branch, states = self._scan(h_bi, delta, B_freq, C_freq, return_states=return_aux)
            branch_outputs.append(y_branch)
            aux[f"delta_{name}"] = delta
            aux[f"y_{name}"] = y_branch
            if return_aux and states is not None:
                branch_states[name] = states

        stacked = torch.stack(branch_outputs, dim=2)
        gate_input = torch.cat(branch_outputs, dim=-1)
        gates = self.freq_gate(gate_input).view(
            gate_input.size(0), gate_input.size(1), self.config.freq_groups, self.config.d_inner
        )
        gates = F.softmax(gates, dim=2)
        y_freq = (gates * stacked).sum(dim=2)
        aux["gates"] = gates
        aux["y_freq"] = y_freq
        if return_aux:
            aux["branch_states"] = branch_states
        return y_freq, aux

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        h_bi, h_fwd, h_bwd, bid_aux = self._bidirectional_ssm(x, return_aux=return_aux)
        y_freq, freq_aux = self._frequency_ssm(h_bi, return_aux=return_aux)
        consistency_map = (h_fwd - h_bwd).pow(2).sum(dim=-1)
        consistency_loss = consistency_map.mean()

        if not return_aux:
            return y_freq, consistency_loss

        aux = {
            "h_fwd": h_fwd,
            "h_bwd": h_bwd,
            "h_bi": h_bi,
            "delta_fused": bid_aux["delta_fused"],
            "delta_low": freq_aux["delta_low"],
            "delta_mid": freq_aux["delta_mid"],
            "delta_high": freq_aux["delta_high"],
            "y_low": freq_aux["y_low"],
            "y_mid": freq_aux["y_mid"],
            "y_high": freq_aux["y_high"],
            "y_freq": freq_aux["y_freq"],
            "gates": freq_aux["gates"],
            "consistency_map": consistency_map,
            "consistency_loss": consistency_loss,
        }
        if "states_fwd" in bid_aux:
            aux["states_fwd"] = bid_aux["states_fwd"]
            aux["states_bwd"] = bid_aux["states_bwd"]
        if "branch_states" in freq_aux:
            aux["branch_states"] = freq_aux["branch_states"]
        return y_freq, consistency_loss, aux

    def selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        BX = deltaB * x.unsqueeze(-1)
        hs = pscan(deltaA, BX)
        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        y = y + D * x
        return y, hs if return_states else None

    def selective_scan_seq(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, length, _ = x.shape
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        BX = deltaB * x.unsqueeze(-1)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=x.device, dtype=x.dtype)
        hs = []
        for t in range(length):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
        hs_tensor = torch.stack(hs, dim=1)
        y = (hs_tensor @ C.unsqueeze(-1)).squeeze(3)
        y = y + D * x
        return y, hs_tensor if return_states else None


class BiMambaFreqResidualBlock(nn.Module):
    def __init__(self, config: BiMambaFreqConfig, shared_output_proj: LowRankLinear):
        super().__init__()
        self.config = config
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, False)
        self.mixer = BiMambaFreq(config)
        self.W_up_shared = shared_output_proj
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        normed = self.norm(x)
        if return_aux:
            y_freq, consistency_loss, aux = self.mixer(normed, return_aux=True)
        else:
            y_freq, consistency_loss = self.mixer(normed, return_aux=False)
            aux = None

        residual_update = self.dropout(self.W_up_shared(y_freq))
        out = x + residual_update

        if not return_aux:
            return out, y_freq, consistency_loss

        aux["residual_update"] = residual_update
        return out, y_freq, consistency_loss, aux


class BiMambaFreqModel(nn.Module):
    def __init__(self, config: BiMambaFreqConfig):
        super().__init__()
        self.config = config
        self.W_down_shared = LowRankLinear(
            config.input_dim, config.d_model, rank=config.input_rank, bias=config.bias
        )
        self.W_up_shared = LowRankLinear(
            config.d_inner, config.d_model, rank=config.model_rank, bias=config.bias
        )
        self.layers = nn.ModuleList(
            [BiMambaFreqResidualBlock(config, shared_output_proj=self.W_up_shared) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 1),
        )
        self.volatility_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 1),
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.pool == "mean":
            return x.mean(dim=1)
        if self.config.pool == "last":
            return x[:, -1]
        raise ValueError(f"Unsupported pool mode: {self.config.pool}")

    def forward(self, x: torch.Tensor, return_interpret: bool = False) -> Dict[str, torch.Tensor]:
        hidden = self.W_down_shared(x)
        consistency_terms: List[torch.Tensor] = []

        if return_interpret:
            interpret: Dict[str, List[torch.Tensor]] = {
                "delta_fused": [],
                "delta_low": [],
                "delta_mid": [],
                "delta_high": [],
                "h_fwd": [],
                "h_bwd": [],
                "h_bi": [],
                "y_low": [],
                "y_mid": [],
                "y_high": [],
                "y_freq": [],
                "gates": [],
                "consistency_map": [],
            }

        for layer in self.layers:
            if return_interpret:
                hidden, _, consistency_loss, aux = layer(hidden, return_aux=True)
                interpret["delta_fused"].append(aux["delta_fused"])
                interpret["delta_low"].append(aux["delta_low"])
                interpret["delta_mid"].append(aux["delta_mid"])
                interpret["delta_high"].append(aux["delta_high"])
                interpret["h_fwd"].append(aux["h_fwd"])
                interpret["h_bwd"].append(aux["h_bwd"])
                interpret["h_bi"].append(aux["h_bi"])
                interpret["y_low"].append(aux["y_low"])
                interpret["y_mid"].append(aux["y_mid"])
                interpret["y_high"].append(aux["y_high"])
                interpret["y_freq"].append(aux["y_freq"])
                interpret["gates"].append(aux["gates"])
                interpret["consistency_map"].append(aux["consistency_map"])
            else:
                hidden, _, consistency_loss = layer(hidden, return_aux=False)
            consistency_terms.append(consistency_loss)

        features = self.final_norm(hidden)
        pooled = self._pool(features)
        logits = self.classifier(pooled)
        classification = torch.sigmoid(logits)
        volatility = F.relu(self.volatility_head(pooled))
        consistency_loss = torch.stack(consistency_terms).mean()

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "classification": classification,
            "volatility": volatility,
            "features": pooled,
            "consistency_loss": consistency_loss,
        }
        if not return_interpret:
            return outputs

        outputs["interpret"] = {
            "delta_fused": torch.stack(interpret["delta_fused"], dim=1),
            "delta_low": torch.stack(interpret["delta_low"], dim=1),
            "delta_mid": torch.stack(interpret["delta_mid"], dim=1),
            "delta_high": torch.stack(interpret["delta_high"], dim=1),
            "h_fwd": torch.stack(interpret["h_fwd"], dim=1),
            "h_bwd": torch.stack(interpret["h_bwd"], dim=1),
            "h_bi": torch.stack(interpret["h_bi"], dim=1),
            "y_low": torch.stack(interpret["y_low"], dim=1),
            "y_mid": torch.stack(interpret["y_mid"], dim=1),
            "y_high": torch.stack(interpret["y_high"], dim=1),
            "y_freq": torch.stack(interpret["y_freq"], dim=1),
            "gates": torch.stack(interpret["gates"], dim=1),
            "consistency_map": torch.stack(interpret["consistency_map"], dim=1),
        }
        return outputs


def bimamba_freq_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    cls_target: torch.Tensor,
    vol_target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> Dict[str, torch.Tensor]:
    logits = outputs["logits"]
    volatility = outputs["volatility"]
    cls_target = cls_target.to(dtype=logits.dtype).view_as(logits)
    vol_target = vol_target.to(dtype=volatility.dtype).view_as(volatility)

    cls_loss = F.binary_cross_entropy_with_logits(logits, cls_target)
    vol_loss = F.mse_loss(volatility, vol_target)
    consistency_loss = outputs["consistency_loss"]
    loss = alpha * cls_loss + beta * vol_loss + gamma * consistency_loss
    return {
        "loss": loss,
        "cls_loss": cls_loss,
        "vol_loss": vol_loss,
        "consistency_loss": consistency_loss,
    }
