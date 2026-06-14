import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mambapy.mamba import RMSNorm
from mambapy.pscan import pscan


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def _split_group_sizes(total: int, n_groups: int) -> List[int]:
    base = total // n_groups
    remainder = total % n_groups
    return [base + (1 if i < remainder else 0) for i in range(n_groups)]


def _group_slices(total: int, n_groups: int) -> List[slice]:
    sizes = _split_group_sizes(total, n_groups)
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    return [slice(offsets[i], offsets[i + 1]) for i in range(n_groups)]


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


@dataclass
class BiMambaFreqConfig:
    input_dim: int
    d_model: int
    n_layers: int
    num_classes: int = 3

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
    freq_group_dt: Tuple[float, ...] = (0.5, 1.0, 2.0)
    low_rank: Optional[int] = None
    bidirectional_merge: str = "concat"
    project_concat_for_residual: bool = True
    pool: str = "mean"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        self.d_inner = self.expand_factor * self.d_model
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        if self.low_rank is None:
            self.low_rank = max(1, self.d_model // 4)
        if self.freq_groups <= 0:
            raise ValueError("freq_groups must be >= 1")
        if len(self.freq_group_dt) != self.freq_groups:
            raise ValueError("len(freq_group_dt) must equal freq_groups")
        if self.bidirectional_merge not in {"concat", "sum"}:
            raise ValueError("bidirectional_merge must be one of {'concat', 'sum'}")
        self.bi_output_dim = self.d_model * 2 if self.bidirectional_merge == "concat" else self.d_model
        self.group_names = ["high", "mid", "low"][: self.freq_groups]
        if len(self.group_names) < self.freq_groups:
            self.group_names.extend([f"group_{i}" for i in range(len(self.group_names), self.freq_groups)])


class BiMambaFreqCore(nn.Module):
    def __init__(self, config: BiMambaFreqConfig):
        super().__init__()
        self.config = config
        self.group_slices = _group_slices(config.d_inner, config.freq_groups)

        self.in_proj = LowRankLinear(config.d_model, 2 * config.d_inner, rank=config.low_rank, bias=config.bias)
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {config.dt_init}")

        with torch.no_grad():
            targets = torch.tensor(config.freq_group_dt, dtype=self.dt_proj.bias.dtype)
            grouped = []
            for target, group_slice in zip(targets, self.group_slices):
                group_width = group_slice.stop - group_slice.start
                grouped.append(torch.full((group_width,), float(target), dtype=self.dt_proj.bias.dtype))
            dt_targets = torch.cat(grouped).clamp(min=config.dt_init_floor)
            self.dt_proj.bias.copy_(_inverse_softplus(dt_targets))

        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = LowRankLinear(config.d_inner, config.d_model, rank=config.low_rank, bias=config.bias)

        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps, False)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, False)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, False)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

    def _apply_layernorms(
        self, dt: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        _, length, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        x_branch = self.conv1d(x_branch.transpose(1, 2))[:, :, :length].transpose(1, 2)
        x_branch = F.silu(x_branch)
        if return_aux:
            y_inner, aux = self.ssm(x_branch, return_aux=True)
        else:
            y_inner = self.ssm(x_branch, return_aux=False)
            aux = None
        z_gate = F.silu(z_branch)
        gated = y_inner * z_gate
        output = self.out_proj(gated)

        if not return_aux:
            return output

        aux = aux or {}
        aux["inner_output"] = gated
        aux["projected_output"] = output
        return output, aux

    def ssm(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.x_proj(x)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(1, 2)
        delta = F.softplus(delta.transpose(1, 2) + self.dt_proj.bias)

        if self.config.pscan:
            y, hs = self.selective_scan(x, delta, A, B, C, D, return_states=return_aux)
        else:
            y, hs = self.selective_scan_seq(x, delta, A, B, C, D, return_states=return_aux)

        if not return_aux:
            return y

        hs_summary = hs.mean(dim=-1)
        group_states = {
            name: hs_summary[:, :, group_slice]
            for name, group_slice in zip(self.config.group_names, self.group_slices)
        }
        return y, {"delta": delta, "state_summary": hs_summary, "group_states": group_states, "raw_states": hs}

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


class BiMambaFreq(nn.Module):
    def __init__(self, config: BiMambaFreqConfig):
        super().__init__()
        self.config = config
        self.mamba = BiMambaFreqCore(config)

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if return_aux:
            h_fwd, aux_fwd = self.mamba(x, return_aux=True)
            h_bwd_rev, aux_bwd = self.mamba(torch.flip(x, dims=[1]), return_aux=True)
        else:
            h_fwd = self.mamba(x)
            h_bwd_rev = self.mamba(torch.flip(x, dims=[1]))
            aux_fwd = None
            aux_bwd = None

        h_bwd = torch.flip(h_bwd_rev, dims=[1])
        if self.config.bidirectional_merge == "concat":
            h_bi = torch.cat([h_fwd, h_bwd], dim=-1)
        else:
            h_bi = h_fwd + h_bwd

        if not return_aux:
            return h_bi

        aux_bwd = dict(aux_bwd)
        aux_bwd["delta"] = torch.flip(aux_bwd["delta"], dims=[1])
        aux_bwd["state_summary"] = torch.flip(aux_bwd["state_summary"], dims=[1])
        aux_bwd["inner_output"] = torch.flip(aux_bwd["inner_output"], dims=[1])
        aux_bwd["projected_output"] = torch.flip(aux_bwd["projected_output"], dims=[1])
        aux_bwd["raw_states"] = torch.flip(aux_bwd["raw_states"], dims=[1])
        aux_bwd["group_states"] = {
            name: torch.flip(state, dims=[1]) for name, state in aux_bwd["group_states"].items()
        }

        divergence = 1.0 - F.cosine_similarity(h_fwd, h_bwd, dim=-1, eps=1e-8)
        aux = {
            "h_fwd": h_fwd,
            "h_bwd": h_bwd,
            "h_bi": h_bi,
            "bi_divergence": divergence,
            "delta_fwd": aux_fwd["delta"],
            "delta_bwd": aux_bwd["delta"],
            "group_states_fwd": aux_fwd["group_states"],
            "group_states_bwd": aux_bwd["group_states"],
            "state_summary_fwd": aux_fwd["state_summary"],
            "state_summary_bwd": aux_bwd["state_summary"],
        }
        return h_bi, aux


class BiMambaFreqResidualBlock(nn.Module):
    def __init__(self, config: BiMambaFreqConfig):
        super().__init__()
        self.config = config
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, False)
        self.mixer = BiMambaFreq(config)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        if config.bidirectional_merge == "concat" and config.project_concat_for_residual:
            self.residual_proj = LowRankLinear(config.bi_output_dim, config.d_model, rank=config.low_rank, bias=config.bias)
        elif config.bidirectional_merge == "concat":
            raise ValueError("concat merge requires project_concat_for_residual=True to preserve stack width")
        else:
            self.residual_proj = nn.Identity()

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        normed = self.norm(x)
        if return_aux:
            bi_output, aux = self.mixer(normed, return_aux=True)
        else:
            bi_output = self.mixer(normed)
            aux = None

        residual_update = self.dropout(self.residual_proj(bi_output))
        out = x + residual_update

        if not return_aux:
            return out, bi_output

        aux["residual_update"] = residual_update
        return out, bi_output, aux


class BiMambaFreqModel(nn.Module):
    def __init__(self, config: BiMambaFreqConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.d_model) if config.input_dim != config.d_model else nn.Identity()
        self.layers = nn.ModuleList([BiMambaFreqResidualBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.bi_output_dim)
        self.classifier = nn.Linear(config.bi_output_dim, config.num_classes)
        self.volatility_head = nn.Linear(config.bi_output_dim, 1)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.pool == "mean":
            return x.mean(dim=1)
        if self.config.pool == "last":
            return x[:, -1]
        raise ValueError(f"Unsupported pool mode: {self.config.pool}")

    def forward(self, x: torch.Tensor, return_interpret: bool = False) -> Dict[str, torch.Tensor]:
        hidden = self.input_proj(x)
        last_bi = None

        if return_interpret:
            interpret: Dict[str, List[torch.Tensor]] = {
                "delta_fwd": [],
                "delta_bwd": [],
                "bi_divergence": [],
                "h_fwd": [],
                "h_bwd": [],
            }
            group_states_fwd: Dict[str, List[torch.Tensor]] = {name: [] for name in self.config.group_names}
            group_states_bwd: Dict[str, List[torch.Tensor]] = {name: [] for name in self.config.group_names}

        for layer in self.layers:
            if return_interpret:
                hidden, last_bi, aux = layer(hidden, return_aux=True)
                interpret["delta_fwd"].append(aux["delta_fwd"])
                interpret["delta_bwd"].append(aux["delta_bwd"])
                interpret["bi_divergence"].append(aux["bi_divergence"])
                interpret["h_fwd"].append(aux["h_fwd"])
                interpret["h_bwd"].append(aux["h_bwd"])
                for name in self.config.group_names:
                    group_states_fwd[name].append(aux["group_states_fwd"][name])
                    group_states_bwd[name].append(aux["group_states_bwd"][name])
            else:
                hidden, last_bi = layer(hidden, return_aux=False)

        assert last_bi is not None
        features = self.final_norm(last_bi)
        pooled = self._pool(features)

        outputs: Dict[str, torch.Tensor] = {
            "logits": self.classifier(pooled),
            "volatility": self.volatility_head(pooled),
            "features": pooled,
        }
        if not return_interpret:
            return outputs

        outputs["interpret"] = {
            "delta_fwd": torch.stack(interpret["delta_fwd"], dim=1),
            "delta_bwd": torch.stack(interpret["delta_bwd"], dim=1),
            "bi_divergence": torch.stack(interpret["bi_divergence"], dim=1),
            "h_fwd": torch.stack(interpret["h_fwd"], dim=1),
            "h_bwd": torch.stack(interpret["h_bwd"], dim=1),
            "group_states_fwd": {name: torch.stack(values, dim=1) for name, values in group_states_fwd.items()},
            "group_states_bwd": {name: torch.stack(values, dim=1) for name, values in group_states_bwd.items()},
        }
        return outputs
