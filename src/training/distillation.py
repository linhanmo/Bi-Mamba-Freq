import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def cosine_distill_weight(epoch: int, total_epochs: int, max_lambda: float, min_lambda: float = 0.0) -> float:
    if total_epochs <= 1:
        return float(max_lambda)
    progress = min(max(epoch / float(total_epochs - 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lambda + (max_lambda - min_lambda) * cosine)


def soft_target_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must share the same shape")
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature**2)


def _haar_dwt(x: torch.Tensor, levels: int = 3) -> List[torch.Tensor]:
    coeffs: List[torch.Tensor] = []
    current = x
    for _ in range(levels):
        if current.size(1) < 2:
            break
        if current.size(1) % 2 == 1:
            current = F.pad(current, (0, 0, 0, 1))
        even = current[:, 0::2]
        odd = current[:, 1::2]
        approx = (even + odd) / math.sqrt(2.0)
        detail = (even - odd) / math.sqrt(2.0)
        coeffs.append(detail)
        current = approx
    coeffs.append(current)
    return coeffs


def freq_alignment_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    band_weights: Optional[Sequence[float]] = None,
    levels: int = 3,
) -> torch.Tensor:
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError("student_hidden and teacher_hidden must share the same shape")
    student_coeffs = _haar_dwt(student_hidden, levels=levels)
    teacher_coeffs = _haar_dwt(teacher_hidden, levels=levels)
    if band_weights is None:
        band_weights = [1.0] * len(student_coeffs)
    if len(band_weights) != len(student_coeffs):
        raise ValueError("band_weights length must match the number of wavelet bands")
    losses = [float(weight) * F.mse_loss(s_band, t_band) for weight, s_band, t_band in zip(band_weights, student_coeffs, teacher_coeffs)]
    return sum(losses)


def bimambafreq_total_loss(
    task_loss: torch.Tensor,
    distill_loss: torch.Tensor,
    epoch: int,
    total_epochs: int,
    max_lambda: float,
    min_lambda: float = 0.0,
) -> Tuple[torch.Tensor, float]:
    weight = cosine_distill_weight(epoch=epoch, total_epochs=total_epochs, max_lambda=max_lambda, min_lambda=min_lambda)
    return task_loss + task_loss.new_tensor(weight) * distill_loss, weight
