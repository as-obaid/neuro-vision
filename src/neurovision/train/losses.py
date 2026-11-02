from typing import Literal

import torch
import torch.nn.functional as F

from neurovision.data.labels import LOCATIONS


def label_weights(kind: Literal["uniform", "competition"]) -> torch.Tensor:
    """Loss weight per target, in TARGETS order. "competition" copies the score: Aneurysm Present
    weighs as much as the 13 locations together."""
    weights = torch.ones(len(LOCATIONS) + 1)
    if kind == "competition":
        weights[-1] = len(LOCATIONS)
    return weights


def weighted_mean(losses: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    """(batch, 14) element losses: mean over the batch, then over targets by `weights`."""
    per_label = losses.mean(0)
    if weights is None:
        return per_label.mean()
    return (per_label * weights).sum() / weights.sum()


def bce(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return weighted_mean(losses, weights)
