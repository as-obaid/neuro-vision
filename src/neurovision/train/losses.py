from typing import Literal

import numpy as np
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


def positive_weights(targets: np.ndarray, kind: Literal["sqrt"] | None) -> torch.Tensor | None:
    """Weight of a positive per target, from the training split's (n, 14) labels: the square
    root of negatives / positives for each location; Aneurysm Present, near balanced, keeps 1."""
    if kind is None:
        return None
    positives = targets.sum(0)
    if (positives == 0).any():
        raise ValueError("a target has no positive in the training split")
    weights = torch.tensor(np.sqrt((len(targets) - positives) / positives), dtype=torch.float32)
    weights[-1] = 1.0
    return weights


def weighted_mean(losses: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    """(batch, 14) element losses: mean over the batch, then over targets by `weights`."""
    per_label = losses.mean(0)
    if weights is None:
        return per_label.mean()
    return (per_label * weights).sum() / weights.sum()


def bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    return weighted_mean(losses, weights)
