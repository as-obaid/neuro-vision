from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss

from neurovision.data.labels import LOCATIONS

LossWeights = Literal["uniform", "competition", "first_place"]
# "first_place": 0.1 x mean location BCE + 0.05 x AP BCE = 0.15 x the weighted mean with AP at 6.5
FIRST_PLACE_AP = 6.5
FIRST_PLACE_SCALE = 0.15


def label_weights(kind: LossWeights) -> torch.Tensor:
    """Loss weight per target, in TARGETS order. "competition" copies the score: Aneurysm Present
    weighs as much as the 13 locations together. "first_place" sets the 1st-place ratio of
    Aneurysm Present to one location (0.05 to 0.1 / 13); `classification_scale` sets its size."""
    weights = torch.ones(len(LOCATIONS) + 1)
    if kind == "competition":
        weights[-1] = len(LOCATIONS)
    elif kind == "first_place":
        weights[-1] = FIRST_PLACE_AP
    return weights


def classification_scale(kind: LossWeights) -> float:
    """Factor on the weighted-mean BCE in the training objective; 1 except for "first_place"."""
    return FIRST_PLACE_SCALE if kind == "first_place" else 1.0


def balanced_bce(logits: torch.Tensor, target: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Location-map loss: mean BCE over target cells plus mean BCE over background cells.

    `logits` and `target` are (batch, 1, D, H, W); `keep` (batch,) is 0 for a series whose map is
    unknown (a positive with no expert point inside the input), which then adds nothing. As in
    the 1st-place solution, the two means are summed and a batch with no target cell adds only
    the background term.
    """
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    k = keep.to(losses.dtype).view(-1, *([1] * (losses.ndim - 1))).expand_as(losses)
    pos, neg = target * k, (1 - target) * k
    pos_loss = (losses * pos).sum() / pos.sum().clamp(min=1.0)
    neg_loss = (losses * neg).sum() / neg.sum().clamp(min=1.0)
    return pos_loss + neg_loss


_DICE_CE = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False, batch=True)


def seg_loss(logits: torch.Tensor, target: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Vessel-class loss of the full version: DiceCE as in the step-1 pretraining (softmax, no
    background in the Dice, Dice over the batch), on the series with a mask (`keep` 1) only.

    `logits` (n, 27, D, H, W), `target` (n, 1, D, H, W) class ids; a batch without a mask
    returns a zero that still carries a gradient.
    """
    kept = keep > 0.5
    if not bool(kept.any()):
        return logits.sum() * 0.0
    loss: torch.Tensor = _DICE_CE(logits[kept], target[kept])
    return loss


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
