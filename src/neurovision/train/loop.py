"""Training epoch, evaluation and checkpoints."""

import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import AveragedModel

from neurovision.infer.predict import predict_volumes
from neurovision.train.losses import bce


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: Optimizer,
    scheduler: LRScheduler,
    ema: AveragedModel,
    accum_steps: int,
    device: torch.device,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, n_batches, pending = 0.0, 0, 0

    def step() -> None:
        optimizer.step()
        scheduler.step()
        ema.update_parameters(model)
        optimizer.zero_grad(set_to_none=True)

    for x, y in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x.to(device))
        loss = bce(logits.float(), y.to(device))
        (loss / accum_steps).backward()
        total += loss.item()
        n_batches += 1
        pending += 1
        if pending == accum_steps:
            step()
            pending = 0
    if pending:
        step()
    return total / n_batches


def evaluate(
    model: nn.Module, loader: Iterable[tuple[torch.Tensor, torch.Tensor]], device: torch.device
) -> tuple[np.ndarray, float]:
    probs, targets = [], []
    for x, y in loader:
        probs.append(predict_volumes(model.to(device), x))
        targets.append(y)
    p = np.concatenate(probs)
    loss = F.binary_cross_entropy(torch.from_numpy(p), torch.cat(targets)).item()
    return p, loss


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: AveragedModel,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    best_score: float,
) -> None:
    """`epoch` is the last completed epoch."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
        },
        tmp,
    )
    os.replace(tmp, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    ema: AveragedModel,
    optimizer: Optimizer,
    scheduler: LRScheduler,
) -> tuple[int, float]:
    """Returns (next epoch, best score)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state["epoch"] + 1, state["best_score"]
