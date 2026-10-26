"""Training epoch, evaluation, checkpoints and the fold-level `fit`."""

import json
import math
import os
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.data import DataLoader
from monai.transforms import Randomizable
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import AveragedModel

from neurovision.config import Config, TrainConfig
from neurovision.data.dataset import CacheDataset
from neurovision.data.labels import TARGETS
from neurovision.evaluate.metric import score
from neurovision.infer.predict import predict_volumes
from neurovision.models.resnet3d import build_model
from neurovision.track import log
from neurovision.track.meta import run_meta
from neurovision.train.ema import make_ema
from neurovision.train.losses import bce
from neurovision.train.schedule import cosine_with_warmup

HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_score",
    "lr",
    "train_seconds",
    "val_seconds",
]


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
    total = torch.zeros((), device=device)
    n_batches, pending = 0, 0

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
        total += loss.detach()
        n_batches += 1
        pending += 1
        if pending == accum_steps:
            step()
            pending = 0
    if n_batches == 0:
        raise ValueError("empty loader")
    if pending:
        # backward divided by accum_steps; a short final group needs its own mean
        for p in model.parameters():
            if p.grad is not None:
                p.grad *= accum_steps / pending
        step()
    return total.item() / n_batches


def evaluate(
    model: nn.Module, loader: Iterable[tuple[torch.Tensor, torch.Tensor]], device: torch.device
) -> tuple[np.ndarray, float]:
    model.to(device)
    probs, targets = [], []
    for x, y in loader:
        probs.append(predict_volumes(model, x))
        targets.append(y)
    if not probs:
        raise ValueError("empty loader")
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


def train_loader(
    ds: CacheDataset, cfg: TrainConfig, seed: int, epoch: int, pin_memory: bool
) -> DataLoader:
    """Shuffled loader whose order and augmentation streams depend only on `seed + epoch`."""
    if isinstance(ds.transform, Randomizable):
        ds.transform.set_random_state(seed=seed + epoch)  # the num_workers=0 stream
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed + epoch),
    )


def fit(
    config: Config,
    fold: int,
    run_dir: Path,
    train_ds: CacheDataset,
    val_ds: CacheDataset,
    val_uids: list[str],
    prior: np.ndarray,
) -> None:
    """Train one fold, resuming from `run_dir/last.pt`; write the run directory files."""
    cfg = config.train
    assert cfg is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    torch.manual_seed(config.seed)
    model = build_model().to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(math.ceil(len(train_ds) / cfg.batch_size) / cfg.accum_steps)
    scheduler = cosine_with_warmup(
        optimizer, cfg.warmup_epochs * steps_per_epoch, cfg.epochs * steps_per_epoch
    )
    ema = make_ema(model, cfg.ema_decay)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=pin
    )

    first_epoch, best_score = 0, -math.inf
    history_path = run_dir / "history.csv"
    history = pd.DataFrame(columns=HISTORY_COLUMNS)
    if (run_dir / "last.pt").exists():
        first_epoch, best_score = load_checkpoint(
            run_dir / "last.pt", model, ema, optimizer, scheduler
        )
        # a crash after the history write but before last.pt would otherwise repeat an epoch
        history = pd.read_csv(history_path).query("epoch < @first_epoch")

    run = log.start(config, fold, run_dir)
    y_true = val_ds.targets
    for epoch in range(first_epoch, cfg.epochs):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader(train_ds, cfg, config.seed, epoch, pin),
            optimizer,
            scheduler,
            ema,
            cfg.accum_steps,
            device,
        )
        t1 = time.perf_counter()
        probs, val_loss = evaluate(ema, val_loader, device)
        val_score = score(y_true, probs)
        t2 = time.perf_counter()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_score": val_score,
            "lr": optimizer.param_groups[0]["lr"],
            "train_seconds": t1 - t0,
            "val_seconds": t2 - t1,
        }
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        history.to_csv(history_path, index=False)
        run.log({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        if val_score > best_score:
            best_score = val_score
            torch.save(ema.module.state_dict(), run_dir / "best.pt")
        save_checkpoint(run_dir / "last.pt", model, ema, optimizer, scheduler, epoch, best_score)

    best = build_model()
    best.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
    probs, _ = evaluate(best, val_loader, device)
    oof = pd.DataFrame(
        np.tile(prior, (len(val_uids), 1)),
        index=pd.Index(val_uids, name="SeriesInstanceUID"),
        columns=list(TARGETS),
    )
    oof.loc[val_ds.uids] = probs
    oof.to_csv(run_dir / "oof.csv")

    meta = run_meta(config) | {
        "train_seconds_per_epoch": float(history["train_seconds"].mean()),
        "val_seconds_per_epoch": float(history["val_seconds"].mean()),
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20 if pin else None,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    run.finish({"best_val_score": best_score})
