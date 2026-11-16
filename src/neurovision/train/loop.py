"""Training epoch, evaluation, checkpoints and the fold-level `fit`."""

import json
import math
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

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
from neurovision.models.heads import MapModel
from neurovision.models.resnet3d import NEW_LAYERS, Backbone, Head, build_model
from neurovision.models.vessel_readout import VesselReadout
from neurovision.paths import atomic_write, store
from neurovision.track import log
from neurovision.track.meta import run_meta
from neurovision.train.ema import make_ema
from neurovision.train.losses import (
    balanced_bce,
    bce,
    classification_scale,
    label_weights,
    positive_weights,
    seg_loss,
    weighted_mean,
)
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
    loader: Iterable[tuple[torch.Tensor, ...]],
    optimizer: Optimizer,
    scheduler: LRScheduler,
    ema: AveragedModel,
    accum_steps: int,
    device: torch.device,
    weights: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
) -> float:
    """One pass on the classification loss; returns its mean."""
    loss, _ = train_epoch(
        model, loader, optimizer, scheduler, ema, accum_steps, device, weights, pos_weight
    )
    return loss


def train_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, ...]],
    optimizer: Optimizer,
    scheduler: LRScheduler,
    ema: AveragedModel,
    accum_steps: int,
    device: torch.device,
    weights: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
    cls_scale: float = 1.0,
    map_weight: float | None = None,
    seg_weight: float | None = None,
) -> tuple[float, float | None]:
    """One pass; returns (mean classification BCE, mean location-map or seg loss, or None).

    With `map_weight`, batches are (x, y, map, keep), the model is a `MapModel`, and the
    objective is cls_scale x BCE + map_weight x `balanced_bce` on the map. With `seg_weight`,
    batches are (x, y, class map, keep), the model is a `VesselReadout`, and the objective is
    cls_scale x BCE + seg_weight x `seg_loss`.
    """
    if map_weight is not None and seg_weight is not None:
        raise ValueError("a run has the location map or the vessel classes, not both")
    aux_weight = map_weight if map_weight is not None else seg_weight
    model.train()
    weights = None if weights is None else weights.to(device)
    pos_weight = None if pos_weight is None else pos_weight.to(device)
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=device)
    total_map = torch.zeros((), device=device)
    n = 0

    def step() -> None:
        optimizer.step()
        scheduler.step()
        ema.update_parameters(model)
        optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if map_weight is not None:
                assert isinstance(model, MapModel)
                logits, aux_logits = model.forward_with_map(x)
            elif seg_weight is not None:
                assert isinstance(model, VesselReadout)
                logits, aux_logits = model.forward_with_seg(x)
            else:
                logits = model(x)
        cls = bce(logits.float(), y, weights, pos_weight)
        loss = cls if cls_scale == 1 else cls_scale * cls
        if aux_weight is not None:
            aux_loss = (balanced_bce if map_weight is not None else seg_loss)(
                aux_logits.float(), batch[2].to(device), batch[3].to(device)
            )
            loss = loss + aux_weight * aux_loss
            total_map += aux_loss.detach()
        (loss / accum_steps).backward()
        total += cls.detach()
        n += 1
        if n % accum_steps == 0:
            step()
    if n == 0:
        raise ValueError("empty loader")
    if tail := n % accum_steps:
        # backward divided by accum_steps; a short final group needs its own mean
        for p in model.parameters():
            if p.grad is not None:
                p.grad *= accum_steps / tail
        step()
    return total.item() / n, None if aux_weight is None else total_map.item() / n


def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    weights: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
) -> tuple[np.ndarray, float]:
    model.to(device)
    probs, targets = [], []
    for x, y in loader:
        probs.append(predict_volumes(model, x))
        targets.append(y)
    if not probs:
        raise ValueError("empty loader")
    p = np.concatenate(probs)
    y = torch.cat(targets)
    scale = None if pos_weight is None else torch.where(y > 0.5, pos_weight.cpu(), 1.0)
    losses = F.binary_cross_entropy(torch.from_numpy(p), y, weight=scale, reduction="none")
    loss = weighted_mean(losses, None if weights is None else weights.cpu()).item()
    return p, loss


def peak_memory_mib(device: torch.device) -> float | None:
    return torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None


def atomic_save(path: Path, obj: object) -> None:
    atomic_write(path, lambda tmp: torch.save(obj, tmp))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: AveragedModel,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    best_score: float,
    peak_mib: float | None,
) -> None:
    """`epoch` is the last completed epoch."""
    atomic_save(
        path,
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "peak_mib": peak_mib,
        },
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    ema: AveragedModel,
    optimizer: Optimizer,
    scheduler: LRScheduler,
) -> tuple[int, float, float | None]:
    """Returns (next epoch, best score, peak GPU MiB so far); old files have no peak."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state["epoch"] + 1, state["best_score"], state.get("peak_mib")


def train_loader(
    ds: CacheDataset, cfg: TrainConfig, seed: int, epoch: int, pin_memory: bool
) -> DataLoader:
    """Shuffled loader whose order and augmentation streams depend only on `seed + epoch`."""
    if isinstance(ds.transform, Randomizable):
        ds.transform.set_random_state(seed=seed + epoch)  # the num_workers=0 stream
    ds.set_flip_seed(seed + epoch)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed + epoch),
    )


def val_loader(ds: CacheDataset, cfg: TrainConfig, pin_memory: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )


def fit(
    config: Config,
    fold: int,
    run_dir: Path,
    train_ds: CacheDataset,
    val_ds: CacheDataset,
    val_uids: list[str],
    prior: np.ndarray,
    counts: Mapping[str, int],
) -> None:
    """Train one fold, resuming from `run_dir/last.pt`; write the run directory files."""
    cfg = config.train
    assert cfg is not None
    if (run_dir / "meta.json").exists():  # written last: the run is complete
        return
    meta = run_meta(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    torch.manual_seed(config.seed)
    weights_dir = store() / "weights" if cfg.init != "random" else None
    if cfg.vessel_run is not None:  # the pretraining run's encoder for this fold
        weights_dir = store() / "runs" / cfg.vessel_run / f"fold{fold}"
    aux_map = cfg.aux_map_weight is not None
    model = build_model(cfg.backbone, cfg.init, weights_dir, cfg.head, aux_map).to(device)
    optimizer = AdamW(
        param_groups(model, cfg.lr, cfg.backbone_lr_scale), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    steps_per_epoch = math.ceil(math.ceil(len(train_ds) / cfg.batch_size) / cfg.accum_steps)
    schedule_epochs = cfg.schedule_epochs or cfg.epochs  # a longer cosine stops partway down
    scheduler = cosine_with_warmup(
        optimizer, cfg.warmup_epochs * steps_per_epoch, schedule_epochs * steps_per_epoch
    )
    ema = make_ema(model, cfg.ema_decay)
    valid = val_loader(val_ds, cfg, pin)
    weights = label_weights(cfg.loss_weights)
    cls_scale = classification_scale(cfg.loss_weights)
    pos_weight = positive_weights(train_ds.targets, cfg.pos_weight)
    seg = cfg.seg_weight is not None
    aux_column = "train_map_loss" if aux_map else "train_seg_loss" if seg else None
    columns = [*HISTORY_COLUMNS, *([aux_column] if aux_column else [])]

    first_epoch, best_score, peak, rows = _resume(run_dir, model, ema, optimizer, scheduler)
    history_path = run_dir / "history.csv"
    run = log.start(config, fold, run_dir)
    try:
        for epoch in range(first_epoch, cfg.epochs):
            if stalled([r["val_loss"] for r in rows], cfg.early_stop_patience):
                break
            set_backbone_trainable(model, epoch >= cfg.freeze_epochs)
            t0 = time.perf_counter()
            train_loss, map_loss = train_epoch(
                model,
                train_loader(train_ds, cfg, config.seed, epoch, pin),
                optimizer,
                scheduler,
                ema,
                cfg.accum_steps,
                device,
                weights,
                pos_weight,
                cls_scale,
                cfg.aux_map_weight,
                cfg.seg_weight,
            )
            t1 = time.perf_counter()
            probs, val_loss = evaluate(ema, valid, device, weights, pos_weight)
            val_score = score(val_ds.targets, probs)
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
            if aux_column is not None and map_loss is not None:
                row[aux_column] = map_loss
            rows.append(row)
            pd.DataFrame(rows, columns=columns).to_csv(history_path, index=False)
            run.log({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            if val_score > best_score:
                best_score = val_score
                atomic_save(run_dir / "best.pt", ema.module.state_dict())
            current = peak_memory_mib(device)
            if current is not None:
                peak = max(peak or 0.0, current)
            save_checkpoint(
                run_dir / "last.pt", model, ema, optimizer, scheduler, epoch, best_score, peak
            )
        meta |= {"counts": dict(counts), "peak_gpu_memory_mib": peak}
        _write_outputs(
            meta,
            run_dir,
            val_ds,
            val_uids,
            prior,
            valid,
            rows,
            device,
            cfg.backbone,
            cfg.head,
            aux_map,
        )
    finally:
        run.finish({"best_val_score": best_score})


def stalled(val_losses: list[float], patience: int | None) -> bool:
    """True once the last `patience` epochs have not lowered the best validation loss."""
    if patience is None or not val_losses:
        return False
    return len(val_losses) - 1 - int(np.argmin(val_losses)) >= patience


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze every layer but the heads (fc, aux); BatchNorm statistics still update."""
    for name, p in model.named_parameters():
        if not name.startswith(NEW_LAYERS):
            p.requires_grad_(trainable)


def param_groups(model: nn.Module, lr: float, backbone_lr_scale: float) -> list[dict[str, Any]]:
    """One group at the optimizer's rate; or the heads (fc, aux) first at lr, the rest at lr x
    scale.

    The head comes first so history's `lr` column (group 0) is the base rate in every run.
    """
    if backbone_lr_scale == 1:
        return [{"params": list(model.parameters())}]
    named = list(model.named_parameters())
    return [
        {"params": [p for n, p in named if n.startswith(NEW_LAYERS)], "lr": lr},
        {
            "params": [p for n, p in named if not n.startswith(NEW_LAYERS)],
            "lr": lr * backbone_lr_scale,
        },
    ]


def _resume(
    run_dir: Path,
    model: nn.Module,
    ema: AveragedModel,
    optimizer: Optimizer,
    scheduler: LRScheduler,
) -> tuple[int, float, float | None, list[dict[str, float]]]:
    """(first epoch, best score, peak GPU MiB, history rows) from `last.pt`, or a fresh start."""
    if not (run_dir / "last.pt").exists():
        return 0, -math.inf, None, []
    first_epoch, best_score, peak = load_checkpoint(
        run_dir / "last.pt", model, ema, optimizer, scheduler
    )
    # a crash after the history write but before last.pt would otherwise repeat an epoch
    history = pd.read_csv(run_dir / "history.csv")
    return first_epoch, best_score, peak, history[history["epoch"] < first_epoch].to_dict("records")


def _write_outputs(
    meta: dict[str, Any],
    run_dir: Path,
    val_ds: CacheDataset,
    val_uids: list[str],
    prior: np.ndarray,
    valid: DataLoader,
    rows: list[dict[str, float]],
    device: torch.device,
    backbone: Backbone,
    head: Head,
    aux_map: bool = False,
) -> None:
    """oof.csv first, meta.json last: meta.json marks the run complete."""
    best = build_model(backbone, head=head, aux_map=aux_map)
    best.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
    probs, _ = evaluate(best, valid, device)
    oof = pd.DataFrame(
        np.tile(prior, (len(val_uids), 1)),
        index=pd.Index(val_uids, name="SeriesInstanceUID"),
        columns=list(TARGETS),
    )
    oof.loc[val_ds.uids] = probs
    atomic_write(run_dir / "oof.csv", oof.to_csv)

    meta |= {
        "train_seconds_per_epoch": float(np.mean([r["train_seconds"] for r in rows])),
        "val_seconds_per_epoch": float(np.mean([r["val_seconds"] for r in rows])),
    }
    atomic_write(run_dir / "meta.json", lambda tmp: tmp.write_text(json.dumps(meta, indent=2)))
