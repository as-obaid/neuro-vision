"""Vessel pretraining (step 1): the r3d_18 U-Net learns the 27 Shi classes on one fold's training
series, from Kinetics weights; its encoder then starts the classifier (`init: vessel`).

Usage (GPU node): PYTHONPATH=src python scripts/pretrain_vessels.py --fold 0 [--run NAME]
Trains only on series outside `--fold` (the fold's validation series are never seen), validates
mean Dice on up to 200 of the fold's series, resumes from `last.pt`, and writes
`runs/<run>/fold<k>/` (history.csv, encoder.pt, meta.json last); `--run` defaults to
`vessel_r3d18_160`.
"""

import argparse
import json
import math
import re
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.losses import DiceCELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from neurovision.data.dataset import load_index
from neurovision.data.folds import load_folds
from neurovision.data.preprocess import cache_name
from neurovision.data.vessels import N_SEG_CLASSES, VesselDataset
from neurovision.models.resnet3d import build_model
from neurovision.models.unet import R3DUNet, encoder_state
from neurovision.paths import atomic_write, store
from neurovision.train.loop import atomic_save
from neurovision.train.schedule import cosine_with_warmup

RUN = "vessel_r3d18_160"
SIZE = 160
EPOCHS, WARMUP, BATCH, LR, WD, FLIP, WORKERS, N_VAL = 12, 1, 4, 3e-4, 0.01, 0.5, 8, 200
ARTERIES = list(range(1, 14))
ANEURYSMS = list(range(14, N_SEG_CLASSES))


def split(root: Path, fold: int) -> tuple[list[str], list[str], Path, Path]:
    cache = root / "cache" / cache_name(SIZE, "mm")
    seg_dir = root / "cache" / f"{cache.name}-vessels"
    index = load_index(cache)
    ok = set(index.index[index["status"] == "ok"])
    have = {p.stem for p in seg_dir.glob("*.npz")}
    folds = load_folds(root)
    usable = [u for u in folds.index if u in ok and u in have]
    train = [u for u in usable if folds[u] != fold]
    val = [u for u in usable if folds[u] == fold][:N_VAL]
    return train, val, cache, seg_dir


def dice_by_class(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Dice per class over the whole validation set (argmax prediction); NaN if never present."""
    inter = torch.zeros(N_SEG_CLASSES, device=device)
    total = torch.zeros(N_SEG_CLASSES, device=device)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(x).argmax(1)
            for c in range(1, N_SEG_CLASSES):
                p, t = pred == c, y == c
                inter[c] += (p & t).sum()
                total[c] += p.sum() + t.sum()
    dice = (2 * inter / total).cpu().numpy()
    dice[total.cpu().numpy() == 0] = np.nan
    return dice


def _run_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError(f"'{value}' is not a plain folder name")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--run", type=_run_name, default=RUN, help="folder under runs/")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = store()
    run_dir = root / "runs" / args.run / f"fold{args.fold}"
    if (run_dir / "meta.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    train, val, cache, seg_dir = split(root, args.fold)
    print(f"vessel pretraining fold {args.fold}: train {len(train)}, validation {len(val)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = R3DUNet(build_model("r3d18", "kinetics", root / "weights")).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = math.ceil(len(train) / BATCH)
    scheduler = cosine_with_warmup(optimizer, WARMUP * steps, EPOCHS * steps)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False, batch=True)
    train_ds = VesselDataset(cache, seg_dir, train, FLIP)
    val_ds = VesselDataset(cache, seg_dir, val)
    val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=WORKERS)
    first, rows = 0, []
    if (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        first = ck["epoch"] + 1
        history = pd.read_csv(run_dir / "history.csv").filter(regex=r"^(?!Unnamed)")
        rows = history.to_dict("records")[:first]
    for epoch in range(first, EPOCHS):
        train_ds.set_flip_seed(epoch)
        loader = DataLoader(
            train_ds,
            batch_size=BATCH,
            shuffle=True,
            generator=torch.Generator().manual_seed(epoch),
            num_workers=WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        model.train()
        t0, total, n = time.perf_counter(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
            loss = loss_fn(logits.float(), y.unsqueeze(1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total, n = total + loss.item(), n + 1
        t1 = time.perf_counter()
        dice = dice_by_class(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total / n,
            "dice_arteries": float(np.nanmean(dice[ARTERIES])),
            "dice_aneurysms": float(np.nanmean(dice[ANEURYSMS])),
            "lr": optimizer.param_groups[0]["lr"],
            "train_seconds": t1 - t0,
            "val_seconds": time.perf_counter() - t1,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        atomic_write(run_dir / "history.csv", partial(pd.DataFrame(rows).to_csv, index=False))
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }
        atomic_save(run_dir / "last.pt", state)
    atomic_save(run_dir / "encoder.pt", encoder_state(model))
    meta = {
        "run": args.run,
        "fold": args.fold,
        "train_series": len(train),
        "val_series": len(val),
        "epochs": EPOCHS,
        "batch": BATCH,
        "lr": LR,
        "weight_decay": WD,
        "flip_lr": FLIP,
        "init": "kinetics r3d_18",
        "loss": "DiceCE (MONAI, no background, batch Dice)",
    }
    atomic_write(run_dir / "meta.json", lambda tmp: tmp.write_text(json.dumps(meta, indent=2)))


if __name__ == "__main__":
    main()
