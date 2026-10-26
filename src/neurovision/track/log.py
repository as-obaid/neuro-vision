"""Offline Weights & Biases logging for one run."""

import os
from pathlib import Path
from typing import Any

import wandb

from neurovision.config import Config


class Run:
    def __init__(self, run: Any) -> None:
        self._run = run

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._run.log(metrics, step=step)

    def finish(self, summary: dict[str, float]) -> None:
        self._run.summary.update(summary)
        self._run.finish()


def start(config: Config, fold: int, run_dir: Path) -> Run:
    run = wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ.get("WANDB_ENTITY"),
        name=f"{config.name}-fold{fold}",
        group=config.name,
        config=config.model_dump(),
        dir=run_dir,
        mode="offline",
    )
    return Run(run)
