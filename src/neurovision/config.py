"""YAML run configs with `extends:` inheritance."""

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GbmConfig(_Frozen):
    max_iter: PositiveInt
    learning_rate: PositiveFloat
    max_depth: PositiveInt


class TrainConfig(_Frozen):
    epochs: PositiveInt
    batch_size: PositiveInt
    accum_steps: PositiveInt
    lr: PositiveFloat
    weight_decay: float
    warmup_epochs: int
    ema_decay: float
    loss_weights: Literal["uniform"]
    num_workers: int


class AugmentConfig(_Frozen):
    prob: float
    noise_std: float
    gamma: tuple[PositiveFloat, PositiveFloat]
    rotate_deg: float
    scale: float
    lowres_zoom: tuple[PositiveFloat, PositiveFloat]


class Config(_Frozen):
    name: str
    kind: Literal["prior", "gbm", "cnn"]
    seed: int
    labels: Literal["corrected", "original"]
    gbm: GbmConfig | None = None
    train: TrainConfig | None = None
    augment: AugmentConfig | None = None

    @model_validator(mode="after")
    def _sections_match_kind(self) -> Self:
        used = {"prior": set(), "gbm": {"gbm"}, "cnn": {"train", "augment"}}[self.kind]
        for section in ("gbm", "train", "augment"):
            present = getattr(self, section) is not None
            if present != (section in used):
                verb = "needs" if section in used else "must not have"
                raise ValueError(f"kind {self.kind!r} {verb} section {section!r}")
        return self


def _merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = dict(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve(path: Path, chain: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in chain:
        names = " -> ".join(p.name for p in (*chain, path))
        raise ValueError(f"extends cycle: {names}")
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _merge(_resolve(path.parent / parent, (*chain, path)), raw)


def resolve(path: Path) -> dict[str, Any]:
    return _resolve(path, ())


def load_config(path: Path) -> Config:
    data = resolve(path)
    if "name" in data:
        raise ValueError(f"{path}: `name` is set from the file name, remove it from the YAML")
    return Config(name=path.stem, **data)
