"""YAML run configs with `extends:` inheritance."""

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator


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
    weight_decay: float = Field(ge=0)
    warmup_epochs: int = Field(ge=0)
    ema_decay: float = Field(gt=0, lt=1)
    loss_weights: Literal["uniform", "competition"]
    # "sqrt": a positive weighs sqrt(negatives / positives) within each location label
    pos_weight: Literal["sqrt"] | None = None
    num_workers: int = Field(ge=0)
    # stop once validation loss has not improved for this many epochs; None trains every epoch
    early_stop_patience: PositiveInt | None = None
    backbone: Literal["resnet18", "resnet34", "resnet50", "r3d18"] = "resnet18"
    # "imagenet": torchvision's 2D ResNet weights inflated to 3D (resnet* backbones);
    # "kinetics": torchvision's r3d_18 Kinetics-400 weights (r3d18); the head starts random
    init: Literal["random", "imagenet", "kinetics"] = "random"
    # the first this many epochs train only the head; every layer trains after
    freeze_epochs: int = Field(default=0, ge=0)
    # every layer but the head trains at lr x this
    backbone_lr_scale: float = Field(default=1.0, gt=0, le=1)
    # edge length of the cached cube the model trains on: reads cache/b2-{input_size}
    input_size: Literal[128, 160] = 128
    # "mlp": 512 -> 256 -> 14 with LayerNorm, ReLU and dropout, random start; "linear": one layer
    head: Literal["linear", "mlp"] = "linear"

    @model_validator(mode="after")
    def _warmup_fits_in_training(self) -> Self:
        if self.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must be less than epochs")
        if self.freeze_epochs >= self.epochs:
            raise ValueError("freeze_epochs must be less than epochs")
        return self

    @model_validator(mode="after")
    def _init_fits_backbone(self) -> Self:
        if self.init == "imagenet" and not self.backbone.startswith("resnet"):
            raise ValueError("init imagenet needs a resnet backbone")
        if self.init == "kinetics" and self.backbone != "r3d18":
            raise ValueError("init kinetics needs the r3d18 backbone")
        return self


class AugmentConfig(_Frozen):
    prob: float = Field(ge=0, le=1)
    noise_std: float = Field(ge=0)
    gamma: tuple[PositiveFloat, PositiveFloat]
    rotate_deg: float = Field(ge=0)
    scale: float = Field(ge=0, lt=1)
    lowres_zoom: tuple[PositiveFloat, PositiveFloat]
    flip_lr: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def _ranges_are_ordered(self) -> Self:
        for name in ("gamma", "lowres_zoom"):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"{name} must be (low, high) with low <= high")
        if self.lowres_zoom[1] > 1:
            raise ValueError("lowres_zoom high must be at most 1")
        return self


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
