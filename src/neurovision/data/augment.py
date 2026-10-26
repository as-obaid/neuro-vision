"""Training augmentation built from the config's `augment` block."""

import math

from monai.transforms import (
    Compose,
    RandAdjustContrast,
    RandAffine,
    RandGaussianNoise,
    RandSimulateLowResolution,
)

from neurovision.config import AugmentConfig


def build_augment(cfg: AugmentConfig) -> Compose:
    rotate = math.radians(cfg.rotate_deg)
    return Compose(
        [
            RandGaussianNoise(prob=cfg.prob, std=cfg.noise_std),
            RandAdjustContrast(prob=cfg.prob, gamma=cfg.gamma),
            RandAffine(
                prob=cfg.prob,
                rotate_range=(rotate,) * 3,
                scale_range=(cfg.scale,) * 3,
                padding_mode="zeros",
            ),
            RandSimulateLowResolution(prob=cfg.prob, zoom_range=cfg.lowres_zoom),
        ]
    )
