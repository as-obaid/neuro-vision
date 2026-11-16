"""Training augmentation built from the config's `augment` block."""

import math

from monai.transforms import (
    Compose,
    RandAdjustContrast,
    RandAdjustContrastd,
    RandAffine,
    RandAffined,
    RandGaussianNoise,
    RandGaussianNoised,
    RandSimulateLowResolution,
    RandSimulateLowResolutiond,
)

from neurovision.config import AugmentConfig

IMAGE, MAP = "image", "map"


def build_augment(cfg: AugmentConfig, with_map: bool = False) -> Compose:
    """Array pipeline on the volume; with `with_map`, the same transforms on a dict whose
    location map ("map") follows the image through the affine (nearest) and nothing else."""
    rotate = math.radians(cfg.rotate_deg)
    if with_map:
        return Compose(
            [
                RandGaussianNoised(keys=IMAGE, prob=cfg.prob, std=cfg.noise_std),
                RandAdjustContrastd(keys=IMAGE, prob=cfg.prob, gamma=cfg.gamma),
                RandAffined(
                    keys=(IMAGE, MAP),
                    prob=cfg.prob,
                    rotate_range=(rotate,) * 3,
                    scale_range=(cfg.scale,) * 3,
                    padding_mode="zeros",
                    mode=("bilinear", "nearest"),
                ),
                RandSimulateLowResolutiond(keys=IMAGE, prob=cfg.prob, zoom_range=cfg.lowres_zoom),
            ]
        )
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
