import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from neurovision.data.preprocess import SIZE, Resample, preprocess_series

logger = logging.getLogger(__name__)


def predict_volumes(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Probabilities, float32 `(n, 14)`; `x` is moved to the model's device."""
    model.eval()
    device = next(model.parameters()).device
    # float32 on purpose: bf16 logits tie and distort AUC; also blocks a caller's autocast
    with torch.inference_mode(), torch.autocast(device.type, enabled=False):
        logits = model(x.to(device))
    return torch.sigmoid(logits).cpu().numpy()


def predict_series(
    series_dir: Path,
    model: nn.Module,
    prior: np.ndarray,
    size: int = SIZE,
    resample: Resample = "cube",
) -> np.ndarray:
    """Probabilities, `(14,)`, for one DICOM series; `prior` when the series cannot be read.

    `size` and `resample` are the training config's `train.input_size` and `train.resample`.
    """
    try:
        volume = preprocess_series(series_dir, size, resample)
    except Exception as error:  # a Kaggle run must never crash on one bad series
        logger.warning("series unreadable (%s); returning the prior", type(error).__name__)
        return prior.astype(np.float32)
    x = torch.from_numpy(volume.astype(np.float32))[None, None]
    return predict_volumes(model, x)[0]
