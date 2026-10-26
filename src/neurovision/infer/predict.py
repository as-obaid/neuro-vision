import numpy as np
import torch
from torch import nn


def predict_volumes(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Probabilities, float32 `(n, 14)`; `x` is moved to the model's device."""
    model.eval()
    device = next(model.parameters()).device
    # float32 on purpose: bf16 logits tie and distort AUC; also blocks a caller's autocast
    with torch.inference_mode(), torch.autocast(device.type, enabled=False):
        logits = model(x.to(device))
    return torch.sigmoid(logits).cpu().numpy()
