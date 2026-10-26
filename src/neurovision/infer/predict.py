import numpy as np
import torch
from torch import nn


def predict_volumes(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Probabilities, float32 `(n, 14)`; `x` is moved to the model's device."""
    model.eval()
    device = next(model.parameters()).device
    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"),
    ):
        logits = model(x.to(device))
    return torch.sigmoid(logits.float()).cpu().numpy()
