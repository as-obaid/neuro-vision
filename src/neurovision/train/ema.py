from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


def make_ema(model: nn.Module, decay: float) -> AveragedModel:
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=True)
