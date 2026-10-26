import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(optimizer: Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    if warmup_steps < 0 or total_steps <= warmup_steps:
        raise ValueError(f"need 0 <= warmup_steps < total_steps, got {warmup_steps}, {total_steps}")

    def factor(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, factor)
