import time

import pytest
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from neurovision.models.resnet3d import build_model
from neurovision.train.ema import make_ema
from neurovision.train.loop import train_one_epoch
from neurovision.train.schedule import cosine_with_warmup

BATCH, ACCUM, WARMUP, TIMED = 2, 4, 3, 20


def _loader(n_batches: int) -> DataLoader[tuple[torch.Tensor, ...]]:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n_batches * BATCH, 1, 128, 128, 128, generator=g)
    y = torch.randint(0, 2, (n_batches * BATCH, 14), generator=g).float()
    return DataLoader(TensorDataset(x, y), batch_size=BATCH)


@pytest.mark.gpu
def test_b2_step_memory_and_speed() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    model = build_model().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = cosine_with_warmup(optimizer, 1, 100)
    ema = make_ema(model, 0.999)

    train_one_epoch(model, _loader(WARMUP), optimizer, scheduler, ema, ACCUM, device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    loader = _loader(TIMED)
    start = time.perf_counter()
    loss = train_one_epoch(model, loader, optimizer, scheduler, ema, ACCUM, device)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    per_batch = elapsed / TIMED
    print(f"gpu: {torch.cuda.get_device_name()}")
    print(f"peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"seconds per batch: {per_batch:.3f}")
    print(f"seconds per optimizer step: {per_batch * ACCUM:.3f}")
    assert torch.isfinite(torch.tensor(loss))
