from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, TensorDataset

from neurovision.infer.predict import predict_volumes
from neurovision.models.resnet3d import build_model
from neurovision.train.ema import make_ema
from neurovision.train.loop import evaluate, load_checkpoint, save_checkpoint, train_one_epoch
from neurovision.train.schedule import cosine_with_warmup

CPU = torch.device("cpu")


def _data(n: int, size: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, size, size, size, generator=g)
    y = torch.randint(0, 2, (n, 14), generator=g).float()
    return x, y


def _setup(lr: float = 1e-3, total: int = 100) -> tuple[nn.Module, AdamW, LambdaLR, AveragedModel]:
    torch.manual_seed(0)
    model = build_model()
    opt = AdamW(model.parameters(), lr=lr)
    sched = cosine_with_warmup(opt, 0, total)
    return model, opt, sched, make_ema(model, 0.9)


def _linear_step(batches: list[tuple[torch.Tensor, torch.Tensor]], accum: int) -> torch.Tensor:
    torch.manual_seed(0)
    linear = nn.Linear(8, 14)
    model = nn.Sequential(nn.Flatten(), linear)
    opt = SGD(model.parameters(), lr=0.1)
    sched = cosine_with_warmup(opt, 0, 10)
    train_one_epoch(model, batches, opt, sched, make_ema(model, 0.9), accum, CPU)
    return linear.weight.detach().clone()


def _pair(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 2, 2, 2, generator=g), torch.randint(0, 2, (n, 14), generator=g).float()


def test_one_step_changes_weights_and_gives_finite_loss() -> None:
    model, opt, sched, ema = _setup()
    before = [p.detach().clone() for p in model.parameters()]
    x, y = _data(2)
    loss = train_one_epoch(
        model, DataLoader(TensorDataset(x, y), batch_size=2), opt, sched, ema, 1, CPU
    )
    assert np.isfinite(loss)
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True))


@pytest.mark.parametrize(("batches", "accum", "steps"), [(4, 2, 2), (5, 2, 3)])
def test_optimizer_steps_once_per_accumulation_group(batches: int, accum: int, steps: int) -> None:
    model, opt, sched, ema = _setup()
    x, y = _data(batches)
    train_one_epoch(
        model, DataLoader(TensorDataset(x, y), batch_size=1), opt, sched, ema, accum, CPU
    )
    assert sched.last_epoch == steps


def test_tiny_batch_overfit() -> None:
    model, opt, sched, ema = _setup()
    loader = DataLoader(TensorDataset(*_data(4)), batch_size=4)
    losses = [train_one_epoch(model, loader, opt, sched, ema, 1, CPU) for _ in range(60)]
    assert losses[-1] < 0.3 * losses[0]


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model, opt, sched, ema = _setup()
    loader = DataLoader(TensorDataset(*_data(2)), batch_size=1)
    train_one_epoch(model, loader, opt, sched, ema, 1, CPU)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, ema, opt, sched, epoch=3, best_score=0.7, peak_mib=12.5)
    assert not list(tmp_path.glob("*.tmp"))

    model2, opt2, sched2, ema2 = _setup()
    assert load_checkpoint(path, model2, ema2, opt2, sched2) == (4, 0.7, 12.5)
    for a, b in zip(model.state_dict().values(), model2.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    for a, b in zip(ema.state_dict().values(), ema2.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert sched2.last_epoch == sched.last_epoch == 2
    state, state2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert state.keys() == state2.keys()
    assert torch.equal(state[0]["exp_avg"], state2[0]["exp_avg"])


def test_predict_volumes_returns_probabilities() -> None:
    probs = predict_volumes(build_model(), _data(3)[0])
    assert probs.shape == (3, 14)
    assert probs.dtype == np.float32
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_evaluate_returns_probabilities_and_bce() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(8**3, 14))
    x, y = _data(5, size=8)
    probs, loss = evaluate(model, DataLoader(TensorDataset(x, y), batch_size=2), CPU)
    assert probs.shape == (5, 14)
    expected = nn.functional.binary_cross_entropy(torch.from_numpy(probs), y).item()
    assert loss == pytest.approx(expected, rel=1e-4)


def test_accumulation_matches_one_big_batch() -> None:
    x, y = _pair(2, 1)
    big = _linear_step([(x, y)], 1)
    split = _linear_step([(x[:1], y[:1]), (x[1:], y[1:])], 2)
    torch.testing.assert_close(split, big, atol=1e-6, rtol=0)


def test_partial_last_group_divides_by_its_own_size() -> None:
    batches = [_pair(1, 1), _pair(1, 2), _pair(1, 3)]
    got = _linear_step(batches, 2)

    torch.manual_seed(0)
    linear = nn.Linear(8, 14)
    ref = nn.Sequential(nn.Flatten(), linear)
    opt = SGD(ref.parameters(), lr=0.1)
    sched = cosine_with_warmup(opt, 0, 10)
    for group in (batches[:2], batches[2:]):
        for x, y in group:
            (nn.functional.binary_cross_entropy_with_logits(ref(x), y) / len(group)).backward()
        opt.step()
        sched.step()
        opt.zero_grad()
    torch.testing.assert_close(got, linear.weight.detach(), atol=1e-6, rtol=0)


def test_train_and_evaluate_reject_empty_loader() -> None:
    model, opt, sched, ema = _setup()
    with pytest.raises(ValueError):
        train_one_epoch(model, [], opt, sched, ema, 1, CPU)
    with pytest.raises(ValueError):
        evaluate(model, [], CPU)


def test_predict_volumes_runs_in_float32_so_close_logits_stay_distinct() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(1, 14, bias=False))
    nn.init.constant_(model[1].weight, 1.0)
    # 1.0 and 1.0 + 2**-12 differ in float32 but collapse to one bf16 value
    x = torch.tensor([1.0, 1.0 + 2**-12]).reshape(2, 1)
    with torch.autocast("cpu", dtype=torch.bfloat16):  # a caller's autocast must not leak in
        probs = predict_volumes(model, x)
    assert probs[0, 0] != probs[1, 0]
