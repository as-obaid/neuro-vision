import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from neurovision.train.ema import make_ema
from neurovision.train.losses import bce, label_weights
from neurovision.train.schedule import cosine_with_warmup


def test_bce_matches_torch_mean_bce_with_logits() -> None:
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 14, generator=g)
    targets = torch.randint(0, 2, (4, 14), generator=g).float()
    assert torch.allclose(bce(logits, targets), F.binary_cross_entropy_with_logits(logits, targets))
    assert torch.allclose(
        bce(logits, targets, label_weights("uniform")),
        F.binary_cross_entropy_with_logits(logits, targets),
    )


def test_competition_weights_give_aneurysm_present_half_the_loss() -> None:
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 14, generator=g)
    targets = torch.randint(0, 2, (4, 14), generator=g).float()
    per_label = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(0)
    expected = 0.5 * per_label[13] + 0.5 * per_label[:13].mean()
    assert torch.allclose(bce(logits, targets, label_weights("competition")), expected)


def test_ema_follows_closed_form_after_three_updates() -> None:
    decay = 0.9
    model = nn.Linear(1, 1, bias=False)
    ema = make_ema(model, decay)
    w0 = 1.0
    model.weight.data.fill_(w0)
    ema.update_parameters(model)  # first call copies the weights
    shadow = w0
    for w in (2.0, 3.0, 4.0):
        model.weight.data.fill_(w)
        ema.update_parameters(model)
        shadow = decay * shadow + (1 - decay) * w
    assert float(ema.module.weight.detach()) == pytest.approx(shadow)


def _lrs(warmup: int, total: int) -> list[float]:
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-4)
    sched = cosine_with_warmup(opt, warmup, total)
    lrs = []
    for _ in range(total + 1):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return lrs


def test_schedule_warms_up_then_cosines_to_zero() -> None:
    lrs = _lrs(warmup=10, total=110)
    assert lrs[0] == 0.0
    assert lrs[10] == pytest.approx(1e-4)
    assert lrs[60] == pytest.approx(0.5e-4)
    assert lrs[110] == pytest.approx(0.0, abs=1e-12)
    assert lrs[5] == pytest.approx(0.5e-4)
    after = lrs[10:]
    assert all(a >= b for a, b in zip(after, after[1:], strict=False))
    assert math.isfinite(sum(lrs))


@pytest.mark.parametrize(("warmup", "total"), [(10, 10), (10, 5), (-1, 10), (0, -1)])
def test_schedule_rejects_bad_steps(warmup: int, total: int) -> None:
    opt = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=1e-4)
    with pytest.raises(ValueError):
        cosine_with_warmup(opt, warmup, total)


def test_ema_moves_batchnorm_buffers_toward_live_model() -> None:
    model = nn.BatchNorm1d(1)
    ema = make_ema(model, 0.5)
    ema.update_parameters(model)  # first call copies
    model.running_mean.fill_(4.0)
    model.running_var.fill_(9.0)
    ema.update_parameters(model)
    assert float(ema.module.running_mean) == pytest.approx(2.0)
    assert float(ema.module.running_var) == pytest.approx(5.0)
