import numpy as np
import pytest

from neurovision.evaluate.bootstrap import METRICS, paired_bootstrap


def make_case(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 14)) < 0.3).astype(np.int8)
    y[:2] = [[0] * 14, [1] * 14]  # every column has both classes
    p = np.clip(y * 0.3 + rng.random((n, 14)), 0, 1)
    return y, p


def test_identical_predictions_differ_by_exactly_zero() -> None:
    y, p = make_case()
    result = paired_bootstrap(y, p, p.copy(), n=50)
    for metric in METRICS:
        assert result[metric] == {
            "difference": 0.0,
            "mean": 0.0,
            "lo": 0.0,
            "hi": 0.0,
            "p_le_0": 1.0,
        }
    assert result["resamples"] == 50


def test_perfect_minus_constant_is_one_half_on_every_resample() -> None:
    y, _ = make_case()
    result = paired_bootstrap(y, y.astype(float), np.full(y.shape, 0.3), n=40)
    for metric in METRICS:
        assert result[metric]["difference"] == pytest.approx(0.5)
        assert result[metric]["lo"] == pytest.approx(0.5)
        assert result[metric]["hi"] == pytest.approx(0.5)
        assert result[metric]["p_le_0"] == 0.0


def test_ci_brackets_the_point_difference() -> None:
    y, p = make_case(seed=1)
    noisy = np.clip(p + np.random.default_rng(2).normal(0, 0.3, p.shape), 0, 1)
    result = paired_bootstrap(y, p, noisy, n=200)
    for metric in METRICS:
        r = result[metric]
        assert r["lo"] <= r["difference"] <= r["hi"]
        assert r["difference"] > 0


def test_resamples_without_both_classes_are_skipped_and_counted() -> None:
    y, p = make_case(n=40)
    y[:, 3] = 0
    y[5, 3] = 1  # one positive: many resamples miss it
    result = paired_bootstrap(y, p, p[::-1].copy(), n=200)
    assert 0 < result["used"] < result["resamples"] == 200


def test_same_seed_same_result_and_shapes_checked() -> None:
    y, p = make_case()
    q = p[::-1].copy()
    assert paired_bootstrap(y, p, q, n=30, seed=4) == paired_bootstrap(y, p, q, n=30, seed=4)
    with pytest.raises(ValueError, match="shape"):
        paired_bootstrap(y, p, q[:-1], n=10)
