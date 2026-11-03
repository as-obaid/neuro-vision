import numpy as np
import pytest

from neurovision.evaluate.curves import calibration, operating_point, operating_points


def test_counts_and_rates_at_a_threshold() -> None:
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    p = np.array([0.1, 0.6, 0.2, 0.7, 0.4, 0.9, 0.8, 0.3])
    point = operating_point(y, p, 0.5)
    assert (point["tp"], point["fp"], point["tn"], point["fn"]) == (3, 1, 3, 1)
    assert point["sensitivity"] == 0.75
    assert point["specificity"] == 0.75
    assert point["ppv"] == 0.75
    assert point["npv"] == 0.75
    assert point["f1"] == pytest.approx(0.75)
    assert point["threshold"] == 0.5


def test_perfect_predictor_has_full_sensitivity_and_specificity() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.4).astype(int)
    p = np.where(y == 1, 0.6 + 0.4 * rng.random(200), 0.5 * rng.random(200))
    points = operating_points(y, p)
    for point in points.values():
        assert point["fp"] + point["tn"] == int((y == 0).sum())
        assert point["tp"] + point["fn"] == int((y == 1).sum())
    assert points["youden"]["sensitivity"] == 1.0
    assert points["youden"]["specificity"] == 1.0
    assert points["sensitivity_0.90"]["sensitivity"] >= 0.90
    assert points["sensitivity_0.90"]["specificity"] == 1.0


def test_sensitivity_point_is_the_highest_threshold_reaching_the_target() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.4).astype(int)
    p = np.clip(0.25 * y + rng.random(500), 0, 1)
    point = operating_points(y, p, sensitivity=0.90)["sensitivity_0.90"]
    assert point["sensitivity"] >= 0.90
    higher = p[p > point["threshold"]].min()
    assert operating_point(y, p, higher)["sensitivity"] < 0.90


def test_calibration_of_a_calibrated_constant_is_zero() -> None:
    y = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    result = calibration(y, np.full(10, 0.3), bins=1)
    assert result["ece"] == pytest.approx(0.0)
    assert result["bins"] == [
        {"mean_predicted": pytest.approx(0.3), "observed": pytest.approx(0.3), "count": 10}
    ]
    assert result["brier"] == pytest.approx(np.mean((0.3 - y) ** 2))
    assert sum(b["count"] for b in calibration(y, np.full(10, 0.3), bins=5)["bins"]) == 10


def test_calibration_bins_are_equal_count_and_ece_is_weighted() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.6, 0.8])
    result = calibration(y, p, bins=2)
    assert [b["count"] for b in result["bins"]] == [2, 2]
    assert result["bins"][0]["mean_predicted"] == pytest.approx(0.2)
    assert result["bins"][0]["observed"] == 0.0
    assert result["bins"][1]["observed"] == 1.0
    assert result["ece"] == pytest.approx(0.5 * 0.2 + 0.5 * 0.3)
