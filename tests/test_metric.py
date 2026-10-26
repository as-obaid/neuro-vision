import contextlib

import numpy as np
import pandas as pd
import pytest

from neurovision.data.labels import TARGETS
from neurovision.evaluate.metric import label_aucs, score
from tests import kaggle_metric

WEIGHTS = [1.0] * 13 + [13.0]


def official(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    solution = pd.DataFrame(y_true, columns=list(TARGETS))
    submission = pd.DataFrame(y_prob, columns=list(TARGETS))
    solution.insert(0, "id", range(len(solution)))
    submission.insert(0, "id", range(len(submission)))
    return kaggle_metric.score(solution, submission, "id", class_weights=WEIGHTS)


def make_case(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(40, 400))
    prevalence = rng.uniform(0.05, 0.6, size=14)
    y_true = (rng.random((n, 14)) < prevalence).astype(np.int8)
    y_true[:2] = [[0] * 14, [1] * 14]  # every column has both classes
    y_prob = np.clip(y_true * rng.uniform(0, 0.6) + rng.random((n, 14)), 0, 1)
    if seed % 3 == 0:
        y_prob = np.round(y_prob, 1)  # many ties
    return y_true, y_prob


@pytest.mark.parametrize("seed", range(24))
def test_score_matches_official_code(seed):
    y_true, y_prob = make_case(seed)
    assert score(y_true, y_prob) == pytest.approx(official(y_true, y_prob), abs=1e-6)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_single_class_column_is_an_error_where_official_code_fails_or_returns_nan():
    y_true, y_prob = make_case(0)
    y_true[:, 3] = 0
    # The official code raises on older scikit-learn and returns nan on the pinned one.
    with contextlib.suppress(kaggle_metric.ParticipantVisibleError):
        assert np.isnan(official(y_true, y_prob))
    with pytest.raises(ValueError, match="both classes"):
        score(y_true, y_prob)


def test_constant_predictions_score_exactly_half():
    y_true, _ = make_case(1)
    assert score(y_true, np.full(y_true.shape, 0.3)) == 0.5


def test_perfect_predictor_scores_one():
    y_true, _ = make_case(2)
    assert score(y_true, y_true.astype(float)) == 1.0


def test_perfect_detector_with_shuffled_locations_scores_near_three_quarters():
    rng = np.random.default_rng(3)
    n = 4000
    y_true = (rng.random((n, 14)) < 0.3).astype(np.int8)
    y_prob = rng.random((n, 14))
    y_prob[:, 13] = y_true[:, 13]
    assert score(y_true, y_prob) == pytest.approx(0.75, abs=0.02)


def test_label_aucs_names_every_target_and_weights_aneurysm_present():
    y_true, y_prob = make_case(4)
    aucs = label_aucs(y_true, y_prob)
    assert list(aucs) == list(TARGETS)
    expected = (13 * aucs[TARGETS[13]] + sum(aucs[t] for t in TARGETS[:13])) / 26
    assert score(y_true, y_prob) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("shape_true, shape_prob", [((10, 13), (10, 13)), ((10, 14), (9, 14))])
def test_wrong_shape_raises_value_error(shape_true, shape_prob):
    with pytest.raises(ValueError, match="shape"):
        score(np.zeros(shape_true), np.zeros(shape_prob))
