import numpy as np
import pandas as pd
import pytest

from neurovision.data.labels import ANEURYSM_PRESENT, LOCATIONS, TARGETS
from neurovision.evaluate.subgroups import by_modality, summarize_subset
from tests.label_fakes import fake_labels


def test_perfect_predictions_score_one_within_every_scan_type() -> None:
    labels = fake_labels(n=800)
    probs = labels[list(TARGETS)].astype(float)
    result = by_modality(labels, probs)
    assert set(result) == {"CTA", "MRA", "MRI T1post", "MRI T2"}
    assert sum(r["n"] for r in result.values()) == 800
    assert sum(r["positives"] for r in result.values()) == labels[ANEURYSM_PRESENT].sum()
    for r in result.values():
        assert r["ap_auc"] == 1.0
        assert r["location_auc"] == 1.0
        assert r["location_columns"] == len(LOCATIONS)
        assert r["score"] == 1.0


def test_single_class_columns_are_left_out_and_counted() -> None:
    labels = fake_labels(n=400)
    labels.loc[labels["Modality"] == "MRA", LOCATIONS[0]] = 0
    labels.loc[labels["Modality"] == "MRI T2", ANEURYSM_PRESENT] = 1
    probs = pd.DataFrame(
        np.random.default_rng(0).random((400, len(TARGETS))),
        index=labels.index,
        columns=list(TARGETS),
    )
    result = by_modality(labels, probs)
    assert result["MRA"]["location_columns"] == len(LOCATIONS) - 1
    assert result["MRA"]["score"] is None
    assert result["MRI T2"]["ap_auc"] is None
    assert result["MRI T2"]["score"] is None
    assert result["CTA"]["score"] is not None


def test_subset_summary_matches_its_counts() -> None:
    labels = fake_labels(n=300)
    y = labels[list(TARGETS)].to_numpy()
    summary = summarize_subset(y, np.full(y.shape, 0.2))
    assert summary["n"] == 300
    assert summary["positives"] == int(y[:, -1].sum())
    assert summary["ap_auc"] == 0.5
    assert summary["score"] == pytest.approx(0.5)


def test_probs_must_cover_the_labels() -> None:
    labels = fake_labels(n=100)
    probs = labels[list(TARGETS)].astype(float).iloc[1:]
    with pytest.raises(ValueError, match="missing"):
        by_modality(labels, probs)
