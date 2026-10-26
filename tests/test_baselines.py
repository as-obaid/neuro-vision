import numpy as np

from neurovision.baselines.b0_prior import predict_prior
from neurovision.baselines.b1_gbm import predict_gbm
from neurovision.config import GbmConfig
from neurovision.data.labels import TARGETS
from tests.label_fakes import fake_labels

GBM = GbmConfig(max_iter=20, learning_rate=0.1, max_depth=2)


def test_prior_is_training_mean_for_every_validation_row() -> None:
    labels = fake_labels()
    train, val = labels.iloc[:150], labels.iloc[150:]
    prob = predict_prior(train, val)
    assert prob.shape == (50, 14)
    expected = train[list(TARGETS)].mean().to_numpy()
    np.testing.assert_allclose(prob, np.tile(expected, (50, 1)))


def test_gbm_output_shape_and_range() -> None:
    labels = fake_labels()
    prob = predict_gbm(labels.iloc[:150], labels.iloc[150:], GBM, seed=0)
    assert prob.shape == (50, 14)
    assert ((prob >= 0) & (prob <= 1)).all()


def test_gbm_one_class_target_gets_constant_rate() -> None:
    labels = fake_labels()
    train, val = labels.iloc[:150].copy(), labels.iloc[150:]
    train[TARGETS[0]] = 0
    train.loc[train.index[0], TARGETS[1]] = 1
    train[TARGETS[2]] = 1
    prob = predict_gbm(train, val, GBM, seed=0)
    assert (prob[:, 0] == 0).all()
    assert (prob[:, 2] == 1).all()
    assert prob[:, 1].std() > 0


def test_gbm_seed_is_reproducible() -> None:
    labels = fake_labels()
    a = predict_gbm(labels.iloc[:150], labels.iloc[150:], GBM, seed=3)
    b = predict_gbm(labels.iloc[:150], labels.iloc[150:], GBM, seed=3)
    np.testing.assert_array_equal(a, b)


def test_gbm_tolerates_category_unseen_in_training() -> None:
    labels = fake_labels()
    val = labels.iloc[150:].copy()
    val["PatientSex"] = "Unknown"
    prob = predict_gbm(labels.iloc[:150], val, GBM, seed=0)
    assert prob.shape == (50, 14)
