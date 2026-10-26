"""B1: gradient boosting on age, sex and modality, one classifier per target."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from neurovision.config import GbmConfig
from neurovision.data.labels import TARGETS

_CATEGORICAL = ("PatientSex", "Modality")


def _features(frame: pd.DataFrame, categories: dict[str, pd.Index]) -> pd.DataFrame:
    features = pd.DataFrame({"PatientAge": frame["PatientAge"].astype(float)})
    for col in _CATEGORICAL:
        # Values unseen in training become NaN, which the model treats as missing.
        seen = frame[col].where(frame[col].isin(categories[col]))
        features[col] = pd.Categorical(seen, categories=categories[col])
    return features


def predict_gbm(train: pd.DataFrame, val: pd.DataFrame, gbm: GbmConfig, seed: int) -> np.ndarray:
    categories = {col: pd.Index(sorted(train[col].dropna().unique())) for col in _CATEGORICAL}
    x_train, x_val = _features(train, categories), _features(val, categories)
    prob = np.zeros((len(val), len(TARGETS)))
    for i, target in enumerate(TARGETS):
        y = train[target]
        if y.nunique() < 2:
            prob[:, i] = y.mean()
            continue
        model = HistGradientBoostingClassifier(
            max_iter=gbm.max_iter,
            learning_rate=gbm.learning_rate,
            max_depth=gbm.max_depth,
            categorical_features="from_dtype",
            random_state=seed,
        )
        prob[:, i] = model.fit(x_train, y).predict_proba(x_val)[:, 1]
    return prob
