"""B0: every series gets each target's prevalence over all labelled series."""

import numpy as np
import pandas as pd

from neurovision.data.labels import TARGETS


def predict_prior(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    # One constant vector, identical in every fold, so pooled out-of-fold AUC is exactly 0.5.
    prior = pd.concat([train, val])[list(TARGETS)].mean().to_numpy(dtype=np.float64)
    return np.tile(prior, (len(val), 1))
