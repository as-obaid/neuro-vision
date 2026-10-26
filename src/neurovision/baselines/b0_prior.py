"""B0: every validation series gets the training-fold rate of each target."""

import numpy as np
import pandas as pd

from neurovision.data.labels import TARGETS


def predict_prior(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    prior = train[list(TARGETS)].mean().to_numpy(dtype=np.float64)
    return np.tile(prior, (len(val), 1))
