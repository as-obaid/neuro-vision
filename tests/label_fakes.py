"""Synthetic label tables shaped like `load_labels` output."""

import numpy as np
import pandas as pd

from neurovision.data.labels import ANEURYSM_PRESENT, TARGETS


def fake_labels(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {t: (rng.random(n) < 0.1).astype("int8") for t in TARGETS},
        index=pd.Index([f"s{i:04d}" for i in range(n)], name="SeriesInstanceUID"),
    )
    frame[ANEURYSM_PRESENT] = (rng.random(n) < 0.4).astype("int8")
    frame["PatientAge"] = rng.integers(20, 90, n)
    frame["PatientSex"] = rng.choice(["Male", "Female"], n)
    frame["Modality"] = rng.choice(["CTA", "MRA", "MRI T1post", "MRI T2"], n)
    return frame
