"""The 14 prediction targets, in Kaggle `LABEL_COLS` (= `train.csv`) order, and label loading."""

import hashlib
from pathlib import Path
from typing import Literal

import pandas as pd

TARGETS: tuple[str, ...] = (
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
    "Aneurysm Present",
)
LOCATIONS = TARGETS[:13]
ANEURYSM_PRESENT = TARGETS[13]


def _lr_swap() -> tuple[int, ...]:
    """Column permutation that exchanges each Left X / Right X pair and fixes every other target."""
    swapped = []
    for name in TARGETS:
        if name.startswith("Left "):
            name = "Right " + name.removeprefix("Left ")
        elif name.startswith("Right "):
            name = "Left " + name.removeprefix("Right ")
        swapped.append(TARGETS.index(name))
    return tuple(swapped)


# Mirroring the head left-right turns a left-sided artery into the right-sided one: `y[LR_SWAP]`.
LR_SWAP: tuple[int, ...] = _lr_swap()

CORRECTED_SHA256 = "4f32f4aa650d145a37b058fe72fa3a8fc2bce4f2e303474f4b2beb751903a365"
_COLUMNS = [*TARGETS, "PatientAge", "PatientSex", "Modality"]


def _read(path: Path, extra: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [c for c in ["SeriesInstanceUID", *TARGETS, *extra] if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    if not frame["SeriesInstanceUID"].is_unique:
        raise ValueError(f"{path.name}: SeriesInstanceUID is not unique")
    if not frame[list(TARGETS)].isin([0, 1]).all().all():
        raise ValueError(f"{path.name}: targets must be 0 or 1")
    frame = frame.set_index("SeriesInstanceUID")
    frame[list(TARGETS)] = frame[list(TARGETS)].astype("int8")
    return frame


def load_labels(store: Path, source: Literal["corrected", "original"]) -> pd.DataFrame:
    train = _read(store / "raw" / "train.csv", ["PatientAge", "PatientSex", "Modality"])
    if source == "corrected":
        path = store / "labels" / "corrected.csv"
        if not path.is_file():
            raise ValueError(f"missing {path}; run scripts/derive_labels.py")
        if hashlib.sha256(path.read_bytes()).hexdigest() != CORRECTED_SHA256:
            raise ValueError(f"{path.name}: sha256 does not match the pinned value")
        corrected = _read(path, [])
        unknown = corrected.index.difference(train.index)
        if len(unknown):
            raise ValueError(f"{len(unknown)} series in {path.name} are not in train.csv")
        train.loc[corrected.index, list(TARGETS)] = corrected[list(TARGETS)]
    return train[_COLUMNS]
