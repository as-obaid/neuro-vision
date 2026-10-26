from pathlib import Path

import numpy as np
import torch
from torch import nn

from neurovision.infer.predict import predict_series
from tests.dicom_fakes import write_series

KW = {"shape": (20, 16, 16), "pixel_spacing": (1.0, 1.0), "slice_spacing": 2.0}
PRIOR = np.linspace(0.1, 0.9, 14)


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(1, 14))


def test_series_gives_fourteen_probabilities(tmp_path: Path) -> None:
    write_series(tmp_path / "s", **KW)
    probs = predict_series(tmp_path / "s", _model(), PRIOR)
    assert probs.shape == (14,)
    assert ((probs > 0) & (probs < 1)).all()
    assert not np.allclose(probs, PRIOR)


def test_broken_geometry_gives_the_prior(tmp_path: Path) -> None:
    positions = [(0.0, 3.0 * i, 3.0 * i) for i in range(20)]
    write_series(
        tmp_path / "s",
        **KW,
        positions=positions,
        multiframe=True,
        top_level_geometry=False,
        frame_orientation=False,
    )
    np.testing.assert_array_equal(predict_series(tmp_path / "s", _model(), PRIOR), PRIOR)


def test_corrupt_file_gives_the_prior(tmp_path: Path) -> None:
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "0.dcm").write_bytes(b"not a dicom file")
    np.testing.assert_array_equal(predict_series(tmp_path / "s", _model(), PRIOR), PRIOR)
