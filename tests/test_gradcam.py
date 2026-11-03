import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
import torch
from PIL import Image
from torch import nn

from neurovision.data.dicom import read_header
from neurovision.data.labels import ANEURYSM_PRESENT, TARGETS
from neurovision.data.preprocess import preprocess_with_crop, series_geometry
from neurovision.evaluate.gradcam import grad_cam
from neurovision.models.resnet3d import build_model
from tests.dicom_fakes import write_series
from tests.script_loader import load_script

SIZE = 16


class _Tiny(nn.Module):
    """One 1x1 conv stage named like the ResNets' last stage, then mean pool and a linear head.

    With a unit filter and a positive head, the evidence for every output is the input's
    positive part, so the CAM must peak on a bright blob.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer4 = nn.Sequential(nn.Conv3d(1, 2, 1, bias=False), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(2, len(TARGETS))
        with torch.no_grad():
            self.layer4[0].weight.fill_(1.0)
            self.fc.weight.fill_(0.5)
            self.fc.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.layer4(x)).flatten(1))


def _blob() -> torch.Tensor:
    x = torch.zeros(1, 1, SIZE, SIZE, SIZE)
    x[0, 0, 3:6, 9:12, 4:7] = 5.0
    return x


def test_cam_has_the_input_shape_and_unit_range() -> None:
    cam = grad_cam(_Tiny(), _blob())
    assert cam.shape == (SIZE, SIZE, SIZE)
    assert cam.dtype == np.float32
    assert cam.min() == pytest.approx(0.0)
    assert cam.max() == pytest.approx(1.0)


def test_cam_peaks_on_the_bright_blob() -> None:
    cam = grad_cam(_Tiny(), _blob())
    assert np.ptp(cam) > 0
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    assert 3 <= peak[0] < 6 and 9 <= peak[1] < 12 and 4 <= peak[2] < 7
    assert cam[0, 0, 0] < 0.05


def test_cam_without_evidence_is_zero() -> None:
    cam = grad_cam(_Tiny(), torch.zeros(1, 1, SIZE, SIZE, SIZE))
    assert not cam.any()


def test_other_outputs_and_model_state_are_left_alone() -> None:
    model = _Tiny().train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    grad_cam(model, _blob(), target=TARGETS.index("Basilar Tip"))
    assert model.training
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_default_target_is_aneurysm_present() -> None:
    model = _Tiny()
    with torch.no_grad():
        model.fc.weight[TARGETS.index(ANEURYSM_PRESENT)].fill_(-0.5)  # evidence against
    assert not grad_cam(model, _blob()).any()
    assert grad_cam(model, _blob(), target=0).any()


def test_one_volume_only() -> None:
    with pytest.raises(ValueError, match="one volume"):
        grad_cam(_Tiny(), torch.zeros(2, 1, SIZE, SIZE, SIZE))


@pytest.mark.parametrize("backbone", ["r3d18", "resnet18"])
def test_backbones_hook_their_last_stage(backbone: str) -> None:
    torch.manual_seed(0)
    model = build_model(backbone)  # type: ignore[arg-type]
    x = torch.randn(1, 1, 32, 32, 32)
    x[0, 0, 8:14, 8:14, 8:14] += 4.0
    cam = grad_cam(model, x)
    assert cam.shape == (32, 32, 32)
    assert cam.min() >= 0.0 and cam.max() <= 1.0


def test_display_point_follows_the_displayed_voxel() -> None:
    script = load_script("gradcam_examples")
    plane = np.zeros((SIZE, SIZE))
    plane[2, 11] = 1.0
    col, row = script.display_point(np.array([2.0, 11.0, 0.0]), SIZE)
    shown = script.display(plane)
    assert shown[int(row), int(col)] == 1.0
    # radiological view: one voxel further along x (RAS+) is one pixel further left on screen
    assert script.display_point(np.array([3.0, 11.0, 0.0]), SIZE)[0] == col - 1


def test_inside_needs_the_margin() -> None:
    script = load_script("gradcam_examples")
    assert script.inside(np.array([8.0, 8.0, 8.0]), SIZE, 5.0)
    assert not script.inside(np.array([2.0, 8.0, 8.0]), SIZE, 5.0)
    assert script.inside(np.array([8.0, 8.0, 15.0]), SIZE, 0.0)  # the last voxel
    assert not script.inside(np.array([8.0, 8.0, 15.8]), SIZE, 0.0)


def test_render_writes_one_figure_without_identifiers(tmp_path: Path) -> None:
    script = load_script("gradcam_examples")
    rng = np.random.default_rng(0)
    examples, cams = [], []
    for scan_type in script.SCAN_TYPES:
        volume = rng.normal(size=(SIZE,) * 3).astype(np.float32)
        probs = np.full(len(TARGETS), 0.5, dtype=np.float32)
        examples.append(
            script.Example(scan_type, "Left Middle Cerebral Artery", (8.0, 8.0, 8.0), volume, probs)
        )
        cams.append(grad_cam(_Tiny(), torch.from_numpy(np.abs(volume))[None, None]))
    script.render(examples, cams, tmp_path / "figure.png")
    image = Image.open(tmp_path / "figure.png")
    assert image.width > 4 * SIZE * script.SCALE
    script.write_overview(examples[0], tmp_path, "overview", ["Left Middle Cerebral Artery"])
    record = json.loads((tmp_path / "overview.json").read_text())
    assert set(record["probabilities"]) == set(TARGETS)
    assert record["true_arteries"] == ["Left Middle Cerebral Artery"]
    assert "SeriesInstanceUID" not in json.dumps(record)


def test_short_names() -> None:
    script = load_script("gradcam_examples")
    assert script.short_name("Right Infraclinoid Internal Carotid Artery") == (
        "Right Infraclinoid ICA"
    )
    assert script.short_name("Anterior Communicating Artery") == "Anterior communicating"


@pytest.mark.parametrize(
    ("orientation", "spacing"),
    [((1, 0, 0, 0, 1, 0), (0.8, 0.8)), ((0, 1, 0, 0, 0, -1), (0.7, 0.9))],  # axial, sagittal
)
def test_box_affine_from_headers_matches_the_crop(
    tmp_path: Path, orientation: tuple[int, ...], spacing: tuple[float, float]
) -> None:
    script = load_script("gradcam_examples")
    pixels = np.zeros((30, 40, 36), dtype=np.uint16)
    pixels[4:26, 6:34, 5:31] = 1000
    write_series(
        tmp_path / "s",
        shape=pixels.shape,
        pixel_spacing=spacing,
        slice_spacing=1.5,
        orientation=orientation,
        pixels=pixels,
    )
    _, geometry, crop = preprocess_with_crop(tmp_path / "s", 24, "mm")
    assert crop is not None
    headers = [read_header(p) for p in sorted((tmp_path / "s").glob("*.dcm"))]
    rebuilt = script.box_affine(series_geometry(headers), crop.offset_mm)
    np.testing.assert_allclose(rebuilt, crop.index_affine, atol=1e-9)


def test_head_chance_is_the_share_of_head_voxels_near_the_point() -> None:
    script = load_script("gradcam_examples")
    volume = np.zeros((40, 40, 40), dtype=np.float32)
    volume[10:30, 10:30, 10:30] = 1.0  # the head: a 20-voxel cube
    near = script.head_chance(volume, np.array([[20.0, 20.0, 20.0]]), 100.0)
    far = script.head_chance(volume, np.array([[0.0, 0.0, 0.0]]), 5.0)
    assert near == pytest.approx(1.0)
    assert far == 0.0


def test_pointing_summary_counts_hits_within_the_radius() -> None:
    script = load_script("gradcam_examples")
    records = pd.DataFrame(
        {
            "scan_type": ["CTA", "CTA", "MRA"],
            "cam_mm": [5.0, 30.0, 10.0],
            "fixed_mm": [25.0, 25.0, 15.0],
            "chance": [0.1, 0.2, 0.3],
        }
    )
    summary = script.pointing_summary(records, 20.0)
    assert summary["all"]["series"] == 3
    assert summary["all"]["cam_hit_rate"] == pytest.approx(2 / 3)
    assert summary["by_scan_type"]["CTA"]["fixed_point_hit_rate"] == 0.0
    assert summary["by_scan_type"]["MRA"]["random_head_voxel_hit_rate"] == pytest.approx(0.3)


def test_box_affine_with_assumed_pixel_spacing(tmp_path: Path) -> None:
    script = load_script("gradcam_examples")
    pixels = np.zeros((20, 64, 48), dtype=np.uint16)
    pixels[3:17, 8:56, 6:42] = 1000
    for path in write_series(tmp_path / "s", shape=pixels.shape, pixel_spacing=(1.0, 1.0),
                             slice_spacing=2.0, pixels=pixels):  # fmt: skip
        ds = pydicom.dcmread(path)
        del ds.PixelSpacing
        ds.save_as(path)
    _, geometry, crop = preprocess_with_crop(tmp_path / "s", 24, "mm")
    assert crop is not None and geometry.spacing_assumed
    headers = [read_header(p) for p in sorted((tmp_path / "s").glob("*.dcm"))]
    rebuilt = script.box_affine(series_geometry(headers), crop.offset_mm)
    np.testing.assert_allclose(rebuilt, crop.index_affine, atol=1e-9)


def test_mirror_axis_finds_the_symmetric_axis() -> None:
    script = load_script("gradcam_examples")
    rng = np.random.default_rng(0)
    half = rng.normal(size=(8, 16, 16)).astype(np.float32)
    volume = np.concatenate([half, half[::-1]], axis=0)  # mirror-symmetric about axis 0
    assert script.mirror_axis(volume) == 0
    assert script.mirror_axis(np.moveaxis(volume, 0, 2)) == 2
