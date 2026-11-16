"""Vessel pretraining: the r3d_18 U-Net, its dataset, and its encoder in the classifier."""

from pathlib import Path

import numpy as np
import pytest
import torch

from neurovision.data.vessels import N_SEG_CLASSES, SEG_LR_SWAP, VesselDataset
from neurovision.models.resnet3d import build_model
from neurovision.models.unet import ENCODER, R3DUNet, encoder_state, load_encoder

pytest.importorskip("torchvision")
SIZE = 32


def test_unet_returns_27_class_logits_at_the_input_size() -> None:
    model = R3DUNet().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, SIZE, SIZE, SIZE))
    assert out.shape == (2, N_SEG_CLASSES, SIZE, SIZE, SIZE)


def test_unet_encoder_names_match_r3d18_and_hand_over_to_the_classifier() -> None:
    torch.manual_seed(0)
    unet = R3DUNet()
    state = encoder_state(unet)
    assert state and all(k.startswith(ENCODER) for k in state)
    classifier = build_model("r3d18", aux_map=True)
    before = classifier.state_dict()["aux.0.weight"].clone()
    load_encoder(classifier, state)
    after = classifier.state_dict()
    for k, v in state.items():
        torch.testing.assert_close(after[k], v)
    torch.testing.assert_close(after["aux.0.weight"], before)


def test_load_encoder_rejects_a_missing_layer() -> None:
    state = encoder_state(R3DUNet())
    del state["layer2.0.conv1.0.weight"]
    with pytest.raises(ValueError, match="layer2.0.conv1.0.weight"):
        load_encoder(build_model("r3d18"), state)


def test_vessel_init_loads_the_encoder_file(tmp_path: Path) -> None:
    state = encoder_state(R3DUNet())
    torch.save(state, tmp_path / "encoder.pt")
    model = build_model("r3d18", "vessel", tmp_path, aux_map=True)
    key = "layer4.1.conv2.0.weight"
    torch.testing.assert_close(model.state_dict()[key], state[key])
    with pytest.raises(ValueError, match="needs"):
        build_model("swin3d_t", "vessel", tmp_path)


def test_vessel_dataset_flips_the_volume_and_swaps_the_side_classes(tmp_path: Path) -> None:
    cache, seg_dir = tmp_path / "cache", tmp_path / "seg"
    cache.mkdir()
    seg_dir.mkdir()
    volume = np.random.default_rng(0).normal(size=(SIZE,) * 3).astype(np.float16)
    np.save(cache / "a.npy", volume)
    seg = np.zeros((SIZE,) * 3, np.uint8)
    seg[2, 5, 6] = 1  # artery 0, a left-sided one; axis 0 is left-right
    np.savez_compressed(seg_dir / "a.npz", seg=seg)
    x, y = VesselDataset(cache, seg_dir, ["a"], flip_lr=0.0)[0]
    assert x.shape == (1, SIZE, SIZE, SIZE) and y.dtype == torch.int64 and y[2, 5, 6] == 1
    fx, fy = VesselDataset(cache, seg_dir, ["a"], flip_lr=1.0)[0]
    torch.testing.assert_close(fx, torch.flip(x, dims=[1]))
    assert fy[SIZE - 3, 5, 6] == SEG_LR_SWAP[1] and int(fy.sum()) == SEG_LR_SWAP[1]


def test_step1_config_changes_only_the_starting_weights() -> None:
    from neurovision.config import load_config

    tune = Path(__file__).resolve().parents[1] / "configs" / "tune"
    parent = load_config(tune / "t_flip_b4_r3d18_mm160_aug0.1_lr6e-4_aux_cos30.yaml")
    child = load_config(tune / "t_aux_cos30_vessel.yaml")
    assert parent.train is not None and child.train is not None
    assert child.train == parent.train.model_copy(
        update={"init": "vessel", "vessel_run": "vessel_r3d18_160"}
    )
