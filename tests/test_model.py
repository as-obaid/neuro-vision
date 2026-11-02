from pathlib import Path

import pytest
import torch

from neurovision.models import resnet3d
from neurovision.models.resnet3d import (
    HEAD,
    WEIGHTS,
    build_model,
    inflate,
    load_imagenet,
    load_kinetics,
    read_imagenet,
    read_weights,
)


def test_resnet18_maps_a_128_cube_to_14_logits() -> None:
    model = build_model().eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 128, 128, 128))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params}")
    assert out.shape == (1, 14)
    assert 0.9 * 33e6 <= n_params <= 1.1 * 33e6


def _imagenet_like(backbone: str = "resnet18") -> dict[str, torch.Tensor]:
    """Random weights with the names and 2D shapes of torchvision's ResNet."""
    gen = torch.Generator().manual_seed(0)
    weights = {}
    for name, t in build_model(backbone).state_dict().items():  # type: ignore[arg-type]
        if name.endswith("downsample.0.bias"):
            continue  # torchvision's shortcut convs have no bias
        if name.startswith(HEAD):
            width = t.shape[-1] if t.ndim == 2 else 0
            shape: tuple[int, ...] = (1000, width) if name.endswith("weight") else (1000,)
        elif t.ndim == 5:
            out, inp, _, h, w = t.shape
            shape = (out, 3 if name == "conv1.weight" else inp, h, w)
        else:
            shape = tuple(t.shape)
        weights[name] = torch.rand(shape, generator=gen) if t.is_floating_point() else t.clone()
    return weights


def test_inflated_filters_sum_over_depth_to_the_2d_filter() -> None:
    weights = _imagenet_like()
    inflated = inflate(weights)
    assert not any(name.startswith(HEAD) for name in inflated)
    assert inflated["conv1.weight"].shape == (64, 1, 7, 7, 7)
    torch.testing.assert_close(
        inflated["conv1.weight"].sum(dim=2), weights["conv1.weight"].sum(dim=1, keepdim=True)
    )
    torch.testing.assert_close(
        inflated["layer1.0.conv1.weight"].sum(dim=2), weights["layer1.0.conv1.weight"]
    )
    torch.testing.assert_close(inflated["bn1.weight"], weights["bn1.weight"])


def test_load_imagenet_fills_every_layer_but_the_head() -> None:
    torch.manual_seed(0)
    model = build_model()
    head = model.state_dict()["fc.weight"].clone()
    load_imagenet(model, _imagenet_like())
    state = model.state_dict()
    torch.testing.assert_close(
        state["layer4.1.bn2.running_var"], _imagenet_like()["layer4.1.bn2.running_var"]
    )
    torch.testing.assert_close(state["fc.weight"], head)
    biases = [v for n, v in state.items() if n.endswith("downsample.0.bias")]
    assert len(biases) == 3
    assert all(torch.count_nonzero(b) == 0 for b in biases)


def test_load_imagenet_rejects_a_missing_layer() -> None:
    weights = _imagenet_like()
    del weights["layer3.0.conv2.weight"]
    with pytest.raises(ValueError, match="layer3.0.conv2.weight"):
        load_imagenet(build_model(), weights)


def test_read_imagenet_rejects_a_file_with_the_wrong_hash(tmp_path: Path) -> None:
    path = tmp_path / "resnet18.pth"
    torch.save(_imagenet_like(), path)
    with pytest.raises(ValueError, match="sha256"):
        read_imagenet(path)


def test_imagenet_init_needs_the_weights_folder() -> None:
    with pytest.raises(ValueError, match="weights"):
        build_model(init="imagenet")


BACKBONES = ["resnet18", "resnet34", "resnet50", "r3d18"]


def _tv(backbone: str) -> None:
    if backbone == "r3d18":
        pytest.importorskip("torchvision")


@pytest.mark.parametrize("backbone", BACKBONES)
def test_every_backbone_maps_a_small_cube_to_14_logits(backbone: str) -> None:
    _tv(backbone)
    model = build_model(backbone).eval()  # type: ignore[arg-type]
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 32, 32, 32))
    assert out.shape == (1, 14)


def test_the_default_backbone_is_resnet18() -> None:
    a, b = build_model(), build_model("resnet18")
    assert list(a.state_dict()) == list(b.state_dict())


@pytest.mark.parametrize("backbone", ["resnet34", "resnet50"])
def test_load_imagenet_fills_every_layer_but_the_head_for_deeper_resnets(backbone: str) -> None:
    torch.manual_seed(0)
    model = build_model(backbone)  # type: ignore[arg-type]
    head = model.state_dict()["fc.weight"].clone()
    weights = _imagenet_like(backbone)
    load_imagenet(model, weights)
    state = model.state_dict()
    torch.testing.assert_close(state["fc.weight"], head)
    torch.testing.assert_close(
        state["layer4.0.bn1.running_var"], weights["layer4.0.bn1.running_var"]
    )
    wanted = "layer4.0.conv3.weight" if backbone == "resnet50" else "layer4.0.conv2.weight"
    assert state[wanted].shape[2] in (1, 3)
    if backbone == "resnet50":  # a 1x1 filter inflates to depth 1
        assert state["layer1.0.conv1.weight"].shape[2:] == (1, 1, 1)
        torch.testing.assert_close(
            state["layer1.0.conv1.weight"][:, :, 0, :, :], weights["layer1.0.conv1.weight"]
        )
    assert all(
        torch.count_nonzero(v) == 0 for n, v in state.items() if n.endswith("downsample.0.bias")
    )


def test_deeper_resnet_load_rejects_a_missing_layer() -> None:
    weights = _imagenet_like("resnet50")
    del weights["layer2.1.conv3.weight"]
    with pytest.raises(ValueError, match="layer2.1.conv3.weight"):
        load_imagenet(build_model("resnet50"), weights)


@pytest.mark.parametrize("backbone", ["resnet18", "resnet34", "resnet50", "r3d18"])
def test_every_pinned_weight_file_has_a_matching_name_and_hash(backbone: str) -> None:
    name, sha = WEIGHTS[backbone]
    assert len(sha) == 64
    assert name.endswith(f"{sha[:8]}.pth")


def test_read_weights_rejects_a_file_with_the_wrong_hash(tmp_path: Path) -> None:
    path = tmp_path / "resnet34.pth"
    torch.save(_imagenet_like("resnet34"), path)
    with pytest.raises(ValueError, match="sha256"):
        read_weights(path, WEIGHTS["resnet34"][1])


def test_read_weights_returns_the_state_when_the_hash_matches(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "w.pth"
    torch.save({"a": torch.ones(2)}, path)
    state = read_weights(path, hashlib.sha256(path.read_bytes()).hexdigest())
    torch.testing.assert_close(state["a"], torch.ones(2))


def test_pretrained_init_needs_the_weights_folder() -> None:
    with pytest.raises(ValueError, match="weights"):
        build_model("resnet34", "imagenet")
    with pytest.raises(ValueError, match="weights"):
        build_model("r3d18", "kinetics")


@pytest.mark.parametrize(
    ("backbone", "init"),
    [("r3d18", "imagenet"), ("resnet18", "kinetics"), ("resnet50", "kinetics")],
)
def test_init_and_backbone_must_agree(backbone: str, init: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs"):
        build_model(backbone, init, tmp_path)  # type: ignore[arg-type]


def test_imagenet_init_loads_the_pinned_file_for_the_backbone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = _imagenet_like("resnet34")
    path = tmp_path / WEIGHTS["resnet34"][0]
    torch.save(weights, path)
    monkeypatch.setitem(resnet3d.WEIGHTS, "resnet34", (path.name, resnet3d._sha256(path)))
    model = build_model("resnet34", "imagenet", tmp_path)
    torch.testing.assert_close(
        model.state_dict()["layer3.2.bn2.running_var"], weights["layer3.2.bn2.running_var"]
    )


class _Tiny(torch.nn.Module):
    """Stand-in for torchvision's r3d_18 names: a 1-channel stem, one block, an fc head."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv3d(1, 4, (3, 7, 7), bias=False), torch.nn.BatchNorm3d(4)
        )
        self.layer1 = torch.nn.Sequential(torch.nn.Conv3d(4, 4, 3, bias=False))
        self.fc = torch.nn.Linear(4, 14)


def _kinetics_like() -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    weights = {k: torch.rand(v.shape, generator=gen) for k, v in _Tiny().state_dict().items()}
    weights["stem.0.weight"] = torch.rand(4, 3, 3, 7, 7, generator=gen)
    weights["fc.weight"] = torch.rand(400, 4, generator=gen)
    weights["fc.bias"] = torch.rand(400, generator=gen)
    return weights


def test_load_kinetics_sums_the_stem_to_one_channel_and_keeps_the_head() -> None:
    torch.manual_seed(0)
    model = _Tiny()
    head = model.state_dict()["fc.weight"].clone()
    weights = _kinetics_like()
    load_kinetics(model, weights)
    state = model.state_dict()
    torch.testing.assert_close(
        state["stem.0.weight"], weights["stem.0.weight"].sum(dim=1, keepdim=True)
    )
    torch.testing.assert_close(state["layer1.0.weight"], weights["layer1.0.weight"])
    torch.testing.assert_close(state["stem.1.running_var"], weights["stem.1.running_var"])
    torch.testing.assert_close(state["fc.weight"], head)


def test_load_kinetics_rejects_a_missing_layer() -> None:
    weights = _kinetics_like()
    del weights["layer1.0.weight"]
    with pytest.raises(ValueError, match="layer1.0.weight"):
        load_kinetics(_Tiny(), weights)


def test_load_kinetics_rejects_an_unexpected_layer() -> None:
    weights = _kinetics_like()
    weights["extra.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="extra.weight"):
        load_kinetics(_Tiny(), weights)


def test_r3d18_builds_with_a_one_channel_stem_and_a_14_way_head() -> None:
    pytest.importorskip("torchvision")
    model = build_model("r3d18")
    assert model.stem[0].weight.shape == (64, 1, 3, 7, 7)
    assert model.fc.weight.shape == (14, 512)
