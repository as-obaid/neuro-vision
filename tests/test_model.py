from pathlib import Path

import pytest
import torch

from neurovision.models.resnet3d import HEAD, build_model, inflate, load_imagenet, read_imagenet


def test_resnet18_maps_a_128_cube_to_14_logits() -> None:
    model = build_model().eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 128, 128, 128))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params}")
    assert out.shape == (1, 14)
    assert 0.9 * 33e6 <= n_params <= 1.1 * 33e6


def _imagenet_like() -> dict[str, torch.Tensor]:
    """Random weights with the names and 2D shapes of torchvision's ResNet-18."""
    gen = torch.Generator().manual_seed(0)
    weights = {}
    for name, t in build_model().state_dict().items():
        if name.endswith("downsample.0.bias"):
            continue  # torchvision's shortcut convs have no bias
        if name.startswith(HEAD):
            shape: tuple[int, ...] = (1000, 512) if name.endswith("weight") else (1000,)
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
        build_model("imagenet")
