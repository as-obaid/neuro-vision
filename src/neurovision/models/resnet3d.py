import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch
from monai.networks.nets import resnet18
from torch import nn

HEAD = "fc."  # the output layer, never pretrained
# torchvision ResNet18_Weights.IMAGENET1K_V1; source in data/manifests/weights.yaml
IMAGENET_FILE = "resnet18-f37072fd.pth"
IMAGENET_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


def build_model(
    init: Literal["random", "imagenet"] = "random", weights_dir: Path | None = None
) -> nn.Module:
    model = resnet18(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=14)
    if init == "imagenet":
        if weights_dir is None:
            raise ValueError("imagenet init needs the weights folder")
        load_imagenet(model, read_imagenet(weights_dir / IMAGENET_FILE))
    return model


def read_imagenet(path: Path) -> dict[str, torch.Tensor]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != IMAGENET_SHA256:
        raise ValueError(f"{path.name}: sha256 does not match the pinned value")
    weights: dict[str, torch.Tensor] = torch.load(path, map_location="cpu", weights_only=True)
    return weights


def inflate(weights_2d: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """torchvision's 2D ResNet-18 weights as MONAI 3D ResNet-18 weights; the layer names match.

    A k x k filter is copied across a depth of k and divided by k, so a volume that is constant in
    depth gives the 2D response. The stem's three colour channels sum to one grey channel.
    """
    inflated = {}
    for name, w in weights_2d.items():
        if name.startswith(HEAD):
            continue
        if w.ndim == 4:
            if name == "conv1.weight":
                w = w.sum(dim=1, keepdim=True)
            depth = w.shape[-1]
            w = w.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth
        inflated[name] = w
    return inflated


def load_imagenet(model: nn.Module, weights_2d: Mapping[str, torch.Tensor]) -> None:
    """Every layer but the head from ImageNet; the shortcut-conv biases torchvision lacks are 0."""
    biases = {name for name in model.state_dict() if name.endswith("downsample.0.bias")}
    missing, unexpected = model.load_state_dict(inflate(weights_2d), strict=False)
    wrong = (set(missing) - biases - {f"{HEAD}weight", f"{HEAD}bias"}) | set(unexpected)
    if wrong:
        raise ValueError(f"ImageNet weights do not fit the model: {sorted(wrong)}")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in biases:
                p.zero_()
