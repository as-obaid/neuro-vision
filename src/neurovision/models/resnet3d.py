import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch
from monai.networks.nets import resnet18, resnet34, resnet50
from torch import nn

HEAD = "fc."  # the output layer, never pretrained
# torchvision ResNet18_Weights.IMAGENET1K_V1; source in data/manifests/weights.yaml
IMAGENET_FILE = "resnet18-f37072fd.pth"
IMAGENET_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


# name -> (file in the store's weights folder, pinned sha256); source in data/manifests/weights.yaml
WEIGHTS = {
    "resnet18": (IMAGENET_FILE, IMAGENET_SHA256),
    "resnet34": (
        "resnet34-b627a593.pth",
        "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f",
    ),
    "resnet50": (
        "resnet50-0676ba61.pth",
        "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a",
    ),
    # torchvision r3d_18 KINETICS400_V1
    "r3d18": (
        "r3d_18-b3b3357e.pth",
        "b3b3357ead25631ec9c57362ff2128a92d0427e01e2cd184951a44380c3f2e9d",
    ),
}
Backbone = Literal["resnet18", "resnet34", "resnet50", "r3d18"]
Init = Literal["random", "imagenet", "kinetics"]
_MONAI = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}
N_CLASSES = 14


def build_model(
    backbone: Backbone = "resnet18", init: Init = "random", weights_dir: Path | None = None
) -> nn.Module:
    if init == "imagenet" and backbone not in _MONAI:
        raise ValueError(f"imagenet init needs a resnet backbone, not {backbone}")
    if init == "kinetics" and backbone != "r3d18":
        raise ValueError(f"kinetics init needs the r3d18 backbone, not {backbone}")
    if init != "random" and weights_dir is None:
        raise ValueError(f"{init} init needs the weights folder")
    model = _r3d18() if backbone == "r3d18" else _resnet(backbone)
    if init != "random":
        assert weights_dir is not None
        name, sha = WEIGHTS[backbone]
        state = read_weights(weights_dir / name, sha)
        (load_kinetics if init == "kinetics" else load_imagenet)(model, state)
    return model


def _resnet(backbone: str) -> nn.Module:
    return _MONAI[backbone](
        pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=N_CLASSES
    )


def _r3d18() -> nn.Module:
    """torchvision's video ResNet-18 with its native stem, a 1-channel input and a 14-way head."""
    from torchvision.models.video import r3d_18

    model = r3d_18(weights=None)
    old = model.stem[0]
    model.stem[0] = nn.Conv3d(
        1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False
    )
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    return model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_weights(path: Path, sha256: str) -> dict[str, torch.Tensor]:
    if _sha256(path) != sha256:
        raise ValueError(f"{path.name}: sha256 does not match the pinned value")
    weights: dict[str, torch.Tensor] = torch.load(path, map_location="cpu", weights_only=True)
    return weights


def read_imagenet(path: Path) -> dict[str, torch.Tensor]:
    return read_weights(path, IMAGENET_SHA256)


def load_kinetics(model: nn.Module, weights: Mapping[str, torch.Tensor]) -> None:
    """Every layer but the head from Kinetics-400; the stem's three colour channels sum to one."""
    state = {k: v for k, v in weights.items() if not k.startswith(HEAD)}
    state["stem.0.weight"] = state["stem.0.weight"].sum(dim=1, keepdim=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    wrong = (set(missing) - {f"{HEAD}weight", f"{HEAD}bias"}) | set(unexpected)
    if wrong:
        raise ValueError(f"Kinetics weights do not fit the model: {sorted(wrong)}")


def inflate(weights_2d: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """torchvision's 2D ResNet weights as MONAI 3D ResNet weights; the layer names match.

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
