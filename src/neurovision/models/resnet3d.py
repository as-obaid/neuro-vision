import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import torch
from monai.networks.nets import resnet18, resnet34, resnet50
from torch import nn

from neurovision.models.heads import MAP_CHANNELS, N_CLASSES, MapModel, map_head
from neurovision.models.transformers import (
    VIDEO_SWINS,
    SwinUNETREncoder,
    VideoSwin,
    load_swinunetr_ssl,
    load_video_swin,
)

__all__ = ["MAP_CHANNELS", "N_CLASSES", "MapModel"]

HEAD = "fc."  # the output layer, never pretrained
AUX = "aux."  # the location-map head, never pretrained
NEW_LAYERS = (HEAD, AUX)  # random at the start, trained at the head rate
LAYER3_CHANNELS = 256  # r3d_18
VESSEL_ENCODER = "encoder.pt"  # in a vessel pretraining run's fold folder
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
    # torchvision Swin3D_{T,S,B}_Weights.KINETICS400_V1 (ImageNet-1k, then Kinetics-400)
    "swin3d_t": (
        "swin3d_t-7615ae03.pth",
        "7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130",
    ),
    "swin3d_s": (
        "swin3d_s-da41c237.pth",
        "da41c237cb1bca0fd4b51a4faa6ab0cf786a2ef6cbb6f6609b53bd5d23afc0f9",
    ),
    "swin3d_b": (
        "swin3d_b_1k-24f7c7c6.pth",
        "24f7c7c6fd97dd52439d72c2208e0785081811389185a4781405b482404e4b9b",
    ),
    # MONAI's self-supervised SwinUNETR encoder (model_swinvit.pt), renamed with its hash
    "swinunetr": (
        "model_swinvit-1cf19eca.pt",
        "1cf19eca3ed0472daf33dedd2f095b118db16edac0fc53d9e878f328ca68c988",
    ),
}
# r3d18_unet: the full version, the step-1 U-Net with a per-artery readout (models.vessel_readout)
Backbone = Literal[
    "resnet18",
    "resnet34",
    "resnet50",
    "r3d18",
    "swin3d_t",
    "swin3d_s",
    "swin3d_b",
    "swinunetr",
    "r3d18_unet",
]
# "ssl": self-supervised CT weights (swinunetr only); "vessel": our own vessel pretraining, read
# from its fold folder `weights_dir`: the encoder (`encoder.pt`) for r3d18, the whole U-Net
# (`last.pt`, once `meta.json` marks the run finished) for r3d18_unet
Init = Literal["random", "imagenet", "kinetics", "ssl", "vessel"]
VESSEL_UNET = "last.pt"
VESSEL_DONE = "meta.json"
_MONAI = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}
KINETICS_BACKBONES = ("r3d18", *VIDEO_SWINS)
MAP_BACKBONES = (*KINETICS_BACKBONES, "swinunetr")  # every backbone but the MONAI ResNets
Head = Literal["linear", "mlp"]
MLP_HIDDEN = 256
MLP_DROPOUT = 0.1


def build_model(
    backbone: Backbone = "resnet18",
    init: Init = "random",
    weights_dir: Path | None = None,
    head: Head = "linear",
    aux_map: bool = False,
) -> nn.Module:
    if aux_map and backbone not in MAP_BACKBONES:
        raise ValueError(f"the location-map head needs r3d18 or a transformer, not {backbone}")
    if backbone == "r3d18_unet":
        return _vessel_readout(init, weights_dir, head)
    if init == "imagenet" and backbone not in _MONAI:
        raise ValueError(f"imagenet init needs a resnet backbone, not {backbone}")
    if init == "kinetics" and backbone not in KINETICS_BACKBONES:
        raise ValueError(f"kinetics init needs r3d18 or a Video Swin backbone, not {backbone}")
    if init == "ssl" and backbone != "swinunetr":
        raise ValueError(f"ssl init needs the swinunetr backbone, not {backbone}")
    if init == "vessel" and backbone != "r3d18":
        raise ValueError(f"vessel init needs the r3d18 backbone, not {backbone}")
    if init != "random" and weights_dir is None:
        raise ValueError(f"{init} init needs the weights folder")
    model = _backbone(backbone, aux_map)
    if head == "mlp":  # random like the linear head; the weight loaders skip every HEAD key
        fc = model.fc
        assert isinstance(fc, nn.Linear)
        model.fc = _mlp_head(fc.in_features)
    if aux_map and backbone == "r3d18":
        model = R3DWithMap(model)
    if init == "vessel":
        assert weights_dir is not None
        from neurovision.models.unet import load_encoder

        encoder = torch.load(weights_dir / VESSEL_ENCODER, map_location="cpu", weights_only=True)
        load_encoder(model, encoder)
    elif init != "random":
        assert weights_dir is not None
        name, sha = WEIGHTS[backbone]
        state = read_weights(weights_dir / name, sha)
        if backbone == "swinunetr":
            load_swinunetr_ssl(model, state)
        elif backbone in VIDEO_SWINS:
            load_video_swin(model, state)
        else:
            (load_kinetics if init == "kinetics" else load_imagenet)(model, state)
    return model


def _vessel_readout(init: Init, weights_dir: Path | None, head: Head) -> nn.Module:
    """The full version's model, random or from a finished vessel pretraining run's U-Net."""
    from neurovision.models.vessel_readout import VesselReadout

    if head != "linear":
        raise ValueError("r3d18_unet has its own readout: head must be linear")
    if init not in ("random", "vessel"):
        raise ValueError(f"r3d18_unet starts random or from vessel pretraining, not {init}")
    model = VesselReadout()
    if init == "vessel":
        if weights_dir is None:
            raise ValueError("vessel init needs the pretraining run's fold folder")
        if not (weights_dir / VESSEL_DONE).exists():
            raise ValueError(f"vessel pretraining in {weights_dir} is not finished (no meta.json)")
        state = torch.load(weights_dir / VESSEL_UNET, map_location="cpu", weights_only=True)
        model.unet.load_state_dict(state["model"])
    return model


def _backbone(backbone: Backbone, aux_map: bool) -> nn.Module:
    """The bare network; a transformer builds its own map head, r3d18's comes after the fc."""
    if backbone == "r3d18":
        return _r3d18()
    if backbone == "swinunetr":
        return SwinUNETREncoder(aux_map)
    if backbone in VIDEO_SWINS:
        return VideoSwin(backbone, aux_map)
    return _resnet(backbone)


def _mlp_head(features: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(features, MLP_HIDDEN),
        nn.LayerNorm(MLP_HIDDEN),
        nn.ReLU(inplace=True),
        nn.Dropout(MLP_DROPOUT),
        nn.Linear(MLP_HIDDEN, N_CLASSES),
    )


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


class R3DWithMap(MapModel):
    """r3d_18 plus a location-map head on layer3; layer names match r3d_18's state dict."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.stem = cast(nn.Module, base.stem)
        self.layer1 = cast(nn.Module, base.layer1)
        self.layer2 = cast(nn.Module, base.layer2)
        self.layer3 = cast(nn.Module, base.layer3)
        self.layer4 = cast(nn.Module, base.layer4)
        self.avgpool = cast(nn.Module, base.avgpool)
        self.fc = cast(nn.Module, base.fc)
        self.aux = map_head(LAYER3_CHANNELS)

    def _features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mid = self.layer3(self.layer2(self.layer1(self.stem(x))))
        logits = self.fc(torch.flatten(self.avgpool(self.layer4(mid)), 1))
        return logits, mid


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
    wrong = {k for k in missing if not k.startswith(NEW_LAYERS)} | set(unexpected)
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
    wrong = {k for k in missing if k not in biases and not k.startswith(HEAD)} | set(unexpected)
    if wrong:
        raise ValueError(f"ImageNet weights do not fit the model: {sorted(wrong)}")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in biases:
                p.zero_()
