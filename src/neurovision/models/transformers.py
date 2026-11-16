"""Transformer backbones with one input channel and a 14-way head.

- Video Swin T, S and B (torchvision, Kinetics-400 weights). The patch embedding keeps axis 0 at
  half size and every patch merge halves only axes 1 and 2.
- The SwinUNETR encoder (MONAI, self-supervised on 5,050 CT scans; Tang et al., CVPR 2022).

The location-map head reads the first feature map with r3d_18 layer3's in-plane stride of 8,
resampled to layer3's grid: Video Swin after its 2nd stage (axis 0 at stride 2, averaged in
pairs), SwinUNETR at stride 8 (axis 0 interpolated up). Both train on the same map target.
"""

from collections.abc import Mapping
from typing import Literal

import torch
from torch import nn

from neurovision.models.heads import N_CLASSES, MapModel, map_head

VideoSwinName = Literal["swin3d_t", "swin3d_s", "swin3d_b"]
VIDEO_SWINS: tuple[VideoSwinName, ...] = ("swin3d_t", "swin3d_s", "swin3d_b")
SWINUNETR_FEATURES = 48  # the self-supervised checkpoint's feature size
NEW_KEYS = ("fc.", "aux.")  # random at the start, never in a checkpoint
_MAP_STAGE = 3  # Video Swin features[:3]: patch stage 1, merge, stage 2
_SWINUNETR_MAP_LEVEL = 2  # SwinUNETR hidden state at stride 8


class VideoSwin(MapModel):
    def __init__(self, name: VideoSwinName, aux_map: bool = False) -> None:
        super().__init__()
        from torchvision.models import video

        base = getattr(video, name)(weights=None)
        old = base.patch_embed.proj
        base.patch_embed.proj = nn.Conv3d(1, old.out_channels, old.kernel_size, old.stride)
        self.patch_embed = base.patch_embed
        self.pos_drop = base.pos_drop
        self.features = base.features
        self.norm = base.norm
        self.fc: nn.Module = nn.Linear(base.num_features, N_CLASSES)
        self.aux = map_head(2 * old.out_channels) if aux_map else None

    def _features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pos_drop(self.patch_embed(x))  # channels last from here: (n, d, h, w, c)
        mid = self.features[:_MAP_STAGE](x)
        out = self.norm(self.features[_MAP_STAGE:](mid)).mean(dim=(1, 2, 3))
        return self.fc(out), mid.permute(0, 4, 1, 2, 3)


class SwinUNETREncoder(MapModel):
    def __init__(self, aux_map: bool = False) -> None:
        super().__init__()
        from monai.networks.nets import SwinUNETR

        self.swinViT = SwinUNETR(
            in_channels=1, out_channels=1, feature_size=SWINUNETR_FEATURES
        ).swinViT
        self.fc: nn.Module = nn.Linear(16 * SWINUNETR_FEATURES, N_CLASSES)
        self.aux = map_head(4 * SWINUNETR_FEATURES) if aux_map else None

    def _features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.swinViT(x, normalize=True)  # strides 2, 4, 8, 16, 32; channels first
        return self.fc(hidden[-1].mean(dim=(2, 3, 4))), hidden[_SWINUNETR_MAP_LEVEL]


def _load(model: nn.Module, state: Mapping[str, torch.Tensor], source: str) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    wrong = {k for k in missing if not k.startswith(NEW_KEYS)} | set(unexpected)
    if wrong:
        raise ValueError(f"{source} weights do not fit the model: {sorted(wrong)}")


def load_video_swin(model: nn.Module, weights: Mapping[str, torch.Tensor]) -> None:
    """Every layer but the Kinetics head; the patch embedding's colour channels sum to one."""
    state = {k: v for k, v in weights.items() if not k.startswith("head.")}
    state["patch_embed.proj.weight"] = state["patch_embed.proj.weight"].sum(dim=1, keepdim=True)
    _load(model, state, "Video Swin Kinetics")


SSL_DROPPED = ("norm.", "rotation_head.", "contrastive_head.", "convTrans3d.")  # pretext parts


def load_swinunetr_ssl(model: nn.Module, checkpoint: Mapping[str, object]) -> None:
    """The encoder from MONAI's self-supervised `model_swinvit.pt`; its MLPs are named fc1/fc2."""
    source = checkpoint["state_dict"]
    assert isinstance(source, Mapping)
    state = {}
    for key, value in source.items():
        name = key.removeprefix("module.")
        if name.startswith(SSL_DROPPED):
            continue
        name = name.replace(".mlp.fc1.", ".mlp.linear1.").replace(".mlp.fc2.", ".mlp.linear2.")
        state[f"swinViT.{name}"] = value
    _load(model, state, "SwinUNETR self-supervised")
