from monai.networks.nets import resnet18
from torch import nn


def build_model() -> nn.Module:
    return resnet18(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=14)
