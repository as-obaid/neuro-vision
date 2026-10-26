import torch

from neurovision.models.resnet3d import build_model


def test_resnet18_maps_a_128_cube_to_14_logits() -> None:
    model = build_model().eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 128, 128, 128))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params}")
    assert out.shape == (1, 14)
    assert 0.9 * 33e6 <= n_params <= 1.1 * 33e6
