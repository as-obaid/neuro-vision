"""Transformer backbones: Video Swin T/S/B (Kinetics-400) and the SwinUNETR encoder (CT SSL)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from neurovision.config import TrainConfig, load_config
from neurovision.models import resnet3d
from neurovision.models.heads import MapModel
from neurovision.models.resnet3d import WEIGHTS, build_model
from neurovision.models.transformers import (
    SwinUNETREncoder,
    VideoSwin,
    load_swinunetr_ssl,
    load_video_swin,
)
from neurovision.train.ema import make_ema
from neurovision.train.loop import param_groups, train_epoch
from neurovision.train.schedule import cosine_with_warmup

pytest.importorskip("torchvision")

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "tune"
COS30 = "t_flip_b4_r3d18_mm160_aug0.1_lr6e-4_aux_cos30.yaml"
SIZE = 32
GRID = (SIZE // 4, SIZE // 8, SIZE // 8)  # r3d_18 layer3 cells
SWINS = ["swin3d_t", "swin3d_s", "swin3d_b"]
STAGE2 = {"swin3d_t": 192, "swin3d_s": 192, "swin3d_b": 256}  # channels after the 2nd stage


def _x(n: int = 2) -> torch.Tensor:
    return torch.randn(n, 1, SIZE, SIZE, SIZE, generator=torch.Generator().manual_seed(0))


# --- shapes --------------------------------------------------------------------------------


@pytest.mark.parametrize("backbone", [*SWINS, "swinunetr"])
def test_transformer_maps_a_cube_to_14_logits(backbone: str) -> None:
    model = build_model(backbone).eval()  # type: ignore[arg-type]
    with torch.no_grad():
        assert model(_x(1)).shape == (1, 14)


@pytest.mark.parametrize("backbone", SWINS)
def test_video_swin_has_a_one_channel_patch_embedding_and_an_fc_head(backbone: str) -> None:
    model = build_model(backbone)  # type: ignore[arg-type]
    assert isinstance(model, VideoSwin)
    assert model.patch_embed.proj.weight.shape[1] == 1
    assert model.fc.out_features == 14
    assert not any(k.startswith(("head.", "aux.")) for k in model.state_dict())


@pytest.mark.parametrize("backbone", [*SWINS, "swinunetr"])
def test_map_head_returns_the_logits_and_a_map_on_the_layer3_grid(backbone: str) -> None:
    torch.manual_seed(0)
    model = build_model(backbone, aux_map=True).eval()  # type: ignore[arg-type]
    assert isinstance(model, MapModel)
    with torch.no_grad():
        logits, map_logits = model.forward_with_map(_x())
        torch.testing.assert_close(model(_x()), logits)
    assert logits.shape == (2, 14)
    assert map_logits.shape == (2, 1, *GRID)


@pytest.mark.parametrize("backbone", SWINS)
def test_video_swin_map_head_reads_the_second_stage(backbone: str) -> None:
    model = build_model(backbone, aux_map=True)  # type: ignore[arg-type]
    assert model.aux[0].in_channels == STAGE2[backbone]  # type: ignore[index, union-attr]


def test_swinunetr_map_head_reads_the_stride_8_features() -> None:
    model = build_model("swinunetr", aux_map=True)
    assert isinstance(model, SwinUNETREncoder)
    assert model.aux[0].in_channels == 192  # type: ignore[index, union-attr]
    assert model.fc.in_features == 768


def test_the_map_needs_the_model_built_with_it() -> None:
    model = build_model("swin3d_t")
    assert isinstance(model, MapModel)
    with pytest.raises(ValueError, match="map head"):
        model.forward_with_map(_x())


def test_map_head_and_fc_train_at_the_head_rate() -> None:
    model = build_model("swin3d_t", aux_map=True)
    head, rest = param_groups(model, 1e-3, 0.1)
    names = {id(p): n for n, p in model.named_parameters()}
    assert {names[id(p)] for p in head["params"]} == {
        n for n in names.values() if n.startswith(("fc.", "aux."))
    }
    assert rest["lr"] == pytest.approx(1e-4)


def test_train_epoch_trains_a_swin_map_head() -> None:
    torch.manual_seed(0)
    model = build_model("swin3d_t", aux_map=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    target = torch.zeros(2, 1, *GRID)
    target[0, 0, 1, 1, 1] = 1
    batch = (_x(), torch.randint(0, 2, (2, 14)).float(), target, torch.ones(2))
    before = model.aux[3].weight.detach().clone()  # type: ignore[index, union-attr]
    cls, map_loss = train_epoch(
        model,
        [batch],
        opt,
        cosine_with_warmup(opt, 0, 10),
        make_ema(model, 0.9),
        1,
        torch.device("cpu"),
        map_weight=1.0,
    )
    assert np.isfinite(cls) and map_loss is not None and np.isfinite(map_loss)
    assert not torch.equal(before, model.aux[3].weight)  # type: ignore[index, union-attr]


# --- Kinetics weights for Video Swin -------------------------------------------------------


def _kinetics_like(backbone: str) -> dict[str, torch.Tensor]:
    """Random weights with the names and shapes of torchvision's Kinetics-400 Video Swin."""
    from torchvision.models import video

    gen = torch.Generator().manual_seed(1)
    state = getattr(video, backbone)(weights=None).state_dict()
    return {
        k: torch.rand(v.shape, generator=gen) if v.is_floating_point() else v.clone()
        for k, v in state.items()
    }


def test_load_video_swin_sums_the_patch_embedding_and_keeps_the_new_heads() -> None:
    torch.manual_seed(0)
    model = build_model("swin3d_t", aux_map=True)
    before = {k: v.clone() for k, v in model.state_dict().items() if k.startswith(("fc.", "aux."))}
    weights = _kinetics_like("swin3d_t")
    assert weights["patch_embed.proj.weight"].shape[1] == 3 and "head.weight" in weights
    load_video_swin(model, weights)
    state = model.state_dict()
    torch.testing.assert_close(
        state["patch_embed.proj.weight"], weights["patch_embed.proj.weight"].sum(1, keepdim=True)
    )
    key = "features.4.1.attn.qkv.weight"
    torch.testing.assert_close(state[key], weights[key])
    torch.testing.assert_close(state["norm.weight"], weights["norm.weight"])
    for k, v in before.items():
        torch.testing.assert_close(state[k], v)


def test_load_video_swin_rejects_a_missing_or_unexpected_layer() -> None:
    weights = _kinetics_like("swin3d_t")
    del weights["features.2.0.mlp.0.weight"]
    with pytest.raises(ValueError, match="features.2.0.mlp.0.weight"):
        load_video_swin(build_model("swin3d_t"), weights)
    weights = _kinetics_like("swin3d_t") | {"extra.weight": torch.zeros(1)}
    with pytest.raises(ValueError, match="extra.weight"):
        load_video_swin(build_model("swin3d_t"), weights)


# --- self-supervised CT weights for SwinUNETR ----------------------------------------------


def _ssl_like() -> dict[str, object]:
    """A checkpoint shaped like MONAI's model_swinvit.pt: 'module.' names, fc1/fc2 MLPs, the
    final norm and the three pretext heads next to the encoder."""
    gen = torch.Generator().manual_seed(2)
    model = build_model("swinunetr")
    assert isinstance(model, SwinUNETREncoder)
    state: dict[str, torch.Tensor] = {}
    for k, v in model.swinViT.state_dict().items():
        name = k.replace(".mlp.linear1.", ".mlp.fc1.").replace(".mlp.linear2.", ".mlp.fc2.")
        state[f"module.{name}"] = torch.rand(v.shape, generator=gen) if v.is_floating_point() else v
    for extra in ("norm", "rotation_head", "contrastive_head", "convTrans3d"):
        state[f"module.{extra}.weight"] = torch.rand(4, generator=gen)
    return {"epoch": 1, "state_dict": state, "optimizer": {}}


def test_load_swinunetr_ssl_renames_the_mlp_and_drops_the_pretext_heads() -> None:
    torch.manual_seed(0)
    model = build_model("swinunetr", aux_map=True)
    fc = model.state_dict()["fc.weight"].clone()
    ckpt = _ssl_like()
    load_swinunetr_ssl(model, ckpt)
    state, src = model.state_dict(), ckpt["state_dict"]
    assert isinstance(src, dict)
    torch.testing.assert_close(
        state["swinViT.layers2.0.blocks.1.mlp.linear1.weight"],
        src["module.layers2.0.blocks.1.mlp.fc1.weight"],
    )
    torch.testing.assert_close(
        state["swinViT.patch_embed.proj.weight"], src["module.patch_embed.proj.weight"]
    )
    torch.testing.assert_close(state["fc.weight"], fc)


def test_load_swinunetr_ssl_rejects_a_missing_layer() -> None:
    ckpt = _ssl_like()
    state = ckpt["state_dict"]
    assert isinstance(state, dict)
    del state["module.layers3.0.blocks.0.attn.qkv.weight"]
    with pytest.raises(ValueError, match="layers3.0.blocks.0.attn.qkv.weight"):
        load_swinunetr_ssl(build_model("swinunetr"), ckpt)


# --- build_model: pinned files and init rules ----------------------------------------------


@pytest.mark.parametrize("backbone", [*SWINS, "swinunetr"])
def test_pinned_transformer_weights_name_their_hash(backbone: str) -> None:
    name, sha = WEIGHTS[backbone]
    assert len(sha) == 64
    assert name.rsplit(".", 1)[0].endswith(sha[:8])


def test_kinetics_init_loads_the_pinned_swin_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = _kinetics_like("swin3d_t")
    path = tmp_path / "swin.pth"
    torch.save(weights, path)
    monkeypatch.setitem(resnet3d.WEIGHTS, "swin3d_t", (path.name, resnet3d._sha256(path)))
    model = build_model("swin3d_t", "kinetics", tmp_path, aux_map=True)
    key = "features.6.1.mlp.3.weight"
    torch.testing.assert_close(model.state_dict()[key], weights[key])


def test_ssl_init_loads_the_pinned_swinunetr_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt = _ssl_like()
    path = tmp_path / "ssl.pt"
    torch.save(ckpt, path)
    monkeypatch.setitem(resnet3d.WEIGHTS, "swinunetr", (path.name, resnet3d._sha256(path)))
    model = build_model("swinunetr", "ssl", tmp_path)
    src = ckpt["state_dict"]
    assert isinstance(src, dict)
    torch.testing.assert_close(
        model.state_dict()["swinViT.layers4.0.blocks.0.norm1.weight"],
        src["module.layers4.0.blocks.0.norm1.weight"],
    )


@pytest.mark.parametrize(
    ("backbone", "init"),
    [("swin3d_t", "imagenet"), ("swin3d_t", "ssl"), ("swinunetr", "kinetics"), ("r3d18", "ssl")],
)
def test_transformer_init_must_fit_the_backbone(backbone: str, init: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs"):
        build_model(backbone, init, tmp_path)  # type: ignore[arg-type]


# --- configs -------------------------------------------------------------------------------


def _train(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "epochs": 10,
        "batch_size": 4,
        "accum_steps": 2,
        "lr": 6e-4,
        "weight_decay": 0.01,
        "warmup_epochs": 1,
        "ema_decay": 0.999,
        "loss_weights": "uniform",
        "num_workers": 0,
        "resample": "mm",
        "aux_map_weight": 1.0,
    }
    return base | overrides


def test_config_accepts_the_map_head_on_every_transformer() -> None:
    for backbone, init in [*((s, "kinetics") for s in SWINS), ("swinunetr", "ssl")]:
        cfg = TrainConfig(**_train(backbone=backbone, init=init))  # type: ignore[arg-type]
        assert (cfg.backbone, cfg.init) == (backbone, init)


@pytest.mark.parametrize(
    ("backbone", "init"), [("swin3d_s", "ssl"), ("swinunetr", "kinetics"), ("swin3d_b", "imagenet")]
)
def test_config_rejects_an_init_the_backbone_has_no_weights_for(backbone: str, init: str) -> None:
    with pytest.raises(ValidationError, match="needs"):
        TrainConfig(**_train(backbone=backbone, init=init))  # type: ignore[arg-type]


@pytest.mark.parametrize("backbone", SWINS)
def test_video_swin_configs_change_only_the_backbone_of_2a_cos30(backbone: str) -> None:
    parent = load_config(CONFIGS / COS30)
    child = load_config(CONFIGS / f"t_aux_cos30_{backbone}.yaml")
    assert parent.train is not None and child.train is not None
    assert child.train == parent.train.model_copy(update={"backbone": backbone})
    assert child.augment == parent.augment and child.seed == parent.seed


def test_swinunetr_config_changes_only_the_backbone_and_its_weights() -> None:
    parent = load_config(CONFIGS / COS30)
    child = load_config(CONFIGS / "t_aux_cos30_swinunetr.yaml")
    assert parent.train is not None and child.train is not None
    assert child.train == parent.train.model_copy(update={"backbone": "swinunetr", "init": "ssl"})
    assert child.augment == parent.augment and child.seed == parent.seed
