"""Full version: the step-1 U-Net with a per-artery readout, its loss, loop and fit."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from neurovision.config import load_config
from neurovision.data.augment import build_augment
from neurovision.data.dataset import CacheDataset
from neurovision.data.labels import LOCATIONS, LR_SWAP, TARGETS
from neurovision.data.vessels import N_SEG_CLASSES, SEG_LR_SWAP, SegCacheDataset
from neurovision.models.resnet3d import build_model
from neurovision.models.unet import R3DUNet
from neurovision.models.vessel_readout import Readout, VesselReadout, masked_average
from neurovision.train.ema import make_ema
from neurovision.train.loop import fit, param_groups, train_epoch
from neurovision.train.losses import seg_loss
from neurovision.train.schedule import cosine_with_warmup
from tests.cnn_fakes import EXCLUDED, FOLDS, UIDS, cnn_labels, write_config, write_store

pytest.importorskip("torchvision")
SIZE = 32
K = len(LOCATIONS)


def test_masked_average_weighs_each_voxel_by_the_mask() -> None:
    w = torch.rand(2, 3, 4, 5, 6)
    f = torch.randn(2, 7, 4, 5, 6)
    got = masked_average(w, f)
    want = torch.einsum("nkdhw,ncdhw->nkc", w, f) / w.sum((2, 3, 4))[..., None]
    assert got.shape == (2, 3, 7)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def _probs(mask_k: int, region: tuple[slice, ...], n: int = 1) -> torch.Tensor:
    """Background everywhere except artery `mask_k` inside `region`."""
    probs = torch.zeros(n, N_SEG_CLASSES, 8, 8, 8)
    probs[:, 0] = 1
    probs[(slice(None), 0, *region)] = 0
    probs[(slice(None), 1 + mask_k, *region)] = 1
    return probs


def test_an_artery_logit_reads_only_inside_that_artery() -> None:
    torch.manual_seed(0)
    readout = Readout(channels=4, bottom=6).eval()
    region = (slice(0, 4), slice(0, 4), slice(0, 4))
    probs = _probs(5, region)
    features = torch.randn(1, 4, 8, 8, 8)
    bottom = torch.randn(1, 6, 2, 2, 2)
    changed = features.clone()
    changed[:, :, 4:, 4:, 4:] += 10.0  # outside artery 5's mask
    a, b = readout(probs, features, bottom), readout(probs, changed, bottom)
    assert a.shape == (1, 14)
    torch.testing.assert_close(a[:, 5], b[:, 5])
    inside = features.clone()
    inside[:, :, :2, :2, :2] += 10.0
    assert not torch.allclose(readout(probs, inside, bottom)[:, 5], a[:, 5])


def test_an_aneurysm_on_an_artery_raises_its_evidence() -> None:
    torch.manual_seed(0)
    readout = Readout(channels=4, bottom=6).eval()
    with torch.no_grad():
        readout.artery_weight.zero_()
        readout.artery_weight[:, -1] = 1.0  # only the aneurysm score
        readout.artery_bias.zero_()
    probs = _probs(2, (slice(0, 4),) * 3)
    probs[0, 1 + 2, 1, 1, 1], probs[0, 1 + K + 2, 1, 1, 1] = 0.2, 0.8
    logits = readout(probs, torch.zeros(1, 4, 8, 8, 8), torch.zeros(1, 6, 2, 2, 2))
    assert logits[0, 2].item() == pytest.approx(0.8)
    assert logits[0, 3].item() == pytest.approx(0.0)


def test_vessel_readout_returns_14_logits_and_27_class_maps() -> None:
    torch.manual_seed(0)
    model = VesselReadout().eval()
    x = torch.randn(2, 1, SIZE, SIZE, SIZE)
    with torch.no_grad():
        logits, seg = model.forward_with_seg(x)
        torch.testing.assert_close(model(x), logits)
    assert logits.shape == (2, 14) and seg.shape == (2, N_SEG_CLASSES, SIZE, SIZE, SIZE)


def test_the_readout_is_the_new_layer_and_the_unet_trains_at_the_backbone_rate() -> None:
    model = build_model("r3d18_unet")
    names = [n for n, _ in model.named_parameters()]
    assert all(n.startswith(("unet.", "fc.")) for n in names)
    assert any(n.startswith("unet.seg.") for n in names) and any(n.startswith("fc.") for n in names)
    head, rest = param_groups(model, 1e-3, 0.1)
    ids = {id(p): n for n, p in model.named_parameters()}
    assert {ids[id(p)] for p in head["params"]} == {n for n in names if n.startswith("fc.")}
    assert rest["lr"] == pytest.approx(1e-4)


def _pretrained(tmp_path: Path, finished: bool = True) -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    state = R3DUNet().state_dict()
    torch.save({"model": state, "epoch": 11}, tmp_path / "last.pt")
    if finished:
        (tmp_path / "meta.json").write_text(json.dumps({"epochs": 12}))
    return state


def test_vessel_init_loads_the_whole_step1_unet(tmp_path: Path) -> None:
    state = _pretrained(tmp_path)
    model = build_model("r3d18_unet", "vessel", tmp_path)
    assert isinstance(model, VesselReadout)
    for key in ("layer4.1.conv2.0.weight", "up1.0.weight", "seg.weight"):
        torch.testing.assert_close(model.unet.state_dict()[key], state[key])


def test_vessel_init_refuses_an_unfinished_pretraining_run(tmp_path: Path) -> None:
    _pretrained(tmp_path, finished=False)
    with pytest.raises(ValueError, match="not finished"):
        build_model("r3d18_unet", "vessel", tmp_path)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"aux_map": True}, "map head"),
        ({"head": "mlp"}, "linear"),
        ({"init": "kinetics"}, "kinetics"),
    ],
)
def test_vessel_readout_rejects_options_it_does_not_have(
    tmp_path: Path, kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_model("r3d18_unet", weights_dir=tmp_path, **kwargs)  # type: ignore[arg-type]


def test_full_config_changes_only_what_the_full_version_needs() -> None:
    tune = Path(__file__).resolve().parents[1] / "configs" / "tune"
    parent = load_config(tune / "t_flip_b4_r3d18_mm160_aug0.1_lr6e-4_aux_cos30.yaml")
    full = load_config(tune / "t_full_cos30.yaml")
    assert parent.train is not None and full.train is not None
    update = {
        "backbone": "r3d18_unet",
        "init": "vessel",
        "vessel_run": "vessel_r3d18_160",
        "resample": "vbox",
        "input_size": 192,
        "aux_map_weight": None,
        "seg_weight": 1.0,
    }
    assert full.train == parent.train.model_copy(update=update)
    assert full.augment == parent.augment and (full.train.epochs, full.train.schedule_epochs) == (
        10,
        30,
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"backbone": "r3d18"},
        {"resample": "mm"},
        {"aux_map_weight": 1.0},
        {"head": "mlp"},
    ],
)
def test_full_config_rejects_what_does_not_fit(bad: dict[str, object]) -> None:
    from pydantic import ValidationError

    from neurovision.config import TrainConfig

    tune = Path(__file__).resolve().parents[1] / "configs" / "tune"
    full = load_config(tune / "t_full_cos30.yaml").train
    assert full is not None
    with pytest.raises(ValidationError):
        TrainConfig(**{**full.model_dump(), **bad})


def test_train_refuses_a_vessel_box_cache_from_another_fold() -> None:
    from tests.script_loader import load_script

    check = load_script("train")._check_box_fold
    index = pd.DataFrame({"box_fold": [0, 0, 1]}, index=pd.Index(["a", "b", "c"]))
    check(index, {"a", "b"}, 0, Path("cache/vbox05-192"))
    with pytest.raises(SystemExit, match="fold"):
        check(index, {"a", "b"}, 1, Path("cache/vbox05-192"))
    with pytest.raises(SystemExit, match="fold"):
        check(index, {"a", "c"}, 0, Path("cache/vbox05-192"))
    with pytest.raises(SystemExit, match="box_fold"):
        check(index.drop(columns="box_fold"), {"a"}, 0, Path("cache/vbox05-192"))


# --- loss ----------------------------------------------------------------------------------


def test_seg_loss_counts_only_series_with_a_mask() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, N_SEG_CLASSES, 6, 6, 6)
    target = torch.randint(0, N_SEG_CLASSES, (2, 1, 6, 6, 6))
    both = seg_loss(logits, target, torch.tensor([1.0, 0.0]))
    first = seg_loss(logits[:1], target[:1], torch.tensor([1.0]))
    torch.testing.assert_close(both, first)
    none = seg_loss(logits.requires_grad_(), target, torch.zeros(2))
    assert none.item() == 0.0 and none.requires_grad


# --- dataset -------------------------------------------------------------------------------


def _store(tmp_path: Path) -> tuple[Path, Path]:
    cache, seg_dir = tmp_path / "cache", tmp_path / "seg"
    cache.mkdir()
    seg_dir.mkdir()
    rng = np.random.default_rng(0)
    for uid in ("a", "b"):
        np.save(cache / f"{uid}.npy", rng.normal(size=(SIZE,) * 3).astype(np.float16))
    seg = np.zeros((SIZE,) * 3, np.uint8)
    seg[2, 5, 6] = 1  # artery 0 (a left one); axis 0 is left-right
    seg[3, 5, 6] = 14  # its aneurysm
    np.savez_compressed(seg_dir / "a.npz", seg=seg)
    return cache, seg_dir


def _targets(n: int) -> np.ndarray:
    t = np.zeros((n, 14), np.float32)
    t[:, 0] = 1  # Left infraclinoid ICA
    return t


def test_seg_items_carry_the_class_map_and_keep(tmp_path: Path) -> None:
    cache, seg_dir = _store(tmp_path)
    ds = SegCacheDataset(cache, ["a", "b"], _targets(2), None, 0.0, seg_dir)
    x, y, m, keep = ds[0]
    assert x.shape == (1, SIZE, SIZE, SIZE) and y.shape == (14,)
    assert m.shape == (1, SIZE, SIZE, SIZE) and m.dtype == torch.int64 and keep.item() == 1.0
    assert m[0, 2, 5, 6] == 1 and m[0, 3, 5, 6] == 14
    _, _, m_b, keep_b = ds[1]
    assert m_b.sum() == 0 and keep_b.item() == 0.0  # no mask file: adds no seg loss


def test_seg_flip_mirrors_the_map_and_swaps_classes_and_labels(tmp_path: Path) -> None:
    cache, seg_dir = _store(tmp_path)
    x, y, m, _ = SegCacheDataset(cache, ["a"], _targets(1), None, 1.0, seg_dir)[0]
    stored = np.load(cache / "a.npy").astype(np.float32)
    np.testing.assert_array_equal(x[0].numpy(), stored[::-1])
    assert m[0, SIZE - 3, 5, 6] == SEG_LR_SWAP[1] and m[0, SIZE - 4, 5, 6] == SEG_LR_SWAP[14]
    assert y[LR_SWAP[0]] == 1 and y[0] == 0


def test_seg_map_follows_the_image_through_the_affine_and_stays_integer(tmp_path: Path) -> None:
    from neurovision.config import AugmentConfig

    cache, seg_dir = _store(tmp_path)
    cfg = AugmentConfig(
        prob=1.0, noise_std=0.0, gamma=(1.0, 1.0), rotate_deg=10, scale=0.2, lowres_zoom=(1.0, 1.0)
    )
    ds = SegCacheDataset(cache, ["a"], _targets(1), build_augment(cfg, with_map=True), 0.0, seg_dir)
    ds.transform.set_random_state(seed=3)  # type: ignore[union-attr]
    x, _, m, _ = ds[0]
    assert x.shape == m.shape == (1, SIZE, SIZE, SIZE)
    assert m.dtype == torch.int64 and set(m.unique().tolist()) <= {0, 1, 14}


# --- loop and fit --------------------------------------------------------------------------


def test_train_epoch_with_the_seg_loss_trains_the_seg_head_and_the_readout() -> None:
    torch.manual_seed(0)
    model = build_model("r3d18_unet")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(2, 1, SIZE, SIZE, SIZE, generator=gen)
    y = torch.randint(0, 2, (2, 14), generator=gen).float()
    m = torch.randint(0, N_SEG_CLASSES, (2, 1, SIZE, SIZE, SIZE), generator=gen)
    seg_before = model.unet.seg.weight.detach().clone()
    fc_before = model.fc.artery_weight.detach().clone()
    cls, seg = train_epoch(
        model,
        [(x, y, m, torch.ones(2))],
        opt,
        cosine_with_warmup(opt, 0, 10),
        make_ema(model, 0.9),
        1,
        torch.device("cpu"),
        seg_weight=1.0,
    )
    assert np.isfinite(cls) and seg is not None and np.isfinite(seg)
    assert not torch.equal(seg_before, model.unet.seg.weight)
    assert not torch.equal(fc_before, model.fc.artery_weight)


def test_fit_with_the_vessel_readout_writes_the_seg_loss_and_a_loadable_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.setenv("WANDB_SILENT", "true")
    write_store(tmp_path / "store")
    path = write_config(tmp_path / "full.yaml", backbone="r3d18_unet", resample="vbox")
    path.write_text(path.read_text().replace("resample: vbox}", "resample: vbox, seg_weight: 1.0}"))
    config = load_config(path)
    labels = cnn_labels()
    ok = [u for u in UIDS if u not in EXCLUDED]
    train = [u for u in ok if FOLDS[u] != 0]
    val = [u for u in ok if FOLDS[u] == 0]
    cache = tmp_path / "store" / "cache" / "b2-128"
    seg_dir = tmp_path / "seg"
    seg_dir.mkdir()
    for uid in train[::2]:
        seg = np.zeros((SIZE,) * 3, np.uint8)
        seg[8:16, 8:16, 8:16] = 1 + UIDS.index(uid) % K
        np.savez_compressed(seg_dir / f"{uid}.npz", seg=seg)
    targets = lambda uids: labels.loc[uids, list(TARGETS)].to_numpy()  # noqa: E731
    assert config.augment is not None
    train_ds = SegCacheDataset(
        cache, train, targets(train), build_augment(config.augment, True), 0.5, seg_dir
    )
    val_ds = CacheDataset(cache, val, targets(val), None)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fit(
        config,
        0,
        run_dir,
        train_ds,
        val_ds,
        [u for u in UIDS if FOLDS[u] == 0],
        np.full(14, 0.3),
        {"train_used": len(train)},
    )
    history = pd.read_csv(run_dir / "history.csv")
    assert "train_seg_loss" in history and history["train_seg_loss"].notna().all()
    assert "train_map_loss" not in history
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    assert any(k.startswith("fc.") for k in state) and any(k.startswith("unet.seg") for k in state)
    assert (run_dir / "meta.json").exists() and (run_dir / "oof.csv").exists()
