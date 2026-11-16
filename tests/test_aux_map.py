"""Location-map head (runs 2a and 2b): points, map target, loss, model, dataset, loop and fit."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from neurovision.config import AugmentConfig, TrainConfig, load_config
from neurovision.data.augment import build_augment
from neurovision.data.dataset import CacheDataset
from neurovision.data.labels import LR_SWAP, TARGETS
from neurovision.data.points import MAP_STRIDE, ball_map, load_points, pool_map
from neurovision.models.resnet3d import R3DWithMap, build_model, load_kinetics
from neurovision.train.ema import make_ema
from neurovision.train.loop import fit, param_groups, train_epoch
from neurovision.train.losses import balanced_bce, bce, classification_scale, label_weights
from neurovision.train.schedule import cosine_with_warmup
from tests.cnn_fakes import EXCLUDED, FOLDS, UIDS, cnn_labels, write_config, write_store

pytest.importorskip("torchvision")

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "tune"
SIZE = 32
GRID = (SIZE // 4, SIZE // 8, SIZE // 8)


# --- points and the map target -------------------------------------------------------------


def test_ball_map_marks_voxels_within_the_radius_only() -> None:
    m = ball_map(np.array([[10.0, 12.0, 14.0]]), SIZE, radius=3.0)
    assert m.dtype == np.uint8
    assert m[10, 12, 14] == 1 and m[13, 12, 14] == 1 and m[14, 12, 14] == 0
    assert m[12, 14, 15] == 1  # offset (2, 2, 1): distance exactly 3
    assert int(m.sum()) == 123  # lattice points within radius 3 of a lattice point


def test_ball_map_clips_at_the_faces_and_skips_far_points() -> None:
    m = ball_map(np.array([[0.0, 0.0, 0.0], [100.0, 5.0, 5.0]]), SIZE, radius=2.0)
    assert m[0, 0, 0] == 1 and m[2, 0, 0] == 1
    assert int(m.sum()) == int(ball_map(np.array([[0.0, 0.0, 0.0]]), SIZE, radius=2.0).sum())


def test_pool_map_takes_any_voxel_per_layer3_cell() -> None:
    m = torch.zeros(1, SIZE, SIZE, SIZE)
    m[0, 5, 9, 30] = 1
    pooled = pool_map(m)
    assert pooled.shape == (1, *GRID)
    assert MAP_STRIDE == (4, 8, 8)
    assert pooled.sum() == 1 and pooled[0, 1, 1, 3] == 1


def test_load_points_keeps_points_inside_the_cube(tmp_path: Path) -> None:
    path = tmp_path / "points.csv"
    pd.DataFrame(
        {
            "SeriesInstanceUID": ["a", "a", "b"],
            "location": ["Basilar Tip"] * 3,
            "x": [3.0, -4.0, 40.0],
            "y": [4.0, 4.0, 4.0],
            "z": [5.0, 5.0, 5.0],
        }
    ).to_csv(path, index=False)
    points = load_points(path, SIZE)
    assert set(points) == {"a"}  # b's only point lies outside a 32 cube
    np.testing.assert_array_equal(points["a"], [[3.0, 4.0, 5.0]])


# --- losses --------------------------------------------------------------------------------


def test_first_place_weights_give_point_one_mean_location_and_point_05_present() -> None:
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(6, 14, generator=gen)
    targets = torch.randint(0, 2, (6, 14), generator=gen).float()
    loss = classification_scale("first_place") * bce(logits, targets, label_weights("first_place"))
    per = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    ).mean(0)
    torch.testing.assert_close(loss, 0.1 * per[:13].mean() + 0.05 * per[13])
    assert classification_scale("uniform") == classification_scale("competition") == 1.0


def test_balanced_bce_sums_the_target_and_background_means() -> None:
    logits = torch.tensor([2.0, -1.0, 0.5, -3.0]).view(1, 1, 4, 1, 1)
    target = torch.tensor([1.0, 0.0, 1.0, 0.0]).view(1, 1, 4, 1, 1)
    each = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    each = each.flatten()
    expected = (each[0] + each[2]) / 2 + (each[1] + each[3]) / 2
    torch.testing.assert_close(balanced_bce(logits, target, torch.ones(1)), expected)


def test_balanced_bce_ignores_a_series_whose_map_is_unknown() -> None:
    logits = torch.randn(2, 1, 2, 2, 2, generator=torch.Generator().manual_seed(1))
    target = torch.zeros(2, 1, 2, 2, 2)
    target[0, 0, 0, 0, 0] = 1
    alone = balanced_bce(logits[:1], target[:1], torch.ones(1))
    both = balanced_bce(logits, target, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(both, alone)


def test_balanced_bce_without_target_cells_is_the_background_mean() -> None:
    logits = torch.randn(1, 1, 2, 2, 2, generator=torch.Generator().manual_seed(2))
    target = torch.zeros_like(logits)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    torch.testing.assert_close(balanced_bce(logits, target, torch.ones(1)), expected)


# --- config --------------------------------------------------------------------------------


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
        "backbone": "r3d18",
        "resample": "mm",
    }
    return base | overrides


def test_map_head_needs_r3d18_and_the_mm_cache() -> None:
    assert TrainConfig(**_train(aux_map_weight=1.0)).aux_map_weight == 1.0  # type: ignore[arg-type]
    for bad in ({"backbone": "resnet18"}, {"resample": "cube"}):
        with pytest.raises(ValidationError, match="aux_map_weight"):
            TrainConfig(**_train(aux_map_weight=1.0, **bad))  # type: ignore[arg-type]
    assert TrainConfig(**_train()).aux_map_weight is None  # type: ignore[arg-type]


def test_runs_2a_and_2b_configs() -> None:
    parent = load_config(CONFIGS / "t_flip_b4_r3d18_mm160_aug0.1_lr6e-4.yaml")
    a = load_config(CONFIGS / "t_flip_b4_r3d18_mm160_aug0.1_lr6e-4_aux.yaml")
    b = load_config(CONFIGS / "t_flip_b4_r3d18_mm160_aug0.1_lr6e-4_aux_w1st.yaml")
    assert parent.train is not None and a.train is not None and b.train is not None
    assert parent.train.aux_map_weight is None
    assert a.train == parent.train.model_copy(update={"aux_map_weight": 1.0})
    assert b.train == a.train.model_copy(update={"loss_weights": "first_place"})
    assert a.augment == parent.augment == b.augment


# --- model ---------------------------------------------------------------------------------


def test_map_model_returns_logits_and_a_layer3_map() -> None:
    torch.manual_seed(0)
    model = build_model("r3d18", aux_map=True)
    assert isinstance(model, R3DWithMap)
    model.eval()
    x = torch.randn(2, 1, SIZE, SIZE, SIZE)
    with torch.no_grad():
        logits, map_logits = model.forward_with_map(x)
        torch.testing.assert_close(model(x), logits)
    assert logits.shape == (2, 14)
    assert map_logits.shape == (2, 1, *GRID)


def test_map_model_keeps_r3d18_names_and_kinetics_fills_all_but_the_heads() -> None:
    torch.manual_seed(0)
    plain = build_model("r3d18")
    model = build_model("r3d18", aux_map=True)
    names = set(model.state_dict())
    assert set(plain.state_dict()) <= names
    assert names - set(plain.state_dict()) == {n for n in names if n.startswith("aux.")}
    weights = {
        k: torch.rand_like(v) if v.is_floating_point() else v.clone()
        for k, v in plain.state_dict().items()
    }
    weights["stem.0.weight"] = torch.rand(64, 3, 3, 7, 7)
    weights["fc.weight"], weights["fc.bias"] = torch.rand(400, 512), torch.rand(400)
    aux_before = model.state_dict()["aux.0.weight"].clone()
    load_kinetics(model, weights)
    key = "layer3.0.conv1.0.weight"
    torch.testing.assert_close(model.state_dict()[key], weights[key])
    torch.testing.assert_close(model.state_dict()["aux.0.weight"], aux_before)


def test_map_head_needs_r3d18() -> None:
    with pytest.raises(ValueError, match="r3d18"):
        build_model("resnet18", aux_map=True)


def test_map_head_trains_at_the_head_rate() -> None:
    model = build_model("r3d18", aux_map=True)
    head, rest = param_groups(model, 1e-3, 0.1)
    names = {id(p): n for n, p in model.named_parameters()}
    head_names = {names[id(p)] for p in head["params"]}
    assert head_names == {n for n in names.values() if n.startswith(("fc.", "aux."))}
    assert head["lr"] == 1e-3 and rest["lr"] == pytest.approx(1e-4)


# --- dataset -------------------------------------------------------------------------------


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    for uid in ("pos", "neg", "lost"):
        np.save(tmp_path / f"{uid}.npy", rng.normal(size=(SIZE,) * 3).astype(np.float16))
    return tmp_path


def _targets(*positive: bool) -> np.ndarray:
    return np.array([[float(p)] * 14 for p in positive])


POINTS = {"pos": np.array([[5.0, 9.0, 30.0]])}


def test_map_items_carry_the_target_and_keep(cache: Path) -> None:
    ds = CacheDataset(cache, ["pos", "neg", "lost"], _targets(True, False, True), None, 0.0, POINTS)
    x, y, m, keep = ds[0]
    assert x.shape == (1, SIZE, SIZE, SIZE) and y.shape == (14,)
    assert m.shape == (1, *GRID) and keep.item() == 1.0
    assert m[0, 1, 1, 3] == 1  # the cell holding voxel (5, 9, 30)
    _, _, m_neg, keep_neg = ds[1]
    assert m_neg.sum() == 0 and keep_neg.item() == 1.0  # a negative: empty map, counted
    _, _, m_lost, keep_lost = ds[2]
    assert m_lost.sum() == 0 and keep_lost.item() == 0.0  # positive, no point: map unknown


def test_flip_mirrors_the_map_with_the_volume(cache: Path) -> None:
    ds = CacheDataset(cache, ["pos"], _targets(True), None, 1.0, POINTS)
    x, _, m, _ = ds[0]
    stored = np.load(cache / "pos.npy").astype(np.float32)
    np.testing.assert_array_equal(x[0].numpy(), stored[::-1])
    mirrored = pool_map(torch.from_numpy(ball_map(np.array([[26.0, 9.0, 30.0]]), SIZE))[None])
    torch.testing.assert_close(m, mirrored)
    assert list(LR_SWAP) != list(range(14))  # labels swap as before


def test_map_follows_the_image_through_the_affine(cache: Path) -> None:
    cfg = AugmentConfig(
        prob=1.0, noise_std=0.0, gamma=(1.0, 1.0), rotate_deg=0, scale=0.3, lowres_zoom=(1.0, 1.0)
    )
    ds = CacheDataset(
        cache, ["pos"], _targets(True), build_augment(cfg, with_map=True), 0.0, POINTS
    )
    ds.transform.set_random_state(seed=3)  # type: ignore[union-attr]
    x, _, m, keep = ds[0]
    assert x.shape == (1, SIZE, SIZE, SIZE) and isinstance(x, torch.Tensor)
    assert m.shape == (1, *GRID) and set(m.unique().tolist()) <= {0.0, 1.0}
    assert keep.item() == 1.0


def test_items_without_points_are_unchanged(cache: Path) -> None:
    ds = CacheDataset(cache, ["pos"], _targets(True), None)
    assert len(ds[0]) == 2


# --- loop and fit --------------------------------------------------------------------------


def _map_batches(n: int = 2) -> list[tuple[torch.Tensor, ...]]:
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(n, 1, SIZE, SIZE, SIZE, generator=gen)
    y = torch.randint(0, 2, (n, 14), generator=gen).float()
    m = torch.zeros(n, 1, *GRID)
    m[0, 0, 1, 1, 1] = 1
    return [(x, y, m, torch.ones(n))]


def test_train_epoch_with_the_map_returns_both_losses_and_trains_the_map_head() -> None:
    torch.manual_seed(0)
    model = build_model("r3d18", aux_map=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = cosine_with_warmup(opt, 0, 10)
    before = model.aux[3].weight.detach().clone()  # type: ignore[index, union-attr]
    cls, map_loss = train_epoch(
        model,
        _map_batches(),
        opt,
        sched,
        make_ema(model, 0.9),
        1,
        torch.device("cpu"),
        map_weight=1.0,
    )
    assert np.isfinite(cls) and map_loss is not None and np.isfinite(map_loss)
    assert not torch.equal(before, model.aux[3].weight)  # type: ignore[index, union-attr]


def test_fit_with_the_map_head_writes_the_map_loss_and_a_loadable_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.setenv("WANDB_SILENT", "true")
    write_store(tmp_path / "store")
    path = write_config(tmp_path / "b2.yaml", backbone="r3d18", resample="mm")
    path.write_text(path.read_text().replace("resample: mm}", "resample: mm, aux_map_weight: 1.0}"))
    config = load_config(path)
    labels = cnn_labels()
    ok = [u for u in UIDS if u not in EXCLUDED]
    train = [u for u in ok if FOLDS[u] != 0]
    val = [u for u in ok if FOLDS[u] == 0]
    cache = tmp_path / "store" / "cache" / "b2-128"
    targets = lambda uids: labels.loc[uids, list(TARGETS)].to_numpy()  # noqa: E731
    points = {u: np.array([[16.0, 16.0, 16.0]]) for u in train[::2]}
    assert config.augment is not None
    train_ds = CacheDataset(
        cache, train, targets(train), build_augment(config.augment, True), 0.5, points
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
    assert "train_map_loss" in history and history["train_map_loss"].notna().all()
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    assert any(k.startswith("aux.") for k in state)
    assert (run_dir / "meta.json").exists() and (run_dir / "oof.csv").exists()


# --- a cosine longer than the run ----------------------------------------------------------


def test_schedule_epochs_must_cover_the_run() -> None:
    assert TrainConfig(**_train(schedule_epochs=30)).schedule_epochs == 30  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="schedule_epochs"):
        TrainConfig(**_train(schedule_epochs=5))  # type: ignore[arg-type]


def test_run_2a_cos30_config_changes_only_the_schedule() -> None:
    a = load_config(CONFIGS / "t_flip_b4_r3d18_mm160_aug0.1_lr6e-4_aux.yaml")
    c = load_config(CONFIGS / "t_flip_b4_r3d18_mm160_aug0.1_lr6e-4_aux_cos30.yaml")
    assert a.train is not None and c.train is not None
    assert c.train == a.train.model_copy(update={"schedule_epochs": 30})
    assert c.train.epochs == 10


def test_fit_on_a_longer_cosine_ends_with_the_rate_still_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.setenv("WANDB_SILENT", "true")
    write_store(tmp_path / "store")
    labels = cnn_labels()
    ok = [u for u in UIDS if u not in EXCLUDED]
    train = [u for u in ok if FOLDS[u] != 0]
    val = [u for u in ok if FOLDS[u] == 0]
    cache = tmp_path / "store" / "cache" / "b2-128"
    targets = lambda uids: labels.loc[uids, list(TARGETS)].to_numpy()  # noqa: E731
    last_lr = {}
    for schedule in (None, 6):
        path = write_config(tmp_path / f"c{schedule}.yaml", epochs=2)
        if schedule is not None:
            text = path.read_text().replace(
                "resample: cube}", f"resample: cube, schedule_epochs: {schedule}}}"
            )
            path.write_text(text)
        config = load_config(path)
        run_dir = tmp_path / f"run{schedule}"
        run_dir.mkdir()
        fit(
            config,
            0,
            run_dir,
            CacheDataset(cache, train, targets(train), None),
            CacheDataset(cache, val, targets(val), None),
            [u for u in UIDS if FOLDS[u] == 0],
            np.full(14, 0.3),
            {"train_used": len(train)},
        )
        last_lr[schedule] = pd.read_csv(run_dir / "history.csv")["lr"].iloc[-1]
    assert last_lr[None] == pytest.approx(0.0, abs=1e-12)  # the 2-epoch cosine reaches 0
    # 1 warm-up epoch, then 1 of 5 cosine epochs: 1e-3 x (1 + cos(pi / 5)) / 2
    assert last_lr[6] == pytest.approx(1e-3 * 0.5 * (1 + np.cos(np.pi / 5)))


def test_b2_aux_final_is_b2_final_plus_the_map_head() -> None:
    final = CONFIGS.parent / "final"
    b2 = load_config(final / "b2_final.yaml")
    aux = load_config(final / "b2_aux_final.yaml")
    assert b2.train is not None and aux.train is not None
    assert aux.train == b2.train.model_copy(update={"aux_map_weight": 1.0})
    assert aux.augment == b2.augment and aux.labels == b2.labels and aux.seed == b2.seed
    assert (aux.train.epochs, aux.train.early_stop_patience, aux.train.schedule_epochs) == (
        30,
        5,
        None,
    )
