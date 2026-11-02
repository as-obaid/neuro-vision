from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from neurovision.config import Config, load_config, resolve

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

GBM = {"max_iter": 200, "learning_rate": 0.05, "max_depth": 3}
TRAIN = {
    "epochs": 30,
    "batch_size": 8,
    "accum_steps": 1,
    "lr": 3.0e-4,
    "weight_decay": 1.0e-2,
    "warmup_epochs": 1,
    "ema_decay": 0.999,
    "loss_weights": "uniform",
    "num_workers": 8,
    "early_stop_patience": 5,
    "pos_weight": None,
    "init": "random",
    "backbone": "resnet18",
    "freeze_epochs": 0,
    "backbone_lr_scale": 1.0,
    "input_size": 128,
    "head": "linear",
    "resample": "cube",
}
AUGMENT = {
    "prob": 0.2,
    "noise_std": 0.1,
    "gamma": [0.7, 1.5],
    "rotate_deg": 10,
    "scale": 0.1,
    "lowres_zoom": [0.5, 1.0],
    "flip_lr": 0.0,
}


def write(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def leaves(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(leaves(value, f"{path}."))
        else:
            out[path] = value
    return out


def changed_leaves(child: dict[str, Any], parent: dict[str, Any]) -> set[str]:
    a, b = leaves(child), leaves(parent)
    return {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}


def parent_of(child: Path) -> Path:
    return child.parent / yaml.safe_load(child.read_text())["extends"]


def check_one_variable(child: Path) -> None:
    diff = changed_leaves(resolve(child), resolve(parent_of(child)))
    assert len(diff) == 1, f"{child.name} differs from its parent in {sorted(diff)}"


def test_extends_chain_of_three_levels(tmp_path: Path) -> None:
    write(tmp_path / "a.yaml", {"seed": 1, "labels": "original"})
    write(tmp_path / "b.yaml", {"extends": "a.yaml", "seed": 2})
    write(tmp_path / "c.yaml", {"extends": "b.yaml", "kind": "prior"})
    assert resolve(tmp_path / "c.yaml") == {"seed": 2, "labels": "original", "kind": "prior"}


def test_extends_is_relative_to_the_file_folder(tmp_path: Path) -> None:
    (tmp_path / "base").mkdir()
    write(tmp_path / "base" / "a.yaml", {"seed": 1})
    write(tmp_path / "base" / "b.yaml", {"extends": "a.yaml"})
    write(tmp_path / "c.yaml", {"extends": "base/b.yaml"})
    assert resolve(tmp_path / "c.yaml") == {"seed": 1}


def test_nested_mappings_merge_key_by_key(tmp_path: Path) -> None:
    write(tmp_path / "p.yaml", {"gbm": {"max_iter": 200, "max_depth": 3}})
    write(tmp_path / "c.yaml", {"extends": "p.yaml", "gbm": {"max_depth": 5}})
    assert resolve(tmp_path / "c.yaml") == {"gbm": {"max_iter": 200, "max_depth": 5}}


def test_lists_are_replaced_not_merged(tmp_path: Path) -> None:
    write(tmp_path / "p.yaml", {"augment": {"gamma": [0.7, 1.5]}})
    write(tmp_path / "c.yaml", {"extends": "p.yaml", "augment": {"gamma": [0.8]}})
    assert resolve(tmp_path / "c.yaml") == {"augment": {"gamma": [0.8]}}


def test_cycle_raises_naming_the_files(tmp_path: Path) -> None:
    write(tmp_path / "a.yaml", {"extends": "b.yaml"})
    write(tmp_path / "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(ValueError, match=r"a\.yaml.*b\.yaml"):
        resolve(tmp_path / "a.yaml")


def test_name_is_the_file_stem(tmp_path: Path) -> None:
    write(tmp_path / "my_run.yaml", {"kind": "prior", "seed": 0, "labels": "corrected"})
    assert load_config(tmp_path / "my_run.yaml").name == "my_run"


def test_name_key_in_yaml_is_an_error(tmp_path: Path) -> None:
    write(tmp_path / "x.yaml", {"name": "y", "kind": "prior", "seed": 0, "labels": "corrected"})
    with pytest.raises(ValueError, match="name"):
        load_config(tmp_path / "x.yaml")


def test_unknown_key_raises(tmp_path: Path) -> None:
    write(tmp_path / "x.yaml", {"kind": "prior", "seed": 0, "labels": "corrected", "oops": 1})
    with pytest.raises(ValidationError):
        load_config(tmp_path / "x.yaml")


def test_unknown_nested_key_raises(tmp_path: Path) -> None:
    write(
        tmp_path / "x.yaml",
        {"kind": "gbm", "seed": 0, "labels": "corrected", "gbm": {**GBM, "oops": 1}},
    )
    with pytest.raises(ValidationError):
        load_config(tmp_path / "x.yaml")


def test_non_positive_value_raises(tmp_path: Path) -> None:
    write(
        tmp_path / "x.yaml",
        {"kind": "gbm", "seed": 0, "labels": "corrected", "gbm": {**GBM, "max_iter": 0}},
    )
    with pytest.raises(ValidationError):
        load_config(tmp_path / "x.yaml")


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("train", "weight_decay", -0.1),
        ("train", "warmup_epochs", -1),
        ("train", "warmup_epochs", 30),
        ("train", "ema_decay", 0),
        ("train", "ema_decay", 1),
        ("train", "ema_decay", 1.5),
        ("train", "num_workers", -1),
        ("train", "early_stop_patience", 0),
        ("train", "loss_weights", "metric"),
        ("train", "pos_weight", "linear"),
        ("train", "init", "medicalnet"),
        ("train", "freeze_epochs", -1),
        ("train", "freeze_epochs", 30),
        ("train", "backbone_lr_scale", 0),
        ("train", "backbone_lr_scale", 1.5),
        ("augment", "prob", -0.1),
        ("augment", "prob", 1.1),
        ("augment", "noise_std", -0.1),
        ("augment", "rotate_deg", -1),
        ("augment", "scale", -0.1),
        ("augment", "scale", 1),
        ("augment", "gamma", [1.5, 0.7]),
        ("augment", "gamma", [0, 1.5]),
        ("augment", "lowres_zoom", [1.0, 0.5]),
        ("augment", "lowres_zoom", [0.5, 1.5]),
        ("augment", "flip_lr", -0.1),
        ("augment", "flip_lr", 1.1),
    ],
)
def test_out_of_range_cnn_value_raises(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    sections = {"train": dict(TRAIN), "augment": dict(AUGMENT)}
    sections[section][key] = value
    write(tmp_path / "x.yaml", {"kind": "cnn", "seed": 0, "labels": "corrected", **sections})
    with pytest.raises(ValidationError):
        load_config(tmp_path / "x.yaml")


@pytest.mark.parametrize(
    "sections",
    [
        {"kind": "prior", "gbm": GBM},
        {"kind": "prior", "train": TRAIN, "augment": AUGMENT},
        {"kind": "gbm"},
        {"kind": "gbm", "gbm": GBM, "train": TRAIN},
        {"kind": "gbm", "gbm": GBM, "augment": AUGMENT},
        {"kind": "cnn", "augment": AUGMENT},
        {"kind": "cnn", "train": TRAIN},
        {"kind": "cnn", "train": TRAIN, "augment": AUGMENT, "gbm": GBM},
    ],
)
def test_missing_or_extra_section_for_kind_raises(tmp_path: Path, sections: dict[str, Any]) -> None:
    write(tmp_path / "x.yaml", {"seed": 0, "labels": "corrected", **sections})
    with pytest.raises(ValidationError):
        load_config(tmp_path / "x.yaml")


def test_config_is_frozen(tmp_path: Path) -> None:
    write(tmp_path / "x.yaml", {"kind": "prior", "seed": 0, "labels": "corrected"})
    cfg = load_config(tmp_path / "x.yaml")
    with pytest.raises(ValidationError):
        cfg.seed = 1  # type: ignore[misc]


@pytest.mark.parametrize("stem", ["b0", "b1", "b2"])
def test_base_configs_load(stem: str) -> None:
    cfg = load_config(CONFIGS / "base" / f"{stem}.yaml")
    assert isinstance(cfg, Config)
    assert cfg.name == stem
    assert cfg.seed == 0
    assert cfg.labels == "corrected"


def test_base_config_values_match_the_contract() -> None:
    b1 = load_config(CONFIGS / "base" / "b1.yaml")
    b2 = load_config(CONFIGS / "base" / "b2.yaml")
    assert b1.kind == "gbm"
    assert b1.gbm is not None and b1.gbm.model_dump() == GBM
    assert b2.kind == "cnn"
    assert b2.train is not None and b2.train.model_dump() == TRAIN
    assert b2.augment is not None
    assert b2.augment.model_dump() == {
        **AUGMENT,
        "gamma": (0.7, 1.5),
        "lowres_zoom": (0.5, 1.0),
    }


def test_stage2_configs_change_one_variable() -> None:
    for child in sorted((CONFIGS / "stage2").glob("*.yaml")):
        check_one_variable(child)


TUNE = sorted((CONFIGS / "tune").glob("*.yaml"))
# configs with their own exact-diff tests: the reference, the batch control, the Kinetics arm,
# the batch-16 rate push
TUNE_MULTI_LEAF = {
    "t_r0.yaml",
    "t_flip_b4.yaml",
    "t_flip_b4_r3d18.yaml",
    "t_flip_b4_r3d18_mm160_aug0.1_b16_lr1.2e-3.yaml",
}


def test_tune_configs_train_ten_epochs_without_early_stop() -> None:
    assert TUNE
    for path in TUNE:
        assert path.stem.startswith("t_")
        cfg = load_config(path)
        assert cfg.train is not None
        assert cfg.train.epochs == 10, path.name
        assert cfg.train.early_stop_patience is None, path.name


def test_tune_configs_change_one_variable() -> None:
    for child in TUNE:
        if child.name not in TUNE_MULTI_LEAF:
            check_one_variable(child)


def test_tune_reference_changes_only_the_schedule() -> None:
    diff = changed_leaves(
        resolve(CONFIGS / "tune" / "t_r0.yaml"),
        resolve(CONFIGS / "stage2" / "b2_imagenet_2lr.yaml"),
    )
    assert diff == {"train.epochs", "train.early_stop_patience"}


def test_one_variable_check_passes_for_one_changed_leaf(tmp_path: Path) -> None:
    write(tmp_path / "p.yaml", {"seed": 0, "gbm": {"max_iter": 200, "max_depth": 3}})
    write(tmp_path / "c.yaml", {"extends": "p.yaml", "gbm": {"max_depth": 5}})
    check_one_variable(tmp_path / "c.yaml")


def test_one_variable_check_fails_for_two_changed_leaves(tmp_path: Path) -> None:
    write(tmp_path / "p.yaml", {"seed": 0, "gbm": {"max_iter": 200, "max_depth": 3}})
    write(tmp_path / "c.yaml", {"extends": "p.yaml", "seed": 1, "gbm": {"max_depth": 5}})
    with pytest.raises(AssertionError):
        check_one_variable(tmp_path / "c.yaml")


def test_one_variable_check_fails_for_no_change(tmp_path: Path) -> None:
    write(tmp_path / "p.yaml", {"seed": 0})
    write(tmp_path / "c.yaml", {"extends": "p.yaml"})
    with pytest.raises(AssertionError):
        check_one_variable(tmp_path / "c.yaml")


def test_input_size_defaults_to_128_and_rejects_unsupported_values(tmp_path: Path) -> None:
    train = {k: v for k, v in TRAIN.items() if k != "input_size"}
    body = {"kind": "cnn", "seed": 0, "labels": "corrected", "augment": AUGMENT}
    write(tmp_path / "x.yaml", {**body, "train": train})
    cfg = load_config(tmp_path / "x.yaml")
    assert cfg.train is not None and cfg.train.input_size == 128
    write(tmp_path / "y.yaml", {**body, "train": {**train, "input_size": 160}})
    cfg = load_config(tmp_path / "y.yaml")
    assert cfg.train is not None and cfg.train.input_size == 160
    write(tmp_path / "z.yaml", {**body, "train": {**train, "input_size": 100}})
    with pytest.raises(ValidationError):
        load_config(tmp_path / "z.yaml")


def test_t_bb0_3_160_changes_only_the_input_size() -> None:
    diff = changed_leaves(
        resolve(CONFIGS / "tune" / "t_bb0.3_160.yaml"), resolve(CONFIGS / "tune" / "t_bb0.3.yaml")
    )
    assert diff == {"train.input_size"}


def _tune_diff(child: str, parent: str) -> set[str]:
    return changed_leaves(resolve(CONFIGS / "tune" / child), resolve(CONFIGS / "tune" / parent))


def test_batch_control_changes_only_batch_and_accumulation() -> None:
    assert _tune_diff("t_flip_b4.yaml", "t_flip.yaml") == {"train.batch_size", "train.accum_steps"}
    cfg = load_config(CONFIGS / "tune" / "t_flip_b4.yaml")
    assert cfg.train is not None
    assert cfg.train.batch_size * cfg.train.accum_steps == 8


def test_resnet_backbone_configs_change_only_the_backbone() -> None:
    for name in ("r34", "r50"):
        stem = f"t_flip_b4_{name}"
        assert _tune_diff(f"{stem}.yaml", "t_flip_b4.yaml") == {"train.backbone"}
    cfg = load_config(CONFIGS / "tune" / "t_flip_b4_r50.yaml")
    assert cfg.train is not None and cfg.train.backbone == "resnet50"


def test_kinetics_config_changes_only_backbone_and_init() -> None:
    diff = _tune_diff("t_flip_b4_r3d18.yaml", "t_flip_b4.yaml")
    assert diff == {"train.backbone", "train.init"}
    cfg = load_config(CONFIGS / "tune" / "t_flip_b4_r3d18.yaml")
    assert cfg.train is not None
    assert (cfg.train.backbone, cfg.train.init) == ("r3d18", "kinetics")


def test_batch16_push_changes_only_batch_accumulation_and_rate() -> None:
    stem = "t_flip_b4_r3d18_mm160_aug0.1"
    diff = _tune_diff(f"{stem}_b16_lr1.2e-3.yaml", f"{stem}_lr6e-4.yaml")
    assert diff == {"train.batch_size", "train.accum_steps", "train.lr"}
    cfg = load_config(CONFIGS / "tune" / f"{stem}_b16_lr1.2e-3.yaml")
    assert cfg.train is not None
    assert (cfg.train.batch_size, cfg.train.accum_steps, cfg.train.lr) == (16, 1, 1.2e-3)


def test_init_must_fit_the_backbone() -> None:
    for backbone, init in [("r3d18", "imagenet"), ("resnet18", "kinetics")]:
        train = {**TRAIN, "backbone": backbone, "init": init}
        with pytest.raises(ValueError, match="needs"):
            Config.model_validate({"kind": "cnn", "train": train})


def test_head_defaults_to_linear_and_accepts_mlp(tmp_path: Path) -> None:
    train = {k: v for k, v in TRAIN.items() if k != "head"}
    body = {"kind": "cnn", "seed": 0, "labels": "corrected", "augment": AUGMENT}
    write(tmp_path / "x.yaml", {**body, "train": train})
    cfg = load_config(tmp_path / "x.yaml")
    assert cfg.train is not None and cfg.train.head == "linear"
    write(tmp_path / "y.yaml", {**body, "train": {**train, "head": "mlp"}})
    cfg = load_config(tmp_path / "y.yaml")
    assert cfg.train is not None and cfg.train.head == "mlp"
    write(tmp_path / "z.yaml", {**body, "train": {**train, "head": "deep"}})
    with pytest.raises(ValidationError):
        load_config(tmp_path / "z.yaml")


def test_mlp_head_config_changes_only_the_head() -> None:
    assert _tune_diff("t_flip_b4_r34_mlp.yaml", "t_flip_b4_r34.yaml") == {"train.head"}
    cfg = load_config(CONFIGS / "tune" / "t_flip_b4_r34_mlp.yaml")
    assert cfg.train is not None and (cfg.train.backbone, cfg.train.head) == ("resnet34", "mlp")


def test_half_augmentation_config_changes_only_the_probability() -> None:
    assert _tune_diff("t_flip_b4_r34_aug0.1.yaml", "t_flip_b4_r34.yaml") == {"augment.prob"}
    cfg = load_config(CONFIGS / "tune" / "t_flip_b4_r34_aug0.1.yaml")
    assert cfg.augment is not None and cfg.augment.prob == 0.1


def test_resample_defaults_to_cube_and_accepts_mm(tmp_path: Path) -> None:
    train = {k: v for k, v in TRAIN.items() if k != "resample"}
    body = {"kind": "cnn", "seed": 0, "labels": "corrected", "augment": AUGMENT}
    write(tmp_path / "x.yaml", {**body, "train": train})
    cfg = load_config(tmp_path / "x.yaml")
    assert cfg.train is not None and cfg.train.resample == "cube"
    write(tmp_path / "y.yaml", {**body, "train": {**train, "resample": "mm"}})
    cfg = load_config(tmp_path / "y.yaml")
    assert cfg.train is not None and cfg.train.resample == "mm"
    write(tmp_path / "z.yaml", {**body, "train": {**train, "resample": "crop"}})
    with pytest.raises(ValidationError):
        load_config(tmp_path / "z.yaml")


def test_mm_config_changes_only_the_resample() -> None:
    assert _tune_diff("t_flip_b4_r3d18_mm160.yaml", "t_flip_b4_r3d18_160.yaml") == {
        "train.resample"
    }
    cfg = load_config(CONFIGS / "tune" / "t_flip_b4_r3d18_mm160.yaml")
    assert cfg.train is not None
    assert (cfg.train.resample, cfg.train.input_size, cfg.train.backbone) == ("mm", 160, "r3d18")
