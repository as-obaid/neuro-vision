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
    "batch_size": 2,
    "accum_steps": 4,
    "lr": 1.0e-4,
    "weight_decay": 1.0e-2,
    "warmup_epochs": 1,
    "ema_decay": 0.999,
    "loss_weights": "uniform",
    "num_workers": 8,
}
AUGMENT = {
    "prob": 0.2,
    "noise_std": 0.1,
    "gamma": [0.7, 1.5],
    "rotate_deg": 10,
    "scale": 0.1,
    "lowres_zoom": [0.5, 1.0],
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
