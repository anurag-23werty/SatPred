from pathlib import Path

import pytest
from pydantic import ValidationError

from sathealth.config import Config, DuplicatePolicy, load_config

DEFAULT = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_default_yaml_matches_code_defaults() -> None:
    assert load_config(DEFAULT) == Config()


def test_overrides_are_typed() -> None:
    cfg = load_config(
        DEFAULT,
        overrides=[
            "windows.size=128",
            "dataset.spacecraft=[SMAP]",
            "dataset.duplicate_policy=union",
        ],
    )
    assert cfg.windows.size == 128
    assert cfg.dataset.spacecraft == ("SMAP",)
    assert cfg.dataset.duplicate_policy is DuplicatePolicy.UNION


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="windos"):
        load_config(None, overrides=["windos.size=3"])


@pytest.mark.parametrize(
    "override",
    [
        "windows.size=1",
        "preprocess.val_fraction=0.7",
        "preprocess.value_columns=[0,0]",
        "dataset.spacecraft=[HUBBLE]",
    ],
)
def test_invalid_values_are_rejected(override: str) -> None:
    with pytest.raises(ValidationError):
        load_config(None, overrides=[override])


def test_malformed_override() -> None:
    with pytest.raises(ValueError, match=r"key\.sub=value"):
        load_config(None, overrides=["windows.size"])


def test_config_is_immutable() -> None:
    cfg = Config()
    with pytest.raises(ValidationError):
        cfg.seed = 1  # type: ignore[misc]


def test_to_dict_is_json_safe() -> None:
    assert Config().to_dict()["paths"]["raw_dir"] == "data/raw/telemanom"
