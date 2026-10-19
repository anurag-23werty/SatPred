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


def test_evaluation_and_baseline_settings() -> None:
    cfg = load_config(
        DEFAULT,
        overrides=["evaluation.threshold=val_sigma", "baselines.iforest.window=16"],
    )
    assert cfg.evaluation.threshold == "val_sigma"
    assert cfg.baselines.iforest.window == 16
    assert cfg.evaluation.exclude_subsets[1] == ("constant_train", "nominal_out_of_range")
    with pytest.raises(ValidationError):
        load_config(None, overrides=["evaluation.quantile=1.5"])


def test_deep_and_tracking_settings() -> None:
    cfg = load_config(DEFAULT, overrides=["deep.lstm.window=32", "tracking.mlflow_uri=file:./m"])
    assert cfg.deep.lstm.window == 32
    assert cfg.deep.training.device == "cpu"
    assert cfg.tracking.mlflow_uri == "file:./m"
    assert load_config(DEFAULT) == load_config(None), "YAML mirrors the code defaults"
    with pytest.raises(ValidationError, match="divisible"):
        load_config(None, overrides=["deep.transformer.heads=5"])


def test_decision_settings() -> None:
    cfg = load_config(DEFAULT, overrides=["decision.fusion=max", "decision.prune=0.13"])
    assert cfg.decision.fusion == "max"
    assert cfg.decision.prune == 0.13
    assert cfg.decision.inputs == ("lstm", "transformer")
    assert load_config(DEFAULT) == load_config(None), "YAML mirrors the code defaults"
    with pytest.raises(ValidationError, match="alarm_off"):
        load_config(None, overrides=["decision.alarm_off=4"])
    with pytest.raises(ValidationError, match="severity_warning"):
        load_config(None, overrides=["decision.severity_warning=7"])
    with pytest.raises(ValidationError):
        load_config(None, overrides=["decision.name=Has Space"])


def test_live_profile_only_adds_streaming_recalibration() -> None:
    live = load_config(DEFAULT.parent / "live.yaml")
    assert live.decision.name == "fused-adaptive"
    assert live.decision.recalibrate_every == 100
    assert live.decision.model_dump(exclude={"name", "recalibrate_every"}) == (
        load_config(DEFAULT).decision.model_dump(exclude={"name", "recalibrate_every"})
    )


def test_explain_settings() -> None:
    cfg = load_config(DEFAULT, overrides=["explain.horizon=100"])
    assert cfg.explain.horizon == 100
    assert cfg.explain.band_level == 0.9
    assert cfg.paths.models_dir.name == "models"
    with pytest.raises(ValidationError):
        load_config(None, overrides=["explain.band_level=0.2"])


def test_published_run_fingerprints_stay_stable() -> None:
    """Adding a setting at its neutral value must not orphan published runs (ADR 0009)."""
    from sathealth.config import fingerprint_view
    from sathealth.pipeline.decide import decision_fingerprint
    from sathealth.pipeline.evaluate import evaluation_fingerprint

    default = load_config(DEFAULT)
    assert evaluation_fingerprint(default, "limit") == "df4222df29cc"
    assert evaluation_fingerprint(default, "lstm") == "749cc37f8ff3"
    live = load_config(DEFAULT.parent / "live.yaml")
    assert decision_fingerprint(live) == "ca6f56643841"
    assert "prime_steps" not in fingerprint_view(default.evaluation)
    primed = load_config(DEFAULT, overrides=["evaluation.prime_steps=256"])
    assert fingerprint_view(primed.evaluation)["prime_steps"] == 256
    assert evaluation_fingerprint(primed, "lstm") != "749cc37f8ff3"
