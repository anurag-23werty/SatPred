import json
from pathlib import Path

import numpy as np
import pytest

from sathealth.config import Config, load_config
from sathealth.data.download import Manifest
from sathealth.pipeline.decide import (
    _kind_agreement,
    alarm_mask,
    decide,
    decision_dir_for,
    describe,
    global_oracle,
)
from sathealth.pipeline.evaluate import evaluate_detector
from sathealth.pipeline.prepare import prepare_dataset


@pytest.fixture
def config(raw_root: Path, raw_manifest: Manifest, tmp_path: Path) -> Config:
    cfg = load_config(
        None,
        overrides=[
            f"paths.raw_dir={raw_root}",
            f"paths.processed_dir={tmp_path / 'processed'}",
            f"paths.runs_dir={tmp_path / 'runs'}",
            f"paths.decisions_dir={tmp_path / 'decisions'}",
            "evaluation.bootstrap=20",
            "decision.inputs=[zscore, limit]",
        ],
    )
    prepare_dataset(cfg, manifest=raw_manifest)
    for name in ("zscore", "limit", "iforest"):
        evaluate_detector(cfg, name)
    return cfg


def test_decision_run_outputs(config: Config) -> None:
    result = decide(config)
    assert not result.reused
    summary = result.summary
    assert summary["kind"] == "decision"
    assert summary["detector"] == "fused"
    assert set(summary["inputs"]) == {"zscore", "limit"}
    assert summary["subsets"]["all"]["n_channels"] == 3
    assert summary["within_budget"] == (
        summary["subsets"]["all"]["fp_events_per_1k_steps"] <= summary["alarm_budget"]
    )
    assert 0 <= summary["alarm_fraction_mean"] <= 1
    assert summary["severity_counts"].keys() == {"watch", "warning", "critical"}
    per_channel = json.loads((result.run_dir / "per_channel.json").read_text())
    assert per_channel[0]["calibration"]["zscore"]["method"] == "evt"
    events = json.loads((result.run_dir / "events" / "A-1.json").read_text())
    assert any(e["start"] <= 320 and e["end"] >= 300 for e in events), "injected spike found"
    assert {"severity", "kind", "contributions", "peak_surprise"} <= events[0].keys()
    with np.load(result.run_dir / "scores" / "A-1.npz") as npz:
        assert npz["surprise"].shape == npz["alarm"].shape == (400,)


def test_reuse_force_and_fingerprint(config: Config) -> None:
    first = decide(config)
    assert decide(config).reused
    assert not decide(config, force=True).reused
    other = config.model_copy(
        update={"decision": config.decision.model_copy(update={"fusion": "max"})}
    )
    assert decision_dir_for(other) != first.run_dir


def test_missing_scores_are_actionable(config: Config) -> None:
    cfg = config.model_copy(
        update={"decision": config.decision.model_copy(update={"inputs": ("lstm",)})}
    )
    with pytest.raises(FileNotFoundError, match="sathealth evaluate lstm"):
        decide(cfg)


def test_ndt_and_prune_path(config: Config) -> None:
    d = config.decision.model_copy(update={"threshold": "ndt", "prune": 0.13, "ndt_window": 100})
    s = np.r_[np.full(300, 0.5), np.full(10, 9.0), np.full(300, 0.5)]
    s += np.random.default_rng(0).random(s.size) * 0.1
    mask = alarm_mask(s, d)
    assert mask[300:310].all()
    assert mask.sum() < 30


def test_global_oracle_uses_one_level_for_all_channels() -> None:
    labels = {"a": np.array([0, 1, 1, 0], dtype=bool), "b": np.array([0, 0, 1, 0], dtype=bool)}
    s = {"a": np.array([0.0, 5.0, 5.0, 0.0]), "b": np.array([1.0, 1.0, 5.0, 1.0])}
    out = global_oracle(s, labels, {"a": [(1, 2)], "b": [(2, 2)]})
    assert out["composite_f1"] == 1.0
    assert 1.0 < out["level"] <= 5.0


def test_kind_agreement_and_describe(config: Config) -> None:
    pairs = [("point", "point"), ("point", "contextual"), ("contextual", "contextual")]
    out = _kind_agreement(pairs)
    assert out["accuracy"] == pytest.approx(2 / 3)
    assert out["majority_baseline"] == pytest.approx(2 / 3)
    assert out["confusion"]["point->contextual"] == 1
    assert _kind_agreement([]) == {"n": 0}
    assert "mean fusion" in describe(config.decision)
    ndt = config.decision.model_copy(update={"threshold": "ndt", "prune": 0.13})
    assert "dynamic threshold, pruning 0.13" in describe(ndt)
