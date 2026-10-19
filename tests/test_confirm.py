import json
from pathlib import Path

import numpy as np
import pytest

from sathealth.config import Config, load_config
from sathealth.data.opssat import load_segments, prepare_channel
from sathealth.pipeline.ablation import VARIANTS
from sathealth.pipeline.confirm import (
    HYPOTHESES,
    ChannelScores,
    render_confirmation,
    segment_rows,
    trainable,
    write_confirmation,
)


@pytest.fixture
def config(opssat_csv: Path, tmp_path: Path) -> Config:
    raw = opssat_csv.parent
    return load_config(
        None,
        overrides=[
            f"confirmation.raw_dir={raw}",
            f"confirmation.out_dir={tmp_path / 'out'}",
            "deep.training.epochs=2",
            "deep.lstm.window=16",
            "deep.lstm.hidden=8",
            "deep.transformer.window=16",
            "deep.transformer.d_model=8",
            "deep.transformer.heads=2",
            "deep.transformer.layers=1",
            "evaluation.bootstrap=50",
        ],
    )


def test_trainable_rule(config: Config) -> None:
    segments = load_segments(config.confirmation.raw_dir / "segments.csv")
    assert trainable(prepare_channel("CH-A", segments["CH-A"]), config)
    assert not trainable(prepare_channel("CH-B", segments["CH-B"]), config)


def test_segment_rows_are_composite_counts() -> None:
    scores = ChannelScores("X", {}, {}, [(0, 3), (4, 7), (8, 9)], [True, False, True])
    pred = np.array([0, 1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    np.testing.assert_array_equal(
        segment_rows(pred, scores), [[2, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )
    assert scores.labels.tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 1, 1]


def test_confirmation_end_to_end(config: Config, tmp_path: Path) -> None:
    out = write_confirmation(config, tmp_path / "C.md", download=False)
    result = json.loads((config.confirmation.out_dir / "results.json").read_text())
    assert result["channels"] == ["CH-A"]
    assert result["excluded_channels"] == ["CH-B"]
    names = {r["name"] for r in result["rows"]}
    assert {v.name for v in VARIANTS} | {"limit", "zscore", "random"} == names
    assert set(result["verdicts"]) == set(HYPOTHESES)
    assert result["best_baseline"] in {"limit", "zscore"}
    best = next(r for r in result["rows"] if r["name"] == result["best_baseline"])
    assert best["delta_vs_best_baseline"] == 0.0
    for row in result["rows"]:
        assert 0 <= row["composite_f1"] <= 1
        assert 0 <= row["segment_auc_pr"] <= 1
    text = out.read_text()
    assert "| C1 |" in text
    assert "excluded by the pre-registered rule: CH-B" in text
    assert render_confirmation(result) == text


def test_variants_ignore_the_callers_decision_settings(config: Config, tmp_path: Path) -> None:
    """Running with a live-style config must not turn every variant adaptive."""
    from sathealth.pipeline.confirm import run_confirmation

    live = config.model_copy(
        update={"decision": config.decision.model_copy(update={"recalibrate_every": 50})}
    )
    plain = {r["name"]: r for r in run_confirmation(config, download=False)["rows"]}
    adaptive = {r["name"]: r for r in run_confirmation(live, download=False)["rows"]}
    assert plain == adaptive
    assert (config.confirmation.out_dir / "scores" / "CH-A.npz").is_file()
