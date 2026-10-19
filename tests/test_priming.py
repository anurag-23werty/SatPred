import json
from pathlib import Path

import numpy as np
import pytest

from sathealth.config import Config
from sathealth.detectors.zscore import RobustZScore
from sathealth.pipeline.evaluate import evaluate_detector, score_primed
from sathealth.pipeline.priming import render_priming, write_priming


def test_score_primed_changes_only_context_dependent_scores() -> None:
    rng = np.random.default_rng(0)
    history, x = rng.normal(size=(100, 1)), rng.normal(size=(50, 1))
    det = RobustZScore().fit(history, history)
    np.testing.assert_array_equal(score_primed(det, history, x, 32), det.score(x))
    np.testing.assert_array_equal(score_primed(det, history, x, 0), det.score(x))
    np.testing.assert_array_equal(score_primed(det, history[:0], x, 32), det.score(x))


def test_priming_comparison_end_to_end(explain_config: Config, tmp_path: Path) -> None:
    cold = explain_config.model_copy(
        update={"decision": explain_config.decision.model_copy(update={"recalibrate_every": 50})}
    )
    primed = cold.model_copy(
        update={"evaluation": cold.evaluation.model_copy(update={"prime_steps": 32})}
    )
    evaluate_detector(primed, "lstm")
    out = write_priming(cold, primed, tmp_path / "P.md")
    result = json.loads((primed.paths.decisions_dir / "priming.json").read_text())
    assert set(result["early"]) == {"cold", "primed"}
    assert result["runs"]["cold"] != result["runs"]["primed"]
    assert isinstance(result["P1"], bool)
    assert isinstance(result["P2"], bool)
    assert out.read_text() == render_priming(result)
    with pytest.raises(KeyError):
        render_priming({})
