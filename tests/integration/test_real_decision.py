"""Pin the Phase 4 findings on the real dataset (``make test-data``).

Uses the fast z-score detector so the CI real-data job needs no deep-model runs.
"""

import json
from pathlib import Path

import pytest

from sathealth.config import Config, load_config
from sathealth.pipeline.decide import decide
from sathealth.pipeline.evaluate import evaluate_detector
from sathealth.pipeline.prepare import prepare_dataset

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "telemanom"

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not (RAW / "labeled_anomalies.csv").is_file(),
        reason="real dataset not downloaded (run `make download`)",
    ),
]


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    tmp = tmp_path_factory.mktemp("phase4")
    cfg = load_config(
        None,
        overrides=[
            f"paths.raw_dir={RAW}",
            f"paths.processed_dir={tmp / 'processed'}",
            f"paths.runs_dir={tmp / 'runs'}",
            f"paths.decisions_dir={tmp / 'decisions'}",
            "evaluation.bootstrap=100",
            "decision.inputs=[zscore]",
        ],
    )
    prepare_dataset(cfg)
    evaluate_detector(cfg, "zscore")
    return cfg


def test_validation_calibration_does_not_transfer_to_test(config: Config) -> None:
    """Surprise >= 3 should fire on 0.1% of nominal steps; on many channels it fires far
    more often, because the test period behaves differently from the end of training."""
    per_channel = json.loads((decide(config).run_dir / "per_channel.json").read_text())
    assert len(per_channel) == 80
    shifted = [c["chan_id"] for c in per_channel if c["nominal_alarm_fraction"] > 0.01]
    assert len(shifted) >= 15, "at least 10x the designed rate on many channels"


def test_streaming_recalibration_reduces_alarms(config: Config) -> None:
    static = decide(config).summary
    adaptive_cfg = config.model_copy(
        update={
            "decision": config.decision.model_copy(
                update={"name": "adaptive", "recalibrate_every": 100}
            )
        }
    )
    adaptive = decide(adaptive_cfg).summary
    assert adaptive["nominal_alarm_fraction_mean"] < static["nominal_alarm_fraction_mean"]
    assert (
        adaptive["subsets"]["all"]["fp_events_per_1k_steps"]
        <= static["subsets"]["all"]["fp_events_per_1k_steps"]
    )


def test_events_carry_severity_type_and_evidence(config: Config) -> None:
    run = decide(config).run_dir
    events = [e for path in (run / "events").glob("*.json") for e in json.loads(path.read_text())]
    assert events
    assert {e["severity"] for e in events} <= {"watch", "warning", "critical"}
    assert {e["kind"] for e in events} == {"point", "contextual"}
    assert all(set(e["contributions"]) == {"zscore"} for e in events)
