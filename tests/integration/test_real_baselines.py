"""Pin the protocol-level findings of Phase 2 on the real dataset (``make test-data``)."""

import json
from pathlib import Path

import pytest

from sathealth.config import Config, load_config
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
    tmp = tmp_path_factory.mktemp("phase2")
    cfg = load_config(
        None,
        overrides=[
            f"paths.raw_dir={RAW}",
            f"paths.processed_dir={tmp / 'processed'}",
            f"paths.runs_dir={tmp / 'runs'}",
            "evaluation.bootstrap=200",
        ],
    )
    prepare_dataset(cfg)
    return cfg


def test_point_adjust_inflates_random_scores(config: Config) -> None:
    overall = evaluate_detector(config, "random").summary["subsets"]["all"]
    assert overall["point_adjusted_f1"] > 0.8, "random looks excellent under point-adjust"
    assert overall["event"]["f1"] < 0.15, "and is exposed by event-wise F1"


def test_every_threshold_comes_from_validation(config: Config) -> None:
    run = evaluate_detector(config, "zscore").run_dir
    per_channel = json.loads((run / "per_channel.json").read_text())
    assert len(per_channel) == 80
    assert {c["threshold_source"] for c in per_channel} == {"val"}


def test_limit_check_false_events_concentrate_on_shifted_channel(config: Config) -> None:
    run = evaluate_detector(config, "limit").run_dir
    per_channel = {c["chan_id"]: c for c in json.loads((run / "per_channel.json").read_text())}
    total_fp = sum(c["events"]["fp"] for c in per_channel.values())
    a5 = per_channel["A-5"]
    assert "nominal_out_of_range" in a5["flags"]
    assert a5["events"]["fp"] / total_fp > 0.5
