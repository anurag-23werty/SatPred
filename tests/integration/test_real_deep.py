"""Pin the Phase 3 findings on real channels (``make test-data``).

A full deep evaluation takes ~12 minutes on 7 cores, too long for CI, so these tests train
the LSTM on two channels where Phase 3 found the clearest difference to the limit check.
"""

from pathlib import Path

import numpy as np
import pytest

from sathealth.config import Config, load_config
from sathealth.detectors.registry import build_detector
from sathealth.evaluation.metrics import average_precision, event_counts
from sathealth.evaluation.thresholds import fit_threshold
from sathealth.pipeline.prepare import (
    PreparedChannel,
    load_prepared,
    prepare_dataset,
    run_dir_for,
)

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
    tmp = tmp_path_factory.mktemp("phase3")
    cfg = load_config(
        None,
        overrides=[
            f"paths.raw_dir={RAW}",
            f"paths.processed_dir={tmp / 'processed'}",
            f"paths.runs_dir={tmp / 'runs'}",
        ],
    )
    prepare_dataset(cfg)
    return cfg


def _channel(config: Config, chan_id: str) -> PreparedChannel:
    return load_prepared(run_dir_for(config), chan_id)


def _detected_events(config: Config, name: str, channel: PreparedChannel) -> tuple[int, float]:
    det = build_detector(name, config).fit(channel.train, channel.val)
    val, test = det.score(channel.val), det.score(channel.test)
    threshold = fit_threshold(val, np.zeros(0), method="val_quantile", quantile=0.999, sigma=4)
    segments = [(s.start, s.end) for s in channel.segments]
    counts = event_counts(test > threshold.value, segments)
    return counts.tp, average_precision(test, channel.test_labels)


@pytest.mark.parametrize("chan_id", ["G-3", "P-3"])
def test_lstm_finds_anomalies_the_limit_check_misses(config: Config, chan_id: str) -> None:
    channel = _channel(config, chan_id)
    limit_tp, limit_ap = _detected_events(config, "limit", channel)
    lstm_tp, lstm_ap = _detected_events(config, "lstm", channel)
    assert limit_tp == 0, "the anomaly stays inside the static limits"
    assert lstm_tp == len(channel.segments)
    assert lstm_ap > limit_ap + 0.3


def test_deep_training_is_reproducible_on_real_data(config: Config) -> None:
    fast = config.model_copy(
        update={
            "deep": config.deep.model_copy(
                update={"training": config.deep.training.model_copy(update={"epochs": 2})}
            )
        }
    )
    channel = _channel(fast, "M-1")  # MSL: 55 features
    runs = [build_detector("lstm", fast).fit(channel.train, channel.val) for _ in range(2)]
    np.testing.assert_array_equal(runs[0].score(channel.test), runs[1].score(channel.test))
