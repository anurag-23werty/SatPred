"""Pin the Phase 6 guarantees on the real dataset (``make test-data``).

Uses the fast z-score detector so the CI real-data job can run it.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sathealth.config import Config, load_config
from sathealth.decision.events import outside_range
from sathealth.ops.streaming import StreamingDecider
from sathealth.pipeline.decide import decide
from sathealth.pipeline.evaluate import evaluate_detector, evaluation_dir_for
from sathealth.pipeline.prepare import (
    load_prepared,
    load_run_manifest,
    prepare_dataset,
    run_dir_for,
)
from sathealth.service.app import create_app

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
    tmp = tmp_path_factory.mktemp("phase6")
    cfg = load_config(
        None,
        overrides=[
            f"paths.raw_dir={RAW}",
            f"paths.processed_dir={tmp / 'processed'}",
            f"paths.runs_dir={tmp / 'runs'}",
            f"paths.decisions_dir={tmp / 'decisions'}",
            f"paths.models_dir={tmp / 'models'}",
            f"service.db_path={tmp / 'service.db'}",
            "evaluation.bootstrap=20",
            "decision.inputs=[zscore]",
            "decision.recalibrate_every=100",
        ],
    )
    prepare_dataset(cfg)
    evaluate_detector(cfg, "zscore")
    return cfg


def test_live_decisions_equal_batch_decisions_on_every_channel(config: Config) -> None:
    run = decide(config).run_dir
    d = config.decision
    data = run_dir_for(config)
    for chan in load_run_manifest(data)["channels"]:
        channel = load_prepared(data, chan)
        with np.load(evaluation_dir_for(config, "zscore") / "scores" / f"{chan}.npz") as npz:
            val, test = npz["val"].astype(np.float64), npz["test"].astype(np.float64)
        train = channel.train[:, [0]]
        oor = outside_range(channel.test[:, [0]], train.min(0), train.max(0), d.point_margin)
        decider = StreamingDecider(d, {"zscore": val})
        events = []
        for t, score in enumerate(test):
            result = decider.step({"zscore": score}, bool(oor[t]))
            if result.closed:
                events.append(result.closed.to_dict())
        if (last := decider.finish()) is not None:
            events.append(last.to_dict())
        assert events == json.loads((run / "events" / f"{chan}.json").read_text()), chan


def test_service_replays_all_channels(config: Config) -> None:
    with TestClient(create_app(config)) as client:
        assert client.get("/readyz").json()["channels"] == 80
        updates = client.post("/api/replay/step", json={"steps": 500}).json()["updates"]
        assert updates
        health = client.get("/api/health").json()
        assert len(health["channels"]) == 80
        assert 0 <= health["overall"] <= 100
