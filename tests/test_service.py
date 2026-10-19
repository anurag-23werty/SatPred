import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sathealth.config import Config, load_config
from sathealth.data.download import Manifest
from sathealth.pipeline.evaluate import evaluate_detector
from sathealth.pipeline.prepare import prepare_dataset
from sathealth.service.app import Broker, create_app


def _config(raw_root: Path, tmp_path: Path, *extra: str) -> Config:
    return load_config(
        None,
        overrides=[
            f"paths.raw_dir={raw_root}",
            f"paths.processed_dir={tmp_path / 'processed'}",
            f"paths.runs_dir={tmp_path / 'runs'}",
            f"paths.models_dir={tmp_path / 'models'}",
            f"service.db_path={tmp_path / 'service.db'}",
            "decision.inputs=[zscore]",
            "decision.recalibrate_every=100",
            *extra,
        ],
    )


@pytest.fixture
def config(raw_root: Path, raw_manifest: Manifest, tmp_path: Path) -> Config:
    cfg = _config(raw_root, tmp_path, "service.show_labels=true")
    prepare_dataset(cfg, manifest=raw_manifest)
    evaluate_detector(cfg, "zscore")
    return cfg


@pytest.fixture
def client(config: Config) -> Iterator[TestClient]:
    with TestClient(create_app(config)) as c:
        yield c


def _alert(client: TestClient) -> dict[str, object]:
    client.post("/api/replay", json={"run": False})
    updates = client.post("/api/replay/step", json={"steps": 400}).json()["updates"]
    closed = [u for u in updates if u["type"] == "closed" and u["chan_id"] == "A-1"]
    assert closed, "the injected spike in A-1 must raise an alert"
    return closed[0]


def test_probes_status_and_metrics(client: TestClient) -> None:
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz").json()
    assert ready["status"] == "ready"
    assert ready["channels"] == 3
    status = client.get("/api/status").json()
    assert (status["step"], status["running"]) == (0, False)
    text = client.get("/metrics").text
    assert "# TYPE sathealth_replay_steps_total counter" in text
    assert "sathealth_health_index_min 100.0" in text


def test_replay_controls(client: TestClient) -> None:
    started = client.post("/api/replay", json={"channels": ["A-1"], "speed": 50, "run": False})
    assert started.json()["channels"] == 1
    assert client.put("/api/replay/speed", json={"speed": 7}).json()["speed"] == 7
    assert client.post("/api/replay/resume").json()["running"] is True
    assert client.post("/api/replay/pause").json()["running"] is False
    stepped = client.post("/api/replay/step", json={"steps": 10}).json()
    assert stepped["status"]["step"] == 10
    assert client.post("/api/replay", json={"channels": ["Z-9"]}).status_code == 404
    assert client.put("/api/replay/speed", json={"speed": 0}).status_code == 422


def test_replayed_stream_explained_alert_operator_verdict(client: TestClient) -> None:
    """The Phase 6 exit criterion, through the HTTP API."""
    alert = _alert(client)
    event_id = alert["event_id"]
    assert alert["start"] <= 320
    assert alert["end"] >= 300
    event = client.get(f"/api/events/{event_id}").json()
    assert event["status"] == "closed"
    explanation = client.get(f"/api/events/{event_id}/explanation").json()
    assert explanation["grounding_violations"] == []
    assert explanation["as_of_step"] == 400
    assert explanation["sentences"][0]["text"].startswith(f"{event['severity'].capitalize()} alert")
    assert "counterfactual" not in explanation, "no LSTM was evaluated in this setup"
    assert client.get(f"/api/events/{event_id}/explanation").json() == explanation  # cached
    posted = client.post(
        f"/api/events/{event_id}/verdict",
        json={"verdict": "confirmed", "operator": "op", "note": "spike"},
    )
    assert posted.status_code == 201
    stored = client.get(f"/api/events/{event_id}/verdicts").json()
    assert [(v["verdict"], v["operator"], v["note"]) for v in stored] == [
        ("confirmed", "op", "spike")
    ]
    assert client.get(f"/api/events/{event_id}").json()["verdict"] == "confirmed"
    summary = client.get("/api/feedback/summary").json()
    assert summary["all"]["confirmed"] == 1
    exported = client.get("/api/feedback/export").json()
    assert [e["id"] for e in exported] == [event_id]


def test_event_queries_and_errors(client: TestClient) -> None:
    _alert(client)
    events = client.get("/api/events").json()
    assert events
    assert all(e["replay_id"] == events[0]["replay_id"] for e in events)
    assert client.get("/api/events", params={"status": "closed", "chan_id": "A-1"}).json()
    assert client.get("/api/events", params={"severity": "nope"}).status_code == 422
    assert client.get("/api/events/999").status_code == 404
    assert client.get("/api/events/999/explanation").status_code == 404
    assert (
        client.post(
            "/api/events/999/verdict", json={"verdict": "confirmed", "operator": "op"}
        ).status_code
        == 404
    )
    bad = client.post(
        f"/api/events/{events[0]['id']}/verdict", json={"verdict": "x", "operator": ""}
    )
    assert bad.status_code == 422


def test_health_channels_and_telemetry(client: TestClient) -> None:
    _alert(client)
    health = client.get("/api/health").json()
    assert {s["subsystem"] for s in health["subsystems"]} == {"A", "C", "M"}
    assert health["overall"] == min(c["health"] for c in health["channels"])
    assert health["healthy_channels"] == sum(c["health"] >= 90 for c in health["channels"])
    assert health["total_channels"] == 3
    channels = client.get("/api/channels").json()
    assert {c["chan_id"] for c in channels} == {"A-1", "C-1", "M-1"}
    t = client.get("/api/channels/A-1/telemetry", params={"window": 100}).json()
    assert (t["start"], t["now"]) == (300, 400)
    assert len(t["values"]) == len(t["surprise"]) == len(t["alarm"]) == len(t["labels"]) == 100
    assert client.get("/api/channels/Z-9/telemetry").status_code == 404


def test_labels_are_hidden_by_default(raw_root: Path, tmp_path: Path, config: Config) -> None:
    hidden = config.model_copy(
        update={"service": config.service.model_copy(update={"show_labels": False})}
    )
    with TestClient(create_app(hidden)) as c:
        c.post("/api/replay/step", json={"steps": 5})
        assert "labels" not in c.get("/api/channels/A-1/telemetry").json()


def test_live_stream_pushes_alerts(client: TestClient) -> None:
    with client.stream("GET", "/api/stream") as stream:
        lines = stream.iter_lines()
        assert next(lines) == "event: status"
        client.post("/api/replay/step", json={"steps": 400})
        kinds = []
        for line in lines:
            if line.startswith("event: "):
                kinds.append(line.removeprefix("event: "))
            if line.startswith("data: ") and kinds and kinds[-1] == "alert":
                assert {"event_id", "chan_id", "severity"} <= json.loads(line[6:]).keys()
                break
    assert "alert" in kinds


def test_console_is_served(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "operator console" in page.text
    assert client.get("/static/console.js").status_code == 200
    assert client.get("/static/console.css").status_code == 200
    assert "/api/events/{event_id}/verdict" in client.get("/openapi.json").json()["paths"]


def test_not_ready_without_saved_scores(
    raw_root: Path, raw_manifest: Manifest, tmp_path: Path
) -> None:
    cfg = _config(raw_root, tmp_path)
    prepare_dataset(cfg, manifest=raw_manifest)  # no `evaluate` run
    with TestClient(create_app(cfg)) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/readyz").status_code == 503
        assert "sathealth evaluate zscore" in c.get("/api/status").json()["detail"]


def test_broker_drops_for_stalled_clients() -> None:
    broker = Broker()
    queue = broker.subscribe()
    for i in range(1001):
        broker.publish("alert", {"i": i})
    assert queue.qsize() == 1000
    assert broker.dropped == 1
    broker.unsubscribe(queue)
    assert broker.subscribers == 0


def test_background_runner_advances_a_running_replay(client: TestClient) -> None:
    import time

    client.post("/api/replay", json={"speed": 2000, "run": True})
    deadline = time.monotonic() + 5
    while client.get("/api/status").json()["step"] == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    status = client.get("/api/status").json()
    assert status["step"] > 0
    client.post("/api/replay/pause")


def test_demo_runs_the_exit_criterion(config: Config, tmp_path: Path) -> None:
    from sathealth.service.demo import run_demo

    result = run_demo(config, db_path=tmp_path / "demo.db")
    assert result.events_opened >= 1
    assert result.explanation["grounding_violations"] == []
    assert result.stored_verdict["verdict"] == "confirmed"
    assert result.stored_verdict["event_id"] == result.first_alert["event_id"]
    assert result.channel_steps_per_second > 0


def test_drift_and_rebaseline_endpoints(client: TestClient) -> None:
    client.post("/api/replay", json={"run": False})
    client.post("/api/replay/step", json={"steps": 200})
    drift = client.get("/api/drift").json()
    assert {d["chan_id"] for d in drift} == {"A-1", "C-1", "M-1"}
    assert all({"psi", "alarm_share", "drifting", "reasons"} <= d.keys() for d in drift)
    assert "drifting" in client.get("/api/channels").json()[0]
    done = client.post("/api/channels/A-1/rebaseline", json={"operator": "op", "note": "new mode"})
    assert done.status_code == 201
    assert done.json()["steps"] == 500
    audit = client.get("/api/channels/A-1/rebaselines").json()
    assert [(a["operator"], a["note"], a["step"]) for a in audit] == [("op", "new mode", 200)]
    assert client.post("/api/channels/Z-9/rebaseline", json={"operator": "op"}).status_code == 404
    assert client.post("/api/channels/A-1/rebaseline", json={"operator": ""}).status_code == 422
    client.post("/api/replay", json={"run": False})
    early = client.post("/api/channels/A-1/rebaseline", json={"operator": "op"})
    assert early.status_code == 409, "too little history to re-baseline on"
