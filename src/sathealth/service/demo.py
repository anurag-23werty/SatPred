"""Headless end-to-end demo: replayed stream → explained alert → operator verdict stored.

Runs the same engine and store as the service, without a browser or HTTP, and reports the
replay throughput and explanation latency. This is the Phase 6 exit criterion as a command.

The demo explains the first closed critical alert that starts after the models' warm-up
window. Each test series starts without history, so the first ``window`` steps produce a
start-up transient (10% of the live pipeline's events, mostly false; ADR 0008). Those
alerts are still raised and stored; the demo just does not pick one as its example.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sathealth.config import Config
from sathealth.service.replay import ReplayEngine
from sathealth.service.store import Store

CHUNK = 250


@dataclass(frozen=True)
class DemoResult:
    replay_id: int
    channels: int
    steps: int
    replay_seconds: float
    events_opened: int
    first_alert: dict[str, Any]
    explanation: dict[str, Any]
    explanation_seconds: float
    verdict_id: int
    stored_verdict: dict[str, Any]

    @property
    def channel_steps_per_second(self) -> float:
        return self.channels * self.steps / max(self.replay_seconds, 1e-9)


def run_demo(
    config: Config, *, db_path: Path, channels: list[str] | None = None, operator: str = "demo"
) -> DemoResult:
    """Replay until the first *closed* critical alert (or the end), explain it, judge it."""
    store = Store(db_path)
    try:
        engine = ReplayEngine(config, store)
        replay_id = engine.reset(channels)
        warm_up = config.deep.lstm.window
        started = time.perf_counter()
        target: dict[str, Any] | None = None
        while target is None and not engine.finished:
            for update in engine.advance(CHUNK):
                if (
                    update["type"] == "closed"
                    and update["severity"] == "critical"
                    and update["start"] >= warm_up
                ):
                    target = update
                    break
        replay_seconds = time.perf_counter() - started
        if target is None:
            closed = store.events(replay_id, status="closed", limit=1)
            if not closed:
                raise RuntimeError("the replay raised no alert")
            row = closed[0]
            target = {**row, "event_id": row["id"]}
        t0 = time.perf_counter()
        explanation = engine.explain(target["event_id"])
        explanation_seconds = time.perf_counter() - t0
        verdict_id = store.add_verdict(
            target["event_id"], "confirmed", operator, "headless demo verdict"
        )
        stored = store.verdicts(target["event_id"])[-1]
        return DemoResult(
            replay_id=replay_id,
            channels=engine.status()["channels"],
            steps=engine.step,
            replay_seconds=replay_seconds,
            events_opened=engine.counters.events_opened,
            first_alert=target,
            explanation=explanation,
            explanation_seconds=explanation_seconds,
            verdict_id=verdict_id,
            stored_verdict=stored,
        )
    finally:
        store.close()
