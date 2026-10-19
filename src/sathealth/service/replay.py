"""Replay engine: telemetry steps in, alerts, health and explanations out.

Every channel has its own :class:`~sathealth.ops.streaming.StreamingDecider` and
:class:`~sathealth.ops.health.HealthTracker`. :meth:`ReplayEngine.advance` moves all
channels forward by ``n`` steps and returns the alert updates (event opened or closed),
which the service stores and pushes to the console.

What is replayed are the detector scores saved by ``sathealth evaluate``: each is causal
(computed from past steps only). Calibration, alarms, events, health and explanations are
computed live. Explanations only use data up to the current step.
"""

import dataclasses
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sathealth.config import Config
from sathealth.data.schema import BoolArray, FloatArray
from sathealth.decision.events import Event, outside_range
from sathealth.detectors.lstm import LSTMDetector
from sathealth.explain.counterfactual import HorizonBand
from sathealth.logging import get_logger
from sathealth.ops.drift import DriftMonitor
from sathealth.ops.health import HealthTracker, step_weight, subsystem, time_to_threshold
from sathealth.ops.streaming import StreamingDecider
from sathealth.pipeline.evaluate import evaluation_dir_for
from sathealth.pipeline.explain import channel_band, explain_event, fitted_lstm
from sathealth.pipeline.prepare import (
    PreparedChannel,
    load_prepared,
    load_run_manifest,
    run_dir_for,
)
from sathealth.service.store import Store

log = get_logger(__name__)

HEALTHY = 90.0  # a channel at or above this index counts as healthy in the summary


@dataclass(frozen=True)
class ChannelFeed:
    chan_id: str
    spacecraft: str
    val: dict[str, FloatArray]
    test: dict[str, FloatArray]
    out_of_range: BoolArray
    values: FloatArray  # first value column, dataset units
    labels: BoolArray
    train_values: FloatArray  # first value column of the training split, dataset units

    @property
    def length(self) -> int:
        return int(self.values.size)


def load_feed(config: Config, chan_id: str) -> ChannelFeed:
    d = config.decision
    channel = load_prepared(run_dir_for(config), chan_id)
    val, test = {}, {}
    for name in d.inputs:
        path = evaluation_dir_for(config, name) / "scores" / f"{chan_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"no saved {name!r} scores; run `sathealth evaluate {name}`")
        with np.load(path) as npz:
            val[name] = npz["val"].astype(np.float64)
            test[name] = npz["test"].astype(np.float64)
    cols = list(config.preprocess.value_columns)
    train = channel.train[:, cols]
    oor = outside_range(channel.test[:, cols], train.min(axis=0), train.max(axis=0), d.point_margin)
    p = channel.scaler
    values = channel.test[:, cols[0]] * p.scale[0] + p.center[0]
    train_values = channel.train[:, cols[0]] * p.scale[0] + p.center[0]
    return ChannelFeed(
        chan_id, channel.spacecraft, val, test, oor, values, channel.test_labels, train_values
    )


@dataclass
class _ChannelState:
    feed: ChannelFeed
    decider: StreamingDecider
    health: HealthTracker
    drift: DriftMonitor
    alarm: list[bool] = field(default_factory=list)
    open_ids: dict[int, int] = field(default_factory=dict)  # event start -> event id


@dataclass
class Counters:
    steps: int = 0
    events_opened: int = 0
    events_closed: int = 0
    explanations: int = 0
    explanation_seconds: float = 0.0


class ReplayEngine:
    def __init__(self, config: Config, store: Store) -> None:
        self.config = config
        self.store = store
        self.counters = Counters()
        self._lock = threading.RLock()
        self._feeds: dict[str, ChannelFeed] = {}
        self._states: dict[str, _ChannelState] = {}
        self._bands: dict[str, tuple[LSTMDetector | None, HorizonBand | None]] = {}
        self.replay_id: int | None = None
        self.step = 0
        self.total = 0
        self.speed = config.service.speed
        self.running = False

    # --- lifecycle -----------------------------------------------------------------------

    def available_channels(self) -> list[str]:
        manifest = load_run_manifest(run_dir_for(self.config))
        configured = self.config.service.channels
        return [c for c in manifest["channels"] if configured is None or c in configured]

    def reset(self, channels: Sequence[str] | None = None) -> int:
        """Start a new replay from step 0 on ``channels`` (default: the configured set)."""
        chosen = list(channels) if channels else self.available_channels()
        unknown = set(chosen) - set(self.available_channels())
        if unknown:
            raise KeyError(f"unknown channels: {sorted(unknown)}")
        s = self.config.service
        with self._lock:
            if self.replay_id is not None:
                self.store.finish_replay(self.replay_id)
            states = {}
            for chan in chosen:
                if chan not in self._feeds:
                    self._feeds[chan] = load_feed(self.config, chan)
                feed = self._feeds[chan]
                states[chan] = _ChannelState(
                    feed,
                    StreamingDecider(self.config.decision, feed.val),
                    HealthTracker(s.health_window, s.health_sample_every),
                    DriftMonitor(feed.train_values, s.drift_window),
                )
            self._states = states
            self.step = 0
            self.total = max(st.feed.length for st in states.values())
            self.replay_id = self.store.start_replay(self.config.decision.name, chosen)
            self.running = False
        log.info("replay reset", replay_id=self.replay_id, channels=len(chosen))
        return self.replay_id

    @property
    def finished(self) -> bool:
        return self.replay_id is not None and self.step >= self.total

    def advance(self, n: int) -> list[dict[str, Any]]:
        """Move every channel forward ``n`` steps; return alert updates in step order."""
        if self.replay_id is None:
            self.reset()
        replay_id = self.replay_id
        assert replay_id is not None
        updates: list[dict[str, Any]] = []
        with self._lock:
            for _ in range(n):
                if self.step >= self.total:
                    break
                t = self.step
                for chan, st in self._states.items():
                    if t < st.feed.length:
                        updates += self._step_channel(replay_id, chan, st, t)
                        if t == st.feed.length - 1:
                            updates += self._close(replay_id, chan, st, st.decider.finish(), t)
                self.step += 1
                self.counters.steps += len(self._states)
            if self.step >= self.total:
                self.running = False
                self.store.finish_replay(replay_id)
        return updates

    def _step_channel(
        self, replay_id: int, chan: str, st: _ChannelState, t: int
    ) -> list[dict[str, Any]]:
        feed = st.feed
        result = st.decider.step(
            {name: feed.test[name][t] for name in feed.test}, bool(feed.out_of_range[t])
        )
        st.alarm.append(result.alarm)
        st.health.update(step_weight(result.fused, result.alarm, self.config.decision))
        st.drift.update(float(feed.values[t]), result.alarm)
        updates = self._close(replay_id, chan, st, result.closed, t)
        if result.opened is not None:
            event_id = self.store.open_event(replay_id, chan, feed.spacecraft, result.opened)
            st.open_ids[result.opened.start] = event_id
            self.counters.events_opened += 1
            updates.append(self._update("opened", event_id, chan, feed, result.opened, t))
        return updates

    def _close(
        self, replay_id: int, chan: str, st: _ChannelState, event: Event | None, t: int
    ) -> list[dict[str, Any]]:
        if event is None:
            return []
        event_id = self.store.close_event(replay_id, chan, event)
        st.open_ids.pop(event.start, None)
        self.counters.events_closed += 1
        return [self._update("closed", event_id, chan, st.feed, event, t)]

    @staticmethod
    def _update(
        kind: str, event_id: int, chan: str, feed: ChannelFeed, event: Event, t: int
    ) -> dict[str, Any]:
        return {
            "type": kind,
            "event_id": event_id,
            "chan_id": chan,
            "spacecraft": feed.spacecraft,
            "step": t,
            **event.to_dict(),
        }

    # --- read models ---------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "replay_id": self.replay_id,
            "running": self.running,
            "finished": self.finished,
            "step": self.step,
            "total": self.total,
            "speed": self.speed,
            "channels": len(self._states),
            "decision": self.config.decision.name,
        }

    def channel_health(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "chan_id": chan,
                    "spacecraft": st.feed.spacecraft,
                    "subsystem": subsystem(chan),
                    "health": st.health.index,
                    "in_alarm": st.health.in_alarm,
                    "open_events": len(st.open_ids),
                    "drifting": st.drift.report().drifting,
                }
                for chan, st in self._states.items()
            ]

    def health(self) -> dict[str, Any]:
        """Channels, subsystems (weakest link) and a labelled trend extrapolation."""
        channels = self.channel_health()
        threshold = self.config.service.health_threshold
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in channels:
            groups.setdefault(row["subsystem"], []).append(row)
        subsystems = []
        with self._lock:
            for name, rows in sorted(groups.items()):
                worst = min(rows, key=lambda r: r["health"])
                history = self._states[worst["chan_id"]].health.history
                trend = (
                    time_to_threshold(
                        np.array([h[0] for h in history], dtype=np.float64),
                        np.array([h[1] for h in history], dtype=np.float64),
                        threshold,
                    )
                    if history
                    else None
                )
                subsystems.append(
                    {
                        "subsystem": name,
                        "health": worst["health"],
                        "weakest_channel": worst["chan_id"],
                        "channels": len(rows),
                        "in_alarm": sum(r["in_alarm"] for r in rows),
                        "trend": None if trend is None else dataclasses.asdict(trend),
                    }
                )
        overall = min((c["health"] for c in channels), default=100.0)
        healthy = sum(c["health"] >= HEALTHY for c in channels)
        return {
            "overall": overall,
            "healthy_channels": healthy,
            "total_channels": len(channels),
            "subsystems": subsystems,
            "channels": channels,
        }

    def drift(self) -> list[dict[str, Any]]:
        """Label-free drift report per channel (ADR 0009 D)."""
        with self._lock:
            return [
                {"chan_id": chan, **dataclasses.asdict(st.drift.report())}
                for chan, st in self._states.items()
            ]

    def rebaseline(self, chan: str, operator: str, note: str = "") -> dict[str, Any]:
        """Accept the channel's recent behaviour as its new normal (operator decision)."""
        steps = self.config.service.rebaseline_steps
        with self._lock:
            st = self._states.get(chan)
            if st is None or self.replay_id is None:
                raise KeyError(f"channel {chan!r} is not in this replay")
            closed = st.decider.rebaseline(steps)
            updates = self._close(self.replay_id, chan, st, closed, max(self.step - 1, 0))
            closed_id = updates[0]["event_id"] if updates else None
            audit_id = self.store.add_rebaseline(
                self.replay_id, chan, self.step, steps, operator, note, closed_id
            )
        log.info("re-baselined", channel=chan, operator=operator, step=self.step)
        return {
            "id": audit_id,
            "chan_id": chan,
            "step": self.step,
            "steps": steps,
            "closed_event_id": closed_id,
            "updates": updates,
        }

    def telemetry(self, chan: str, window: int = 600) -> dict[str, Any]:
        with self._lock:
            st = self._states.get(chan)
            if st is None:
                raise KeyError(f"channel {chan!r} is not in this replay")
            now = min(self.step, st.feed.length)
            start = max(0, now - window)
            out: dict[str, Any] = {
                "chan_id": chan,
                "spacecraft": st.feed.spacecraft,
                "start": start,
                "now": now,
                "values": [round(float(v), 5) for v in st.feed.values[start:now]],
                "surprise": [round(float(v), 3) for v in st.decider.fused[start:now]],
                "alarm": st.alarm[start:now],
                "alarm_on": self.config.decision.alarm_on,
            }
            if self.config.service.show_labels:
                out["labels"] = [bool(x) for x in st.feed.labels[start:now]]
            return out

    # --- explanations --------------------------------------------------------------------

    def explain(self, event_id: int) -> dict[str, Any]:
        """Explanation of an event using data up to the current step only (cached once the
        event is closed)."""
        cached = self.store.explanation(event_id)
        if cached is not None:
            return cached
        row = self.store.event(event_id)
        if row is None:
            raise KeyError(f"no event {event_id}")
        started = time.perf_counter()
        with self._lock:
            st = self._states.get(row["chan_id"])
            if st is None or row["replay_id"] != self.replay_id:
                raise KeyError(f"event {event_id} is not part of the current replay")
            now = min(self.step, st.feed.length)
            surprise = st.decider.fused[:now].copy()
        channel = load_prepared(run_dir_for(self.config), row["chan_id"])
        visible = dataclasses.replace(
            channel, test=channel.test[:now], test_labels=channel.test_labels[:now]
        )
        det, band = self._model(row["chan_id"], channel)
        event = {
            k: row[k]
            for k in (
                "start",
                "end",
                "peak_step",
                "peak_surprise",
                "severity",
                "kind",
                "contributions",
            )
        }
        explanation, arrays = explain_event(self.config, det, band, visible, event, surprise)
        explanation["as_of_step"] = now
        explanation["status"] = row["status"]
        explanation["series"] = {
            k: [round(float(x), 5) for x in v[:, 0]] for k, v in arrays.items() if v.ndim == 2
        }
        if row["status"] == "closed":
            self.store.save_explanation(event_id, explanation)
        self.counters.explanations += 1
        self.counters.explanation_seconds += time.perf_counter() - started
        return explanation

    def _model(
        self, chan: str, channel: PreparedChannel
    ) -> tuple[LSTMDetector | None, HorizonBand | None]:
        """The verified LSTM and its band, or (None, None) when no LSTM was evaluated."""
        if chan in self._bands:
            return self._bands[chan]
        try:
            det = fitted_lstm(self.config, channel)
        except FileNotFoundError:
            self._bands[chan] = (None, None)
            return None, None
        rng = np.random.default_rng([self.config.seed, int.from_bytes(chan.encode(), "little")])
        pair = (det, channel_band(self.config, det, channel, rng))
        self._bands[chan] = pair
        return pair
