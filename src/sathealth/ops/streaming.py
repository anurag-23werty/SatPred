"""The Phase 4 decision layer, one telemetry step at a time.

:func:`sathealth.pipeline.decide.decide` processes a whole series at once. A live system
receives one step at a time and must raise the alert the moment it fires. This class
implements the same rule incrementally:

* calibration on validation scores, refit every ``recalibrate_every`` steps on the
  validation anchor plus the most recent non-alarm scores (exactly the block schedule of
  :func:`sathealth.decision.adaptive.adaptive_surprises`);
* fusion and hysteresis with carried state;
* event merging: an event stays open through gaps of up to ``merge_gap`` steps and closes
  once a longer gap is certain (``merge_gap + 1`` quiet steps after the last alarm);
* final events built by the same :func:`~sathealth.decision.events.build_events` as the
  batch path, so they are identical field by field (pinned by a parity test).

Telemanom's dynamic threshold and pruning look at a whole window of future scores and are
therefore not supported here.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from sathealth.config import DecisionConfig
from sathealth.data.schema import FloatArray
from sathealth.decision.calibration import TailCalibrator
from sathealth.decision.events import Event, build_events
from sathealth.decision.fusion import fuse


@dataclass(frozen=True)
class StepResult:
    step: int
    fused: float
    surprises: dict[str, float]
    alarm: bool  # hysteresis state after this step
    opened: Event | None = None  # an alert fired at this step (provisional values)
    closed: Event | None = None  # an event is final as of this step


@dataclass
class _OpenEvent:
    start: int
    last_alarm: int
    announced: bool = False


@dataclass
class StreamingDecider:
    config: DecisionConfig
    validation: Mapping[str, FloatArray]  # nominal validation scores per input detector
    step_index: int = 0
    _active: bool = False
    _cal: dict[str, TailCalibrator] = field(default_factory=dict)
    _recent: dict[str, FloatArray] = field(default_factory=dict)
    _accepted: dict[str, list[float]] = field(default_factory=dict)
    _fused: list[float] = field(default_factory=list)
    _per_input: dict[str, list[float]] = field(default_factory=dict)
    _oor: list[bool] = field(default_factory=list)
    _open: _OpenEvent | None = None

    def __post_init__(self) -> None:
        d = self.config
        if d.threshold != "hysteresis" or d.prune is not None:
            raise ValueError("streaming supports the hysteresis rule without pruning only")
        if set(self.validation) != set(d.inputs):
            raise ValueError(f"validation scores for {sorted(self.validation)}, need {d.inputs}")
        for name in d.inputs:
            self._recent[name] = np.zeros(0)
            self._accepted[name] = []
            self._per_input[name] = []
        self._refit()

    def _refit(self) -> None:
        d = self.config
        for name in d.inputs:
            reference = np.r_[
                np.asarray(self.validation[name], dtype=np.float64), self._recent[name]
            ]
            self._cal[name] = TailCalibrator(d.calibration, d.tail_quantile, d.max_surprise).fit(
                reference
            )

    def _maybe_recalibrate(self) -> None:
        every = self.config.recalibrate_every
        if every is None or self.step_index == 0 or self.step_index % every:
            return
        window = self.config.recalibrate_window
        for name in self.config.inputs:
            merged = np.r_[self._recent[name], np.asarray(self._accepted[name])]
            self._recent[name] = merged[-window:]
            self._accepted[name] = []
        self._refit()

    def _event(self, start: int, end: int) -> Event:
        span = slice(start, end + 1)
        fused = np.asarray(self._fused[span])
        (event,) = build_events(
            np.ones(end - start + 1, dtype=np.bool_),
            fused,
            {k: np.asarray(v[span]) for k, v in self._per_input.items()},
            np.asarray(self._oor[span], dtype=np.bool_),
            warning=self.config.severity_warning,
            critical=self.config.severity_critical,
        )
        shift = start
        return Event(
            start=event.start + shift,
            end=event.end + shift,
            peak_step=event.peak_step + shift,
            peak_surprise=event.peak_surprise,
            severity=event.severity,
            kind=event.kind,
            contributions=event.contributions,
        )

    def step(self, scores: Mapping[str, float], out_of_range: bool = False) -> StepResult:
        """Consume the detector scores of the next step; return what changed."""
        d = self.config
        self._maybe_recalibrate()
        t = self.step_index
        surprises = {
            name: float(self._cal[name].surprise(np.array([scores[name]], dtype=np.float64))[0])
            for name in d.inputs
        }
        fused = float(fuse([np.array([surprises[n]]) for n in d.inputs], d.fusion)[0])
        self._active = fused >= d.alarm_off if self._active else fused >= d.alarm_on
        if not self._active and d.recalibrate_every is not None:
            for name in d.inputs:
                self._accepted[name].append(float(scores[name]))
        self._fused.append(fused)
        for name in d.inputs:
            self._per_input[name].append(surprises[name])
        self._oor.append(bool(out_of_range))

        opened = closed = None
        if self._active:
            if self._open is not None and t - self._open.last_alarm - 1 <= d.merge_gap:
                self._open.last_alarm = t
            else:
                closed = self._close()
                self._open = _OpenEvent(start=t, last_alarm=t)
            if not self._open.announced and t - self._open.start + 1 >= d.min_duration:
                self._open.announced = True
                opened = self._event(self._open.start, t)
        elif self._open is not None and t - self._open.last_alarm - 1 >= d.merge_gap:
            closed = self._close()
        self.step_index += 1
        return StepResult(t, fused, surprises, self._active, opened, closed)

    def _close(self) -> Event | None:
        current, self._open = self._open, None
        if current is None:
            return None
        if current.last_alarm - current.start + 1 < self.config.min_duration:
            return None
        return self._event(current.start, current.last_alarm)

    def finish(self) -> Event | None:
        """End of stream: close the event that is still open, if any."""
        return self._close()

    @property
    def fused(self) -> FloatArray:
        return np.asarray(self._fused, dtype=np.float64)
