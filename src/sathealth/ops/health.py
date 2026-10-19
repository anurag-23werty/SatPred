"""Health index per channel and subsystem, and an honestly labelled trend extrapolation.

**Health index.** ``100 × (1 − w)``, where ``w`` is the severity-weighted share of the last
``window`` steps spent in alarm (watch 0.25, warning 0.5, critical 1, by the fused surprise
of each step). 100 means no alarm recently; 0 means a critical alarm throughout the window.
It only moves when alarms do, so an operator can always trace it back to events.

**Subsystems.** NASA anonymised SMAP/MSL; the channel family letter is the only grouping in
the data and stands in for the subsystem. A subsystem's index is the *minimum* over its
channels (the weakest link), reported with the number of channels currently in alarm.

**No remaining useful life.** SMAP/MSL contain no run-to-failure histories, so no RUL model
can be validated on them. :func:`time_to_threshold` instead extrapolates the recent linear
trend of the health index to a threshold, with a residual-bootstrap interval, and is
labelled an extrapolation wherever it is shown.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from sathealth.config import DecisionConfig
from sathealth.data.schema import FloatArray

SEVERITY_WEIGHTS = {"watch": 0.25, "warning": 0.5, "critical": 1.0}


def step_weight(fused: float, alarm: bool, d: DecisionConfig) -> float:
    if not alarm:
        return 0.0
    if fused >= d.severity_critical:
        return SEVERITY_WEIGHTS["critical"]
    if fused >= d.severity_warning:
        return SEVERITY_WEIGHTS["warning"]
    return SEVERITY_WEIGHTS["watch"]


class HealthTracker:
    """Rolling health index of one channel, plus a sparse history for trends."""

    def __init__(self, window: int = 500, sample_every: int = 50, history: int = 200) -> None:
        if window < 1 or sample_every < 1:
            raise ValueError("window and sample_every must be >= 1")
        self.window = window
        self.sample_every = sample_every
        self._weights: deque[float] = deque(maxlen=window)
        self._sum = 0.0
        self._steps = 0
        self.history: deque[tuple[int, float]] = deque(maxlen=history)

    def update(self, weight: float) -> float:
        if len(self._weights) == self.window:
            self._sum -= self._weights[0]
        self._weights.append(weight)
        self._sum += weight
        self._steps += 1
        index = self.index
        if self._steps % self.sample_every == 0:
            self.history.append((self._steps - 1, index))
        return index

    @property
    def index(self) -> float:
        if not self._weights:
            return 100.0
        return round(100.0 * (1.0 - max(self._sum, 0.0) / len(self._weights)), 2)

    @property
    def in_alarm(self) -> bool:
        return bool(self._weights) and self._weights[-1] > 0


@dataclass(frozen=True)
class TrendForecast:
    """Steps until the index reaches ``threshold`` at its current linear trend."""

    slope_per_1k_steps: float
    steps_to_threshold: float | None  # None: trend is flat or improving
    interval80: tuple[float, float] | None
    label: str = "trend extrapolation, not a validated remaining-useful-life estimate"


def time_to_threshold(
    steps: FloatArray,
    index: FloatArray,
    threshold: float = 50.0,
    *,
    n_boot: int = 500,
    seed: int = 0,
    min_downward: float = 0.9,
    max_extrapolation: float = 5.0,
) -> TrendForecast | None:
    """Least-squares trend of the recent index, extrapolated to ``threshold``.

    A time is reported only when the trend is reliably downward (at least ``min_downward``
    of residual-bootstrap slopes are negative) and the extrapolation reaches no further than
    ``max_extrapolation`` times the observed history; otherwise ``steps_to_threshold`` is
    None ("no meaningful trend"). Returns None with fewer than 5 points.
    """
    x = np.asarray(steps, dtype=np.float64)
    y = np.asarray(index, dtype=np.float64)
    if x.size < 5:
        return None
    xc = x - x.mean()
    sxx = float((xc**2).sum())
    if sxx == 0:
        return None
    slope = float((xc * (y - y.mean())).sum() / sxx)
    intercept = float(y.mean() - slope * x.mean())
    flat = TrendForecast(round(slope * 1000, 3), None, None)
    now = x[-1]
    if slope >= 0 or slope * now + intercept <= threshold:
        return flat
    residuals = y - (slope * x + intercept)
    rng = np.random.default_rng(seed)
    yb = slope * x + intercept + rng.choice(residuals, size=(n_boot, x.size), replace=True)
    slopes = (yb - yb.mean(axis=1, keepdims=True)) @ xc / sxx
    intercepts = yb.mean(axis=1) - slopes * x.mean()
    if np.mean(slopes < 0) < min_downward:
        return flat
    point = (threshold - intercept) / slope - now
    if point > max_extrapolation * (x[-1] - x[0]):
        return flat
    down = slopes < 0
    times = (threshold - intercepts[down]) / slopes[down] - now
    lo, hi = np.percentile(times, [10, 90])
    return TrendForecast(
        round(slope * 1000, 3),
        round(float(point), 1),
        (round(max(float(lo), 0.0), 1), round(float(hi), 1)),
    )


def subsystem(chan_id: str) -> str:
    """Channel family letter (``A`` for ``A-1``): the only grouping NASA's data provides."""
    return chan_id.split("-", 1)[0]
