"""Label-free drift monitoring per channel.

Two signals, both computable without labels (ADR 0009 D):

* **Population stability index** of the recent values against the training distribution,
  over training-quantile bins: ``PSI = Σ (q − p) · ln(q / p)``. The conventional reading is
  < 0.1 stable, 0.1–0.25 moderate shift, > 0.25 significant shift.
* **Alarm share**: the share of recent steps in alarm, reported as an indicator only.

The first version flagged a channel when PSI exceeded the conventional 0.25 or the alarm
share exceeded 10× the designed rate. On SMAP/MSL that flagged 78 of 80 channels: PSI
assumes independent samples, and autocorrelated telemetry that moves between operating modes
never fills a 1,000-step window with the whole training distribution. The flag is therefore
**self-calibrated per channel**: a channel drifts when its recent values differ from training
more than *any* training window of the same length differs from training as a whole
(and by at least the conventional 0.25). The alarm share no longer raises the flag: it
restates what the alarm system already shows.

Neither signal can tell an anomaly from a new normal. That is the operator's call, and
:meth:`~sathealth.ops.streaming.StreamingDecider.rebaseline` acts on it.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from sathealth.data.schema import FloatArray

PSI_SIGNIFICANT = 0.25
EPS = 1e-4


def quantile_edges(values: FloatArray, bins: int = 10) -> FloatArray:
    """Interior bin edges at the training quantiles (duplicates removed for discrete data)."""
    edges: FloatArray = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
    return edges


def psi(expected: FloatArray, actual: FloatArray) -> float:
    """PSI between two distributions over the same bins (counts or probabilities)."""
    p = np.maximum(
        np.asarray(expected, dtype=np.float64) / max(float(np.sum(expected)), 1e-12), EPS
    )
    q = np.maximum(np.asarray(actual, dtype=np.float64) / max(float(np.sum(actual)), 1e-12), EPS)
    return float(np.sum((q - p) * np.log(q / p)))


@dataclass(frozen=True)
class DriftReport:
    psi: float
    alarm_share: float
    window: int
    drifting: bool
    reasons: tuple[str, ...]
    threshold: float = PSI_SIGNIFICANT


def within_training_psi(
    values: FloatArray, edges: FloatArray, expected: FloatArray, window: int
) -> float:
    """Largest PSI of a training window (half-overlapping) against all of training."""
    bins = np.searchsorted(edges, values, side="right")
    worst = 0.0
    for start in range(0, max(values.size - window, 0) + 1, max(window // 2, 1)):
        chunk = np.bincount(bins[start : start + window], minlength=expected.size)
        worst = max(worst, psi(expected, chunk.astype(np.float64)))
    return worst


class DriftMonitor:
    def __init__(self, train_values: FloatArray, window: int = 1000, bins: int = 10) -> None:
        values = np.asarray(train_values, dtype=np.float64).ravel()
        self.edges = quantile_edges(values, bins)
        self.expected = np.bincount(
            np.searchsorted(self.edges, values, side="right"), minlength=self.edges.size + 1
        ).astype(np.float64)
        self.window = window
        self.threshold = max(
            PSI_SIGNIFICANT, within_training_psi(values, self.edges, self.expected, window)
        )
        self._bins: deque[int] = deque(maxlen=window)
        self._alarms: deque[bool] = deque(maxlen=window)

    def update(self, value: float, alarm: bool) -> None:
        self._bins.append(int(np.searchsorted(self.edges, value, side="right")))
        self._alarms.append(alarm)

    def report(self, min_steps: int = 100) -> DriftReport:
        n = len(self._bins)
        if n < min_steps:
            return DriftReport(0.0, 0.0, n, False, (), round(self.threshold, 4))
        actual = np.bincount(
            np.fromiter(self._bins, dtype=np.int64), minlength=self.expected.size
        ).astype(np.float64)
        value_psi = psi(self.expected, actual)
        share = float(np.mean(np.fromiter(self._alarms, dtype=bool)))
        reasons = []
        if value_psi > self.threshold:
            reasons.append(
                f"values differ from training more than any training window does "
                f"(PSI {value_psi:.2f} > {self.threshold:.2f})"
            )
        return DriftReport(
            round(value_psi, 4),
            round(share, 4),
            n,
            bool(reasons),
            tuple(reasons),
            round(self.threshold, 4),
        )
