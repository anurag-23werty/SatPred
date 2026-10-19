"""Isolation Forest on hand-crafted window features.

Each window of ``window`` steps is summarised per value column by: last value, mean, std,
min, max, net change (last - first) and mean absolute step change (roughness), plus the
number of commands that fired in the window, since the same telemetry level can be nominal
right after a command and anomalous without one.

Two lessons from building this baseline, both pinned by tests:

* A feature for "a command fired at the last step" was removed. Isolation Forest isolates
  *rare* values by design, so every nominal command became a false alarm.
* Isolation Forest **saturates outside the training range**: splits are drawn inside the
  range seen in training, so a value far beyond it lands in the same leaf as the most extreme
  training point. It can say "unusual" but not "much further out than ever before". The
  forecasting models of Phase 3 do not have this ceiling.

Features are computed vectorised over zero-copy window views (cumulative sums for command
counts), so a full SMAP/MSL pass takes seconds. Isolation Forest splits on one feature at a
time, so features need no scaling. Score = ``-score_samples`` (higher = more isolated).
"""

from typing import Any, Self

import numpy as np
from sklearn.ensemble import IsolationForest

from sathealth.data.schema import FloatArray
from sathealth.detectors.base import Detector, align_window_scores
from sathealth.features.windowing import n_windows, sliding_windows

FEATURE_NAMES = ("last", "mean", "std", "min", "max", "delta", "roughness")


def window_features(x: FloatArray, value_columns: tuple[int, ...], window: int) -> FloatArray:
    """``(T, D)`` -> ``(N, 7 * V + 1)`` feature matrix, one row per window."""
    count = n_windows(x.shape[0], window)
    if count == 0:
        raise ValueError(f"series of {x.shape[0]} steps is shorter than window {window}")
    values = np.ascontiguousarray(x[:, list(value_columns)], dtype=np.float64)
    w = sliding_windows(values, window)  # (N, window, V) view
    steps = np.abs(np.diff(values, axis=0))  # (T-1, V)
    csum_steps = np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(steps, axis=0)])
    roughness = (csum_steps[window - 1 :] - csum_steps[:count]) / (window - 1)
    per_value = [
        w[:, -1],
        w.mean(axis=1),
        w.std(axis=1),
        w.min(axis=1),
        w.max(axis=1),
        w[:, -1] - w[:, 0],
        roughness,
    ]

    command_cols = [c for c in range(x.shape[1]) if c not in value_columns]
    if command_cols:
        fired = x[:, command_cols].sum(axis=1)
        csum = np.concatenate([[0.0], np.cumsum(fired)])
        in_window = csum[window:] - csum[:count]
    else:
        in_window = np.zeros(count)
    features: FloatArray = np.column_stack([*per_value, in_window])
    return features


class IsolationForestDetector(Detector):
    name = "iforest"

    def __init__(
        self,
        value_columns: tuple[int, ...] = (0,),
        window: int = 32,
        n_estimators: int = 200,
        max_samples: int = 256,
        seed: int = 0,
    ) -> None:
        super().__init__(value_columns)
        self.window = window
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.seed = seed
        self._model: IsolationForest | None = None

    def fit(self, train: FloatArray, val: FloatArray) -> Self:
        features = window_features(train, self.value_columns, self.window)
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, features.shape[0]),
            random_state=self.seed,
        ).fit(features)
        return self

    def score(self, x: FloatArray) -> FloatArray:
        if self._model is None:
            raise RuntimeError("IsolationForestDetector is not fitted")
        if x.shape[0] < self.window:  # too short for one window: no evidence either way
            return np.zeros(x.shape[0])
        window_scores = -self._model.score_samples(
            window_features(x, self.value_columns, self.window)
        )
        return align_window_scores(window_scores, x.shape[0], self.window)

    def params(self) -> dict[str, Any]:
        return {
            **super().params(),
            "window": self.window,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "seed": self.seed,
        }
