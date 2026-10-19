"""Detector interface.

Every detector maps a prepared ``(T, D)`` series to a point-wise score of shape ``(T,)``,
where higher means more anomalous. Fitting sees only nominal data (train, plus validation
for detectors that need early stopping); thresholds are fit separately by
:mod:`sathealth.evaluation.thresholds`, so detectors never see labels.

Window-based detectors score windows. :func:`align_window_scores` assigns each window's score
to its last time step, the moment the detector could have raised the alarm in a live stream.
The first ``size - 1`` steps reuse the first window's score.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Self

import numpy as np

from sathealth.data.schema import FloatArray
from sathealth.features.windowing import window_end_index


class Detector(ABC):
    name: ClassVar[str]

    def __init__(self, value_columns: tuple[int, ...] = (0,)) -> None:
        self.value_columns = value_columns

    @abstractmethod
    def fit(self, train: FloatArray, val: FloatArray) -> Self: ...

    @abstractmethod
    def score(self, x: FloatArray) -> FloatArray: ...

    def params(self) -> dict[str, Any]:
        """Settings that change the scores; part of the evaluation run fingerprint."""
        return {"value_columns": list(self.value_columns)}

    def diagnostics(self) -> dict[str, Any]:
        """Per-channel fit details (e.g. training curves) stored next to the results."""
        return {}

    def _values(self, x: FloatArray) -> FloatArray:
        if x.ndim != 2:
            raise ValueError(f"expected (T, D) array, got shape {x.shape}")
        values: FloatArray = np.asarray(x[:, list(self.value_columns)], dtype=np.float64)
        return values


def align_window_scores(
    window_scores: FloatArray, length: int, size: int, stride: int = 1
) -> FloatArray:
    """Window scores -> point scores of length ``length`` (end-aligned, forward-filled)."""
    ends = window_end_index(length, size, stride)
    if ends.size != window_scores.size:
        raise ValueError(f"{window_scores.size} window scores for {ends.size} windows")
    out = np.full(length, np.nan)
    if ends.size == 0:
        return out
    out[ends] = window_scores
    out[: ends[0]] = window_scores[0]
    # forward-fill gaps left by stride > 1 and the tail after the last window
    filled = np.where(np.isnan(out), 0, np.arange(length))
    np.maximum.accumulate(filled, out=filled)
    result: FloatArray = out[filled]
    return result
