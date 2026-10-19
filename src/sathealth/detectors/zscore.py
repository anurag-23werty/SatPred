"""Robust z-score: ``|x - median| / (IQR / 1.349)`` per value column, max over columns.

Like the limit check it looks at one time step at a time, but it ranks every step by how
unusual it is instead of a hard band, so it gives a threshold-free score for AUC-PR.
"""

from typing import Self

import numpy as np

from sathealth.data.schema import FloatArray
from sathealth.detectors.base import Detector


class RobustZScore(Detector):
    name = "zscore"

    def __init__(self, value_columns: tuple[int, ...] = (0,), min_scale: float = 1e-6) -> None:
        super().__init__(value_columns)
        self.min_scale = min_scale
        self._center: FloatArray | None = None
        self._scale: FloatArray | None = None

    def fit(self, train: FloatArray, val: FloatArray) -> Self:
        values = self._values(train)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
        scale = (q3 - q1) / 1.349
        scale = np.where(scale < self.min_scale, values.std(axis=0), scale)
        self._center = median
        self._scale = np.where(scale < self.min_scale, 1.0, scale)
        return self

    def score(self, x: FloatArray) -> FloatArray:
        if self._center is None or self._scale is None:
            raise RuntimeError("RobustZScore is not fitted")
        result: FloatArray = (np.abs(self._values(x) - self._center) / self._scale).max(axis=1)
        return result
