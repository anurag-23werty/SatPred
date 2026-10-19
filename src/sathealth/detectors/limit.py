"""Static limit check: what most operations teams run today.

Limits are the training range widened by ``margin`` x range on each side. The score is the
distance outside the band (0 inside), so the strict ``score > threshold`` rule reproduces a
classic out-of-limits alarm. It cannot see drifts or broken correlations that stay inside the
limits, which is exactly the gap the learned detectors must close.
"""

from typing import Any, Self

import numpy as np

from sathealth.data.schema import FloatArray
from sathealth.detectors.base import Detector


class LimitCheck(Detector):
    name = "limit"

    def __init__(self, value_columns: tuple[int, ...] = (0,), margin: float = 0.05) -> None:
        super().__init__(value_columns)
        self.margin = margin
        self._lo: FloatArray | None = None
        self._hi: FloatArray | None = None

    def fit(self, train: FloatArray, val: FloatArray) -> Self:
        values = self._values(train)
        lo, hi = values.min(axis=0), values.max(axis=0)
        # A flat channel has zero range; widen by `margin` in absolute (scaled) units instead.
        width = np.where(hi - lo > 0, hi - lo, 1.0) * self.margin
        self._lo, self._hi = lo - width, hi + width
        return self

    def score(self, x: FloatArray) -> FloatArray:
        if self._lo is None or self._hi is None:
            raise RuntimeError("LimitCheck is not fitted")
        values = self._values(x)
        outside = np.maximum(self._lo - values, values - self._hi)
        result: FloatArray = np.maximum(outside, 0.0).max(axis=1)
        return result

    def params(self) -> dict[str, Any]:
        return {**super().params(), "margin": self.margin}
