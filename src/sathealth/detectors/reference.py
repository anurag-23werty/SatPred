"""Random scores: the floor any real detector must clear.

It exists to make metric inflation visible. Under point-adjusted F1, uniformly random scores
already look competitive on SMAP/MSL (Kim et al., AAAI 2022), while event-wise and composite
F1 stay low. Showing both side by side is the cheapest honesty check in the report.
"""

from typing import Any, Self

import numpy as np

from sathealth.data.schema import FloatArray
from sathealth.detectors.base import Detector


class RandomScore(Detector):
    name = "random"

    def __init__(self, value_columns: tuple[int, ...] = (0,), seed: int = 0) -> None:
        super().__init__(value_columns)
        self.seed = seed

    def fit(self, train: FloatArray, val: FloatArray) -> Self:
        return self

    def score(self, x: FloatArray) -> FloatArray:
        rng = np.random.default_rng([self.seed, x.shape[0]])  # deterministic per length
        scores: FloatArray = rng.random(int(x.shape[0]))
        return scores

    def params(self) -> dict[str, Any]:
        return {**super().params(), "seed": self.seed}
