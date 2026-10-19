"""Turn raw detector scores into tail probabilities learnt from nominal validation scores.

A raw score means nothing on its own: 0.8 may be routine for one channel and alarming for
another, and an LSTM error is not comparable with a Transformer error. Calibration maps every
score ``x`` to ``p = P(score >= x | nominal)``, estimated on the validation tail, and reports
the *surprise* ``-log10 p``. Surprise 3 means "seen once in 1,000 nominal steps", whatever
the channel or detector.

* ``ecdf``: the empirical tail ``(#{val >= x} + 1) / (n + 1)``. It cannot go below
  ``1 / (n + 1)``: with ~500 validation rows, nothing is ever rarer than p = 0.002.
* ``evt`` (peaks over threshold, Pickands-Balkema-de Haan): the empirical tail up to the
  ``tail_quantile`` ``u``, then a Generalised Pareto distribution fit to the excesses over
  ``u``. This extrapolates beyond the largest validation score, which is exactly where alarms
  live. Parameters come from probability-weighted moments (Hosking & Wallis 1987), which are
  far more stable than maximum likelihood on the 20-30 excesses we have per channel.
  The shape is clamped to ``[0, 0.5]``. A negative shape asserts a hard upper endpoint
  ("larger scores are impossible"), which 20-30 excesses cannot support. On Gaussian scores
  it makes the calibration overconfident far out in the tail (surprise 7.2 where the truth
  is 5.5, pinned by a test), and overconfidence means false alarms.
* ``none``: the raw score is passed through (for reference methods that need raw errors).
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from sathealth.data.schema import FloatArray

Method = Literal["evt", "ecdf", "none"]
MIN_EXCESSES = 5
SHAPE_MIN, SHAPE_MAX = 0.0, 0.5


@dataclass(frozen=True)
class GeneralizedPareto:
    """GPD with ``shape`` xi and ``scale`` sigma; survival ``(1 + xi y / sigma)^(-1/xi)``."""

    shape: float
    scale: float

    def survival(self, excess: FloatArray) -> FloatArray:
        y = np.maximum(excess, 0.0) / self.scale
        if abs(self.shape) < 1e-9:
            result: FloatArray = np.exp(-y)
            return result
        base = 1.0 + self.shape * y
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(base > 0, np.power(np.maximum(base, 1e-300), -1.0 / self.shape), 0.0)
        result = out
        return result


def fit_gpd_pwm(
    excess: FloatArray, shape_bounds: tuple[float, float] = (SHAPE_MIN, SHAPE_MAX)
) -> GeneralizedPareto:
    """Probability-weighted-moment GPD fit (Hosking & Wallis 1987), shape clamped to bounds.

    When the shape is clamped, the scale is refit so the mean excess is preserved
    (mean excess of a GPD = scale / (1 - shape)).

    With fewer than ``MIN_EXCESSES`` excesses the shape is not identifiable and an exponential
    tail (shape 0) with the mean excess as scale is used instead.
    """
    y = np.sort(np.asarray(excess, dtype=np.float64))
    n = y.size
    if n == 0:
        raise ValueError("no excesses to fit")
    a0 = float(y.mean())
    if n < MIN_EXCESSES or a0 <= 0:
        return GeneralizedPareto(0.0, max(a0, 1e-12))
    # unbiased estimator of E[Y (1 - F(Y))]
    a1 = float(np.sum(y * (n - np.arange(1, n + 1)) / (n - 1)) / n)
    denom = a0 - 2.0 * a1
    low, high = shape_bounds
    if denom <= 0:  # extremely heavy tail: estimator breaks down
        return GeneralizedPareto(high, a0 * (1 - high))
    raw = 2.0 - a0 / denom  # xi = -k in Hosking & Wallis' notation
    if low <= raw <= high:
        return GeneralizedPareto(raw, max(2.0 * a0 * a1 / denom, 1e-12))
    shape = float(np.clip(raw, low, high))
    return GeneralizedPareto(shape, max(a0 * (1 - shape), 1e-12))


class TailCalibrator:
    """Fit on nominal validation scores, then map any scores to surprise ``-log10 p``."""

    def __init__(self, method: Method = "evt", tail_quantile: float = 0.95, cap: float = 12.0):
        self.method = method
        self.tail_quantile = tail_quantile
        self.cap = cap
        self._sorted: FloatArray = np.zeros(0)
        self.u = 0.0
        self.zeta = 0.0
        self.gpd: GeneralizedPareto | None = None

    def fit(self, val_scores: FloatArray) -> "TailCalibrator":
        finite = np.asarray(val_scores, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("no finite validation scores to calibrate on")
        self._sorted = np.sort(finite)
        if self.method == "evt":
            self.u = float(np.quantile(finite, self.tail_quantile))
            excess = finite[finite > self.u] - self.u
            self.zeta = excess.size / finite.size
            if excess.size == 0:  # flat tail (e.g. constant scores): any excess is extreme
                self.zeta = 1.0 / (finite.size + 1)
                self.gpd = GeneralizedPareto(0.0, max(float(finite.std()), 1e-12))
            else:
                self.gpd = fit_gpd_pwm(excess)
        return self

    def tail_probability(self, scores: FloatArray) -> FloatArray:
        x = np.asarray(scores, dtype=np.float64)
        n = self._sorted.size
        if n == 0:
            raise RuntimeError("TailCalibrator is not fitted")
        n_ge = n - np.searchsorted(self._sorted, x, side="left")
        p: FloatArray = np.asarray((n_ge + 1.0) / (n + 1.0), dtype=np.float64)
        if self.method == "evt" and self.gpd is not None:
            tail = self.zeta * self.gpd.survival(x - self.u)
            p = np.where(x > self.u, tail, p).astype(np.float64)
        return p

    def surprise(self, scores: FloatArray) -> FloatArray:
        if self.method == "none":
            return np.asarray(scores, dtype=np.float64).copy()
        p = np.maximum(self.tail_probability(scores), 10.0 ** (-self.cap))
        result: FloatArray = np.minimum(-np.log10(p), self.cap)
        return result

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"method": self.method, "n": int(self._sorted.size)}
        if self.method == "evt" and self.gpd is not None:
            out |= {
                "u": self.u,
                "zeta": self.zeta,
                "shape": self.gpd.shape,
                "scale": self.gpd.scale,
            }
        return out
