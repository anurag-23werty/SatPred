"""What would the signal have done if nothing had gone wrong?

A one-step forecast is a poor counterfactual: once an anomaly starts, the forecaster's input
contains the anomaly and its forecast follows it. Instead the LSTM is **rolled out** from
the last step before the onset: each predicted value is fed back as the next input, together
with the command flags that were *actually* sent. The result is the trajectory the model
expects given the spacecraft's commanded activity, without the anomaly.

Rollout errors grow with the horizon, so the band is horizon-dependent: the same rollouts are
run from many starts on nominal data (the validation tail), and the 5% and 95% quantiles of
the residual at each horizon form a 90% band. Quantiles are pooled over neighbouring
horizons for stability and made non-decreasing in width.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from sathealth.data.schema import FloatArray

POOL = 4  # pool residuals over horizons h-4 .. h+4


def rollout(
    model: nn.Module,
    history: FloatArray,
    future: FloatArray,
    value_columns: tuple[int, ...],
    window: int,
) -> FloatArray:
    """Batched free-running forecast.

    ``history``: ``(B, L, D)`` real inputs before the rollout (L ≤ window is allowed).
    ``future``: ``(B, H, D)``, the real future; only its command columns are used.
    Returns predicted values ``(B, H, V)``.
    """
    cols = list(value_columns)
    seq = torch.as_tensor(np.asarray(history, dtype=np.float32))
    fut = torch.as_tensor(np.asarray(future, dtype=np.float32))
    out = torch.empty(fut.shape[0], fut.shape[1], len(cols))
    with torch.no_grad():
        for h in range(fut.shape[1]):
            pred = model(seq[:, -window:])[:, -1, :]
            out[:, h] = pred
            nxt = fut[:, h].clone()
            nxt[:, cols] = pred
            seq = torch.cat([seq[:, -(window - 1) :], nxt[:, None]], dim=1)
    result: FloatArray = out.double().numpy()
    return result


@dataclass(frozen=True)
class HorizonBand:
    """Residual quantiles per horizon: ``lower``/``upper`` shaped ``(H, V)``."""

    lower: FloatArray
    upper: FloatArray
    source: str  # "val" or "train" (fallback for very short validation tails)
    n_starts: int

    def at(self, horizons: int) -> tuple[FloatArray, FloatArray]:
        """Band offsets for horizons 1..``horizons``; beyond the calibrated range the last
        (widest) offsets are reused."""
        idx = np.minimum(np.arange(horizons), self.lower.shape[0] - 1)
        return self.lower[idx], self.upper[idx]


def calibrate_band(
    model: nn.Module,
    series: FloatArray,
    value_columns: tuple[int, ...],
    window: int,
    horizon: int,
    n_starts: int,
    rng: np.random.Generator,
    source: str = "val",
    quantiles: tuple[float, float] = (0.05, 0.95),
) -> HorizonBand:
    cols = list(value_columns)
    usable = series.shape[0] - window - 1
    horizon = int(min(horizon, usable))
    if horizon < 10:
        raise ValueError(f"series of {series.shape[0]} steps is too short to calibrate a band")
    last_start = series.shape[0] - horizon
    starts = np.sort(rng.choice(np.arange(window, last_start + 1), size=n_starts, replace=True))
    hist = np.stack([series[s - window : s] for s in starts])
    fut = np.stack([series[s : s + horizon] for s in starts])
    residual = fut[:, :, cols] - rollout(model, hist, fut, value_columns, window)  # (B, H, V)
    lower = np.empty((horizon, len(cols)))
    upper = np.empty((horizon, len(cols)))
    for h in range(horizon):
        pooled = residual[:, max(0, h - POOL) : h + POOL + 1].reshape(-1, len(cols))
        lower[h], upper[h] = np.quantile(pooled, quantiles, axis=0)
    # rollout uncertainty does not shrink with the horizon
    lower = np.minimum.accumulate(lower, axis=0)
    upper = np.maximum.accumulate(upper, axis=0)
    return HorizonBand(lower, upper, source, n_starts)
