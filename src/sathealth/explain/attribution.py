"""Integrated Gradients for the LSTM forecaster (Sundararajan, Taly & Yan, ICML 2017).

The quantity explained is the squared one-step forecast error at the event peak, as a
function of the 64-step input window (value history and command flags). IG integrates the
gradient along a straight path from a baseline input to the real input:

    IG_i = (x_i - b_i) * mean over alpha in [0, 1] of d f(b + alpha (x - b)) / d x_i

It satisfies *completeness*: the attributions sum to ``f(x) - f(b)``. The remaining gap is
returned as a check on the integration (trapezoidal rule over ``steps`` intervals).

The baseline is all zeros: the training median of the (robust-scaled) value and no
commands. The summaries answer the two operator questions separately: *which input*
(feature importance, summed over time) and *which moment* (lag importance, summed over
features; lag 1 is the most recent step).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from sathealth.data.schema import FloatArray

TargetFn = Callable[[Tensor], Tensor]  # (B, W, D) -> (B,)


def forecast_error_fn(model: nn.Module, actual: FloatArray) -> TargetFn:
    """Squared error of the next-step forecast against the value that actually followed."""
    target = torch.as_tensor(np.asarray(actual, dtype=np.float32))

    def f(x: Tensor) -> Tensor:
        prediction = model(x)[:, -1, :]
        error: Tensor = ((prediction - target) ** 2).sum(dim=-1)
        return error

    return f


@dataclass(frozen=True)
class Attribution:
    values: FloatArray  # (W, D), signed
    target: float  # f(x)
    baseline_target: float  # f(baseline)

    @property
    def completeness_gap(self) -> float:
        """|sum(IG) - (f(x) - f(b))| relative to |f(x) - f(b)|."""
        delta = self.target - self.baseline_target
        return float(abs(self.values.sum() - delta) / max(abs(delta), 1e-12))

    def feature_importance(self) -> FloatArray:
        result: FloatArray = np.abs(self.values).sum(axis=0)
        return result

    def lag_importance(self) -> FloatArray:
        """Importance per lag, index 0 = lag 1 (the most recent input step)."""
        result: FloatArray = np.abs(self.values).sum(axis=1)[::-1]
        return result

    def top_features(self, k: int = 3) -> list[int]:
        return [int(i) for i in np.argsort(-self.feature_importance(), kind="stable")[:k]]

    def top_lags(self, k: int = 3) -> list[int]:
        return [int(i) + 1 for i in np.argsort(-self.lag_importance(), kind="stable")[:k]]


def integrated_gradients(
    f: TargetFn, window: FloatArray, baseline: FloatArray | None = None, steps: int = 64
) -> Attribution:
    x = torch.as_tensor(np.asarray(window, dtype=np.float32))
    b = torch.zeros_like(x) if baseline is None else torch.as_tensor(baseline, dtype=torch.float32)
    alphas = torch.linspace(0.0, 1.0, steps + 1)[:, None, None]
    path = (b + alphas * (x - b)).requires_grad_(True)
    with torch.enable_grad():
        out = f(path)
        (grads,) = torch.autograd.grad(out.sum(), path)
    avg = ((grads[:-1] + grads[1:]) / 2).mean(dim=0)  # trapezoidal rule
    values = ((x - b) * avg).detach().double().numpy()
    return Attribution(values, float(out[-1].detach()), float(out[0].detach()))


def deletion_effect(
    f: TargetFn,
    window: FloatArray,
    attribution: Attribution,
    fraction: float,
    rng: np.random.Generator,
    *,
    rank: Literal["abs", "positive"] = "abs",
) -> tuple[float, float]:
    """Relative error reduction when the top-``fraction`` attributed inputs, or a random set
    of the same size, are reset to the baseline (zeros). Faithful attributions reduce more.

    ``rank="abs"`` ranks by |IG| (the pre-registered test). It also selects inputs that
    *lower* the error, so resetting them can raise it; ``rank="positive"`` ranks by signed
    IG and only removes what pushes the error up.
    """
    signed = attribution.values.ravel()
    flat = np.abs(signed) if rank == "abs" else signed
    k = max(1, round(fraction * flat.size))
    top = np.argsort(-flat, kind="stable")[:k]
    random = rng.choice(flat.size, size=k, replace=False)
    batch = np.repeat(np.asarray(window, dtype=np.float32)[None], 3, axis=0)
    batch[1].reshape(-1)[top] = 0.0
    batch[2].reshape(-1)[random] = 0.0
    with torch.no_grad():
        err = f(torch.as_tensor(batch)).double().numpy()
    base = max(float(err[0]), 1e-12)
    return float((err[0] - err[1]) / base), float((err[0] - err[2]) / base)
