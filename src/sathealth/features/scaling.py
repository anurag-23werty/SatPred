"""Per-channel scalers fit on nominal training data only.

* ``robust``: ``(x - median) / (IQR / 1.349)``. The 1.349 factor makes the scale equal to the
  standard deviation for Gaussian data, so scores stay interpretable as σ-units, while a few
  unlabelled anomalies hiding in the training data cannot inflate the scale.
* ``standard``: ``(x - mean) / std``.
* ``none``: identity, kept so experiments can switch scaling off from config.

Degenerate spread (constant signals, 16 channels in this dataset) falls back
IQR → std → 1.0 instead of dividing by ~0, which would turn the first nominal wiggle in test
into a 10^6σ "anomaly". Only ``columns`` are scaled; command flags pass through untouched.
Parameters serialise to plain JSON so every prepared run records exactly how it was scaled.
"""

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from sathealth.data.schema import FloatArray

ScalerKind = Literal["robust", "standard", "none"]
_IQR_TO_STD = 1.349


@dataclass(frozen=True)
class ScalerParams:
    kind: ScalerKind
    columns: tuple[int, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    degenerate: tuple[bool, ...]  # True where the fitted spread fell back to a default

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ScalerParams":
        return cls(
            kind=raw["kind"],
            columns=tuple(raw["columns"]),
            center=tuple(float(v) for v in raw["center"]),
            scale=tuple(float(v) for v in raw["scale"]),
            degenerate=tuple(bool(v) for v in raw["degenerate"]),
        )


class ChannelScaler:
    def __init__(
        self, kind: ScalerKind = "robust", columns: tuple[int, ...] = (0,), min_scale: float = 1e-6
    ) -> None:
        self.kind = kind
        self.columns = columns
        self.min_scale = min_scale
        self._params: ScalerParams | None = None

    @classmethod
    def from_params(cls, params: ScalerParams) -> "ChannelScaler":
        scaler = cls(params.kind, params.columns)
        scaler._params = params
        return scaler

    @property
    def params(self) -> ScalerParams:
        if self._params is None:
            raise RuntimeError("scaler is not fitted; call fit() on training data first")
        return self._params

    def fit(self, x: FloatArray) -> "ChannelScaler":
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError(f"expected non-empty (T, D) array, got shape {x.shape}")
        cols = x[:, list(self.columns)]
        if self.kind == "none":
            center, scale = np.zeros(cols.shape[1]), np.ones(cols.shape[1])
            degenerate = np.zeros(cols.shape[1], dtype=bool)
        else:
            std = cols.std(axis=0)
            if self.kind == "robust":
                q1, center, q3 = np.quantile(cols, [0.25, 0.5, 0.75], axis=0)
                scale = (q3 - q1) / _IQR_TO_STD
                scale = np.where(scale < self.min_scale, std, scale)  # e.g. mostly-flat signals
            else:
                center, scale = cols.mean(axis=0), std
            degenerate = scale < self.min_scale
            scale = np.where(degenerate, 1.0, scale)
        self._params = ScalerParams(
            kind=self.kind,
            columns=self.columns,
            center=tuple(float(v) for v in center),
            scale=tuple(float(v) for v in scale),
            degenerate=tuple(bool(v) for v in degenerate),
        )
        return self

    def transform(self, x: FloatArray) -> FloatArray:
        p = self.params
        out = np.array(x, dtype=np.float64, copy=True)
        out[:, list(p.columns)] = (out[:, list(p.columns)] - np.asarray(p.center)) / np.asarray(
            p.scale
        )
        return out

    def inverse_transform(self, x: FloatArray) -> FloatArray:
        p = self.params
        out = np.array(x, dtype=np.float64, copy=True)
        out[:, list(p.columns)] = out[:, list(p.columns)] * np.asarray(p.scale) + np.asarray(
            p.center
        )
        return out

    def fit_transform(self, x: FloatArray) -> FloatArray:
        return self.fit(x).transform(x)
