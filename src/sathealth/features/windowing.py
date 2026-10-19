"""Sliding windows and chronological splits.

Windows are **zero-copy, read-only views** (``sliding_window_view``), so a channel with
8 640 steps x 25 features and window 100 costs no extra memory instead of ~170 MB.
Splitting happens on rows *before* windowing, so no window can straddle train and
validation; ``gap`` optionally discards rows between them to also break autocorrelation.

Window ``i`` covers rows ``[i*stride, i*stride + size)``; :func:`window_end_index` maps each
window back to the time step its score is attributed to (its last row), which later phases
use to turn window scores into point-wise anomaly scores.
"""

from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from sathealth.data.schema import BoolArray, FloatArray

LabelRule = Literal["any", "last"]


def n_windows(length: int, size: int, stride: int = 1) -> int:
    if size < 1 or stride < 1:
        raise ValueError("size and stride must be >= 1")
    return 0 if length < size else (length - size) // stride + 1


def sliding_windows(x: FloatArray, size: int, stride: int = 1) -> FloatArray:
    """``(T, D)`` -> read-only view ``(N, size, D)``."""
    if x.ndim != 2:
        raise ValueError(f"expected (T, D) array, got shape {x.shape}")
    count = n_windows(x.shape[0], size, stride)
    if count == 0:
        return np.empty((0, size, x.shape[1]), dtype=x.dtype)
    view = sliding_window_view(x, window_shape=size, axis=0)  # (T-size+1, D, size)
    return view.transpose(0, 2, 1)[::stride]


def window_labels(
    mask: BoolArray, size: int, stride: int = 1, rule: LabelRule = "any"
) -> BoolArray:
    """One label per window: ``any`` anomalous step, or the ``last`` step's label. O(T)."""
    count = n_windows(mask.shape[0], size, stride)
    if count == 0:
        return np.zeros(0, dtype=np.bool_)
    starts = np.arange(count) * stride
    if rule == "last":
        return mask[starts + size - 1].copy()
    csum = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
    labels: BoolArray = (csum[starts + size] - csum[starts]) > 0
    return labels


def window_end_index(length: int, size: int, stride: int = 1) -> NDArray[np.int64]:
    """Row index of the last step in each window."""
    return np.arange(n_windows(length, size, stride), dtype=np.int64) * stride + size - 1


def chronological_split(length: int, val_fraction: float, gap: int = 0) -> tuple[slice, slice]:
    """Row slices ``(train, val)``: val is the last ``val_fraction`` of the series."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    n_val = round(length * val_fraction)
    val_start = length - n_val
    train_end = max(0, val_start - gap)
    return slice(0, train_end), slice(val_start, length)
