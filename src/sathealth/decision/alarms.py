"""Turn a (fused) surprise series into an alarm mask.

* :func:`hysteresis`: enter alarm at ``on``, stay until the surprise drops below ``off``.
  A score hovering around a single level otherwise splits one incident into many alarms.
  Causal, so it works on a live stream.
* :func:`merge_gaps` and :func:`drop_short`: the event builder's clean-up.
* :func:`ndt_mask` and :func:`prune`: Telemanom's non-parametric dynamic threshold and its
  pruning step (Hundman et al., KDD 2018), the standard reference on SMAP/MSL. Both look at a
  whole window of *test* scores before deciding, so they are batch methods, not live ones.
"""

import itertools

import numpy as np

from sathealth.data.schema import BoolArray, FloatArray
from sathealth.evaluation.metrics import mask_to_events

NDT_Z = np.arange(2.5, 12.0, 0.5)


def hysteresis(
    surprise: FloatArray, on: float, off: float, *, start_active: bool = False
) -> BoolArray:
    """``start_active`` continues an alarm from a previous block of the same stream."""
    if off > on:
        raise ValueError("off must be <= on")
    mask = np.zeros(surprise.size, dtype=np.bool_)
    active = start_active
    for i, value in enumerate(surprise):
        active = value >= off if active else value >= on
        mask[i] = active
    return mask


def merge_gaps(mask: BoolArray, gap: int) -> BoolArray:
    """Fill runs of at most ``gap`` False steps that sit between two alarm runs."""
    out = mask.copy()
    if gap <= 0:
        return out
    runs = mask_to_events(mask)
    for (_, end), (start, _) in itertools.pairwise(runs):
        if start - end - 1 <= gap:
            out[end + 1 : start] = True
    return out


def drop_short(mask: BoolArray, min_duration: int) -> BoolArray:
    out = mask.copy()
    if min_duration <= 1:
        return out
    for start, end in mask_to_events(mask):
        if end - start + 1 < min_duration:
            out[start : end + 1] = False
    return out


def _best_epsilon(errors: FloatArray) -> float:
    """Telemanom's epsilon: maximise the relative drop in mean and std of what stays below,
    penalised by the number of flagged steps and (squared) sequences."""
    mean, std = float(errors.mean()), float(errors.std())
    if mean <= 0 or std <= 0:
        return float("inf")
    best_score, best_eps = -np.inf, float("inf")
    for z in NDT_Z:
        eps = mean + z * std
        above = errors > eps
        n_above = int(above.sum())
        if n_above == 0:
            continue
        below = errors[~above]
        n_seq = len(mask_to_events(above))
        score = ((mean - below.mean()) / mean + (std - below.std()) / std) / (n_above + n_seq**2)
        if score > best_score:
            best_score, best_eps = score, float(eps)
    return best_eps


def ndt_mask(scores: FloatArray, window: int, step: int) -> BoolArray:
    """Union of the steps above each window's dynamic threshold."""
    n = scores.size
    mask = np.zeros(n, dtype=np.bool_)
    starts = range(0, max(n - window, 0) + 1, step) if n > window else range(1)
    for start in starts:
        chunk = scores[start : start + window]
        mask[start : start + chunk.size] |= chunk > _best_epsilon(chunk)
    if n > window and (n - window) % step:  # cover the tail of the series
        chunk = scores[n - window :]
        mask[n - window :] |= chunk > _best_epsilon(chunk)
    return mask


def prune(mask: BoolArray, scores: FloatArray, margin: float) -> BoolArray:
    """Telemanom pruning: rank alarm runs by peak score, add the highest non-alarm score, and
    keep runs only down to the last relative drop of at least ``margin`` between neighbours."""
    runs = mask_to_events(mask)
    if not runs:
        return mask.copy()
    peaks = np.array([scores[s : e + 1].max() for s, e in runs])
    rest = scores[~mask]
    floor = float(rest.max()) if rest.size else 0.0
    order = np.argsort(-peaks, kind="mergesort")
    ranked = np.r_[peaks[order], floor]
    with np.errstate(divide="ignore", invalid="ignore"):
        drops = np.where(ranked[:-1] > 0, (ranked[:-1] - ranked[1:]) / ranked[:-1], 0.0)
    significant = np.flatnonzero(drops >= margin)
    keep = set(order[: significant[-1] + 1].tolist()) if significant.size else set()
    out = np.zeros_like(mask)
    for idx in keep:
        start, end = runs[idx]
        out[start : end + 1] = True
    return out
