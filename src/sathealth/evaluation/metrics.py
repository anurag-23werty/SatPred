"""Anomaly detection metrics for time series, with the definitions spelled out.

Labels are inclusive ``[start, end]`` segments; predictions are a boolean mask. A
*predicted event* is a maximal run of ``True`` (runs separated by ``<= min_gap`` steps merge).

* **Point-wise**: ordinary precision/recall over time steps. Long anomalies dominate it.
* **Event-wise** (Hundman et al., KDD 2018): a labelled segment is a TP if any predicted step
  hits it, otherwise an FN. A predicted event touching no segment is an FP. Precision is
  ``TP / (TP + FP)``. This is the protocol of the dataset's own paper.
* **Composite** (Garg et al., IEEE TNNLS 2021): harmonic mean of *point-wise precision* and
  *event-wise recall*. It rewards finding every event and punishes flooding the operator with
  alarm steps. This is our headline metric.
* **Point-adjusted** (Xu et al., WWW 2018): if any step of a segment is hit, the whole
  segment counts as detected, then point-wise F1. Widely reported but strongly inflated:
  random scores can reach high values (Kim et al., AAAI 2022). Reported for comparison only.

``early_tolerance`` lets a hit up to that many steps before a labelled start count as a
detection (and not an FP). Detection delay is measured from the labelled start.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sathealth.data.schema import BoolArray, FloatArray

Segment = tuple[int, int]  # inclusive


def mask_to_events(mask: BoolArray, min_gap: int = 0) -> list[Segment]:
    """Maximal runs of True as inclusive segments; runs <= ``min_gap`` steps apart merge."""
    if mask.size == 0 or not mask.any():
        return []
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2] - 1
    events: list[Segment] = []
    for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
        if events and s - events[-1][1] - 1 <= min_gap:
            events[-1] = (events[-1][0], e)
        else:
            events.append((s, e))
    return events


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


@dataclass(frozen=True)
class PointCounts:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return _safe_div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _safe_div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        return f1(self.precision, self.recall)

    def __add__(self, other: "PointCounts") -> "PointCounts":
        return PointCounts(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn
        )


@dataclass(frozen=True)
class EventCounts:
    tp: int  # labelled segments detected
    fn: int  # labelled segments missed
    fp: int  # predicted events touching no labelled segment

    @property
    def precision(self) -> float:
        return _safe_div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _safe_div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        return f1(self.precision, self.recall)

    def __add__(self, other: "EventCounts") -> "EventCounts":
        return EventCounts(self.tp + other.tp, self.fn + other.fn, self.fp + other.fp)


def point_counts(pred: BoolArray, labels: BoolArray) -> PointCounts:
    if pred.shape != labels.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs labels {labels.shape}")
    p, y = pred.astype(bool), labels.astype(bool)
    return PointCounts(
        tp=int((p & y).sum()),
        fp=int((p & ~y).sum()),
        fn=int((~p & y).sum()),
        tn=int((~p & ~y).sum()),
    )


def _expanded(segments: list[Segment], tolerance: int) -> list[Segment]:
    return [(max(0, s - tolerance), e) for s, e in segments]


def event_counts(
    pred: BoolArray, segments: list[Segment], *, early_tolerance: int = 0, min_gap: int = 0
) -> EventCounts:
    windows = _expanded(segments, early_tolerance)
    tp = sum(bool(pred[s : e + 1].any()) for s, e in windows)
    fp = 0
    for ps, pe in mask_to_events(pred, min_gap):
        if not any(ps <= e and pe >= s for s, e in windows):
            fp += 1
    return EventCounts(tp=tp, fn=len(segments) - tp, fp=fp)


def point_adjust(pred: BoolArray, segments: list[Segment]) -> BoolArray:
    """Copy of ``pred`` where every segment with at least one hit is filled in completely."""
    adjusted = pred.astype(bool).copy()
    for s, e in segments:
        if adjusted[s : e + 1].any():
            adjusted[s : e + 1] = True
    return adjusted


def detection_delays(
    pred: BoolArray, segments: list[Segment], early_tolerance: int = 0
) -> list[int]:
    """Steps from labelled start to first hit, for detected segments (negative = early)."""
    delays = []
    for (s, _), (ws, we) in zip(segments, _expanded(segments, early_tolerance), strict=True):
        hits = np.flatnonzero(pred[ws : we + 1])
        if hits.size:
            delays.append(int(ws + hits[0] - s))
    return delays


def average_precision(scores: FloatArray, labels: BoolArray) -> float:
    """Area under the precision-recall curve (step-wise, identical to scikit-learn's AP)."""
    y = labels.astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_y = scores[order], y[order]
    # Evaluate only at the last index of each block of tied scores.
    last_of_tie = np.r_[np.flatnonzero(np.diff(sorted_scores)), sorted_y.size - 1]
    tps = np.cumsum(sorted_y)[last_of_tie]
    precision = tps / (last_of_tie + 1)
    recall = tps / n_pos
    recall_steps = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_steps * precision))


@dataclass(frozen=True)
class ChannelMetrics:
    """Everything we report for one channel at one operating threshold."""

    points: PointCounts
    events: EventCounts
    adjusted: PointCounts
    delays: tuple[int, ...]
    average_precision: float
    n_steps: int

    @property
    def composite_f1(self) -> float:
        return f1(self.points.precision, self.events.recall)


def evaluate_predictions(
    pred: BoolArray,
    labels: BoolArray,
    scores: FloatArray,
    segments: list[Segment],
    *,
    early_tolerance: int = 0,
    min_gap: int = 0,
) -> ChannelMetrics:
    return ChannelMetrics(
        points=point_counts(pred, labels),
        events=event_counts(pred, segments, early_tolerance=early_tolerance, min_gap=min_gap),
        adjusted=point_counts(point_adjust(pred, segments), labels),
        delays=tuple(detection_delays(pred, segments, early_tolerance)),
        average_precision=average_precision(scores, labels),
        n_steps=int(labels.size),
    )


def oracle_event_f1(
    scores: FloatArray,
    segments: list[Segment],
    n_thresholds: int = 200,
    *,
    early_tolerance: int = 0,
    min_gap: int = 0,
) -> tuple[float, float]:
    """Best event-wise F1 over candidate thresholds **chosen with labels**: an upper bound.

    Returns ``(best_f1, threshold)``. Never use this threshold as an operating point.
    """
    candidates: NDArray[np.float64] = np.unique(
        np.quantile(scores, np.linspace(0.0, 1.0, n_thresholds))
    )
    best = (0.0, float(candidates[-1]))
    for thr in candidates:
        counts = event_counts(
            scores > thr, segments, early_tolerance=early_tolerance, min_gap=min_gap
        )
        if counts.f1 > best[0]:
            best = (counts.f1, float(thr))
    return best
