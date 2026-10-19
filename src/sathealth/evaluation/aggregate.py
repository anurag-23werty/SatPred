"""Aggregate per-channel results into dataset-level numbers with uncertainty.

Counts are **micro-averaged** (TP/FP/FN summed over channels before computing ratios), as
in Hundman et al. 2018, so a channel with many events weighs more than a channel with one.
Macro event F1 (the mean of per-channel F1s) is reported next to it because one channel
with hundreds of false events can dominate the micro number: on SMAP/MSL, channel A-5 alone
produces 90% of the limit check's false events. Average precision and oracle F1 are macro
means (they are not count-based).

95% confidence intervals come from a percentile bootstrap that resamples *channels* with
replacement: channels are the independent units, while time steps within a channel are not.

To compare two detectors, :func:`paired_composite_delta` resamples the *same* channels for
both. Two detectors are scored on identical channels, so their errors are correlated; a
paired interval for the difference is much tighter than two separate intervals, and "the two
separate intervals do not overlap" is a needlessly strict test.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sathealth.data.schema import FloatArray
from sathealth.evaluation.metrics import ChannelMetrics, EventCounts, PointCounts, f1


@dataclass(frozen=True)
class ChannelResult:
    chan_id: str
    spacecraft: str
    metrics: ChannelMetrics
    oracle_event_f1: float
    threshold: float
    threshold_source: str
    flags: tuple[str, ...] = ()  # quality checks raised for this channel in Phase 1

    def to_dict(self) -> dict[str, Any]:
        m = self.metrics
        return {
            "chan_id": self.chan_id,
            "spacecraft": self.spacecraft,
            "flags": list(self.flags),
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "n_steps": m.n_steps,
            "events": {
                "tp": m.events.tp,
                "fn": m.events.fn,
                "fp": m.events.fp,
                "precision": m.events.precision,
                "recall": m.events.recall,
                "f1": m.events.f1,
            },
            "points": {
                "tp": m.points.tp,
                "fp": m.points.fp,
                "fn": m.points.fn,
                "tn": m.points.tn,
                "precision": m.points.precision,
                "recall": m.points.recall,
                "f1": m.points.f1,
            },
            "composite_f1": m.composite_f1,
            "point_adjusted_f1": m.adjusted.f1,
            "average_precision": m.average_precision,
            "delays": list(m.delays),
            "oracle_event_f1": self.oracle_event_f1,
        }


def _micro(points: PointCounts, events: EventCounts) -> tuple[float, float]:
    return events.f1, f1(points.precision, events.recall)


def _bootstrap(
    results: Sequence[ChannelResult], n: int, seed: int
) -> dict[str, tuple[float, float]]:
    if n == 0 or len(results) < 2:
        return {}
    ev = np.array(
        [(r.metrics.events.tp, r.metrics.events.fn, r.metrics.events.fp) for r in results],
        dtype=np.float64,
    )
    pt = np.array([(r.metrics.points.tp, r.metrics.points.fp) for r in results], dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(results), size=(n, len(results)))
    tp, fn, fp = (ev[idx, k].sum(axis=1) for k in range(3))
    ptp, pfp = (pt[idx, k].sum(axis=1) for k in range(2))
    with np.errstate(divide="ignore", invalid="ignore"):
        e_prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        e_rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        p_prec = np.where(ptp + pfp > 0, ptp / (ptp + pfp), 0.0)
        e_f1 = np.where(e_prec + e_rec > 0, 2 * e_prec * e_rec / (e_prec + e_rec), 0.0)
        c_f1 = np.where(p_prec + e_rec > 0, 2 * p_prec * e_rec / (p_prec + e_rec), 0.0)

    def ci(x: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(x, [2.5, 97.5])
        return float(lo), float(hi)

    return {"event_f1": ci(e_f1), "composite_f1": ci(c_f1)}


def summarize(
    results: Sequence[ChannelResult], *, bootstrap: int = 1000, seed: int = 0
) -> dict[str, Any]:
    if not results:
        return {"n_channels": 0}
    points = sum((r.metrics.points for r in results), PointCounts(0, 0, 0, 0))
    events = sum((r.metrics.events for r in results), EventCounts(0, 0, 0))
    adjusted = sum((r.metrics.adjusted for r in results), PointCounts(0, 0, 0, 0))
    delays = [d for r in results for d in r.metrics.delays]
    steps = sum(r.metrics.n_steps for r in results)
    event_f1, composite = _micro(points, events)
    return {
        "n_channels": len(results),
        "n_events": events.tp + events.fn,
        "n_steps": steps,
        "event": {
            "tp": events.tp,
            "fn": events.fn,
            "fp": events.fp,
            "precision": events.precision,
            "recall": events.recall,
            "f1": event_f1,
        },
        "point": {"precision": points.precision, "recall": points.recall, "f1": points.f1},
        "composite_f1": composite,
        "event_f1_macro": float(np.mean([r.metrics.events.f1 for r in results])),
        "point_adjusted_f1": adjusted.f1,
        "average_precision_macro": float(np.mean([r.metrics.average_precision for r in results])),
        "oracle_event_f1_macro": float(np.mean([r.oracle_event_f1 for r in results])),
        "delay_median_steps": float(np.median(delays)) if delays else None,
        "fp_events_per_1k_steps": 1000 * events.fp / steps if steps else 0.0,
        "ci95": _bootstrap(results, bootstrap, seed),
    }


def summarize_subsets(
    results: Sequence[ChannelResult],
    *,
    exclude_subsets: Sequence[Sequence[str]] = (),
    bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Summaries for all channels, each spacecraft, and each "all minus flagged" subset."""
    subsets: dict[str, list[ChannelResult]] = {"all": list(results)}
    for craft in sorted({r.spacecraft for r in results}):
        subsets[craft] = [r for r in results if r.spacecraft == craft]
    for checks in exclude_subsets:
        name = "excluding " + "+".join(checks)
        subsets[name] = [r for r in results if not set(r.flags) & set(checks)]
    return {name: summarize(rs, bootstrap=bootstrap, seed=seed) for name, rs in subsets.items()}


COMPOSITE_COLUMNS = ("point_tp", "point_fp", "event_tp", "event_fn")


def composite_counts(per_channel: Sequence[dict[str, Any]]) -> FloatArray:
    """``(C, 4)`` counts behind composite F1 from ``per_channel.json`` rows."""
    return np.array(
        [
            (c["points"]["tp"], c["points"]["fp"], c["events"]["tp"], c["events"]["fn"])
            for c in per_channel
        ],
        dtype=np.float64,
    )


def _composite_f1(sums: FloatArray) -> FloatArray:
    """Composite F1 from summed counts, vectorised over the leading axes."""
    ptp, pfp, etp, efn = (sums[..., k] for k in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(ptp + pfp > 0, ptp / (ptp + pfp), 0.0)
        recall = np.where(etp + efn > 0, etp / (etp + efn), 0.0)
        result: FloatArray = np.where(
            precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0
        )
    return result


def _paired(point: float, boot: FloatArray | None) -> dict[str, Any]:
    if boot is None:
        return {"delta": point, "ci95": None, "p_not_better": None}
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "delta": point,
        "ci95": (float(lo), float(hi)),
        "p_not_better": float(np.mean(boot <= 0)),
    }


def paired_mean_delta(
    a: FloatArray, b: FloatArray, *, bootstrap: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Mean of a per-channel metric (e.g. AUC-PR) for A minus B, paired over channels."""
    if a.shape != b.shape:
        raise ValueError(f"channel counts differ: {a.shape} vs {b.shape}")
    diff = a - b
    if bootstrap == 0 or a.shape[0] < 2:
        return _paired(float(diff.mean()), None)
    idx = np.random.default_rng(seed).integers(0, a.shape[0], size=(bootstrap, a.shape[0]))
    return _paired(float(diff.mean()), diff[idx].mean(axis=1))


def paired_composite_delta(
    a: FloatArray, b: FloatArray, *, bootstrap: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Composite F1 of A minus B, with a 95% CI from resampling the same channels for both.

    ``a`` and ``b`` are :func:`composite_counts` arrays over identical channels in identical
    order. ``p_not_better`` is the bootstrap fraction where A does not beat B.
    """
    if a.shape != b.shape:
        raise ValueError(f"channel counts differ: {a.shape} vs {b.shape}")
    delta = float(_composite_f1(a.sum(axis=0)) - _composite_f1(b.sum(axis=0)))
    if bootstrap == 0 or a.shape[0] < 2:
        return _paired(delta, None)
    idx = np.random.default_rng(seed).integers(0, a.shape[0], size=(bootstrap, a.shape[0]))
    boot = _composite_f1(a[idx].sum(axis=1)) - _composite_f1(b[idx].sum(axis=1))
    return _paired(delta, boot)
