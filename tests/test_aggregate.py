import numpy as np
import pytest

from sathealth.evaluation.aggregate import (
    ChannelResult,
    composite_counts,
    paired_composite_delta,
    paired_mean_delta,
    summarize,
    summarize_subsets,
)
from sathealth.evaluation.metrics import ChannelMetrics, EventCounts, PointCounts


def _result(
    chan: str,
    craft: str,
    events: EventCounts,
    points: PointCounts,
    flags: tuple[str, ...] = (),
    delays: tuple[int, ...] = (),
) -> ChannelResult:
    metrics = ChannelMetrics(
        points=points,
        events=events,
        adjusted=points,
        delays=delays,
        average_precision=0.5,
        n_steps=1000,
    )
    return ChannelResult(
        chan,
        craft,
        metrics,
        oracle_event_f1=0.8,
        threshold=1.0,
        threshold_source="val",
        flags=flags,
    )


RESULTS = [
    _result(
        "A-1", "SMAP", EventCounts(tp=3, fn=1, fp=1), PointCounts(30, 10, 10, 950), delays=(2, 4)
    ),
    _result(
        "A-2",
        "SMAP",
        EventCounts(tp=0, fn=1, fp=3),
        PointCounts(0, 30, 20, 950),
        flags=("constant_train",),
    ),
    _result("M-1", "MSL", EventCounts(tp=1, fn=0, fp=0), PointCounts(10, 0, 0, 990), delays=(10,)),
]


def test_micro_averaging() -> None:
    s = summarize(RESULTS, bootstrap=0)
    assert s["event"] == {
        "tp": 4,
        "fn": 2,
        "fp": 4,
        "precision": 0.5,
        "recall": pytest.approx(4 / 6),
        "f1": pytest.approx(4 / 7),
    }
    assert s["point"]["precision"] == pytest.approx(40 / 80)
    assert s["composite_f1"] == pytest.approx(2 * 0.5 * (4 / 6) / (0.5 + 4 / 6))
    assert s["delay_median_steps"] == 4.0
    assert s["fp_events_per_1k_steps"] == pytest.approx(4 / 3)
    assert s["n_events"] == 6
    assert s["event_f1_macro"] == pytest.approx((0.75 + 0.0 + 1.0) / 3)


def test_bootstrap_ci_brackets_point_estimate_and_is_seeded() -> None:
    s1 = summarize(RESULTS, bootstrap=2000, seed=1)
    s2 = summarize(RESULTS, bootstrap=2000, seed=1)
    lo, hi = s1["ci95"]["event_f1"]
    assert lo <= s1["event"]["f1"] <= hi
    assert s1["ci95"] == s2["ci95"]


def test_subsets() -> None:
    subsets = summarize_subsets(
        RESULTS, exclude_subsets=[["constant_train"], ["constant_train", "other"]], bootstrap=0
    )
    assert list(subsets) == [
        "all",
        "MSL",
        "SMAP",
        "excluding constant_train",
        "excluding constant_train+other",
    ]
    assert subsets["excluding constant_train"]["n_channels"] == 2
    assert subsets["MSL"]["event"]["f1"] == 1.0


def test_empty_summary() -> None:
    assert summarize([]) == {"n_channels": 0}


def test_channel_result_serialises() -> None:
    d = RESULTS[0].to_dict()
    assert d["events"]["f1"] == pytest.approx(0.75)
    assert d["flags"] == []


def _counts(rows: list[tuple[int, int, int, int]]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


def test_paired_delta_point_estimate_and_identity() -> None:
    # A: point precision 8/10, event recall 3/4 -> 2*.8*.75/1.55; B: 4/10 and 2/4.
    a = _counts([(4, 1, 2, 0), (4, 1, 1, 1)])
    b = _counts([(2, 3, 1, 1), (2, 3, 1, 1)])
    out = paired_composite_delta(a, b, bootstrap=200, seed=0)
    assert out["delta"] == pytest.approx(2 * 0.8 * 0.75 / 1.55 - 2 * 0.4 * 0.5 / 0.9)
    same = paired_composite_delta(a, a, bootstrap=200, seed=0)
    assert same["delta"] == 0.0
    assert same["ci95"] == (0.0, 0.0)
    assert same["p_not_better"] == 1.0


def test_paired_delta_is_tighter_than_unpaired_intervals() -> None:
    """Channel difficulty varies a lot; a constant improvement is only visible when paired."""
    rng = np.random.default_rng(0)
    hard = rng.integers(1, 18, size=40)
    b = np.column_stack([hard, 20 - hard, hard // 4, 5 - hard // 4]).astype(np.float64)
    a = b + np.array([2.0, -2.0, 0.0, 0.0])  # 2 fewer false-alarm steps everywhere
    out = paired_composite_delta(a, b, bootstrap=500, seed=1)
    assert out["delta"] > 0
    assert out["ci95"][0] > 0
    assert out["p_not_better"] < 0.025


def test_paired_delta_edge_cases() -> None:
    a = _counts([(1, 0, 1, 0)])
    assert paired_composite_delta(a, a, bootstrap=100)["ci95"] is None
    assert paired_composite_delta(a, a, bootstrap=0)["p_not_better"] is None
    with pytest.raises(ValueError, match="channel counts differ"):
        paired_composite_delta(a, _counts([(1, 0, 1, 0), (1, 0, 1, 0)]))


def test_composite_counts_reads_per_channel_rows() -> None:
    row = {"points": {"tp": 3, "fp": 1}, "events": {"tp": 1, "fn": 2}}
    np.testing.assert_array_equal(composite_counts([row]), [[3, 1, 1, 2]])


def test_paired_mean_delta() -> None:
    a = np.array([0.9, 0.5, 0.7, 0.4])
    out = paired_mean_delta(a, a - 0.1, bootstrap=200, seed=0)
    assert out["delta"] == pytest.approx(0.1)
    assert out["ci95"] == pytest.approx((0.1, 0.1))
    assert out["p_not_better"] == 0.0
    assert paired_mean_delta(a, a, bootstrap=0)["ci95"] is None
    with pytest.raises(ValueError, match="channel counts differ"):
        paired_mean_delta(a, a[:2])
