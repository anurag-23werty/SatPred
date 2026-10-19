import pytest

from sathealth.evaluation.aggregate import ChannelResult, summarize, summarize_subsets
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
