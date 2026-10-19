import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from sathealth.evaluation.metrics import (
    EventCounts,
    PointCounts,
    average_precision,
    detection_delays,
    evaluate_predictions,
    event_counts,
    mask_to_events,
    oracle_event_f1,
    point_adjust,
    point_counts,
)


def _mask(n: int, *ranges: tuple[int, int]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in ranges:
        m[s : e + 1] = True
    return m


def test_mask_to_events_and_merging() -> None:
    m = _mask(20, (2, 4), (6, 6), (15, 19))
    assert mask_to_events(m) == [(2, 4), (6, 6), (15, 19)]
    assert mask_to_events(m, min_gap=1) == [(2, 6), (15, 19)]
    assert mask_to_events(np.zeros(5, dtype=bool)) == []


def test_point_counts() -> None:
    counts = point_counts(_mask(10, (0, 3)), _mask(10, (2, 5)))
    assert counts == PointCounts(tp=2, fp=2, fn=2, tn=4)
    assert counts.precision == counts.recall == counts.f1 == 0.5
    with pytest.raises(ValueError, match="shape"):
        point_counts(np.zeros(3, dtype=bool), np.zeros(4, dtype=bool))


def test_event_counts_hundman_definition() -> None:
    segments = [(10, 20), (40, 50), (70, 80)]
    pred = _mask(100, (15, 15), (45, 60), (90, 95), (0, 2))  # hits 2 segments, 2 false events
    counts = event_counts(pred, segments)
    assert counts == EventCounts(tp=2, fn=1, fp=2)
    assert counts.precision == 0.5
    assert counts.recall == pytest.approx(2 / 3)


def test_early_tolerance_turns_early_alarm_into_detection() -> None:
    segments = [(50, 60)]
    pred = _mask(100, (45, 47))
    assert event_counts(pred, segments) == EventCounts(tp=0, fn=1, fp=1)
    assert event_counts(pred, segments, early_tolerance=5) == EventCounts(tp=1, fn=0, fp=0)
    assert detection_delays(pred, segments, early_tolerance=5) == [-5]


def test_detection_delay() -> None:
    assert detection_delays(_mask(100, (55, 56)), [(50, 60), (80, 90)]) == [5]


def test_point_adjust_inflates_a_single_hit() -> None:
    labels = _mask(100, (10, 59))
    pred = _mask(100, (30, 30))
    assert point_counts(pred, labels).recall == pytest.approx(0.02)
    adjusted = point_counts(point_adjust(pred, [(10, 59)]), labels)
    assert adjusted.recall == 1.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_average_precision_matches_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    labels = rng.random(500) > 0.8
    scores = np.round(rng.random(500) + labels * 0.3, 2)  # rounding creates ties
    assert average_precision(scores, labels) == pytest.approx(
        average_precision_score(labels, scores)
    )


def test_average_precision_without_positives() -> None:
    assert average_precision(np.arange(5.0), np.zeros(5, dtype=bool)) == 0.0


def test_composite_f1_combines_point_precision_and_event_recall() -> None:
    labels = _mask(100, (10, 19), (50, 59))
    pred = _mask(100, (10, 11))  # precise, but finds only one of two events
    metrics = evaluate_predictions(pred, labels, pred.astype(float), [(10, 19), (50, 59)])
    assert metrics.points.precision == 1.0
    assert metrics.events.recall == 0.5
    assert metrics.composite_f1 == pytest.approx(2 / 3)
    assert metrics.n_steps == 100


def test_oracle_finds_separating_threshold() -> None:
    scores = np.zeros(100)
    scores[40:45] = 5.0
    scores[80] = 1.0  # a distractor below the right threshold
    best_f1, thr = oracle_event_f1(scores, [(40, 44)])
    assert best_f1 == 1.0
    assert 1.0 <= thr < 5.0


def test_counts_add() -> None:
    assert EventCounts(1, 2, 3) + EventCounts(1, 1, 1) == EventCounts(2, 3, 4)
    assert PointCounts(1, 1, 1, 1) + PointCounts(1, 0, 0, 0) == PointCounts(2, 1, 1, 1)
