import numpy as np
import pytest

from sathealth.config import DecisionConfig
from sathealth.ops.health import HealthTracker, step_weight, subsystem, time_to_threshold


def test_health_index_is_severity_weighted_share_of_window() -> None:
    tracker = HealthTracker(window=4, sample_every=2)
    assert tracker.index == 100.0
    for w in (0.0, 1.0, 0.5, 0.0):
        tracker.update(w)
    assert tracker.index == pytest.approx(100 * (1 - 1.5 / 4))
    tracker.update(0.0)  # the oldest step (0.0) leaves the window: still 1.5
    assert tracker.index == pytest.approx(100 * (1 - 1.5 / 4))
    tracker.update(0.0)  # now the 1.0 leaves
    assert tracker.index == pytest.approx(100 * (1 - 0.5 / 4))
    assert not tracker.in_alarm
    assert list(tracker.history) == [(1, 50.0), (3, 62.5), (5, 87.5)]  # partial window at step 1
    with pytest.raises(ValueError, match="window"):
        HealthTracker(window=0)


def test_step_weight_follows_severity_levels() -> None:
    d = DecisionConfig()
    assert step_weight(9.0, False, d) == 0.0
    assert step_weight(3.5, True, d) == 0.25
    assert step_weight(5.0, True, d) == 0.5
    assert step_weight(7.0, True, d) == 1.0


def test_downward_trend_gives_time_with_interval() -> None:
    rng = np.random.default_rng(0)
    x = np.arange(0, 1000, 50, dtype=np.float64)
    y = 100 - 0.03 * x + rng.normal(0, 0.3, x.size)  # reaches 50 at x ~ 1667
    forecast = time_to_threshold(x, y, 50.0)
    assert forecast is not None
    assert forecast.steps_to_threshold == pytest.approx(1667 - 950, rel=0.15)
    assert forecast.interval80 is not None
    lo, hi = forecast.interval80
    assert lo < forecast.steps_to_threshold < hi
    assert "not a validated remaining-useful-life" in forecast.label


@pytest.mark.parametrize(
    "y",
    [
        np.full(20, 100.0),  # flat
        100 - np.linspace(0, 0.5, 20),  # barely downward: would extrapolate far too far
        100 + np.random.default_rng(1).normal(0, 5, 20),  # noise, no reliable trend
    ],
)
def test_no_meaningful_trend_reports_no_time(y: np.ndarray) -> None:
    forecast = time_to_threshold(np.arange(20, dtype=np.float64) * 50, y, 50.0)
    assert forecast is not None
    assert forecast.steps_to_threshold is None


def test_trend_edge_cases() -> None:
    assert time_to_threshold(np.arange(3.0), np.arange(3.0)) is None
    assert time_to_threshold(np.zeros(6), np.arange(6.0)) is None
    below = time_to_threshold(np.arange(6.0), np.full(6, 40.0) - np.arange(6.0))
    assert below is not None
    assert below.steps_to_threshold is None
    assert subsystem("A-12") == "A"
