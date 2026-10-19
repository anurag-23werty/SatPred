import numpy as np
import pytest

from sathealth.explain.onset import estimate_onset


def test_walks_back_while_surprise_stays_elevated() -> None:
    s = np.array([0.2, 0.5, 1.2, 1.8, 2.5, 3.5, 4.0])
    assert estimate_onset(s, alarm_start=5) == 2


def test_abrupt_start_and_lookback_cap() -> None:
    assert estimate_onset(np.array([0.0, 0.0, 5.0]), alarm_start=2) == 2
    s = np.full(100, 2.0)
    assert estimate_onset(s, alarm_start=90, max_lookback=10) == 80
    assert estimate_onset(s, alarm_start=90, level=3.0) == 90


def test_rejects_alarm_outside_series() -> None:
    with pytest.raises(ValueError, match="outside"):
        estimate_onset(np.zeros(5), alarm_start=5)
