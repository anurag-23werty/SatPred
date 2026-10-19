import numpy as np
import pytest

from sathealth.ops.drift import DriftMonitor, psi, quantile_edges


def test_psi_basics() -> None:
    assert psi(np.array([10, 10]), np.array([5, 5])) == pytest.approx(0.0)
    assert psi(np.array([50, 50]), np.array([90, 10])) > 0.25
    edges = quantile_edges(np.r_[np.zeros(50), np.ones(50)], bins=10)
    assert len(edges) < 9, "discrete data: duplicate quantile edges are merged"


def test_stable_stream_is_not_flagged_and_shift_is() -> None:
    rng = np.random.default_rng(0)
    monitor = DriftMonitor(rng.normal(size=5000), window=500)
    assert monitor.report().window == 0
    for v in rng.normal(size=500):
        monitor.update(float(v), alarm=False)
    stable = monitor.report()
    assert not stable.drifting
    assert stable.psi < 0.1
    for v in rng.normal(loc=2.0, size=500):
        monitor.update(float(v), alarm=False)
    shifted = monitor.report()
    assert shifted.drifting
    assert "PSI" in shifted.reasons[0]


def test_alarm_share_is_reported_but_does_not_raise_the_flag() -> None:
    rng = np.random.default_rng(1)
    monitor = DriftMonitor(rng.normal(size=1000), window=200)
    for i, v in enumerate(rng.normal(size=200)):
        monitor.update(float(v), alarm=i % 20 == 0)  # 5% of steps in alarm
    report = monitor.report()
    assert report.alarm_share == pytest.approx(0.05)
    assert not report.drifting


def test_threshold_is_calibrated_on_training_windows() -> None:
    """A training series that moves between modes sets a higher bar than iid noise."""
    rng = np.random.default_rng(2)
    modes = np.r_[rng.normal(0, 1, 2000), rng.normal(3, 1, 2000)]
    moded = DriftMonitor(modes, window=500)
    iid = DriftMonitor(rng.normal(size=4000), window=500)
    assert moded.threshold > iid.threshold >= 0.25
    for v in rng.normal(3, 1, 500):  # one known mode: not drift
        moded.update(float(v), alarm=False)
    assert not moded.report().drifting
    for v in rng.normal(9, 1, 500):  # a level never seen in training: drift
        moded.update(float(v), alarm=False)
    assert moded.report().drifting
