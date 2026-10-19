import numpy as np

from sathealth.config import DecisionConfig
from sathealth.decision.adaptive import adaptive_surprises
from sathealth.decision.alarms import hysteresis
from sathealth.decision.calibration import TailCalibrator


def _adaptive_cfg(**kw: object) -> DecisionConfig:
    return DecisionConfig(recalibrate_every=50, recalibrate_window=500, **kw)  # type: ignore[arg-type]


def test_adaptive_recalibration_absorbs_a_nominal_level_shift() -> None:
    rng = np.random.default_rng(7)
    val = rng.gumbel(size=600)
    test = rng.gumbel(size=3000) + 3.0  # same process, shifted: test period differs from val
    d = _adaptive_cfg(inputs=("a",))
    static = TailCalibrator().fit(val).surprise(test)
    (adaptive,) = adaptive_surprises([val], [test], d, every=50)
    assert (static[1500:] >= 3).mean() > 0.03, "static calibration floods with alarms"
    assert (adaptive[1500:] >= 3).mean() < 0.01, "adaptive calibration settles"


def test_adaptive_never_learns_an_ongoing_anomaly_as_normal() -> None:
    rng = np.random.default_rng(8)
    val = rng.normal(size=600)
    test = rng.normal(size=3000)
    test[1000:2500] += 8.0  # long anomaly: in alarm, so excluded from the reference
    (adaptive,) = adaptive_surprises([val], [test], _adaptive_cfg(inputs=("a",)), every=50)
    assert (adaptive[1000:2500] >= 3).mean() > 0.95


def test_adaptive_is_causal() -> None:
    rng = np.random.default_rng(9)
    val, test = rng.normal(size=400), rng.normal(size=1000)
    changed = test.copy()
    changed[700:] += 20
    d = _adaptive_cfg(inputs=("a",))
    (first,) = adaptive_surprises([val], [test], d, every=50)
    (second,) = adaptive_surprises([val], [changed], d, every=50)
    np.testing.assert_array_equal(first[:700], second[:700])


def test_hysteresis_can_continue_an_alarm() -> None:
    s = np.array([2.5, 1.0])
    assert hysteresis(s, 3, 2, start_active=True).tolist() == [True, False]
