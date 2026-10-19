import math

import numpy as np
import pytest

from sathealth.decision.calibration import GeneralizedPareto, TailCalibrator, fit_gpd_pwm


def _normal_tail(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2))


@pytest.mark.parametrize(("shape", "scale"), [(0.2, 1.5), (-0.2, 1.0)])
def test_pwm_recovers_gpd_parameters(shape: float, scale: float) -> None:
    u = np.random.default_rng(0).random(50_000)
    sample = scale / shape * ((1 - u) ** -shape - 1)  # inverse CDF of the GPD
    fit = fit_gpd_pwm(sample, shape_bounds=(-0.5, 0.5))
    assert fit.shape == pytest.approx(shape, abs=0.03)
    assert fit.scale == pytest.approx(scale, rel=0.05)


def test_negative_shapes_are_clamped_keeping_the_mean_excess() -> None:
    u = np.random.default_rng(6).random(50_000)
    sample = 1.0 / -0.3 * ((1 - u) ** 0.3 - 1)  # bounded tail, shape -0.3
    fit = fit_gpd_pwm(sample)
    assert fit.shape == 0.0
    assert fit.scale == pytest.approx(sample.mean())


def test_pwm_exponential_and_small_samples() -> None:
    sample = np.random.default_rng(1).exponential(2.0, size=50_000)
    fit = fit_gpd_pwm(sample)
    assert fit.shape == pytest.approx(0.0, abs=0.03)
    assert fit.scale == pytest.approx(2.0, rel=0.05)
    assert fit_gpd_pwm(np.array([1.0, 3.0])) == GeneralizedPareto(0.0, 2.0)
    with pytest.raises(ValueError, match="no excesses"):
        fit_gpd_pwm(np.zeros(0))


def test_gpd_survival_bounded_support() -> None:
    gpd = GeneralizedPareto(shape=-0.5, scale=1.0)  # upper endpoint at y = 2
    np.testing.assert_allclose(gpd.survival(np.array([0.0, 1.0, 2.5])), [1.0, 0.25, 0.0])
    np.testing.assert_allclose(GeneralizedPareto(0.0, 2.0).survival(np.array([2.0])), [math.e**-1])


def test_evt_extrapolates_beyond_the_validation_maximum() -> None:
    val = np.random.default_rng(2).normal(size=5_000)
    evt = TailCalibrator("evt", tail_quantile=0.95).fit(val)
    ecdf = TailCalibrator("ecdf").fit(val)
    x = np.array([4.0, 4.5])  # beyond almost every validation score
    truth = -np.log10([_normal_tail(v) for v in x])
    s = evt.surprise(x)
    assert np.all(s <= truth + 0.3), "never overconfident: that would mean false alarms"
    assert np.all(s >= truth - 1.5)
    assert ecdf.surprise(x).max() <= np.log10(val.size + 1) + 1e-9  # ECDF cannot go further
    assert s.min() > ecdf.surprise(x).max()
    assert s[1] > s[0]


def test_calibrated_alarm_rate_matches_the_nominal_risk() -> None:
    rng = np.random.default_rng(3)
    cal = TailCalibrator("evt", tail_quantile=0.95).fit(rng.gumbel(size=20_000))
    fresh = cal.surprise(rng.gumbel(size=400_000))
    rate = float((fresh >= 3).mean())
    assert 0.5e-3 < rate < 2e-3, "surprise >= 3 should fire about once per 1,000 nominal steps"


def test_surprise_is_monotone_capped_and_documented() -> None:
    val = np.random.default_rng(4).exponential(size=500)
    cal = TailCalibrator("evt", cap=12.0).fit(val)
    x = np.linspace(-1, 200, 500)
    s = cal.surprise(x)
    assert np.all(np.diff(s) >= -1e-12)
    assert s.max() == 12.0
    assert s.min() == pytest.approx(0.0, abs=0.01)
    info = cal.to_dict()
    assert info["method"] == "evt"
    assert 0.04 < info["zeta"] < 0.06


def test_flat_validation_scores_make_any_excess_extreme() -> None:
    cal = TailCalibrator("evt").fit(np.zeros(300))  # e.g. a limit check that never fired
    s = cal.surprise(np.array([0.0, 0.5]))
    assert s[0] < 0.01
    assert s[1] == 12.0


def test_calibrator_errors_and_passthrough() -> None:
    with pytest.raises(ValueError, match="no finite"):
        TailCalibrator().fit(np.array([np.nan]))
    with pytest.raises(RuntimeError, match="not fitted"):
        TailCalibrator().surprise(np.zeros(3))
    raw = np.array([0.1, 5.0])
    np.testing.assert_array_equal(TailCalibrator("none").fit(raw).surprise(raw), raw)
    assert TailCalibrator("ecdf").fit(raw).to_dict() == {"method": "ecdf", "n": 2}
