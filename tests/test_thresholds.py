import numpy as np
import pytest

from sathealth.evaluation.thresholds import fit_threshold


def test_quantile_threshold_limits_nominal_alarm_rate() -> None:
    rng = np.random.default_rng(0)
    val = np.abs(rng.normal(size=100_000))
    thr = fit_threshold(val, val, quantile=0.99)
    assert thr.source == "val"
    fresh = np.abs(rng.normal(size=100_000))
    assert (fresh > thr.value).mean() == pytest.approx(0.01, abs=0.002)


def test_sigma_threshold() -> None:
    val = np.array([0.0, 2.0] * 50)
    thr = fit_threshold(val, val, method="val_sigma", sigma=3.0)
    assert thr.value == pytest.approx(1.0 + 3.0 * 1.0)


def test_short_validation_falls_back_to_train() -> None:
    thr = fit_threshold(np.ones(10), np.full(500, 2.0), min_rows=50)
    assert (thr.source, thr.value, thr.n_rows) == ("train", 2.0, 500)


def test_zero_scores_give_strict_zero_threshold() -> None:
    thr = fit_threshold(np.zeros(100), np.zeros(100))
    assert thr.value == 0.0
    assert not (np.zeros(5) > thr.value).any()


def test_non_finite_scores_are_ignored_and_empty_rejected() -> None:
    val = np.r_[np.full(99, 1.0), np.nan]
    assert fit_threshold(val, val, min_rows=1).value == 1.0
    with pytest.raises(ValueError, match="finite"):
        fit_threshold(np.full(60, np.nan), np.zeros(1))
