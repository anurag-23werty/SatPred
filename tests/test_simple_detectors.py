import numpy as np
import pytest

from sathealth.detectors.limit import LimitCheck
from sathealth.detectors.reference import RandomScore
from sathealth.detectors.zscore import RobustZScore


def _series(values: list[float]) -> np.ndarray:
    return np.column_stack([np.asarray(values, dtype=float), np.zeros(len(values))])


def test_limit_check_scores_distance_outside_band() -> None:
    det = LimitCheck(margin=0.1).fit(_series([0.0, 10.0]), _series([5.0]))
    scores = det.score(_series([5.0, 10.5, 12.0, -3.0]))
    np.testing.assert_allclose(scores, [0.0, 0.0, 1.0, 2.0])  # band is [-1, 11]


def test_limit_check_on_flat_channel_uses_absolute_margin() -> None:
    det = LimitCheck(margin=0.05).fit(_series([-1.0] * 10), _series([-1.0]))
    np.testing.assert_allclose(det.score(_series([-1.0, -1.04, -0.5])), [0.0, 0.0, 0.45])


def test_limit_check_ignores_command_columns() -> None:
    det = LimitCheck().fit(_series([0.0, 1.0]), _series([0.5]))
    x = _series([0.5])
    x[0, 1] = 99.0
    assert det.score(x).tolist() == [0.0]


def test_robust_zscore() -> None:
    rng = np.random.default_rng(0)
    train = _series(rng.normal(3.0, 2.0, 50_000).tolist())
    det = RobustZScore().fit(train, train)
    assert det.score(_series([3.0, 9.0]))[1] == pytest.approx(3.0, rel=0.05)


def test_robust_zscore_flat_channel_does_not_explode() -> None:
    det = RobustZScore().fit(_series([1.0] * 20), _series([1.0]))
    assert det.score(_series([1.0, 3.0])).tolist() == [0.0, 2.0]


def test_random_is_deterministic_and_uniform() -> None:
    det = RandomScore(seed=3).fit(np.zeros((1, 1)), np.zeros((1, 1)))
    a, b = det.score(np.zeros((5000, 2))), det.score(np.zeros((5000, 2)))
    np.testing.assert_array_equal(a, b)
    assert 0.45 < a.mean() < 0.55
    assert det.params()["seed"] == 3


@pytest.mark.parametrize("det", [LimitCheck(), RobustZScore()])
def test_unfitted_detectors_raise(det: LimitCheck | RobustZScore) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        det.score(np.zeros((2, 2)))
