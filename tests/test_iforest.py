import numpy as np
import pytest

from sathealth.detectors.iforest import FEATURE_NAMES, IsolationForestDetector, window_features


def _series(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    value = np.sin(t / 10) + 0.05 * rng.normal(size=n)
    commands = np.zeros((n, 3))
    commands[t % 40 == 0, 1] = 1.0
    return np.column_stack([value, commands])


def test_window_features_match_naive() -> None:
    x = _series(60)
    feats = window_features(x, (0,), window=8)
    assert feats.shape == (53, len(FEATURE_NAMES) + 1)
    w = x[10:18]
    v = w[:, 0]
    naive = [
        v[-1],
        v.mean(),
        v.std(),
        v.min(),
        v.max(),
        v[-1] - v[0],
        np.abs(np.diff(v)).mean(),
        w[:, 1:].sum(),
    ]
    np.testing.assert_allclose(feats[10], naive)


def test_window_features_too_short() -> None:
    with pytest.raises(ValueError, match="shorter than window"):
        window_features(np.zeros((5, 2)), (0,), window=8)


def test_detects_injected_level_shift() -> None:
    det = IsolationForestDetector(window=16, n_estimators=100, seed=0)
    det.fit(_series(2000, seed=1), _series(200, seed=2))
    test = _series(600, seed=3)
    test[300:340, 0] += 3.0
    scores = det.score(test)
    assert scores.shape == (600,)
    assert scores[300:356].mean() > np.quantile(scores[:280], 0.99)


def test_nominal_commands_are_not_false_alarms() -> None:
    det = IsolationForestDetector(window=16, n_estimators=100, seed=0)
    det.fit(_series(2000, seed=1), _series(200, seed=2))
    scores = det.score(_series(600, seed=3))
    command_steps = np.arange(40, 600, 40)
    assert scores[command_steps].mean() < np.quantile(scores, 0.9)


def test_scores_saturate_beyond_training_range() -> None:
    """Documents a known Isolation Forest limitation (see module docstring)."""
    det = IsolationForestDetector(window=16, n_estimators=100, seed=0)
    det.fit(_series(2000, seed=1), _series(200, seed=2))
    moderate, extreme = _series(200, seed=3), _series(200, seed=3)
    moderate[100:140, 0] += 3.0
    extreme[100:140, 0] += 300.0
    np.testing.assert_allclose(det.score(moderate)[120:140], det.score(extreme)[120:140])


def test_is_deterministic_for_a_seed() -> None:
    train, test = _series(500, 1), _series(200, 2)
    a = IsolationForestDetector(window=8, seed=4).fit(train, train).score(test)
    b = IsolationForestDetector(window=8, seed=4).fit(train, train).score(test)
    np.testing.assert_array_equal(a, b)


def test_short_input_scores_zero_and_unfitted_raises() -> None:
    det = IsolationForestDetector(window=8)
    with pytest.raises(RuntimeError, match="not fitted"):
        det.score(_series(20))
    det.fit(_series(100), _series(10))
    assert det.score(_series(5)).tolist() == [0.0] * 5
    assert det.params()["window"] == 8
