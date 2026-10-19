import numpy as np
import pytest

from sathealth.decision.alarms import drop_short, hysteresis, merge_gaps, ndt_mask, prune
from sathealth.decision.fusion import fuse


def test_fusion_methods() -> None:
    a, b = np.array([1.0, 4.0]), np.array([3.0, 0.0])
    np.testing.assert_array_equal(fuse([a, b], "mean"), [2.0, 2.0])
    np.testing.assert_array_equal(fuse([a, b], "max"), [3.0, 4.0])
    np.testing.assert_array_equal(fuse([a, b], "min"), [1.0, 0.0])
    with pytest.raises(ValueError, match="nothing"):
        fuse([])
    with pytest.raises(ValueError, match="unknown"):
        fuse([a], "median")  # type: ignore[arg-type]


def test_hysteresis_keeps_one_alarm_through_a_dip() -> None:
    s = np.array([0, 3.2, 2.5, 3.1, 2.4, 1.0, 2.5, 3.0])
    assert hysteresis(s, on=3, off=2).tolist() == [0, 1, 1, 1, 1, 0, 0, 1]
    assert hysteresis(s, on=3, off=3).tolist() == [0, 1, 0, 1, 0, 0, 0, 1]
    with pytest.raises(ValueError, match="off"):
        hysteresis(s, on=2, off=3)


def test_merge_gaps_and_drop_short() -> None:
    mask = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 1], dtype=bool)
    assert merge_gaps(mask, 2).astype(int).tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
    assert merge_gaps(mask, 0).tolist() == mask.tolist()
    assert drop_short(mask, 2).astype(int).tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    assert drop_short(mask, 1).tolist() == mask.tolist()


def test_ndt_flags_a_clear_excursion_only() -> None:
    rng = np.random.default_rng(5)
    errors = np.abs(rng.normal(size=3000)) * 0.1 + 0.1
    errors[2000:2030] += 3.0
    mask = ndt_mask(errors, window=1000, step=100)
    assert mask[2000:2030].all()
    assert mask.sum() < 60
    assert not ndt_mask(np.ones(50), window=100, step=10).any()  # flat: nothing to find


def test_prune_drops_runs_close_to_the_noise_floor() -> None:
    scores = np.full(100, 1.0)
    scores[10:13] = 10.0  # clear anomaly
    scores[50:52] = 1.05  # barely above the floor
    mask = scores > 1.01
    pruned = prune(mask, scores, margin=0.13)
    assert pruned[10:13].all()
    assert not pruned[50:52].any()
    assert not prune(np.zeros(5, dtype=bool), np.ones(5), 0.13).any()
