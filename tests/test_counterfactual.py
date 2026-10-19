import numpy as np
import pytest
import torch
from torch import nn

from sathealth.explain.counterfactual import HorizonBand, calibrate_band, rollout


class _Persistence(nn.Module):
    """Predicts the last value plus 1 for every command fired at the last step."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nxt = x[:, :, :1] + x[:, :, 1:].sum(dim=-1, keepdim=True)
        return nxt


def test_rollout_feeds_back_predictions_and_uses_real_commands() -> None:
    history = np.zeros((1, 3, 2))
    future = np.zeros((1, 4, 2))
    future[0, 1, 1] = 1.0  # a command sent at the second future step
    future[0, :, 0] = 99.0  # future values must be ignored
    out = rollout(_Persistence(), history, future, (0,), window=3)
    np.testing.assert_array_equal(out[0, :, 0], [0.0, 0.0, 1.0, 1.0])


def test_rollout_accepts_short_history() -> None:
    out = rollout(_Persistence(), np.ones((2, 1, 2)) * [1, 0], np.zeros((2, 3, 2)), (0,), 5)
    assert out.shape == (2, 3, 1)
    np.testing.assert_array_equal(out[:, :, 0], np.ones((2, 3)))


def test_band_is_monotone_and_extends_beyond_calibration() -> None:
    rng = np.random.default_rng(0)
    series = np.column_stack([np.cumsum(rng.normal(size=400)), np.zeros(400)])
    band = calibrate_band(_Persistence(), series, (0,), window=8, horizon=50, n_starts=30, rng=rng)
    assert band.lower.shape == (50, 1)
    assert np.all(np.diff(band.upper[:, 0]) >= 0)
    assert np.all(np.diff(band.lower[:, 0]) <= 0)
    assert band.upper[-1, 0] > band.upper[0, 0], "random walk: uncertainty grows"
    low, high = band.at(70)
    assert low.shape == (70, 1)
    assert high[-1, 0] == band.upper[-1, 0]


def test_band_needs_enough_data() -> None:
    with pytest.raises(ValueError, match="too short"):
        calibrate_band(_Persistence(), np.zeros((15, 2)), (0,), 8, 50, 5, np.random.default_rng(0))
    band = HorizonBand(np.zeros((1, 1)), np.ones((1, 1)), "val", 1)
    assert band.at(3)[1].tolist() == [[1.0], [1.0], [1.0]]
