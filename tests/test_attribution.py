import numpy as np
import torch

from sathealth.explain.attribution import (
    Attribution,
    deletion_effect,
    forecast_error_fn,
    integrated_gradients,
)
from sathealth.models.networks import LSTMForecaster
from sathealth.models.runtime import seed_torch


def test_ig_is_exact_for_a_linear_function() -> None:
    weights = torch.tensor([[1.0, -2.0], [0.5, 3.0], [0.0, 1.0]])  # (W=3, D=2)

    def f(x: torch.Tensor) -> torch.Tensor:
        return (x * weights).sum(dim=(1, 2))

    x = np.array([[1.0, 1.0], [2.0, -1.0], [4.0, 0.5]])
    attr = integrated_gradients(f, x, steps=8)
    np.testing.assert_allclose(attr.values, x * weights.numpy(), atol=1e-5)
    assert attr.completeness_gap < 1e-5


def test_ig_completeness_on_the_lstm_forecaster() -> None:
    seed_torch(0)
    model = LSTMForecaster(4, 1, hidden=8, layers=2, dropout=0.0).eval()
    window = np.random.default_rng(0).normal(size=(16, 4))
    attr = integrated_gradients(forecast_error_fn(model, np.array([2.0])), window, steps=64)
    assert attr.values.shape == (16, 4)
    assert attr.completeness_gap < 0.02


def test_importance_summaries() -> None:
    values = np.zeros((4, 3))
    values[3, 2] = -5.0  # most recent step, feature 2
    values[0, 1] = 2.0  # oldest step, feature 1
    attr = Attribution(values, target=1.0, baseline_target=0.0)
    assert attr.top_features(2) == [2, 1]
    assert attr.top_lags(2) == [1, 4]
    np.testing.assert_array_equal(attr.lag_importance(), [5.0, 0.0, 0.0, 2.0])


def test_deletion_of_top_inputs_beats_random() -> None:
    weights = torch.zeros(10, 2)
    weights[9, 0] = 5.0  # the only input that matters

    def f(x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = ((x * weights).sum(dim=(1, 2)) - 0.0) ** 2
        return out

    window = np.ones((10, 2))
    attr = integrated_gradients(f, window)
    top, random = deletion_effect(f, window, attr, 0.05, np.random.default_rng(0))
    assert top == 1.0
    assert random < top
