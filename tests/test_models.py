import numpy as np
import pytest
import torch

from sathealth.config import TrainingConfig
from sathealth.models.networks import LSTMForecaster, TransformerAutoencoder, sinusoidal_encoding
from sathealth.models.runtime import resolve_device, seed_torch
from sathealth.models.scoring import ewma
from sathealth.models.training import train_model


def test_ewma_matches_reference_recursion() -> None:
    x = np.random.default_rng(0).random(50)
    alpha = 2 / (10 + 1)
    expected = [x[0]]
    for value in x[1:]:
        expected.append(alpha * value + (1 - alpha) * expected[-1])
    np.testing.assert_allclose(ewma(x, 10), expected)


def test_ewma_is_causal_and_span_one_is_identity() -> None:
    x = np.random.default_rng(1).random(40)
    changed = x.copy()
    changed[30:] += 100
    np.testing.assert_array_equal(ewma(x, 5)[:30], ewma(changed, 5)[:30])
    np.testing.assert_array_equal(ewma(x, 1), x)
    assert ewma(np.zeros(0), 5).size == 0
    with pytest.raises(ValueError, match="span"):
        ewma(x, 0)


def test_resolve_device() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "mps", "cuda"}
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")
    if not torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match="MPS"):
            resolve_device("mps")


def test_seed_torch_is_reproducible() -> None:
    seed_torch(3)
    first = torch.randn(4)
    seed_torch(3)
    assert torch.equal(first, torch.randn(4))
    assert torch.equal(
        torch.randperm(10, generator=seed_torch(5)), torch.randperm(10, generator=seed_torch(5))
    )


def test_lstm_forecaster_is_causal() -> None:
    seed_torch(0)
    net = LSTMForecaster(n_features=3, n_outputs=1, hidden=8, layers=2, dropout=0.0).eval()
    x = torch.randn(2, 20, 3)
    changed = x.clone()
    changed[:, 12:] += 5.0
    with torch.no_grad():
        out, out_changed = net(x), net(changed)
    assert out.shape == (2, 20, 1)
    torch.testing.assert_close(out[:, :12], out_changed[:, :12])
    assert not torch.allclose(out[:, 12:], out_changed[:, 12:])


def test_transformer_autoencoder_shapes_and_prefixes() -> None:
    seed_torch(0)
    net = TransformerAutoencoder(
        4, 2, window=16, d_model=8, heads=2, layers=1, feedforward=16, latent=3, dropout=0.0
    ).eval()
    with torch.no_grad():
        assert net(torch.randn(5, 16, 4)).shape == (5, 16, 2)
        assert net(torch.randn(1, 3, 4)).shape == (1, 3, 2)  # shorter prefix


def test_sinusoidal_encoding() -> None:
    enc = sinusoidal_encoding(10, 6)
    assert enc.shape == (10, 6)
    torch.testing.assert_close(enc[0], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))


def _mse(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    loss: torch.Tensor = torch.nn.functional.mse_loss(model(batch), batch)
    return loss


def test_train_model_learns_and_restores_best() -> None:
    gen = seed_torch(0)
    model = torch.nn.Linear(3, 3)
    train = torch.randn(64, 3)
    cfg = TrainingConfig(epochs=40, batch_size=16, learning_rate=1e-2, patience=3)
    history = train_model(model, _mse, train, torch.randn(16, 3), cfg, gen)
    assert history.monitor == "val"
    assert history.train_loss[-1] < history.train_loss[0]
    assert history.best_loss == min(history.val_loss)
    assert history.val_loss[history.best_epoch - 1] == history.best_loss
    assert history.to_dict()["best_epoch"] == history.best_epoch


def test_train_model_stops_early_and_falls_back_to_train_loss() -> None:
    gen = seed_torch(0)
    cfg = TrainingConfig(epochs=50, learning_rate=1e-9, patience=2, min_delta=1.0)
    history = train_model(
        torch.nn.Linear(2, 2), _mse, torch.randn(8, 2), torch.zeros(0, 2), cfg, gen
    )
    assert history.monitor == "train"
    assert history.val_loss == ()
    assert history.stopped_early
    assert len(history.train_loss) == 3  # best at epoch 1, then `patience` stale epochs


def test_train_model_needs_windows() -> None:
    with pytest.raises(ValueError, match="no training windows"):
        train_model(
            torch.nn.Linear(2, 2),
            _mse,
            torch.zeros(0, 2),
            torch.zeros(0, 2),
            TrainingConfig(),
            seed_torch(0),
        )
