"""The two network architectures of Phase 3.

Both read windows of shape ``(B, L, D)``: every feature of the channel, i.e. the scaled
telemetry value(s) *and* the one-hot command flags, because commands explain many nominal
level changes. Both predict only the value columns ``(B, L, V)``.

* :class:`LSTMForecaster`: next-step forecaster in the style of Telemanom (Hundman et al.,
  KDD 2018). Output ``t`` predicts input ``t + 1``; the LSTM is causal, so the output at
  position ``t`` only depends on steps ``<= t``.
* :class:`TransformerAutoencoder`: reconstructs the whole window from a ``latent``-sized
  code, mean-pooled over time. The bottleneck stops it from learning the identity, which an
  attention model with direct input-output paths otherwise learns quickly. Positional
  encodings are sinusoidal so prefixes shorter than the training window work unchanged.
"""

import math

import torch
from torch import Tensor, nn


class LSTMForecaster(nn.Module):
    def __init__(
        self, n_features: int, n_outputs: int, hidden: int, layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, n_outputs)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.lstm(x)
        prediction: Tensor = self.head(out)
        return prediction


def sinusoidal_encoding(length: int, d_model: int) -> Tensor:
    """Fixed positional encoding of Vaswani et al. 2017, shape ``(length, d_model)``."""
    position = torch.arange(length, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(1e4) / d_model))
    enc = torch.zeros(length, d_model)
    enc[:, 0::2] = torch.sin(position * freq)
    enc[:, 1::2] = torch.cos(position * freq[: d_model // 2])
    return enc


class TransformerAutoencoder(nn.Module):
    position: Tensor

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        *,
        window: int,
        d_model: int,
        heads: int,
        layers: int,
        feedforward: int,
        latent: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.register_buffer("position", sinusoidal_encoding(window, d_model), persistent=False)

        def stack() -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model, heads, feedforward, dropout, batch_first=True, norm_first=True
            )
            return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

        self.encoder = stack()
        self.to_latent = nn.Linear(d_model, latent)
        self.from_latent = nn.Linear(latent, d_model)
        self.decoder = stack()
        self.head = nn.Linear(d_model, n_outputs)

    def forward(self, x: Tensor) -> Tensor:
        length = x.shape[1]
        position: Tensor = self.position[:length]
        code = self.to_latent(self.encoder(self.embed(x) + position).mean(dim=1))  # (B, latent)
        queries = self.from_latent(code)[:, None, :] + position  # (B, L, d_model)
        reconstruction: Tensor = self.head(self.decoder(queries))
        return reconstruction
