"""Bottlenecked Transformer autoencoder.

Each window of ``window`` steps is compressed to a ``latent``-sized code and decoded back
(:class:`~sathealth.models.networks.TransformerAutoencoder`). The score at step ``t`` is the
reconstruction error of step ``t`` in the window that *ends* at ``t``, so it is available as
soon as the reading arrives, then smoothed by a causal EWMA.

It complements the forecaster: forecasting asks "is this reading what the recent past
predicts?", reconstruction asks "does this whole window look like nominal behaviour?". The
architecture borrows the Transformer encoder of TranAD (Tuli et al., VLDB 2022) but not its
adversarial two-phase training.

Steps before the first full window are reconstructed from the shorter prefix that exists.
"""

from typing import Any

import numpy as np
from torch import Tensor, nn

from sathealth.config import TrainingConfig, TransformerConfig
from sathealth.data.schema import FloatArray
from sathealth.detectors.deep import INFERENCE_BATCH, DeepDetector
from sathealth.features.windowing import sliding_windows
from sathealth.models.networks import TransformerAutoencoder


class TransformerDetector(DeepDetector):
    name = "transformer"

    def __init__(
        self,
        value_columns: tuple[int, ...] = (0,),
        config: TransformerConfig | None = None,
        training: TrainingConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.config = config or TransformerConfig()
        super().__init__(
            value_columns,
            window=self.config.window,
            smoothing=self.config.smoothing,
            training=training,
            seed=seed,
        )

    def _build(self, n_features: int) -> nn.Module:
        c = self.config
        return TransformerAutoencoder(
            n_features,
            len(self.value_columns),
            window=c.window,
            d_model=c.d_model,
            heads=c.heads,
            layers=c.layers,
            feedforward=c.feedforward,
            latent=c.latent,
            dropout=c.dropout,
        )

    def _windows(self, x: FloatArray) -> FloatArray:
        return sliding_windows(x, self.window)

    def _loss(self, model: nn.Module, batch: Tensor) -> Tensor:
        loss: Tensor = nn.functional.mse_loss(model(batch), batch[..., list(self.value_columns)])
        return loss

    def _errors(self, model: nn.Module, x: FloatArray) -> FloatArray:
        length = x.shape[0]
        values = self._values(x)
        recon = np.empty_like(values)
        for t in range(min(self.window - 1, length)):  # prefixes shorter than a window
            recon[t] = model(self._tensor(x[None, : t + 1]))[0, -1].cpu().numpy()
        windows = sliding_windows(x, self.window)  # window i ends at row i + window - 1
        for start in range(0, windows.shape[0], INFERENCE_BATCH):
            batch = self._tensor(windows[start : start + INFERENCE_BATCH])
            first = self.window - 1 + start
            recon[first : first + batch.shape[0]] = model(batch)[:, -1].cpu().numpy()
        result: FloatArray = np.abs(values - recon).max(axis=1)
        return result

    def params(self) -> dict[str, Any]:
        return {**super().params(), "architecture": self.config.model_dump(mode="json")}
