"""Telemanom-style LSTM forecaster (Hundman et al., KDD 2018).

The network sees ``window`` steps of every feature and predicts the next value. It is trained
sequence-to-sequence: output ``t`` of a window predicts input ``t + 1``, which gives
``window`` targets per window instead of one. The anomaly score at step ``t`` is the
absolute error of the forecast made at ``t - 1``, smoothed by a causal EWMA.

Unlike the Isolation Forest, the error grows with the size of the deviation, so a reading far
outside the training range scores higher than one just outside it (pinned by a test).

The first ``window - 1`` steps of a series have less history than a full window; the causal
LSTM forecasts them from whatever prefix exists, in one pass. Step 0 has no history and
reuses the error of step 1.
"""

from typing import Any

import numpy as np
from torch import Tensor, nn

from sathealth.config import LSTMConfig, TrainingConfig
from sathealth.data.schema import FloatArray
from sathealth.detectors.deep import INFERENCE_BATCH, DeepDetector
from sathealth.features.windowing import sliding_windows
from sathealth.models.networks import LSTMForecaster


class LSTMDetector(DeepDetector):
    name = "lstm"

    def __init__(
        self,
        value_columns: tuple[int, ...] = (0,),
        config: LSTMConfig | None = None,
        training: TrainingConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.config = config or LSTMConfig()
        super().__init__(
            value_columns,
            window=self.config.window,
            smoothing=self.config.smoothing,
            training=training,
            seed=seed,
        )

    def _build(self, n_features: int) -> nn.Module:
        c = self.config
        return LSTMForecaster(n_features, len(self.value_columns), c.hidden, c.layers, c.dropout)

    def _windows(self, x: FloatArray) -> FloatArray:
        return sliding_windows(x, self.window + 1)  # inputs [:-1], targets [1:]

    def _loss(self, model: nn.Module, batch: Tensor) -> Tensor:
        target = batch[:, 1:, list(self.value_columns)]
        loss: Tensor = nn.functional.mse_loss(model(batch[:, :-1]), target)
        return loss

    def _errors(self, model: nn.Module, x: FloatArray) -> FloatArray:
        length = x.shape[0]
        values = self._values(x)
        if length < 2:
            return np.zeros(length)
        forecast = np.empty_like(values)
        head = min(self.window, length) - 1  # steps 1..head are forecast from a prefix
        forecast[1 : head + 1] = model(self._tensor(x[None, :head]))[0].cpu().numpy()
        if length > self.window:  # window i = rows [i, i + window) forecasts row i + window
            windows = sliding_windows(x[:-1], self.window)
            for start in range(0, windows.shape[0], INFERENCE_BATCH):
                batch = self._tensor(windows[start : start + INFERENCE_BATCH])
                rows = slice(self.window + start, self.window + start + batch.shape[0])
                forecast[rows] = model(batch)[:, -1].cpu().numpy()
        errors = np.abs(values - forecast).max(axis=1)
        errors[0] = errors[1]
        result: FloatArray = errors
        return result

    def params(self) -> dict[str, Any]:
        return {**super().params(), "architecture": self.config.model_dump(mode="json")}
