"""Shared machinery of the PyTorch detectors: seeding, training and error smoothing.

A subclass defines the network, its loss on a batch of training windows, the windows it
trains on and its raw per-step error. This base class trains the network under
:func:`~sathealth.models.runtime.seed_torch` (so results are reproducible), turns errors into
scores with a causal EWMA, and keeps the training history for the run directory.
"""

from abc import abstractmethod
from typing import Any, Self

import numpy as np
import torch
from torch import Tensor, nn

from sathealth.config import TrainingConfig
from sathealth.data.schema import FloatArray
from sathealth.detectors.base import Detector
from sathealth.models.runtime import resolve_device, seed_torch
from sathealth.models.scoring import ewma
from sathealth.models.training import TrainingHistory, train_model

INFERENCE_BATCH = 1024


class DeepDetector(Detector):
    def __init__(
        self,
        value_columns: tuple[int, ...] = (0,),
        *,
        window: int,
        smoothing: int,
        training: TrainingConfig | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(value_columns)
        self.window = window
        self.smoothing = smoothing
        self.training = training or TrainingConfig()
        self.seed = seed
        self.device = resolve_device(self.training.device)
        self._model: nn.Module | None = None
        self.history: TrainingHistory | None = None

    @abstractmethod
    def _build(self, n_features: int) -> nn.Module: ...

    @abstractmethod
    def _windows(self, x: FloatArray) -> FloatArray:
        """Training/validation windows ``(N, L, D)`` cut from one split."""

    @abstractmethod
    def _loss(self, model: nn.Module, batch: Tensor) -> Tensor: ...

    @abstractmethod
    def _errors(self, model: nn.Module, x: FloatArray) -> FloatArray:
        """Raw non-negative error per time step, shape ``(T,)``."""

    def _tensor(self, x: FloatArray) -> Tensor:
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device)

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        return self._model

    def fit(self, train: FloatArray, val: FloatArray) -> Self:
        generator = seed_torch(self.seed, self.training.threads)
        model = self._build(train.shape[1]).to(self.device)  # init after seeding
        self.history = train_model(
            model,
            self._loss,
            self._tensor(self._windows(train)),
            self._tensor(self._windows(val)),
            self.training,
            generator,
        )
        self._model = model
        return self

    def score(self, x: FloatArray) -> FloatArray:
        model = self.model
        if x.shape[0] == 0:
            return np.zeros(0)
        with torch.no_grad():
            errors = self._errors(model, x)
        return ewma(errors, self.smoothing)

    def diagnostics(self) -> dict[str, Any]:
        if self.history is None:
            return {}
        n_params = sum(p.numel() for p in self.model.parameters())
        return {"parameters": n_params, **self.history.to_dict()}

    def params(self) -> dict[str, Any]:
        return {
            **super().params(),
            "window": self.window,
            "smoothing": self.smoothing,
            "training": self.training.model_dump(mode="json"),
            "seed": self.seed,
        }
