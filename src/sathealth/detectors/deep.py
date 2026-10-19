"""Shared machinery of the PyTorch detectors: seeding, training and error smoothing.

A subclass defines the network, its loss on a batch of training windows, the windows it
trains on and its raw per-step error. This base class trains the network under
:func:`~sathealth.models.runtime.seed_torch` (so results are reproducible), turns errors into
scores with a causal EWMA, and keeps the training history for the run directory.
"""

from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path
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
        self._n_features = 0
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
        return self.fit_segments([train], [val])

    def fit_segments(self, train: Sequence[FloatArray], val: Sequence[FloatArray]) -> Self:
        """Fit on several separate pieces of nominal data; no window spans two pieces.

        ``fit(train, val)`` is the one-piece case. Separate pieces matter when the data has
        gaps (e.g. OPS-SAT's short telemetry segments): a window across a gap would teach the
        model a jump that never happened.
        """
        n_features = int(train[0].shape[1])
        generator = seed_torch(self.seed, self.training.threads)
        model = self._build(n_features).to(self.device)  # init after seeding
        self.history = train_model(
            model,
            self._loss,
            self._tensor(self._stack(train, n_features)),
            self._tensor(self._stack(val, n_features)),
            self.training,
            generator,
        )
        self._model, self._n_features = model, n_features
        return self

    def _stack(self, pieces: Sequence[FloatArray], n_features: int) -> FloatArray:
        windows = [w for w in (self._windows(p) for p in pieces) if w.shape[0]]
        if not windows:
            empty: FloatArray = self._windows(np.zeros((0, n_features)))
            return empty
        stacked: FloatArray = np.concatenate(windows) if len(windows) > 1 else windows[0]
        return stacked

    def score(self, x: FloatArray) -> FloatArray:
        model = self.model
        if x.shape[0] == 0:
            return np.zeros(0)
        with torch.no_grad():
            errors = self._errors(model, x)
        return ewma(errors, self.smoothing)

    def save(self, path: Path) -> None:
        """Weights plus the settings needed to rebuild the network, written atomically."""
        model = self.model
        n_features = self._n_features
        payload = {"state": model.state_dict(), "n_features": n_features, "params": self.params()}
        tmp = path.with_suffix(path.suffix + ".part")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, tmp)
        tmp.replace(path)

    def load(self, path: Path) -> Self:
        """Restore weights saved by :meth:`save`; settings must match this detector's."""
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if payload["params"] != self.params():
            raise ValueError(f"{path} was saved with different settings")
        model = self._build(int(payload["n_features"])).to(self.device)
        model.load_state_dict(payload["state"])
        model.eval()
        self._model, self._n_features = model, int(payload["n_features"])
        return self

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
