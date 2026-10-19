"""One training loop for every deep detector: Adam, gradient clipping, early stopping.

Early stopping watches the loss on the *validation* windows, the nominal tail of train that
the thresholds are also fit on (no labels involved). The weights of the best epoch are
restored at the end. When validation is too short to form a single window, the training loss
is monitored instead and the run is flagged ``monitor="train"`` in its history.
"""

import copy
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from sathealth.config import TrainingConfig

LossFn = Callable[[nn.Module, Tensor], Tensor]


@dataclass(frozen=True)
class TrainingHistory:
    train_loss: tuple[float, ...]
    val_loss: tuple[float, ...]
    best_epoch: int  # 1-based
    best_loss: float
    monitor: str  # "val" or "train"
    stopped_early: bool
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean_loss(model: nn.Module, loss_fn: LossFn, windows: Tensor, batch_size: int) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, windows.shape[0], batch_size):
            batch = windows[start : start + batch_size]
            total += float(loss_fn(model, batch)) * batch.shape[0]
    return total / windows.shape[0]


def train_model(
    model: nn.Module,
    loss_fn: LossFn,
    train: Tensor,
    val: Tensor,
    cfg: TrainingConfig,
    generator: torch.Generator,
) -> TrainingHistory:
    """Fit ``model`` in place on ``train`` windows ``(N, L, D)``; ``val`` may be empty."""
    if train.shape[0] == 0:
        raise ValueError("no training windows: series shorter than the model window")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    monitor = "val" if val.shape[0] > 0 else "train"
    train_hist: list[float] = []
    val_hist: list[float] = []
    best_loss, best_epoch, best_state = float("inf"), 0, copy.deepcopy(model.state_dict())
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(train.shape[0], generator=generator)
        running = 0.0
        for start in range(0, train.shape[0], cfg.batch_size):
            batch = train[order[start : start + cfg.batch_size]]
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            running += float(loss.detach()) * batch.shape[0]
        train_hist.append(running / train.shape[0])
        if monitor == "val":
            val_hist.append(_mean_loss(model, loss_fn, val, cfg.batch_size))
        current = val_hist[-1] if monitor == "val" else train_hist[-1]
        if current < best_loss - cfg.min_delta:
            best_loss, best_epoch, stale = current, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return TrainingHistory(
        train_loss=tuple(round(v, 6) for v in train_hist),
        val_loss=tuple(round(v, 6) for v in val_hist),
        best_epoch=best_epoch,
        best_loss=round(best_loss, 6),
        monitor=monitor,
        stopped_early=len(train_hist) < cfg.epochs,
        seconds=round(time.perf_counter() - started, 3),
    )
