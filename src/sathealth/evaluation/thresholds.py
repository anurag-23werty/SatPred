"""Label-free operating thresholds fit on *nominal* validation scores.

The validation tail of each train series is assumed nominal, so the score distribution
there tells us what "normal" looks like for this detector on this channel. A threshold is
the level we accept being exceeded on nominal data:

* ``val_quantile``: the ``q`` quantile of validation scores (``q = 0.999`` means we accept
  about 0.1% alarm steps on held-out nominal data).
* ``val_sigma``: ``mean + k * std`` of validation scores.

Test labels are never an input here. Prediction is ``score > threshold`` (strict), so a
detector that outputs exactly 0 for "inside limits" never alarms on in-limit data, even
when the threshold is 0.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from sathealth.data.schema import FloatArray

ThresholdMethod = Literal["val_quantile", "val_sigma"]


@dataclass(frozen=True)
class Threshold:
    value: float
    method: ThresholdMethod
    source: Literal["val", "train"]  # train is the fallback for very short validation tails
    n_rows: int


def fit_threshold(
    val_scores: FloatArray,
    train_scores: FloatArray,
    *,
    method: ThresholdMethod = "val_quantile",
    quantile: float = 0.999,
    sigma: float = 4.0,
    min_rows: int = 50,
) -> Threshold:
    source: Literal["val", "train"] = "val" if val_scores.size >= min_rows else "train"
    scores = val_scores if source == "val" else train_scores
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        raise ValueError("no finite scores to fit a threshold on")
    if method == "val_quantile":
        value = float(np.quantile(finite, quantile))
    else:
        value = float(finite.mean() + sigma * finite.std())
    return Threshold(value=value, method=method, source=source, n_rows=int(finite.size))
