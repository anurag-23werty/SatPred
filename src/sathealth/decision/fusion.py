"""Combine calibrated surprises of several detectors into one series.

Surprises are ``-log10`` tail probabilities, so:

* ``mean`` is the log of the geometric mean of the tail probabilities: every detector has to
  be fairly surprised. Detectors with partly independent errors (a forecaster and an
  autoencoder) rarely raise the same false alarm, so agreement suppresses false alarms.
* ``max`` alarms when *any* detector is surprised (union: more recall, more false alarms).
* ``min`` alarms only when *all* are (intersection: fewest false alarms, misses more).
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np

from sathealth.data.schema import FloatArray

Fusion = Literal["mean", "max", "min"]


def fuse(surprises: Sequence[FloatArray], method: Fusion = "mean") -> FloatArray:
    if not surprises:
        raise ValueError("nothing to fuse")
    stacked = np.vstack(surprises)
    if method == "mean":
        result: FloatArray = stacked.mean(axis=0)
    elif method == "max":
        result = stacked.max(axis=0)
    elif method == "min":
        result = stacked.min(axis=0)
    else:
        raise ValueError(f"unknown fusion {method!r}")
    return result
