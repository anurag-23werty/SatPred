"""Turn raw model errors into anomaly scores.

Raw per-step errors are spiky: one noisy reading makes a large error without anything being
wrong. Telemanom therefore smooths errors with an exponentially weighted moving average
before thresholding. The EWMA here is causal (only past and present steps), so the score at
step ``t`` is available at step ``t`` in a live stream.
"""

import numpy as np
from scipy.signal import lfilter

from sathealth.data.schema import FloatArray


def ewma(errors: FloatArray, span: int) -> FloatArray:
    """Causal EWMA with ``alpha = 2 / (span + 1)``, started at the first error.

    Equivalent to ``pandas.Series(errors).ewm(span=span, adjust=False).mean()``.
    """
    if span < 1:
        raise ValueError("span must be >= 1")
    if span == 1 or errors.size == 0:
        return np.asarray(errors, dtype=np.float64).copy()
    alpha = 2.0 / (span + 1.0)
    x = np.asarray(errors, dtype=np.float64)
    smoothed, _ = lfilter([alpha], [1.0, alpha - 1.0], x, zi=[(1.0 - alpha) * x[0]])
    result: FloatArray = smoothed
    return result
