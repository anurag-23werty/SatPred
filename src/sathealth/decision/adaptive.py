"""Causal recalibration: keep the nominal reference up to date while the stream runs.

Calibration fit once on the validation tail assumes the test period behaves like the end of
training. On SMAP/MSL it does not: the share of test steps in alarm is far above the designed
rate, which is visible without any labels. Following the streaming-EVT idea of SPOT/DSPOT
(Siffer et al., KDD 2017), the reference distribution becomes a sliding window of the most
recent scores that were *not* in alarm, *anchored* by the validation scores:

* the stream is processed in blocks of ``every`` steps;
* each block is scored with calibrators fit on the validation scores plus the last
  ``window`` non-alarm test scores (strictly past data);
* the block's non-alarm scores are then added to the recent window.

The validation anchor matters. An unanchored first version let the reference shrink to a
quiet stretch of the stream; ordinary fluctuations then became "rare", went into alarm, were
excluded from the reference and kept it narrow (on channel F-3 the alarm share went from 0%
to 45%, visible without labels). With the anchor, recalibration can widen what counts as
nominal when the stream is noisier than validation, but never make it narrower.

Alarm steps never enter the reference, so an ongoing anomaly does not become "normal".
Two known prices of adaptivity, reported next to the static variant rather than hidden:
a slow drift that stays below the alarm level can be absorbed, and a permanent level shift
large enough to keep every step in alarm stays in alarm until an operator re-baselines.
"""

from collections.abc import Sequence

import numpy as np

from sathealth.config import DecisionConfig
from sathealth.data.schema import FloatArray
from sathealth.decision.alarms import hysteresis
from sathealth.decision.calibration import TailCalibrator
from sathealth.decision.fusion import fuse


def adaptive_surprises(
    val: Sequence[FloatArray], test: Sequence[FloatArray], d: DecisionConfig, every: int
) -> list[FloatArray]:
    """Per-input test surprises, each block calibrated on past non-alarm scores only."""
    length = test[0].size
    anchors = [np.asarray(v, dtype=np.float64) for v in val]
    recent = [np.zeros(0) for _ in test]
    out = [np.empty(length) for _ in test]
    active = False
    for start in range(0, length, every):
        block = slice(start, min(start + every, length))
        cals = [
            TailCalibrator(d.calibration, d.tail_quantile, d.max_surprise).fit(np.r_[a, r])
            for a, r in zip(anchors, recent, strict=True)
        ]
        scored = [cal.surprise(t[block]) for cal, t in zip(cals, test, strict=True)]
        fused = fuse(scored, d.fusion)
        mask = hysteresis(fused, d.alarm_on, d.alarm_off, start_active=active)
        active = bool(mask[-1])
        for i, s in enumerate(scored):
            out[i][block] = s
            accepted = test[i][block][~mask]
            recent[i] = np.r_[recent[i], accepted][-d.recalibrate_window :]
    return out
