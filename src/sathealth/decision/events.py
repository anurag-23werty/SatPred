"""Alarm runs -> operator-facing events with severity, type and per-detector evidence.

* **Severity** grows with the peak surprise: how improbable the episode is under nominal
  behaviour (``watch`` < ``warning`` < ``critical``).
* **Type** follows the dataset's own taxonomy. ``point``: the event contains a raw value
  outside the training range (with a margin), something a limit check could also see.
  ``contextual``: every value is plausible on its own and only the dynamics are abnormal,
  which is where learned detectors earn their keep.
* **Contributions** record each input detector's peak surprise inside the event; Phase 5
  explanations start from them.
"""

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from sathealth.data.schema import BoolArray, FloatArray
from sathealth.evaluation.metrics import mask_to_events

Severity = Literal["watch", "warning", "critical"]
Kind = Literal["point", "contextual"]


@dataclass(frozen=True)
class Event:
    start: int
    end: int  # inclusive
    peak_step: int
    peak_surprise: float
    severity: Severity
    kind: Kind
    contributions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def severity(peak: float, warning: float, critical: float) -> Severity:
    if peak >= critical:
        return "critical"
    if peak >= warning:
        return "warning"
    return "watch"


def outside_range(
    values: FloatArray, low: FloatArray, high: FloatArray, margin: float
) -> BoolArray:
    """Per step: does any value column leave ``[low, high]`` widened by ``margin * range``?"""
    pad = margin * (high - low)
    result: BoolArray = ((values < low - pad) | (values > high + pad)).any(axis=1)
    return result


def build_events(
    mask: BoolArray,
    surprise: FloatArray,
    per_detector: dict[str, FloatArray],
    out_of_range: BoolArray,
    *,
    warning: float,
    critical: float,
) -> list[Event]:
    events = []
    for start, end in mask_to_events(mask):
        window = slice(start, end + 1)
        peak_step = start + int(np.argmax(surprise[window]))
        peak = float(surprise[peak_step])
        events.append(
            Event(
                start=start,
                end=end,
                peak_step=peak_step,
                peak_surprise=round(peak, 4),
                severity=severity(peak, warning, critical),
                kind="point" if bool(out_of_range[window].any()) else "contextual",
                contributions={
                    k: round(float(v[window].max()), 4) for k, v in per_detector.items()
                },
            )
        )
    return events
