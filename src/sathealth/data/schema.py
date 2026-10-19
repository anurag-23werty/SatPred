"""Domain types for channel telemetry and anomaly labels.

Conventions used everywhere in the package:

* A channel's arrays are shaped ``(T, D)``; ``T`` is time, ``D`` features.
* In the NASA data column 0 is the telemetry value and columns ``1..D-1`` are one-hot
  command flags sent to the spacecraft (``D = 25`` for SMAP, ``55`` for MSL).
* Anomaly segments use **inclusive** ``[start, end]`` indices into the test series, exactly
  as published in ``labeled_anomalies.csv``.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class AnomalyKind(StrEnum):
    POINT = "point"  # isolated extreme values
    CONTEXTUAL = "contextual"  # values plausible in isolation, anomalous in context


@dataclass(frozen=True, order=True)
class AnomalySegment:
    start: int
    end: int  # inclusive
    kind: AnomalyKind = AnomalyKind.CONTEXTUAL

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid segment [{self.start}, {self.end}]")

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class ChannelInfo:
    chan_id: str
    spacecraft: str
    segments: tuple[AnomalySegment, ...]
    num_values: int  # test length declared by the label file

    @property
    def group(self) -> str:
        """Channel family prefix (``A`` in ``A-1``). NASA anonymised what each family means."""
        return self.chan_id.split("-", 1)[0]


def segments_to_mask(segments: tuple[AnomalySegment, ...], length: int) -> BoolArray:
    """Point-wise boolean labels of size ``length`` from inclusive segments."""
    mask = np.zeros(length, dtype=np.bool_)
    for seg in segments:
        if seg.end >= length:
            raise ValueError(f"segment [{seg.start}, {seg.end}] exceeds series length {length}")
        mask[seg.start : seg.end + 1] = True
    return mask


@dataclass(frozen=True)
class ChannelData:
    info: ChannelInfo
    train: FloatArray  # (T_train, D), assumed nominal (no labels are published for train)
    test: FloatArray  # (T_test, D)

    @property
    def chan_id(self) -> str:
        return self.info.chan_id

    @property
    def n_features(self) -> int:
        return int(self.train.shape[1])

    @property
    def test_labels(self) -> BoolArray:
        return segments_to_mask(self.info.segments, int(self.test.shape[0]))
