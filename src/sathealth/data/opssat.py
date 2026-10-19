"""OPSSAT-AD loader: labelled telemetry segments of ESA's OPS-SAT cubesat.

``segments.csv`` holds one row per sample: channel, timestamp, value, segment id, the
segment's label (``anomaly``) and the benchmark's split (``train``). Each segment is a short
univariate fragment labelled as a whole; train and test segments interleave in time.

Preparation follows ADR 0009 (decided from the data layout before any detector was run):

* segments are ordered by their first timestamp within a channel;
* training uses the *nominal* training segments only; the chronologically last
  ``val_fraction`` of their points, by whole segments, is the validation set;
* robust scaling is fitted on the nominal training values (validation excluded);
* every segment stays separate, so no model window ever spans a gap between segments.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sathealth.data.schema import FloatArray
from sathealth.features.scaling import ChannelScaler, ScalerParams


@dataclass(frozen=True)
class Segment:
    segment_id: int
    channel: str
    start_time: str
    values: FloatArray  # (n, 1)
    anomaly: bool
    train: bool

    @property
    def length(self) -> int:
        return int(self.values.shape[0])


@dataclass(frozen=True)
class OpsSatChannel:
    chan_id: str
    train: list[FloatArray]  # nominal training segments, scaled
    val: list[FloatArray]  # nominal validation segments, scaled
    test: list[Segment]  # test segments in time order, values scaled
    scaler: ScalerParams

    @property
    def n_test_anomalous(self) -> int:
        return sum(s.anomaly for s in self.test)


def load_segments(path: Path) -> dict[str, list[Segment]]:
    """All segments per channel, ordered by first timestamp."""
    rows: dict[int, dict[str, object]] = {}
    values: dict[int, list[float]] = defaultdict(list)
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            seg = int(r["segment"])
            if seg not in rows:
                rows[seg] = {
                    "channel": r["channel"],
                    "start": r["timestamp"],
                    "anomaly": r["anomaly"] == "1",
                    "train": r["train"] == "1",
                }
            values[seg].append(float(r["value"]))
    channels: dict[str, list[Segment]] = defaultdict(list)
    for seg, meta in rows.items():
        channels[str(meta["channel"])].append(
            Segment(
                seg,
                str(meta["channel"]),
                str(meta["start"]),
                np.asarray(values[seg], dtype=np.float64)[:, None],
                bool(meta["anomaly"]),
                bool(meta["train"]),
            )
        )
    return {
        c: sorted(s, key=lambda x: (x.start_time, x.segment_id))
        for c, s in sorted(channels.items())
    }


def prepare_channel(
    chan_id: str, segments: list[Segment], val_fraction: float = 0.2, min_scale: float = 1e-6
) -> OpsSatChannel:
    nominal = [s for s in segments if s.train and not s.anomaly]
    if not nominal:
        raise ValueError(f"{chan_id}: no nominal training segments")
    total = sum(s.length for s in nominal)
    train_segs: list[Segment] = []
    seen = 0
    for s in nominal:
        if seen >= (1 - val_fraction) * total and train_segs:
            break
        train_segs.append(s)
        seen += s.length
    val_segs = nominal[len(train_segs) :]
    scaler = ChannelScaler("robust", (0,), min_scale).fit(
        np.concatenate([s.values for s in train_segs])
    )

    def scale(x: FloatArray) -> FloatArray:
        return scaler.transform(x)

    test = [
        Segment(s.segment_id, s.channel, s.start_time, scale(s.values), s.anomaly, s.train)
        for s in segments
        if not s.train
    ]
    return OpsSatChannel(
        chan_id,
        [scale(s.values) for s in train_segs],
        [scale(s.values) for s in val_segs],
        test,
        scaler.params,
    )
