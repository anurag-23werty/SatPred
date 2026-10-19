"""Loader for the NASA SMAP/MSL ("Telemanom") anomaly benchmark.

Raw layout (as produced by ``sathealth download``)::

    <root>/labeled_anomalies.csv
    <root>/train/<chan_id>.npy   float64 (T_train, D)
    <root>/test/<chan_id>.npy    float64 (T_test, D)

Label-file quirks handled here, each verified against the real file:

* ``anomaly_sequences`` is JSON (``[[2149, 2349], ...]``) but ``class`` is not
  (``[contextual, point]``), so they are parsed differently.
* Sequences are not always sorted (``P-1``, ``E-12``, ``P-11``); we sort them.
* ``P-2`` appears twice with different labels; resolved by ``DuplicatePolicy``.
"""

import csv
import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sathealth.config import DatasetConfig, DuplicatePolicy
from sathealth.data.schema import AnomalyKind, AnomalySegment, ChannelData, ChannelInfo
from sathealth.logging import get_logger

log = get_logger(__name__)

LABEL_FILE = "labeled_anomalies.csv"
_REQUIRED_COLUMNS = {"chan_id", "spacecraft", "anomaly_sequences", "class", "num_values"}


@dataclass(frozen=True)
class LabelTable:
    channels: dict[str, ChannelInfo]
    notes: tuple[str, ...] = field(default_factory=tuple)  # human-readable cleaning decisions


def _parse_kinds(raw: str) -> list[AnomalyKind]:
    inner = raw.strip().removeprefix("[").removesuffix("]")
    return [AnomalyKind(token.strip()) for token in inner.split(",") if token.strip()]


def _parse_row(row: dict[str, str]) -> ChannelInfo:
    chan_id = row["chan_id"].strip()
    bounds = json.loads(row["anomaly_sequences"])
    kinds = _parse_kinds(row["class"])
    if len(bounds) != len(kinds):
        raise ValueError(f"{chan_id}: {len(bounds)} sequences but {len(kinds)} classes")
    segments = sorted(
        AnomalySegment(int(start), int(end), kind)
        for (start, end), kind in zip(bounds, kinds, strict=True)
    )
    return ChannelInfo(
        chan_id=chan_id,
        spacecraft=row["spacecraft"].strip(),
        segments=tuple(segments),
        num_values=int(row["num_values"]),
    )


def _merge(segments: list[AnomalySegment]) -> tuple[AnomalySegment, ...]:
    """Union of segments; overlapping or touching ones merge and keep the first kind."""
    merged: list[AnomalySegment] = []
    for seg in sorted(segments):
        if merged and seg.start <= merged[-1].end + 1:
            last = merged[-1]
            merged[-1] = AnomalySegment(last.start, max(last.end, seg.end), last.kind)
        else:
            merged.append(seg)
    return tuple(merged)


def parse_label_file(path: Path, policy: DuplicatePolicy = DuplicatePolicy.DROP) -> LabelTable:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = _REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = [_parse_row(row) for row in reader]

    by_id: dict[str, list[ChannelInfo]] = defaultdict(list)
    for info in rows:
        by_id[info.chan_id].append(info)

    channels: dict[str, ChannelInfo] = {}
    notes: list[str] = []
    for chan_id, infos in by_id.items():
        if len(infos) == 1:
            channels[chan_id] = infos[0]
            continue
        if len({(i.spacecraft, i.num_values) for i in infos}) > 1:
            raise ValueError(f"{chan_id}: duplicate rows disagree on spacecraft/num_values")
        if policy is DuplicatePolicy.DROP:
            notes.append(f"{chan_id}: listed {len(infos)}x with conflicting labels; dropped")
        elif policy is DuplicatePolicy.FIRST:
            channels[chan_id] = infos[0]
            notes.append(f"{chan_id}: listed {len(infos)}x; kept first row")
        else:
            merged = _merge([s for i in infos for s in i.segments])
            channels[chan_id] = ChannelInfo(
                chan_id, infos[0].spacecraft, merged, infos[0].num_values
            )
            notes.append(f"{chan_id}: listed {len(infos)}x; merged into {len(merged)} segments")
    return LabelTable(channels=channels, notes=tuple(notes))


class TelemanomDataset:
    """Read-only view over the raw benchmark, filtered by ``DatasetConfig``."""

    def __init__(self, root: Path, config: DatasetConfig | None = None) -> None:
        self.root = root
        self.config = config or DatasetConfig()
        label_path = root / LABEL_FILE
        if not label_path.is_file():
            raise FileNotFoundError(f"{label_path} not found; run `sathealth download` first")
        table = parse_label_file(label_path, self.config.duplicate_policy)
        for note in table.notes:
            log.info("label cleaning", decision=note)
        self.notes = table.notes

        wanted = set(self.config.channels) if self.config.channels is not None else None
        if wanted is not None and (unknown := wanted - table.channels.keys()):
            raise KeyError(f"unknown channel ids: {sorted(unknown)}")
        self._infos = {
            cid: info
            for cid, info in sorted(table.channels.items())
            if info.spacecraft in self.config.spacecraft and (wanted is None or cid in wanted)
        }

    @property
    def channel_ids(self) -> list[str]:
        return list(self._infos)

    def __len__(self) -> int:
        return len(self._infos)

    def info(self, chan_id: str) -> ChannelInfo:
        return self._infos[chan_id]

    def load(self, chan_id: str) -> ChannelData:
        info = self._infos[chan_id]
        train = np.load(self.root / "train" / f"{chan_id}.npy", allow_pickle=False)
        test = np.load(self.root / "test" / f"{chan_id}.npy", allow_pickle=False)
        for name, arr in (("train", train), ("test", test)):
            if arr.ndim != 2:
                raise ValueError(f"{chan_id}/{name}: expected 2-D array, got shape {arr.shape}")
        return ChannelData(
            info=info,
            train=train.astype(np.float64, copy=False),
            test=test.astype(np.float64, copy=False),
        )

    def __iter__(self) -> Iterator[ChannelData]:
        for chan_id in self._infos:
            yield self.load(chan_id)
