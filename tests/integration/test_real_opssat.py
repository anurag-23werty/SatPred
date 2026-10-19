"""OPSSAT-AD: verified download, structure and the pre-registered exclusion rule
(``make test-data``). Model training for the confirmation itself runs in `sathealth confirm`.
"""

from pathlib import Path

import pytest

from sathealth.config import load_config
from sathealth.data.download import download_dataset, load_manifest
from sathealth.data.opssat import load_segments, prepare_channel
from sathealth.pipeline.confirm import trainable

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "telemanom"

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not (RAW / "labeled_anomalies.csv").is_file(),
        reason="real datasets not requested (run `make download`)",
    ),
]


def test_opssat_layout_and_exclusion_rule() -> None:
    config = load_config(None)
    raw = ROOT / config.confirmation.raw_dir
    report = download_dataset(raw, load_manifest(config.confirmation.manifest), max_workers=1)
    assert report.ok, "segments.csv matches its pinned SHA-256"
    segments = load_segments(raw / "segments.csv")
    assert len(segments) == 9
    assert sum(len(s) for s in segments.values()) == 2123
    channels = {c: prepare_channel(c, s) for c, s in segments.items()}
    excluded = sorted(c for c, ch in channels.items() if not trainable(ch, config))
    assert excluded == ["CADC0886", "CADC0890"]
    kept = [ch for c, ch in channels.items() if c not in excluded]
    assert sum(len(ch.test) for ch in kept) == 523
