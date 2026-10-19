"""Pin the verified facts of the real NASA dataset so silent upstream drift fails loudly.

Run with ``make test-data`` after ``make download``; skipped when the data is absent.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sathealth.config import Config, load_config
from sathealth.data.download import load_manifest, verify_dataset
from sathealth.data.telemanom import TelemanomDataset
from sathealth.features.validation import Severity, validate_dataset
from sathealth.pipeline.prepare import load_prepared, prepare_dataset

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "telemanom"

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not (RAW / "labeled_anomalies.csv").is_file(),
        reason="real dataset not downloaded (run `make download`)",
    ),
]


@pytest.fixture(scope="module")
def dataset() -> TelemanomDataset:
    return TelemanomDataset(RAW)


def test_raw_files_match_pinned_checksums() -> None:
    assert verify_dataset(RAW, load_manifest("telemanom.json")).ok


def test_channel_counts_after_cleaning(dataset: TelemanomDataset) -> None:
    crafts = Counter(dataset.info(c).spacecraft for c in dataset.channel_ids)
    assert crafts == {"SMAP": 53, "MSL": 27}
    assert "P-2" not in dataset.channel_ids
    assert sum(len(dataset.info(c).segments) for c in dataset.channel_ids) == 103


def test_feature_dimensions_and_command_encoding(dataset: TelemanomDataset) -> None:
    for channel in dataset:
        expected = 25 if channel.info.spacecraft == "SMAP" else 55
        assert channel.n_features == expected, channel.chan_id
        assert np.isin(channel.train[:, 1:], (0.0, 1.0)).all(), channel.chan_id


def test_no_error_level_quality_issues(dataset: TelemanomDataset) -> None:
    report = validate_dataset(dataset, Config())
    assert report.errors == []
    constant = {i.chan_id for i in report.issues if i.check == "constant_train"}
    assert {"M-6", "B-1", "D-2"} <= constant
    assert all(i.severity is not Severity.ERROR for i in report.issues)


def test_prepare_real_dataset(tmp_path: Path) -> None:
    cfg = load_config(None, overrides=[f"paths.raw_dir={RAW}", f"paths.processed_dir={tmp_path}"])
    result = prepare_dataset(cfg)
    assert len(result.channels) == 80
    m6 = load_prepared(result.run_dir, "M-6")
    assert m6.scaler.degenerate == (True,)
    assert float(m6.test[:, 0].max()) == pytest.approx(259.1, abs=0.1)
    assert np.isfinite(m6.test).all()
