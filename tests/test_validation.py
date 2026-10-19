from dataclasses import replace
from pathlib import Path

import numpy as np

from sathealth.config import Config, load_config
from sathealth.data.schema import ChannelData
from sathealth.data.telemanom import TelemanomDataset
from sathealth.features.validation import Severity, check_channel, validate_dataset


def _checks(issues: list, severity: Severity) -> set[str]:
    return {i.check for i in issues if i.severity is severity}


def test_clean_channel_has_no_warnings(raw_root: Path) -> None:
    channel = TelemanomDataset(raw_root).load("A-1")
    stats, issues = check_channel(channel, Config())
    assert stats is not None
    assert _checks(issues, Severity.ERROR) == set()
    assert _checks(issues, Severity.WARNING) == set()
    assert stats.n_segments == 2
    assert stats.constant_command_columns == 23  # only the first command column ever fires


def test_constant_channel_is_flagged(raw_root: Path) -> None:
    channel = TelemanomDataset(raw_root).load("C-1")
    _, issues = check_channel(channel, Config())
    assert {"constant_train", "nominal_out_of_range"} <= _checks(issues, Severity.WARNING)


def test_out_of_range_ignores_labelled_anomalies(raw_root: Path) -> None:
    # A-1's test has an injected +4 spike exactly on its labelled [300, 320] segment.
    stats, _ = check_channel(TelemanomDataset(raw_root).load("A-1"), Config())
    assert stats is not None
    assert stats.test_max > stats.train_max + 2
    assert stats.nominal_out_of_range_fraction < 0.05


def test_structural_errors_short_circuit(raw_root: Path) -> None:
    channel = TelemanomDataset(raw_root).load("A-1")
    broken_test = channel.test.copy()
    broken_test[5, 0] = np.nan
    broken = ChannelData(
        info=replace(channel.info, num_values=999), train=channel.train, test=broken_test
    )
    stats, issues = check_channel(broken, Config())
    assert stats is None
    assert _checks(issues, Severity.ERROR) == {"non_finite", "length_mismatch"}


def test_feature_mismatch(raw_root: Path) -> None:
    channel = TelemanomDataset(raw_root).load("A-1")
    broken = ChannelData(info=channel.info, train=channel.train, test=channel.test[:, :10])
    _, issues = check_channel(broken, Config())
    assert "feature_mismatch" in _checks(issues, Severity.ERROR)


def test_short_train_warning(raw_root: Path) -> None:
    cfg = load_config(None, overrides=["windows.size=450"])
    _, issues = check_channel(TelemanomDataset(raw_root).load("A-1"), cfg)
    assert "short_train" in _checks(issues, Severity.WARNING)


def test_low_cardinality_info(raw_root: Path) -> None:
    channel = TelemanomDataset(raw_root).load("A-1")
    train = channel.train.copy()
    train[:, 0] = np.where(np.arange(train.shape[0]) % 2 == 0, 0.0, 1.0)
    _, issues = check_channel(replace(channel, train=train), Config())
    assert "low_cardinality" in _checks(issues, Severity.INFO)


def test_validate_dataset_aggregates(raw_root: Path) -> None:
    report = validate_dataset(TelemanomDataset(raw_root), Config())
    assert [s.chan_id for s in report.stats] == ["A-1", "C-1", "M-1"]
    assert report.errors == []
    assert report.check_counts()["warning:constant_train"] == 1
    assert report.to_dict()["counts"] == report.check_counts()
