"""Per-channel data quality checks with explicit severities.

``ERROR``   the channel cannot be used as-is (pipeline stops when ``fail_on_error``).
``WARNING`` usable, but results on this channel need care (reported in the data card).
``INFO``    a property worth knowing when interpreting models (e.g. discrete telemetry).

The out-of-range check deliberately looks only at *nominal* (unlabelled) test points: labelled
anomalies are supposed to leave the training range, nominal points doing so is distribution
shift that will surface as false alarms.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from sathealth.config import Config
from sathealth.data.schema import ChannelData
from sathealth.data.telemanom import TelemanomDataset


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Issue:
    chan_id: str
    check: str
    severity: Severity
    message: str


@dataclass(frozen=True)
class ChannelStats:
    chan_id: str
    spacecraft: str
    n_train: int
    n_test: int
    n_features: int
    n_segments: int
    anomaly_fraction: float
    train_min: float
    train_max: float
    train_median: float
    train_iqr: float
    test_min: float
    test_max: float
    nominal_out_of_range_fraction: float
    train_unique_values: int
    constant_command_columns: int


@dataclass
class QualityReport:
    stats: list[ChannelStats] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    def by_severity(self, severity: Severity) -> list[Issue]:
        return [i for i in self.issues if i.severity is severity]

    @property
    def errors(self) -> list[Issue]:
        return self.by_severity(Severity.ERROR)

    @property
    def warnings(self) -> list[Issue]:
        return self.by_severity(Severity.WARNING)

    def failed_channels(self) -> set[str]:
        return {i.chan_id for i in self.errors}

    def check_counts(self) -> dict[str, int]:
        return dict(Counter(f"{i.severity}:{i.check}" for i in self.issues))

    def to_dict(self) -> dict[str, Any]:
        return {
            "stats": [asdict(s) for s in self.stats],
            "issues": [asdict(i) for i in self.issues],
            "counts": self.check_counts(),
        }


def check_channel(channel: ChannelData, config: Config) -> tuple[ChannelStats | None, list[Issue]]:
    """Run every check on one channel. Returns ``None`` stats if the channel is unreadable."""
    cid, vcfg = channel.chan_id, config.validation
    train, test = channel.train, channel.test
    issues: list[Issue] = []

    def add(check: str, severity: Severity, message: str) -> None:
        issues.append(Issue(cid, check, severity, message))

    # Structural errors: stop here, later statistics would be meaningless.
    if train.shape[1] != test.shape[1]:
        add(
            "feature_mismatch",
            Severity.ERROR,
            f"train has {train.shape[1]} features, test has {test.shape[1]}",
        )
    if bad := [c for c in config.preprocess.value_columns if c >= train.shape[1]]:
        add("value_column_missing", Severity.ERROR, f"value columns {bad} not in data")
    if not (np.isfinite(train).all() and np.isfinite(test).all()):
        n_bad = int((~np.isfinite(train)).sum() + (~np.isfinite(test)).sum())
        add("non_finite", Severity.ERROR, f"{n_bad} NaN/inf values")
    if test.shape[0] != channel.info.num_values:
        add(
            "length_mismatch",
            Severity.ERROR,
            f"test has {test.shape[0]} rows, label file declares {channel.info.num_values}",
        )
    if any(s.end >= test.shape[0] for s in channel.info.segments):
        add("label_bounds", Severity.ERROR, "an anomaly segment ends past the test series")
    if train.shape[0] == 0:
        add("empty_train", Severity.ERROR, "train series is empty")
    if issues:
        return None, issues

    col = config.preprocess.value_columns[0]
    tr, te = train[:, col], test[:, col]
    labels = channel.test_labels
    q1, median, q3 = np.quantile(tr, [0.25, 0.5, 0.75])
    lo, hi = float(tr.min()), float(tr.max())
    nominal = te[~labels]
    oor = float(((nominal < lo) | (nominal > hi)).mean()) if nominal.size else 0.0
    commands = train[
        :, [c for c in range(train.shape[1]) if c not in config.preprocess.value_columns]
    ]
    n_const_cmd = int((commands.std(axis=0) == 0).sum()) if commands.size else 0
    n_unique = int(np.unique(tr).size)

    stats = ChannelStats(
        chan_id=cid,
        spacecraft=channel.info.spacecraft,
        n_train=int(train.shape[0]),
        n_test=int(test.shape[0]),
        n_features=channel.n_features,
        n_segments=len(channel.info.segments),
        anomaly_fraction=float(labels.mean()),
        train_min=lo,
        train_max=hi,
        train_median=float(median),
        train_iqr=float(q3 - q1),
        test_min=float(te.min()),
        test_max=float(te.max()),
        nominal_out_of_range_fraction=oor,
        train_unique_values=n_unique,
        constant_command_columns=n_const_cmd,
    )

    if tr.std() < config.preprocess.min_scale:
        add(
            "constant_train",
            Severity.WARNING,
            f"telemetry is constant ({lo:.4g}) across all {tr.size} training steps; "
            "no nominal variability to learn from",
        )
    if oor > vcfg.out_of_range_warn_fraction:
        add(
            "nominal_out_of_range",
            Severity.WARNING,
            f"{oor:.1%} of unlabelled test points fall outside the train range "
            f"[{lo:.3g}, {hi:.3g}]: distribution shift, expect false alarms",
        )
    usable = int(train.shape[0] * (1 - config.preprocess.val_fraction))
    if usable < config.windows.size:
        add(
            "short_train",
            Severity.WARNING,
            f"{usable} training rows after the validation split < window size "
            f"{config.windows.size}",
        )
    if stats.anomaly_fraction > vcfg.max_anomaly_fraction:
        add(
            "high_anomaly_fraction",
            Severity.WARNING,
            f"{stats.anomaly_fraction:.1%} of test points are labelled anomalous",
        )
    if stats.n_segments == 0:
        add("no_anomalies", Severity.WARNING, "no labelled anomalies in test")
    if 1 < n_unique <= vcfg.low_cardinality_max:
        add(
            "low_cardinality",
            Severity.INFO,
            f"telemetry takes only {n_unique} distinct values in train (discrete/state signal)",
        )
    if commands.size and not np.isin(commands, (0.0, 1.0)).all():
        add("non_binary_commands", Severity.INFO, "command columns contain values other than 0/1")
    return stats, issues


def validate_dataset(dataset: TelemanomDataset, config: Config) -> QualityReport:
    report = QualityReport()
    for channel in dataset:
        stats, issues = check_channel(channel, config)
        if stats is not None:
            report.stats.append(stats)
        report.issues.extend(issues)
    return report
