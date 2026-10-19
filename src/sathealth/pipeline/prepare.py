"""``prepare``: verified raw data -> validated, split, scaled arrays + a run manifest.

Output layout (one directory per data-affecting config fingerprint)::

    data/processed/telemanom-<fingerprint>/
        manifest.json         provenance: config, code version, git commit, raw checksums,
                              per-channel row counts, scaler parameters and quality issues
        quality_report.json   full validation statistics
        channels/<chan>.npz   train / val / test (float32, scaled), test_labels, segments

Guarantees:

* **Integrity gate**: raw files are re-verified against the pinned SHA-256 manifest first.
* **No leakage**: the scaler is fit on the training rows only; validation and test are
  transformed with those parameters.
* **Idempotent and atomic**: written to a temp dir, renamed into place when complete; an
  existing complete run is reused unless ``force``.
* Window size/stride are not part of the fingerprint: windows are built as views at load
  time, so changing them never requires re-preparing.
"""

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sathealth.config import Config
from sathealth.data.download import Manifest, load_manifest, verify_dataset
from sathealth.data.schema import AnomalyKind, AnomalySegment, BoolArray, FloatArray
from sathealth.data.telemanom import TelemanomDataset
from sathealth.features.scaling import ChannelScaler, ScalerParams
from sathealth.features.validation import QualityReport, validate_dataset
from sathealth.features.windowing import chronological_split, sliding_windows, window_labels
from sathealth.logging import get_logger
from sathealth.utils.hashing import fingerprint
from sathealth.utils.provenance import provenance

log = get_logger(__name__)

MANIFEST_FILE = "manifest.json"
SPLITS = ("train", "val", "test")


class DataIntegrityError(RuntimeError):
    """Raw files are missing or do not match their pinned checksums."""


class DataQualityError(RuntimeError):
    """Validation found ERROR-level issues and ``validation.fail_on_error`` is set."""


@dataclass(frozen=True)
class PreparedChannel:
    chan_id: str
    spacecraft: str
    train: FloatArray
    val: FloatArray
    test: FloatArray
    test_labels: BoolArray
    segments: tuple[AnomalySegment, ...]
    scaler: ScalerParams

    def split(self, name: str) -> FloatArray:
        if name not in SPLITS:
            raise KeyError(f"unknown split {name!r}; expected one of {SPLITS}")
        arr: FloatArray = getattr(self, name)
        return arr

    def windows(self, name: str, size: int, stride: int = 1) -> FloatArray:
        return sliding_windows(self.split(name), size, stride)

    def test_window_labels(self, size: int, stride: int = 1, rule: str = "any") -> BoolArray:
        return window_labels(self.test_labels, size, stride, "last" if rule == "last" else "any")


@dataclass(frozen=True)
class PrepareResult:
    run_dir: Path
    channels: tuple[str, ...]
    reused: bool
    report: QualityReport | None


def data_fingerprint(config: Config) -> str:
    """Hash of exactly the settings that change the stored arrays."""
    return fingerprint(
        {
            "dataset": config.dataset.model_dump(mode="json"),
            "preprocess": config.preprocess.model_dump(mode="json"),
        }
    )


def run_dir_for(config: Config) -> Path:
    return config.paths.processed_dir / f"{config.dataset.name}-{data_fingerprint(config)}"


def _manifest_digest(manifest: Manifest) -> str:
    return fingerprint([(f.path, f.size, f.sha256) for f in manifest.files], length=64)


def prepare_dataset(
    config: Config, *, manifest: Manifest | None = None, verify: bool = True, force: bool = False
) -> PrepareResult:
    run_dir = run_dir_for(config)
    if (run_dir / MANIFEST_FILE).is_file() and not force:
        channels = tuple(json.loads((run_dir / MANIFEST_FILE).read_text())["channels"])
        log.info("reusing prepared run", run_dir=run_dir, channels=len(channels))
        return PrepareResult(run_dir, channels, reused=True, report=None)

    manifest = manifest or load_manifest(config.download.manifest)
    raw_dir = config.paths.raw_dir
    if verify:
        integrity = verify_dataset(raw_dir, manifest, config.download.max_workers)
        if not integrity.ok:
            bad = sorted(p for p, s in integrity.statuses.items() if s.value != "ok")
            raise DataIntegrityError(
                f"{len(bad)} raw files failed verification (e.g. {bad[:3]}); "
                "run `sathealth download` to repair"
            )

    dataset = TelemanomDataset(raw_dir, config.dataset)
    report = validate_dataset(dataset, config)
    for issue in report.errors:
        log.error("data quality", channel=issue.chan_id, check=issue.check, detail=issue.message)
    if report.errors and config.validation.fail_on_error:
        raise DataQualityError(
            f"{len(report.errors)} error-level issues in channels "
            f"{sorted(report.failed_channels())}"
        )
    log.info("validation finished", channels=len(dataset), **report.check_counts())

    issues_by_channel: dict[str, list[dict[str, str]]] = {}
    for issue in report.issues:
        issues_by_channel.setdefault(issue.chan_id, []).append(
            {"check": issue.check, "severity": issue.severity.value, "message": issue.message}
        )

    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".prepare-", dir=config.paths.processed_dir))
    try:
        (tmp / "channels").mkdir()
        channel_meta: dict[str, Any] = {}
        for chan_id in dataset.channel_ids:
            if chan_id in report.failed_channels():
                continue
            channel_meta[chan_id] = _prepare_channel(dataset, chan_id, config, tmp / "channels")
            channel_meta[chan_id]["issues"] = issues_by_channel.get(chan_id, [])

        run_manifest = {
            "run_id": run_dir.name,
            **provenance(),
            "config": config.to_dict(),
            "data_fingerprint": data_fingerprint(config),
            "raw": {
                "manifest": manifest.name,
                "digest": _manifest_digest(manifest),
                "verified": verify,
                "source": manifest.source,
            },
            "label_cleaning": list(dataset.notes),
            "quality_counts": report.check_counts(),
            "channels": channel_meta,
        }
        (tmp / "quality_report.json").write_text(json.dumps(report.to_dict(), indent=2))
        (tmp / MANIFEST_FILE).write_text(json.dumps(run_manifest, indent=2))
        if run_dir.exists():
            shutil.rmtree(run_dir)
        tmp.rename(run_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    log.info("prepared dataset", run_dir=run_dir, channels=len(channel_meta))
    return PrepareResult(run_dir, tuple(channel_meta), reused=False, report=report)


def _prepare_channel(
    dataset: TelemanomDataset, chan_id: str, config: Config, out_dir: Path
) -> dict[str, Any]:
    channel = dataset.load(chan_id)
    pcfg = config.preprocess
    train_rows, val_rows = chronological_split(
        channel.train.shape[0], pcfg.val_fraction, pcfg.split_gap
    )
    scaler = ChannelScaler(pcfg.scaler, pcfg.value_columns, pcfg.min_scale)
    scaler.fit(channel.train[train_rows])
    arrays = {
        "train": scaler.transform(channel.train[train_rows]).astype(np.float32),
        "val": scaler.transform(channel.train[val_rows]).astype(np.float32),
        "test": scaler.transform(channel.test).astype(np.float32),
    }
    segs = channel.info.segments
    # typed as Any: numpy's stub for **kwds collides with its `allow_pickle: bool` keyword
    payload: dict[str, Any] = {
        **arrays,
        "test_labels": channel.test_labels,
        "segments": np.array([(s.start, s.end) for s in segs], dtype=np.int64).reshape(-1, 2),
        "segment_kinds": np.array([s.kind.value for s in segs], dtype="<U16"),
    }
    np.savez_compressed(out_dir / f"{chan_id}.npz", **payload)
    return {
        "spacecraft": channel.info.spacecraft,
        "rows": {name: int(arr.shape[0]) for name, arr in arrays.items()},
        "n_features": channel.n_features,
        "scaler": scaler.params.to_dict(),
    }


def load_run_manifest(run_dir: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((run_dir / MANIFEST_FILE).read_text())
    return manifest


def load_prepared(run_dir: Path, chan_id: str) -> PreparedChannel:
    meta = load_run_manifest(run_dir)["channels"][chan_id]
    with np.load(run_dir / "channels" / f"{chan_id}.npz", allow_pickle=False) as npz:
        segments = tuple(
            AnomalySegment(int(s), int(e), AnomalyKind(str(k)))
            for (s, e), k in zip(npz["segments"], npz["segment_kinds"], strict=True)
        )
        return PreparedChannel(
            chan_id=chan_id,
            spacecraft=meta["spacecraft"],
            train=npz["train"],
            val=npz["val"],
            test=npz["test"],
            test_labels=npz["test_labels"],
            segments=segments,
            scaler=ScalerParams.from_dict(meta["scaler"]),
        )
