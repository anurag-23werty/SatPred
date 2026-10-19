import json
from pathlib import Path

import numpy as np
import pytest

from sathealth.config import Config, load_config
from sathealth.data.download import Manifest
from sathealth.data.telemanom import TelemanomDataset
from sathealth.features.validation import validate_dataset
from sathealth.pipeline.prepare import (
    DataIntegrityError,
    DataQualityError,
    load_prepared,
    prepare_dataset,
    run_dir_for,
)
from sathealth.pipeline.report import render_data_card


def _config(raw_root: Path, tmp_path: Path, *extra: str) -> Config:
    return load_config(
        None,
        overrides=[
            f"paths.raw_dir={raw_root}",
            f"paths.processed_dir={tmp_path / 'processed'}",
            *extra,
        ],
    )


def test_prepare_end_to_end(raw_root: Path, raw_manifest: Manifest, tmp_path: Path) -> None:
    cfg = _config(raw_root, tmp_path)
    result = prepare_dataset(cfg, manifest=raw_manifest)
    assert result.channels == ("A-1", "C-1", "M-1")
    assert not result.reused

    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["label_cleaning"] == ["P-2: listed 2x with conflicting labels; dropped"]
    assert manifest["raw"]["verified"] is True
    assert manifest["channels"]["A-1"]["rows"] == {"train": 400, "val": 100, "test": 400}
    assert {i["check"] for i in manifest["channels"]["C-1"]["issues"]} >= {"constant_train"}

    channel = load_prepared(result.run_dir, "A-1")
    assert channel.train.dtype == np.float32
    assert channel.test_labels.sum() == 42
    assert channel.windows("train", 100).shape == (301, 100, 25)
    assert channel.test_window_labels(100).shape == (301,)


def test_scaler_is_fit_on_train_rows_only(
    raw_root: Path, raw_manifest: Manifest, tmp_path: Path
) -> None:
    result = prepare_dataset(_config(raw_root, tmp_path), manifest=raw_manifest)
    channel = load_prepared(result.run_dir, "A-1")
    raw_train = TelemanomDataset(raw_root).load("A-1").train[:400, 0]
    q1, median, q3 = np.quantile(raw_train, [0.25, 0.5, 0.75])
    assert channel.scaler.center[0] == pytest.approx(median)
    assert channel.scaler.scale[0] == pytest.approx((q3 - q1) / 1.349)
    assert abs(float(np.median(channel.train[:, 0]))) < 1e-6


def test_second_run_reuses_and_force_rebuilds(
    raw_root: Path, raw_manifest: Manifest, tmp_path: Path
) -> None:
    cfg = _config(raw_root, tmp_path)
    first = prepare_dataset(cfg, manifest=raw_manifest)
    assert prepare_dataset(cfg, manifest=raw_manifest).reused
    assert not prepare_dataset(cfg, manifest=raw_manifest, force=True).reused
    assert not list(first.run_dir.parent.glob(".prepare-*")), "temp dirs must be cleaned up"


def test_window_settings_do_not_change_fingerprint(raw_root: Path, tmp_path: Path) -> None:
    base = _config(raw_root, tmp_path)
    assert run_dir_for(base) == run_dir_for(_config(raw_root, tmp_path, "windows.size=7"))
    assert run_dir_for(base) != run_dir_for(_config(raw_root, tmp_path, "preprocess.scaler=none"))


def test_corrupted_raw_file_blocks_prepare(
    raw_root: Path, raw_manifest: Manifest, tmp_path: Path
) -> None:
    (raw_root / "test" / "A-1.npy").write_bytes(b"garbage")
    with pytest.raises(DataIntegrityError, match="sathealth download"):
        prepare_dataset(_config(raw_root, tmp_path), manifest=raw_manifest)


def test_quality_errors_block_prepare(raw_root: Path, tmp_path: Path) -> None:
    broken = np.load(raw_root / "test" / "M-1.npy")
    broken[10, 0] = np.nan
    np.save(raw_root / "test" / "M-1.npy", broken)
    with pytest.raises(DataQualityError, match="M-1"):
        prepare_dataset(_config(raw_root, tmp_path), verify=False)

    lenient = _config(raw_root, tmp_path, "validation.fail_on_error=false")
    result = prepare_dataset(lenient, verify=False)
    assert "M-1" not in result.channels


def test_unknown_split_rejected(raw_root: Path, raw_manifest: Manifest, tmp_path: Path) -> None:
    result = prepare_dataset(_config(raw_root, tmp_path), manifest=raw_manifest)
    with pytest.raises(KeyError, match="unknown split"):
        load_prepared(result.run_dir, "A-1").split("holdout")


def test_data_card_renders(raw_root: Path, raw_manifest: Manifest) -> None:
    dataset = TelemanomDataset(raw_root)
    card = render_data_card(dataset, validate_dataset(dataset, Config()), raw_manifest)
    assert "| SMAP | 2 | 25 |" in card
    assert "P-2: listed 2x" in card
    assert "`constant_train`" in card
    assert "| C-1 | SMAP |" in card
