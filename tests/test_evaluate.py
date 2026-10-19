import json
from pathlib import Path

import numpy as np
import pytest

from sathealth.config import Config, load_config
from sathealth.data.download import Manifest
from sathealth.detectors.registry import DETECTOR_NAMES, build_detector
from sathealth.pipeline.evaluate import evaluate_detector, evaluation_dir_for
from sathealth.pipeline.prepare import prepare_dataset


@pytest.fixture
def config(raw_root: Path, raw_manifest: Manifest, tmp_path: Path) -> Config:
    cfg = load_config(
        None,
        overrides=[
            f"paths.raw_dir={raw_root}",
            f"paths.processed_dir={tmp_path / 'processed'}",
            f"paths.runs_dir={tmp_path / 'runs'}",
            "evaluation.bootstrap=50",
            "baselines.iforest.window=16",
            "baselines.iforest.n_estimators=50",
        ],
    )
    prepare_dataset(cfg, manifest=raw_manifest)
    return cfg


def test_registry_builds_every_detector() -> None:
    assert set(DETECTOR_NAMES) == {"limit", "zscore", "iforest", "random"}
    for name in DETECTOR_NAMES:
        assert build_detector(name, Config()).name == name
    with pytest.raises(KeyError, match="unknown detector"):
        build_detector("magic", Config())


@pytest.mark.parametrize("detector", ["limit", "zscore", "iforest", "random"])
def test_evaluation_run_outputs(config: Config, detector: str) -> None:
    result = evaluate_detector(config, detector)
    assert not result.reused
    subsets = result.summary["subsets"]
    assert set(subsets) == {
        "all",
        "MSL",
        "SMAP",
        "excluding constant_train",
        "excluding constant_train+nominal_out_of_range",
    }
    assert subsets["all"]["n_channels"] == 3
    assert subsets["excluding constant_train"]["n_channels"] == 2
    per_channel = json.loads((result.run_dir / "per_channel.json").read_text())
    assert [c["chan_id"] for c in per_channel] == ["A-1", "C-1", "M-1"]
    assert "constant_train" in per_channel[1]["flags"]
    with np.load(result.run_dir / "scores" / "A-1.npz") as npz:
        assert npz["test"].shape == (400,)
        assert float(npz["threshold"]) == pytest.approx(per_channel[0]["threshold"])


def test_limit_check_finds_injected_spike(config: Config) -> None:
    # A-1's test has a +4 spike exactly on its labelled point anomaly [300, 320].
    per_channel = json.loads(
        (evaluate_detector(config, "limit").run_dir / "per_channel.json").read_text()
    )
    a1 = next(c for c in per_channel if c["chan_id"] == "A-1")
    assert a1["events"]["tp"] >= 1
    assert a1["delays"][0] == 0


def test_reuse_force_and_fingerprint(config: Config) -> None:
    first = evaluate_detector(config, "zscore")
    assert evaluate_detector(config, "zscore").reused
    assert not evaluate_detector(config, "zscore", force=True).reused
    other = load_config(
        None, overrides=[f"paths.runs_dir={config.paths.runs_dir}", "evaluation.quantile=0.99"]
    )
    assert evaluation_dir_for(other, "zscore") != first.run_dir
    assert not list(config.paths.runs_dir.glob(".zscore-*"))


def test_missing_prepared_data(tmp_path: Path) -> None:
    cfg = load_config(
        None, overrides=[f"paths.processed_dir={tmp_path}", f"paths.runs_dir={tmp_path / 'runs'}"]
    )
    with pytest.raises(FileNotFoundError, match="sathealth prepare"):
        evaluate_detector(cfg, "limit")
