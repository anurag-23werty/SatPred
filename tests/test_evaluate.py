import json
from pathlib import Path

import numpy as np
import pytest

from sathealth.config import Config, load_config
from sathealth.data.download import Manifest
from sathealth.detectors.registry import BASELINE_NAMES, DETECTOR_NAMES, build_detector
from sathealth.pipeline.evaluate import TRAINING_FILE, evaluate_detector, evaluation_dir_for
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
            "deep.training.epochs=2",
            "deep.lstm.window=16",
            "deep.lstm.hidden=8",
            "deep.transformer.window=16",
            "deep.transformer.d_model=8",
            "deep.transformer.heads=2",
            "deep.transformer.layers=1",
        ],
    )
    prepare_dataset(cfg, manifest=raw_manifest)
    return cfg


def test_registry_builds_every_detector() -> None:
    assert set(DETECTOR_NAMES) == {"limit", "zscore", "iforest", "random", "lstm", "transformer"}
    assert set(BASELINE_NAMES) < set(DETECTOR_NAMES)
    for name in DETECTOR_NAMES:
        assert build_detector(name, Config()).name == name
    with pytest.raises(KeyError, match="unknown detector"):
        build_detector("magic", Config())


@pytest.mark.parametrize("detector", DETECTOR_NAMES)
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


@pytest.mark.parametrize("detector", ["lstm", "transformer"])
def test_deep_runs_store_training_curves(config: Config, detector: str) -> None:
    result = evaluate_detector(config, detector)
    training = json.loads((result.run_dir / TRAINING_FILE).read_text())
    assert set(training) == {"A-1", "C-1", "M-1"}
    assert training["A-1"]["best_epoch"] in {1, 2}
    assert len(training["A-1"]["train_loss"]) <= 2
    assert result.summary["params"]["training"]["epochs"] == 2
    assert not (evaluate_detector(config, "limit").run_dir / TRAINING_FILE).exists()


def _test_scores(run_dir: Path) -> dict[str, np.ndarray]:
    scores = {}
    for path in sorted((run_dir / "scores").glob("*.npz")):
        with np.load(path) as npz:
            scores[path.stem] = npz["test"]
    return scores


def test_parallel_jobs_give_identical_results(config: Config) -> None:
    serial = evaluate_detector(config, "lstm")
    serial_scores = _test_scores(serial.run_dir)
    parallel = evaluate_detector(config, "lstm", force=True, jobs=2)
    assert parallel.summary["jobs"] == 2
    assert parallel.run_dir == serial.run_dir
    parallel_scores = _test_scores(parallel.run_dir)
    assert set(parallel_scores) == {"A-1", "C-1", "M-1"}
    for chan, scores in serial_scores.items():
        np.testing.assert_array_equal(scores, parallel_scores[chan])
    assert parallel.summary["subsets"] == serial.summary["subsets"]


def test_jobs_must_be_positive(config: Config) -> None:
    with pytest.raises(ValueError, match="jobs"):
        evaluate_detector(config, "limit", force=True, jobs=0)


def test_finished_runs_are_sent_to_mlflow_when_configured(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        "sathealth.pipeline.evaluate.log_to_mlflow", lambda *args: calls.append(args) or "r1"
    )
    evaluate_detector(config, "zscore")
    assert calls == [], "tracking is off by default"
    tracked = config.model_copy(
        update={"tracking": config.tracking.model_copy(update={"mlflow_uri": "file:./m"})}
    )
    result = evaluate_detector(tracked, "zscore", force=True)
    (summary, run_dir, uri, experiment) = calls[0]
    assert (run_dir, uri, experiment) == (result.run_dir, "file:./m", "sathealth")
    assert summary["detector"] == "zscore"
