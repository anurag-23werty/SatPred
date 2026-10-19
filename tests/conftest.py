"""Shared fixtures: a tiny synthetic dataset in the exact raw Telemanom layout."""

from pathlib import Path

import numpy as np
import pytest

from sathealth.config import Config, load_config
from sathealth.data.download import FileSpec, Manifest
from sathealth.utils.hashing import sha256_file

LABELS = """chan_id,spacecraft,anomaly_sequences,class,num_values
A-1,SMAP,"[[300, 320], [120, 140]]","[point, contextual]",400
P-2,SMAP,"[[200, 260]]",[point],400
P-2,SMAP,"[[250, 300]]",[point],400
C-1,SMAP,"[[50, 60]]",[contextual],400
M-1,MSL,"[[100, 180]]",[contextual],350
"""


def _channel(rng: np.random.Generator, n: int, d: int, constant: bool = False) -> np.ndarray:
    t = np.arange(n)
    value = np.full(n, -1.0) if constant else np.sin(t / 15) + 0.05 * rng.normal(size=n)
    commands = np.zeros((n, d - 1))
    commands[t % 50 == 0, 0] = 1.0
    return np.column_stack([value, commands])


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    """Four channels: normal SMAP (A-1), duplicated P-2, constant-in-train C-1, MSL M-1."""
    root = tmp_path / "raw"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    (root / "labeled_anomalies.csv").write_text(LABELS)
    rng = np.random.default_rng(0)
    specs = {
        "A-1": (500, 400, 25, False),
        "P-2": (500, 400, 25, False),
        "C-1": (500, 400, 25, True),
        "M-1": (450, 350, 55, False),
    }
    for chan, (n_train, n_test, d, constant) in specs.items():
        np.save(root / "train" / f"{chan}.npy", _channel(rng, n_train, d, constant))
        test = _channel(rng, n_test, d)
        test[300:321, 0] += 4.0  # injected spike, only matters for A-1's labels
        np.save(root / "test" / f"{chan}.npy", test)
    return root


@pytest.fixture
def raw_manifest(raw_root: Path) -> Manifest:
    files = tuple(
        FileSpec(p.relative_to(raw_root).as_posix(), p.stat().st_size, sha256_file(p))
        for p in sorted(raw_root.rglob("*"))
        if p.is_file()
    )
    return Manifest(name="synthetic", base_url="https://mirror.test", files=files)


EXPLAIN_FAST = [
    "deep.training.epochs=3",
    "deep.lstm.window=16",
    "deep.lstm.hidden=8",
    "decision.inputs=[lstm]",
    "explain.horizon=20",
    "explain.band_starts=5",
    "explain.ig_steps=8",
    "explain.eval_injections=3",
    "explain.eval_coverage_horizon=10",
    "explain.eval_channels_per_spacecraft=1",
    "evaluation.bootstrap=20",
]


@pytest.fixture
def explain_config(raw_root: Path, raw_manifest: Manifest, tmp_path: Path) -> Config:
    """Synthetic dataset with a tiny LSTM evaluated and a decision run on its scores."""
    from sathealth.pipeline.decide import decide
    from sathealth.pipeline.evaluate import evaluate_detector
    from sathealth.pipeline.prepare import prepare_dataset

    cfg = load_config(
        None,
        overrides=[
            f"paths.raw_dir={raw_root}",
            f"paths.processed_dir={tmp_path / 'processed'}",
            f"paths.runs_dir={tmp_path / 'runs'}",
            f"paths.decisions_dir={tmp_path / 'decisions'}",
            f"paths.models_dir={tmp_path / 'models'}",
            f"paths.explanations_dir={tmp_path / 'explanations'}",
            *EXPLAIN_FAST,
        ],
    )
    prepare_dataset(cfg, manifest=raw_manifest)
    evaluate_detector(cfg, "lstm")
    decide(cfg)
    return cfg


def _write_opssat(path: Path, seed: int = 0) -> Path:
    """Two synthetic channels in the OPSSAT-AD layout: ``CH-A`` with segments long enough for
    the models, ``CH-B`` with only short segments."""
    rng = np.random.default_rng(seed)
    lines = ["channel,timestamp,value,label,sampling,anomaly,segment,train"]
    seg_id = 0
    for chan, n_segments, length in (("CH-A", 40, 90), ("CH-B", 6, 10)):
        for k in range(n_segments):
            seg_id += 1
            anomaly = k % 5 == 4
            train = k % 4 != 3
            t = np.arange(length)
            values = np.sin(t / 6) + 0.05 * rng.normal(size=length)
            if anomaly:
                values[length // 2 : length // 2 + 8] += 3.0
            day = 1 + k // 10
            for i, v in enumerate(values):
                stamp = f"2022-06-{day:02d}T{(k % 10):02d}:{i // 60:02d}:{i % 60:02d}.000Z"
                lines.append(
                    f"{chan},{stamp},{v:.6f},{'anomaly' if anomaly else 'normal'},1,"
                    f"{int(anomaly)},{seg_id},{int(train)}"
                )
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def opssat_csv(tmp_path: Path) -> Path:
    """A synthetic ``segments.csv`` in the OPSSAT-AD layout (see ``_write_opssat``)."""
    raw = tmp_path / "opssat"
    raw.mkdir()
    return _write_opssat(raw / "segments.csv")
