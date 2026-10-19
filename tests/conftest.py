"""Shared fixtures: a tiny synthetic dataset in the exact raw Telemanom layout."""

from pathlib import Path

import numpy as np
import pytest

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
