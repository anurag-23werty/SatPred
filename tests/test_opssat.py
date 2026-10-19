from pathlib import Path

import numpy as np
import pytest

from sathealth.data.download import load_manifest
from sathealth.data.opssat import load_segments, prepare_channel


def test_load_and_prepare(opssat_csv: Path) -> None:
    segments = load_segments(opssat_csv)
    assert set(segments) == {"CH-A", "CH-B"}
    a = segments["CH-A"]
    assert len(a) == 40
    assert [s.start_time for s in a] == sorted(s.start_time for s in a)
    assert a[0].values.shape == (90, 1)
    ch = prepare_channel("CH-A", a, val_fraction=0.2)
    nominal_train = [s for s in a if s.train and not s.anomaly]
    assert len(ch.train) + len(ch.val) == len(nominal_train)
    points = [x.shape[0] for x in ch.train]
    assert sum(points) >= 0.8 * sum(s.length for s in nominal_train)
    assert len(ch.test) == sum(not s.train for s in a)
    assert ch.n_test_anomalous == sum(s.anomaly and not s.train for s in a)
    assert abs(float(np.median(np.concatenate(ch.train)))) < 1e-9, "robust scaling on train"


def test_prepare_needs_nominal_training_data(opssat_csv: Path) -> None:
    segments = load_segments(opssat_csv)
    only_test = [s for s in segments["CH-A"] if not s.train]
    with pytest.raises(ValueError, match="no nominal training"):
        prepare_channel("CH-A", only_test)


def test_manifest_is_pinned() -> None:
    manifest = load_manifest("opssat.json")
    (spec,) = manifest.files
    assert spec.path == "segments.csv"
    assert len(spec.sha256) == 64
    assert manifest.url_for(spec).startswith("https://zenodo.org/records/15108715/files/")
