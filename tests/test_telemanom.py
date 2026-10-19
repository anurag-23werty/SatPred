from pathlib import Path

import numpy as np
import pytest

from sathealth.config import DatasetConfig, DuplicatePolicy
from sathealth.data.schema import AnomalyKind, AnomalySegment, segments_to_mask
from sathealth.data.telemanom import LABEL_FILE, TelemanomDataset, parse_label_file


def test_segments_are_sorted_and_typed(raw_root: Path) -> None:
    table = parse_label_file(raw_root / LABEL_FILE)
    a1 = table.channels["A-1"]
    assert a1.segments == (
        AnomalySegment(120, 140, AnomalyKind.CONTEXTUAL),
        AnomalySegment(300, 320, AnomalyKind.POINT),
    )
    assert a1.group == "A"


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (DuplicatePolicy.DROP, None),
        (DuplicatePolicy.FIRST, ((200, 260),)),
        (DuplicatePolicy.UNION, ((200, 300),)),
    ],
)
def test_duplicate_policies(
    raw_root: Path, policy: DuplicatePolicy, expected: tuple[tuple[int, int], ...] | None
) -> None:
    table = parse_label_file(raw_root / LABEL_FILE, policy)
    assert len(table.notes) == 1
    if expected is None:
        assert "P-2" not in table.channels
    else:
        assert tuple((s.start, s.end) for s in table.channels["P-2"].segments) == expected


def test_mismatched_class_count_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / LABEL_FILE
    path.write_text(
        "chan_id,spacecraft,anomaly_sequences,class,num_values\n"
        'X-1,SMAP,"[[1, 2], [5, 6]]",[point],10\n'
    )
    with pytest.raises(ValueError, match="2 sequences but 1 classes"):
        parse_label_file(path)


def test_missing_columns_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / LABEL_FILE
    path.write_text("chan_id,spacecraft\nX-1,SMAP\n")
    with pytest.raises(ValueError, match="missing columns"):
        parse_label_file(path)


def test_segments_to_mask_is_inclusive() -> None:
    mask = segments_to_mask((AnomalySegment(2, 4),), 6)
    assert mask.tolist() == [False, False, True, True, True, False]
    with pytest.raises(ValueError, match="exceeds"):
        segments_to_mask((AnomalySegment(2, 6),), 6)


def test_invalid_segment_rejected() -> None:
    with pytest.raises(ValueError, match="invalid segment"):
        AnomalySegment(5, 4)


def test_dataset_filters_and_loads(raw_root: Path) -> None:
    ds = TelemanomDataset(raw_root, DatasetConfig(spacecraft=("SMAP",)))
    assert ds.channel_ids == ["A-1", "C-1"]  # P-2 dropped, M-1 is MSL
    channel = ds.load("A-1")
    assert channel.train.shape == (500, 25)
    assert channel.test_labels.sum() == 21 + 21
    assert channel.test.dtype == np.float64


def test_dataset_channel_subset(raw_root: Path) -> None:
    ds = TelemanomDataset(raw_root, DatasetConfig(channels=("M-1",)))
    assert [c.chan_id for c in ds] == ["M-1"]
    assert ds.load("M-1").n_features == 55
    with pytest.raises(KeyError, match="Z-9"):
        TelemanomDataset(raw_root, DatasetConfig(channels=("Z-9",)))


def test_missing_label_file_gives_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sathealth download"):
        TelemanomDataset(tmp_path)
