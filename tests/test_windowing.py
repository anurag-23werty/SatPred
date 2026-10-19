import numpy as np
import pytest

from sathealth.features.windowing import (
    chronological_split,
    n_windows,
    sliding_windows,
    window_end_index,
    window_labels,
)


@pytest.mark.parametrize(
    ("length", "size", "stride"), [(10, 3, 1), (10, 3, 2), (10, 10, 1), (9, 10, 1), (100, 7, 5)]
)
def test_window_contents_match_naive(length: int, size: int, stride: int) -> None:
    x = np.arange(length * 2, dtype=float).reshape(length, 2)
    windows = sliding_windows(x, size, stride)
    naive = [x[s : s + size] for s in range(0, length - size + 1, stride)]
    assert windows.shape == (n_windows(length, size, stride), size, 2)
    for got, want in zip(windows, naive, strict=True):
        np.testing.assert_array_equal(got, want)


def test_windows_are_read_only_views() -> None:
    x = np.zeros((1000, 25))
    windows = sliding_windows(x, 100)
    assert np.shares_memory(windows, x)
    with pytest.raises(ValueError, match="read-only"):
        windows[0, 0, 0] = 1.0


@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("rule", ["any", "last"])
def test_window_labels_match_naive(stride: int, rule: str) -> None:
    rng = np.random.default_rng(3)
    mask = rng.random(200) > 0.9
    got = window_labels(mask, 12, stride, rule)  # type: ignore[arg-type]
    starts = range(0, 200 - 12 + 1, stride)
    want = [mask[s : s + 12].any() if rule == "any" else mask[s + 11] for s in starts]
    np.testing.assert_array_equal(got, want)


def test_end_index_points_at_last_row() -> None:
    x = np.arange(20, dtype=float)[:, None]
    ends = window_end_index(20, 5, 3)
    np.testing.assert_array_equal(sliding_windows(x, 5, 3)[:, -1, 0], ends)


def test_too_short_series_gives_empty_outputs() -> None:
    assert sliding_windows(np.zeros((3, 4)), 5).shape == (0, 5, 4)
    assert window_labels(np.zeros(3, dtype=bool), 5).shape == (0,)


def test_split_is_chronological_and_disjoint() -> None:
    train, val = chronological_split(1000, 0.2, gap=50)
    assert (train.start, train.stop, val.start, val.stop) == (0, 750, 800, 1000)
    rows = np.arange(1000)
    train_windows = sliding_windows(rows[train, None].astype(float), 100)
    assert train_windows.max() < val.start, "no train window may touch validation rows"


def test_split_without_validation() -> None:
    train, val = chronological_split(10, 0.0)
    assert (train.stop, val.start, val.stop) == (10, 10, 10)


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="val_fraction"):
        chronological_split(10, 1.0)
    with pytest.raises(ValueError, match=">= 1"):
        n_windows(10, 0)
