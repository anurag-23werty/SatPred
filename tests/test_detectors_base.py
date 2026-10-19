import numpy as np
import pytest

from sathealth.detectors.base import Detector, align_window_scores


def test_align_end_index_and_leading_fill() -> None:
    out = align_window_scores(np.array([1.0, 2.0, 3.0]), length=5, size=3)
    np.testing.assert_array_equal(out, [1.0, 1.0, 1.0, 2.0, 3.0])


def test_align_with_stride_forward_fills() -> None:
    # length 10, size 4, stride 3 -> windows end at 3, 6, 9
    out = align_window_scores(np.array([1.0, 2.0, 3.0]), length=10, size=4, stride=3)
    np.testing.assert_array_equal(out, [1, 1, 1, 1, 1, 1, 2, 2, 2, 3])


def test_align_tail_after_last_window() -> None:
    # length 9, size 4, stride 3 -> windows end at 3, 6; steps 7-8 keep the last score
    out = align_window_scores(np.array([1.0, 2.0]), length=9, size=4, stride=3)
    np.testing.assert_array_equal(out[-3:], [2.0, 2.0, 2.0])


def test_align_rejects_wrong_count() -> None:
    with pytest.raises(ValueError, match="window scores"):
        align_window_scores(np.zeros(2), length=5, size=3)


def test_align_too_short_series_is_nan() -> None:
    assert np.isnan(align_window_scores(np.zeros(0), length=2, size=3)).all()


class _Dummy(Detector):
    name = "dummy"

    def fit(self, train: np.ndarray, val: np.ndarray) -> "_Dummy":
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.abs(self._values(x)).max(axis=1)


def test_detector_contract() -> None:
    det = _Dummy(value_columns=(0,)).fit(np.zeros((5, 3)), np.zeros((2, 3)))
    assert det.score(np.array([[2.0, 9.0, 9.0]])).tolist() == [2.0]
    assert det.params() == {"value_columns": [0]}
    with pytest.raises(ValueError, match="T, D"):
        det.score(np.zeros(3))
