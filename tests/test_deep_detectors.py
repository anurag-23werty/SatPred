import numpy as np
import pytest

from sathealth.config import LSTMConfig, TrainingConfig, TransformerConfig
from sathealth.detectors.deep import DeepDetector
from sathealth.detectors.lstm import LSTMDetector
from sathealth.detectors.transformer import TransformerDetector

FAST = TrainingConfig(epochs=8, batch_size=32, learning_rate=3e-3, patience=3)


def _series(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    value = np.sin(t / 8) + 0.05 * rng.normal(size=n)
    commands = np.zeros((n, 2))
    commands[t % 30 == 0, 0] = 1.0
    return np.column_stack([value, commands])


def _lstm(**kw: int) -> LSTMDetector:
    cfg = LSTMConfig(window=24, hidden=16, layers=1, dropout=0.0, **kw)
    return LSTMDetector((0,), cfg, FAST, seed=0)


def _transformer() -> TransformerDetector:
    cfg = TransformerConfig(
        window=24, d_model=16, heads=2, layers=1, feedforward=32, latent=4, dropout=0.0
    )
    return TransformerDetector((0,), cfg, FAST, seed=0)


@pytest.fixture(scope="module")
def lstm() -> LSTMDetector:
    return _lstm().fit(_series(600, 1), _series(150, 2))


@pytest.fixture(scope="module")
def transformer() -> TransformerDetector:
    return _transformer().fit(_series(600, 1), _series(150, 2))


@pytest.mark.parametrize("name", ["lstm", "transformer"])
def test_detects_injected_level_shift(
    name: str, lstm: LSTMDetector, transformer: TransformerDetector
) -> None:
    det: DeepDetector = lstm if name == "lstm" else transformer
    test = _series(400, 3)
    test[250:270, 0] += 2.0
    scores = det.score(test)
    assert scores.shape == (400,)
    assert np.isfinite(scores).all()
    assert scores[250:275].max() > 2 * scores[50:240].max()


def test_lstm_score_grows_with_deviation(lstm: LSTMDetector) -> None:
    # The Isolation Forest saturates outside the training range; a forecaster must not.
    peaks = []
    for size in (3.0, 30.0):
        test = _series(300, 4)
        test[200:205, 0] += size
        peaks.append(lstm.score(test)[200:210].max())
    assert peaks[1] > 5 * peaks[0]


@pytest.mark.parametrize("make", [_lstm, _transformer])
def test_fit_is_bit_reproducible(make: type[DeepDetector]) -> None:
    train, val, test = _series(300, 1), _series(80, 2), _series(120, 3)
    first = make().fit(train, val).score(test)
    second = make().fit(train, val).score(test)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("length", [1, 2, 10, 24, 25, 60])
@pytest.mark.parametrize("name", ["lstm", "transformer"])
def test_scores_every_step_of_short_series(
    name: str, length: int, lstm: LSTMDetector, transformer: TransformerDetector
) -> None:
    det: DeepDetector = lstm if name == "lstm" else transformer
    scores = det.score(_series(length, 5))
    assert scores.shape == (length,)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


@pytest.mark.parametrize("name", ["lstm", "transformer"])
def test_scores_are_causal(name: str, lstm: LSTMDetector, transformer: TransformerDetector) -> None:
    """The score at step t uses only steps <= t, as it must in a live stream."""
    det: DeepDetector = lstm if name == "lstm" else transformer
    x = _series(90, 6)
    changed = x.copy()
    changed[60:, 0] += 50.0
    np.testing.assert_allclose(det.score(x)[:60], det.score(changed)[:60], rtol=1e-5)
    np.testing.assert_allclose(det.score(x)[:60], det.score(x[:60]), rtol=1e-5)


def test_smoothing_lowers_isolated_spikes(lstm: LSTMDetector) -> None:
    raw = _lstm(smoothing=1)
    raw._model, raw.history = lstm.model, lstm.history
    test = _series(200, 7)
    test[100, 0] += 2.0
    assert lstm.score(test)[100] < raw.score(test)[100]


def test_diagnostics_and_params(lstm: LSTMDetector, transformer: TransformerDetector) -> None:
    diag = lstm.diagnostics()
    assert diag["parameters"] > 0
    assert diag["monitor"] == "val"
    assert 1 <= diag["best_epoch"] <= FAST.epochs
    params = transformer.params()
    assert params["architecture"]["latent"] == 4
    assert params["training"]["epochs"] == FAST.epochs
    assert _lstm().diagnostics() == {}


def test_short_validation_monitors_train_loss() -> None:
    det = _lstm().fit(_series(300, 1), _series(10, 2))  # val shorter than one window
    assert det.diagnostics()["monitor"] == "train"


def test_unfitted_detector_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        _transformer().score(_series(50, 0))
