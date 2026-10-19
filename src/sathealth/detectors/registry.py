"""Build detectors by name from the typed config."""

from collections.abc import Callable

from sathealth.config import Config
from sathealth.detectors.base import Detector
from sathealth.detectors.iforest import IsolationForestDetector
from sathealth.detectors.limit import LimitCheck
from sathealth.detectors.lstm import LSTMDetector
from sathealth.detectors.reference import RandomScore
from sathealth.detectors.transformer import TransformerDetector
from sathealth.detectors.zscore import RobustZScore

_BUILDERS: dict[str, Callable[[Config], Detector]] = {
    "limit": lambda c: LimitCheck(c.preprocess.value_columns, c.baselines.limit.margin),
    "zscore": lambda c: RobustZScore(c.preprocess.value_columns, c.preprocess.min_scale),
    "iforest": lambda c: IsolationForestDetector(
        c.preprocess.value_columns,
        window=c.baselines.iforest.window,
        n_estimators=c.baselines.iforest.n_estimators,
        max_samples=c.baselines.iforest.max_samples,
        seed=c.seed,
    ),
    "random": lambda c: RandomScore(c.preprocess.value_columns, seed=c.seed),
    "lstm": lambda c: LSTMDetector(
        c.preprocess.value_columns, c.deep.lstm, c.deep.training, seed=c.seed
    ),
    "transformer": lambda c: TransformerDetector(
        c.preprocess.value_columns, c.deep.transformer, c.deep.training, seed=c.seed
    ),
}

BASELINE_NAMES: tuple[str, ...] = ("limit", "zscore", "iforest", "random")

DETECTOR_NAMES: tuple[str, ...] = tuple(_BUILDERS)


def build_detector(name: str, config: Config) -> Detector:
    try:
        return _BUILDERS[name](config)
    except KeyError:
        raise KeyError(f"unknown detector {name!r}; choose from {DETECTOR_NAMES}") from None
