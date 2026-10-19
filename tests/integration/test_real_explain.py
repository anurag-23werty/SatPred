"""Pin the Phase 5 findings on real channels (``make test-data``).

Needs the saved Phase 3 LSTM scores (``sathealth evaluate lstm``, ~12 minutes), which the CI
real-data job does not produce, so it is skipped there.
"""

from pathlib import Path

import numpy as np
import pytest

from sathealth.config import Config, load_config
from sathealth.explain.narrative import narrate, ungrounded
from sathealth.pipeline.evaluate import evaluation_dir_for
from sathealth.pipeline.explain import attribute_peak, channel_band, fitted_lstm
from sathealth.pipeline.prepare import load_prepared, run_dir_for

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config(ROOT / "configs" / "live.yaml")

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not (ROOT / evaluation_dir_for(CONFIG, "lstm") / "summary.json").is_file(),
        reason="no saved LSTM evaluation (run `sathealth evaluate lstm`)",
    ),
]


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    tmp = tmp_path_factory.mktemp("phase5")
    paths = CONFIG.paths.model_copy(
        update={
            "raw_dir": ROOT / CONFIG.paths.raw_dir,
            "processed_dir": ROOT / CONFIG.paths.processed_dir,
            "runs_dir": ROOT / CONFIG.paths.runs_dir,
            "models_dir": tmp / "models",  # force a rebuild + verification
        }
    )
    return CONFIG.model_copy(update={"paths": paths})


def test_rebuilt_model_reproduces_the_evaluated_one(config: Config) -> None:
    channel = load_prepared(run_dir_for(config), "G-7")
    det = fitted_lstm(config, channel)  # raises if the rebuild differs
    attr = attribute_peak(config, det, channel, 7667)
    assert attr.completeness_gap < 0.05
    assert attr.top_features(1) == [0], "G-7's model relies on its own recent values"
    band = channel_band(config, det, channel, np.random.default_rng(0))
    assert band.source == "val"
    assert np.all(band.upper >= band.lower)


def test_a_full_narrative_is_grounded() -> None:
    explanation = {
        "chan_id": "G-7",
        "spacecraft": "SMAP",
        "onset": 7594,
        "event": {
            "start": 7596,
            "end": 7697,
            "peak_step": 7667,
            "peak_surprise": 9.5,
            "severity": "critical",
            "kind": "contextual",
            "contributions": {"lstm": 7.0, "transformer": 12.0},
        },
    }
    sentences, facts = narrate(explanation)
    assert ungrounded(sentences, facts) == []
