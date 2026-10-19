from pathlib import Path

import pytest

from sathealth.config import Config, load_config
from sathealth.data.download import Manifest
from sathealth.pipeline.decide import decide
from sathealth.pipeline.evaluate import evaluate_detector
from sathealth.pipeline.leaderboard import load_summaries, write_leaderboard
from sathealth.pipeline.prepare import prepare_dataset


@pytest.fixture
def config(raw_root: Path, raw_manifest: Manifest, tmp_path: Path) -> Config:
    cfg = load_config(
        None,
        overrides=[
            f"paths.raw_dir={raw_root}",
            f"paths.processed_dir={tmp_path / 'processed'}",
            f"paths.runs_dir={tmp_path / 'runs'}",
            f"paths.decisions_dir={tmp_path / 'decisions'}",
            "evaluation.bootstrap=20",
            "decision.inputs=[zscore, limit]",
        ],
    )
    prepare_dataset(cfg, manifest=raw_manifest)
    for name in ("zscore", "limit", "iforest"):
        evaluate_detector(cfg, name)
    return cfg


def test_leaderboard_lists_the_decision_pipeline(config: Config, tmp_path: Path) -> None:
    decide(config)
    names = ["limit", "zscore", "fused"]
    assert [s["detector"] for s in load_summaries(config, names)] == names
    text = write_leaderboard(config, names, tmp_path / "board.md").read_text()
    assert "Decision layer: EVT on zscore + limit" in text
    assert "| fused | all | Composite F1 |" in text
    missing = config.model_copy(
        update={"decision": config.decision.model_copy(update={"name": "other"})}
    )
    with pytest.raises(FileNotFoundError, match="sathealth decide"):
        load_summaries(missing, ["other"])
