from pathlib import Path

import pytest

from sathealth.config import Config, load_config
from sathealth.data.download import Manifest
from sathealth.pipeline.evaluate import evaluate_detector
from sathealth.pipeline.leaderboard import load_summaries, render_leaderboard, write_leaderboard
from sathealth.pipeline.prepare import prepare_dataset


@pytest.fixture
def config(raw_root: Path, raw_manifest: Manifest, tmp_path: Path) -> Config:
    cfg = load_config(
        None,
        overrides=[
            f"paths.raw_dir={raw_root}",
            f"paths.processed_dir={tmp_path / 'processed'}",
            f"paths.runs_dir={tmp_path / 'runs'}",
            "evaluation.bootstrap=20",
        ],
    )
    prepare_dataset(cfg, manifest=raw_manifest)
    return cfg


def test_leaderboard_ranks_by_composite_f1(config: Config, tmp_path: Path) -> None:
    for name in ("random", "limit"):
        evaluate_detector(config, name)
    out = write_leaderboard(config, ["random", "limit"], tmp_path / "BASELINES.md")
    text = out.read_text()
    assert text.startswith("# Baseline results")
    assert "## All (3 channels" in text
    assert "## excluding constant_train+nominal_out_of_range" in text
    assert "random scores reach" in text
    summaries = sorted(
        load_summaries(config, ["random", "limit"]),
        key=lambda s: -s["subsets"]["all"]["composite_f1"],
    )
    first_row = next(
        line
        for line in text.splitlines()
        if line.startswith("| random |") or line.startswith("| limit |")
    )
    assert first_row.startswith(f"| {summaries[0]['detector']} |")


def test_missing_evaluation_is_actionable(config: Config) -> None:
    with pytest.raises(FileNotFoundError, match="sathealth evaluate zscore"):
        load_summaries(config, ["zscore"])


def test_render_requires_summaries() -> None:
    with pytest.raises(ValueError, match="no summaries"):
        render_leaderboard([])
