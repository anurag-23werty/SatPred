from pathlib import Path

import pytest

from sathealth.config import Config, load_config
from sathealth.data.download import Manifest
from sathealth.pipeline.evaluate import evaluate_detector
from sathealth.pipeline.leaderboard import (
    compare_with_baselines,
    load_summaries,
    render_leaderboard,
    write_leaderboard,
)
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
    assert text.startswith("# Detector results")
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


def test_deep_detectors_are_compared_with_the_best_baseline(config: Config, tmp_path: Path) -> None:
    tiny = load_config(
        None,
        overrides=[
            f"paths.raw_dir={config.paths.raw_dir}",
            f"paths.processed_dir={config.paths.processed_dir}",
            f"paths.runs_dir={config.paths.runs_dir}",
            "evaluation.bootstrap=20",
            "deep.training.epochs=1",
            "deep.lstm.window=16",
            "deep.lstm.hidden=4",
        ],
    )
    names = ["limit", "zscore", "random", "lstm"]
    for name in names:
        evaluate_detector(tiny, name)
    summaries = load_summaries(tiny, names)
    rows = compare_with_baselines(tiny, summaries)
    assert [(r["detector"], r["subset"], r["metric"]) for r in rows] == [
        ("lstm", subset, metric)
        for subset in ("all", "SMAP", "MSL")
        for metric in ("composite_f1", "average_precision")
    ]
    by_name = {s["detector"]: s for s in summaries}
    key = {"composite_f1": "composite_f1", "average_precision": "average_precision_macro"}
    for row in rows:
        assert row["baseline"] in {"limit", "zscore"}, "random is a floor, never the rival"
        best = max(
            by_name[b]["subsets"][row["subset"]][key[row["metric"]]] for b in ("limit", "zscore")
        )
        assert row["baseline_value"] == best
        assert row["delta"] == pytest.approx(row["value"] - best)
        if row["metric"] == "average_precision":
            assert row["marginal_cis_separate"] is None
    text = render_leaderboard(summaries, rows)
    assert "## Versus the best baseline" in text
    assert "| lstm | SMAP | Composite F1 |" in text
    assert "| lstm | MSL | AUC-PR |" in text
    assert compare_with_baselines(tiny, [by_name["lstm"]]) == []
