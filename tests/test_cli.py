import json
from pathlib import Path

import pytest

from sathealth.cli import main


def _args(raw_root: Path, tmp_path: Path) -> list[str]:
    return [
        "--set",
        f"paths.raw_dir={raw_root}",
        "--set",
        f"paths.processed_dir={tmp_path / 'processed'}",
    ]


def test_info_prints_resolved_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", "--set", "windows.size=64"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["config"]["windows"]["size"] == 64
    assert out["run_dir"].startswith("data/processed/telemanom-")


def test_bad_override_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", "--set", "windows.size=0"]) == 2
    assert "invalid configuration" in capsys.readouterr().err


def test_prepare_and_report(
    raw_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _args(raw_root, tmp_path)
    assert main(["prepare", "--skip-verify", *args]) == 0
    run_dir = Path(capsys.readouterr().out.strip())
    assert (run_dir / "manifest.json").is_file()

    card = tmp_path / "card.md"
    assert main(["report", "--out", str(card), *args]) == 0
    assert card.read_text().startswith("# Data card")


def test_prepare_missing_data_exits_1(tmp_path: Path) -> None:
    assert main(["prepare", "--skip-verify", *_args(tmp_path / "nope", tmp_path)]) == 1


def test_verify_fails_on_missing_files(tmp_path: Path) -> None:
    assert main(["verify", "--set", f"paths.raw_dir={tmp_path}"]) == 1


def test_evaluate_and_leaderboard(
    raw_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = [
        *_args(raw_root, tmp_path),
        "--set",
        f"paths.runs_dir={tmp_path / 'runs'}",
        "--set",
        "evaluation.bootstrap=10",
    ]
    assert main(["prepare", "--skip-verify", *args]) == 0
    assert main(["evaluate", "limit", "random", *args]) == 0
    out = capsys.readouterr().out
    assert "limit" in out
    assert "composite_f1=" in out
    board = tmp_path / "BASELINES.md"
    assert main(["leaderboard", "limit", "random", "--out", str(board), *args]) == 0
    assert "| limit |" in board.read_text()
    assert main(["leaderboard", "zscore", "--out", str(board), *args]) == 1


def test_evaluate_without_prepared_data_exits_1(tmp_path: Path) -> None:
    overrides = [
        "--set",
        f"paths.processed_dir={tmp_path}",
        "--set",
        f"paths.runs_dir={tmp_path / 'runs'}",
    ]
    assert main(["evaluate", "limit", *overrides]) == 1


def test_detector_selection_defaults_to_all() -> None:
    from sathealth.cli import _detectors, build_parser

    args = build_parser().parse_args(["evaluate"])
    assert _detectors(args.detectors) == ["limit", "zscore", "iforest", "random"]
    assert _detectors(build_parser().parse_args(["evaluate", "limit"]).detectors) == ["limit"]
