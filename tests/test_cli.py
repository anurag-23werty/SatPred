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
    assert _detectors(args.detectors) == [
        "limit",
        "zscore",
        "iforest",
        "random",
        "lstm",
        "transformer",
    ]
    assert _detectors(build_parser().parse_args(["evaluate", "limit"]).detectors) == ["limit"]


def test_detector_groups_expand_without_repeats() -> None:
    from sathealth.cli import _detectors, build_parser

    args = build_parser().parse_args(["evaluate", "deep", "lstm", "limit", "--jobs", "3"])
    assert _detectors(args.detectors) == ["lstm", "transformer", "limit"]
    assert args.jobs == 3
    assert _detectors(["baselines"]) == ["limit", "zscore", "iforest", "random"]


def test_decide_ablation_and_leaderboard_with_decision(
    raw_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = [
        *_args(raw_root, tmp_path),
        "--set",
        f"paths.runs_dir={tmp_path / 'runs'}",
        "--set",
        f"paths.decisions_dir={tmp_path / 'decisions'}",
        "--set",
        "evaluation.bootstrap=10",
        "--set",
        "decision.inputs=[zscore]",
    ]
    assert main(["prepare", "--skip-verify", *args]) == 0
    assert main(["decide", *args]) == 1, "no saved scores yet"
    assert main(["evaluate", "baselines", *args]) == 0
    assert main(["decide", *args]) == 0
    assert "within_budget=" in capsys.readouterr().out
    board = tmp_path / "board.md"
    assert main(["leaderboard", "baselines", "--with-decision", "--out", str(board), *args]) == 0
    assert "| fused |" in board.read_text()
    assert main(["ablation", "--out", str(tmp_path / "d.md"), *args]) == 1, "needs deep runs"


def test_explain_commands(raw_root: Path, tmp_path: Path) -> None:
    args = [*_args(raw_root, tmp_path)]
    for key, value in (
        ("paths.runs_dir", tmp_path / "runs"),
        ("paths.decisions_dir", tmp_path / "decisions"),
        ("paths.models_dir", tmp_path / "models"),
        ("paths.explanations_dir", tmp_path / "explanations"),
        ("deep.training.epochs", 2),
        ("deep.lstm.window", 16),
        ("deep.lstm.hidden", 8),
        ("decision.inputs", "[lstm]"),
        ("explain.horizon", 20),
        ("explain.band_starts", 5),
        ("explain.ig_steps", 8),
        ("explain.eval_injections", 2),
        ("explain.eval_coverage_horizon", 10),
        ("explain.eval_channels_per_spacecraft", 1),
        ("evaluation.bootstrap", 10),
    ):
        args += ["--set", f"{key}={value}"]
    assert main(["prepare", "--skip-verify", *args]) == 0
    assert main(["explain", *args]) == 1, "no decision run yet"
    assert main(["explain-report", "--out", str(tmp_path / "e.md"), *args]) == 1
    assert main(["evaluate", "lstm", *args]) == 0
    assert main(["decide", *args]) == 0
    assert main(["explain", "--channel", "A-1", *args]) == 0
    assert main(["explain-report", "--out", str(tmp_path / "e.md"), *args]) == 0
    assert "## Hypotheses" in (tmp_path / "e.md").read_text()


def test_demo_and_serve_commands(
    raw_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = [
        *_args(raw_root, tmp_path),
        "--set",
        f"paths.runs_dir={tmp_path / 'runs'}",
        "--set",
        "decision.inputs=[zscore]",
        "--set",
        f"service.db_path={tmp_path / 's.db'}",
    ]
    assert main(["prepare", "--skip-verify", *args]) == 0
    assert main(["demo", "--db", str(tmp_path / "d.db"), *args]) == 1, "no saved scores yet"
    assert main(["evaluate", "zscore", *args]) == 0
    assert main(["demo", "--db", str(tmp_path / "d.db"), *args]) == 0
    out = capsys.readouterr().out
    assert "4. Verdict" in out
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.append(kw))
    assert main(["serve", "--port", "8123", *args]) == 0
    assert calls == [{"host": "127.0.0.1", "port": 8123, "log_level": "warning"}]


def test_confirm_command(opssat_csv: Path, tmp_path: Path) -> None:
    args = ["--set", f"confirmation.raw_dir={opssat_csv.parent}"]
    for key, value in (
        ("confirmation.out_dir", tmp_path / "out"),
        ("deep.training.epochs", 1),
        ("deep.lstm.window", 16),
        ("deep.lstm.hidden", 8),
        ("deep.transformer.window", 16),
        ("deep.transformer.d_model", 8),
        ("deep.transformer.heads", 2),
        ("deep.transformer.layers", 1),
        ("evaluation.bootstrap", 20),
    ):
        args += ["--set", f"{key}={value}"]
    out = tmp_path / "C.md"
    assert main(["confirm", "--no-download", "--out", str(out), *args]) == 0
    assert out.read_text().startswith("# Independent confirmation")
    missing = ["--set", f"confirmation.raw_dir={tmp_path / 'none'}"]
    assert main(["confirm", "--no-download", "--out", str(out), *missing]) == 1
