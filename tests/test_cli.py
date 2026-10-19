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
