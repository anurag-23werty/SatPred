from pathlib import Path

import pytest

from sathealth.utils.provenance import git_state, provenance


def test_provenance_fields() -> None:
    stamp = provenance()
    assert set(stamp) == {"created_at", "sathealth_version", "git"}
    assert set(stamp["git"]) == {"commit", "dirty"}


def test_git_state_fails_soft_outside_a_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_state() == {"commit": None, "dirty": None}
