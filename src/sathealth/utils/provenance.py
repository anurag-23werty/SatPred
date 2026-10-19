"""Provenance stamped into every run manifest: code version, git commit, timestamp."""

import subprocess
from datetime import UTC, datetime
from typing import Any

from sathealth import __version__


def git_state() -> dict[str, Any]:
    """Current commit and whether the working tree has uncommitted changes (fail-soft)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def provenance() -> dict[str, Any]:
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sathealth_version": __version__,
        "git": git_state(),
    }
