"""Content hashing used for dataset integrity checks and run fingerprints."""

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: Path) -> str:
    """Streamed SHA-256 of a file; constant memory regardless of file size."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """The SHA-1 git assigns to a file's contents (matches GitHub's API ``sha`` field)."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def fingerprint(obj: Any, length: int = 12) -> str:
    """Stable short hash of any JSON-serialisable object (key order independent)."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:length]
