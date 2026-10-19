import hashlib
import logging
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sathealth.logging import KeyValueFormatter, configure_logging, get_logger
from sathealth.utils.hashing import fingerprint, git_blob_sha1, sha256_file
from sathealth.utils.seed import seed_everything


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    payload = b"x" * (3 << 20) + b"tail"  # spans several read chunks
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_git_blob_sha1_matches_git(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_bytes(b"hello telemetry\n")
    try:
        expected = subprocess.run(
            ["git", "hash-object", str(path)], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    assert git_blob_sha1(path) == expected


def test_fingerprint_is_order_independent_and_sensitive() -> None:
    assert fingerprint({"a": 1, "b": [1, 2]}) == fingerprint({"b": [1, 2], "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})
    assert len(fingerprint({"a": 1}, length=8)) == 8


def test_seed_everything_is_reproducible() -> None:
    first = seed_everything(7).normal(size=5)
    second = seed_everything(7).normal(size=5)
    np.testing.assert_array_equal(first, second)


def test_key_value_formatter_renders_fields() -> None:
    record = logging.LogRecord("sathealth.x", logging.INFO, __file__, 1, "done", None, None)
    record.sathealth_fields = {"path": "a b", "n": 3, "ratio": 0.123456789}
    line = KeyValueFormatter().format(record)
    assert line.endswith('| done path="a b" n=3 ratio=0.123457')


def test_struct_logger_emits_fields(caplog: pytest.LogCaptureFixture) -> None:
    configure_logging("DEBUG")
    logger = logging.getLogger("sathealth")
    logger.propagate = True  # let caplog see records
    try:
        with caplog.at_level(logging.INFO, logger="sathealth"):
            get_logger("sathealth.test").info("hello", channel="A-1")
        assert caplog.records[-1].sathealth_fields == {"channel": "A-1"}
    finally:
        logger.propagate = False


def test_log_handler_follows_swapped_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    configure_logging("INFO")
    replacement = io.StringIO()
    monkeypatch.setattr("sys.stderr", replacement)
    get_logger("sathealth.test").info("after swap", ok=True)
    assert "after swap ok=True" in replacement.getvalue()
