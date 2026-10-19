import hashlib
import json
from pathlib import Path

import pytest

from sathealth.data.download import (
    FileSpec,
    FileStatus,
    Manifest,
    download_dataset,
    load_manifest,
    verify_dataset,
)

FILES = {
    "labeled_anomalies.csv": b"chan_id\n",
    "train/A-1.npy": b"a" * 100,
    "test/A-1.npy": b"b" * 50,
}


def _manifest() -> Manifest:
    specs = tuple(FileSpec(p, len(b), hashlib.sha256(b).hexdigest()) for p, b in FILES.items())
    return Manifest(name="fake", base_url="https://mirror.test/data", files=specs)


class FakeServer:
    """Serves FILES; the first `failures[path]` requests for a path return truncated bytes."""

    def __init__(self, failures: dict[str, int] | None = None) -> None:
        self.failures = dict(failures or {})
        self.calls: list[str] = []

    def __call__(self, url: str, dest: Path, timeout_s: float) -> None:
        path = url.removeprefix("https://mirror.test/data/")
        self.calls.append(path)
        payload = FILES[path]
        if self.failures.get(path, 0) > 0:
            self.failures[path] -= 1
            payload = payload[: len(payload) // 2]  # simulate silent truncation
        dest.write_bytes(payload)


def _no_sleep(_: float) -> None:
    return None


def test_fresh_download_verifies_everything(tmp_path: Path) -> None:
    report = download_dataset(tmp_path, _manifest(), fetch=FakeServer(), sleep=_no_sleep)
    assert report.ok
    assert report.count(FileStatus.DOWNLOADED) == 3
    assert (tmp_path / "train" / "A-1.npy").read_bytes() == FILES["train/A-1.npy"]
    assert not list(tmp_path.rglob("*.part"))


def test_truncated_file_is_retried_then_succeeds(tmp_path: Path) -> None:
    server = FakeServer(failures={"train/A-1.npy": 2})
    report = download_dataset(tmp_path, _manifest(), retries=3, fetch=server, sleep=_no_sleep)
    assert report.ok
    assert server.calls.count("train/A-1.npy") == 3


def test_exhausted_retries_reported_not_raised(tmp_path: Path) -> None:
    server = FakeServer(failures={"test/A-1.npy": 10})
    report = download_dataset(tmp_path, _manifest(), retries=2, fetch=server, sleep=_no_sleep)
    assert not report.ok
    assert report.statuses["test/A-1.npy"] is FileStatus.FAILED
    assert "size" in report.errors["test/A-1.npy"]
    assert not (tmp_path / "test" / "A-1.npy").exists(), "a bad file must never be left in place"


def test_second_run_is_a_no_op(tmp_path: Path) -> None:
    download_dataset(tmp_path, _manifest(), fetch=FakeServer(), sleep=_no_sleep)
    server = FakeServer()
    report = download_dataset(tmp_path, _manifest(), fetch=server, sleep=_no_sleep)
    assert report.count(FileStatus.OK) == 3
    assert server.calls == []


def test_network_error_is_retried(tmp_path: Path) -> None:
    attempts = {"n": 0}
    good = FakeServer()

    def flaky(url: str, dest: Path, timeout_s: float) -> None:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("timed out")
        good(url, dest, timeout_s)

    report = download_dataset(tmp_path, _manifest(), max_workers=1, fetch=flaky, sleep=_no_sleep)
    assert report.ok


def test_verify_detects_missing_and_corrupt(tmp_path: Path) -> None:
    download_dataset(tmp_path, _manifest(), fetch=FakeServer(), sleep=_no_sleep)
    (tmp_path / "train" / "A-1.npy").write_bytes(b"a" * 99 + b"X")  # same size, wrong bytes
    (tmp_path / "test" / "A-1.npy").unlink()
    report = verify_dataset(tmp_path, _manifest())
    assert report.statuses["train/A-1.npy"] is FileStatus.CORRUPT
    assert report.statuses["test/A-1.npy"] is FileStatus.MISSING
    assert report.statuses["labeled_anomalies.csv"] is FileStatus.OK


@pytest.mark.parametrize("bad", ["../etc/passwd", "/abs/path", "a/../../b", ""])
def test_unsafe_manifest_paths_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        FileSpec(bad, 1, "0" * 64)


def test_bundled_manifest_is_complete() -> None:
    manifest = load_manifest("telemanom.json")
    paths = {f.path for f in manifest.files}
    train = {p.split("/")[1] for p in paths if p.startswith("train/")}
    test = {p.split("/")[1] for p in paths if p.startswith("test/")}
    assert "labeled_anomalies.csv" in paths
    assert len(train) == 82
    assert train == test, "every channel needs both a train and a test file"
    assert manifest.source["mirror_commit"] in manifest.base_url
    assert all(len(f.sha256) == 64 for f in manifest.files)


def test_load_manifest_from_path_rejects_duplicates(tmp_path: Path) -> None:
    entry = {"path": "a.npy", "size": 1, "sha256": "0" * 64}
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"name": "x", "base_url": "u", "files": [entry, entry]}))
    with pytest.raises(ValueError, match="twice"):
        load_manifest(path)
