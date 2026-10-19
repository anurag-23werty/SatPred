"""Manifest-driven, checksum-verified dataset download.

Design choices (learned the hard way on this dataset: plain parallel ``curl`` silently
truncated 12 of 164 files):

* Every file has a pinned size and SHA-256 in a committed manifest; nothing is trusted
  until it verifies.
* Writes go to ``<file>.part`` and are atomically renamed only after verification, so an
  interrupted run never leaves a half-written file that looks valid.
* Each file is retried with exponential backoff and jitter; already-valid files are skipped,
  which makes the command idempotent and resumable.
* Manifest paths are sandboxed to the target directory (no ``..`` or absolute paths).
"""

import json
import random
import shutil
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any

from sathealth.logging import get_logger
from sathealth.utils.hashing import sha256_file

log = get_logger(__name__)

Fetcher = Callable[[str, Path, float], None]


class ChecksumError(RuntimeError):
    """Downloaded bytes do not match the manifest."""


@dataclass(frozen=True)
class FileSpec:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        pure = PurePosixPath(self.path)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise ValueError(f"unsafe manifest path: {self.path!r}")


@dataclass(frozen=True)
class Manifest:
    name: str
    base_url: str
    files: tuple[FileSpec, ...]
    source: dict[str, Any] = field(default_factory=dict)

    def url_for(self, spec: FileSpec) -> str:
        return f"{self.base_url.rstrip('/')}/{spec.path}"

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)


def load_manifest(ref: str | Path) -> Manifest:
    """Load a manifest by bundled name (``telemanom.json``) or filesystem path."""
    path = Path(ref)
    if path.exists():
        raw = json.loads(path.read_text())
    else:
        bundled = resources.files("sathealth.data.manifests").joinpath(str(ref))
        raw = json.loads(bundled.read_text())
    files = tuple(FileSpec(f["path"], int(f["size"]), f["sha256"]) for f in raw["files"])
    if len({f.path for f in files}) != len(files):
        raise ValueError("manifest lists the same path twice")
    return Manifest(
        name=raw["name"], base_url=raw["base_url"], files=files, source=raw.get("source", {})
    )


def urllib_fetch(url: str, dest: Path, timeout_s: float) -> None:
    """Stream ``url`` to ``dest`` using only the standard library."""
    request = urllib.request.Request(url, headers={"User-Agent": "sathealth-downloader/1"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response, dest.open("wb") as fh:
        shutil.copyfileobj(response, fh, length=1 << 20)


class FileStatus(StrEnum):
    OK = "ok"  # present and verified, nothing to do
    DOWNLOADED = "downloaded"  # fetched and verified in this run
    MISSING = "missing"  # verify-only: file absent
    CORRUPT = "corrupt"  # verify-only: size or checksum mismatch
    FAILED = "failed"  # download exhausted its retries


@dataclass
class DownloadReport:
    statuses: dict[str, FileStatus] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(s in (FileStatus.OK, FileStatus.DOWNLOADED) for s in self.statuses.values())

    def count(self, status: FileStatus) -> int:
        return sum(1 for s in self.statuses.values() if s is status)

    def summary(self) -> dict[str, int]:
        return {s.value: self.count(s) for s in FileStatus if self.count(s)}


def verify_file(root: Path, spec: FileSpec) -> FileStatus:
    """Cheap size check first, then the full SHA-256."""
    path = root / spec.path
    if not path.is_file():
        return FileStatus.MISSING
    if path.stat().st_size != spec.size or sha256_file(path) != spec.sha256:
        return FileStatus.CORRUPT
    return FileStatus.OK


def verify_dataset(root: Path, manifest: Manifest, max_workers: int = 8) -> DownloadReport:
    report = DownloadReport()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for spec, status in zip(
            manifest.files, pool.map(lambda s: verify_file(root, s), manifest.files), strict=True
        ):
            report.statuses[spec.path] = status
    return report


def _download_one(
    root: Path,
    manifest: Manifest,
    spec: FileSpec,
    *,
    retries: int,
    timeout_s: float,
    backoff_s: float,
    fetch: Fetcher,
    sleep: Callable[[float], None],
) -> FileStatus:
    target = root / spec.path
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    url = manifest.url_for(spec)
    for attempt in range(retries + 1):
        try:
            fetch(url, tmp, timeout_s)
            size = tmp.stat().st_size
            if size != spec.size:
                raise ChecksumError(f"size {size} != expected {spec.size}")
            digest = sha256_file(tmp)
            if digest != spec.sha256:
                raise ChecksumError(f"sha256 {digest[:12]}… != expected {spec.sha256[:12]}…")
            tmp.replace(target)
            return FileStatus.DOWNLOADED
        except (OSError, ChecksumError) as exc:
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise
            delay = backoff_s * 2**attempt + random.uniform(0, backoff_s)
            log.warning(
                "retrying download",
                path=spec.path,
                attempt=attempt + 1,
                error=str(exc),
                delay_s=round(delay, 2),
            )
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def download_dataset(
    root: Path,
    manifest: Manifest,
    *,
    max_workers: int = 8,
    retries: int = 4,
    timeout_s: float = 60.0,
    backoff_s: float = 1.0,
    fetch: Fetcher = urllib_fetch,
    sleep: Callable[[float], None] = time.sleep,
) -> DownloadReport:
    """Bring ``root`` in line with ``manifest``; returns per-file outcomes (never raises)."""
    root.mkdir(parents=True, exist_ok=True)
    report = DownloadReport()

    def work(spec: FileSpec) -> tuple[FileSpec, FileStatus, str | None]:
        if verify_file(root, spec) is FileStatus.OK:
            return spec, FileStatus.OK, None
        try:
            return (
                spec,
                _download_one(
                    root,
                    manifest,
                    spec,
                    retries=retries,
                    timeout_s=timeout_s,
                    backoff_s=backoff_s,
                    fetch=fetch,
                    sleep=sleep,
                ),
                None,
            )
        except (OSError, ChecksumError) as exc:
            return spec, FileStatus.FAILED, str(exc)

    log.info(
        "syncing dataset",
        name=manifest.name,
        files=len(manifest.files),
        mb=round(manifest.total_bytes / 1e6, 1),
        root=root,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for spec, status, error in pool.map(work, manifest.files):
            report.statuses[spec.path] = status
            if error is not None:
                report.errors[spec.path] = error
                log.error("download failed", path=spec.path, error=error)
    log.info("dataset sync finished", **report.summary())
    return report
