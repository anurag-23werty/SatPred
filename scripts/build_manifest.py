"""Regenerate ``src/sathealth/data/manifests/telemanom.json`` for a pinned mirror commit.

Maintainer tool, not part of the runtime. It lists the files at an exact commit through the
GitHub API, checks each local copy against the git blob SHA GitHub reports (downloading any
file that is missing or does not match), and records size + SHA-256 for every file::

    uv run python scripts/build_manifest.py --commit <sha> --local-dir data/raw/telemanom
"""

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

from sathealth.utils.hashing import git_blob_sha1, sha256_file

REPO = "imperial-qore/TranAD"
SUBDIR = "data/SMAP_MSL"


def _api(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True, help="full commit SHA of the mirror repo")
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("src/sathealth/data/manifests/telemanom.json")
    )
    args = parser.parse_args()

    raw_base = f"https://raw.githubusercontent.com/{REPO}/{args.commit}/{SUBDIR}"
    listing: list[dict[str, Any]] = []
    for sub in ("", "/train", "/test"):
        url = f"https://api.github.com/repos/{REPO}/contents/{SUBDIR}{sub}?ref={args.commit}"
        prefix = sub.lstrip("/")
        for item in _api(url):
            if item["type"] == "file" and (
                item["name"].endswith(".npy") or item["name"] == "labeled_anomalies.csv"
            ):
                rel = f"{prefix}/{item['name']}" if prefix else item["name"]
                listing.append({"path": rel, "git_sha": item["sha"], "size": item["size"]})

    files = []
    for entry in sorted(listing, key=lambda e: e["path"]):
        local = args.local_dir / entry["path"]
        if not local.is_file() or git_blob_sha1(local) != entry["git_sha"]:
            local.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{raw_base}/{entry['path']}", local)
            if git_blob_sha1(local) != entry["git_sha"]:
                raise SystemExit(f"{entry['path']}: download does not match git blob sha")
        files.append(
            {"path": entry["path"], "size": local.stat().st_size, "sha256": sha256_file(local)}
        )

    manifest = {
        "name": "telemanom",
        "description": "NASA SMAP and MSL spacecraft telemetry with expert-labelled anomalies "
        "(Hundman et al., KDD 2018).",
        "source": {
            "paper": "https://arxiv.org/abs/1802.04431",
            "original_url": "https://s3-us-west-2.amazonaws.com/telemanom/data.zip",
            "original_status": "HTTP 403 as of 2026-10; see docs/adr/0001-dataset-source.md",
            "labels_upstream": "https://github.com/khundman/telemanom/blob/master/labeled_anomalies.csv",
            "mirror_repo": f"https://github.com/{REPO}",
            "mirror_commit": args.commit,
        },
        "base_url": raw_base,
        "files": files,
    }
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out} with {len(files)} files")


if __name__ == "__main__":
    main()
