# ADR 0001: Source the NASA SMAP/MSL data from a pinned, checksummed mirror

- **Status:** accepted
- **Date:** 2026-10-19

## Context

The benchmark's canonical archive, `https://s3-us-west-2.amazonaws.com/telemanom/data.zip`,
returns **HTTP 403** as of October 2026. Copies on Kaggle need an account and API token. We
need a download that works unattended in CI and is bit-for-bit reproducible.

While evaluating mirrors we downloaded the 164 `.npy` files with parallel `curl`. **12 files
were silently truncated** (exit code 0, wrong size). One of them failed to load, and the
others would have loaded as wrong data if the truncation had landed on a row boundary.

## Decision

1. Use the copy in `imperial-qore/TranAD`, **pinned to commit `7ffb98d0c181`**, not a branch.
2. Before recording any checksum, verify every file against the git blob SHA that the GitHub
   API reports for that commit (`scripts/build_manifest.py`).
3. Commit a manifest (`src/sathealth/data/manifests/telemanom.json`) with path, size and
   SHA-256 for all 165 files. The downloader trusts nothing that fails verification. It writes
   to `.part` files and renames them atomically, retries with exponential backoff and jitter,
   and is idempotent.
4. `labeled_anomalies.csv` from the mirror is byte-identical to upstream
   `khundman/telemanom` master, which we checked at the time of this decision.

## Consequences

- `sathealth download` works with no credentials and is safe to re-run or interrupt.
- `sathealth prepare` re-verifies the checksums first (the integrity gate), so a corrupted
  local copy can never produce results silently.
- If the mirror disappears, any other copy can be dropped in with `--set paths.raw_dir=...`.
  It will pass verification only if it is byte-identical.
- Moving to another mirror means regenerating the manifest. The data fingerprint changes, so
  old prepared runs are not mixed with new ones.
