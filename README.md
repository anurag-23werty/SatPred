# sathealth: explainable satellite health monitoring

AI-based satellite health monitoring with early anomaly detection and explainable alerts.
Built for the **Odisha Seva First Innovation Challenge**.

Satellites downlink thousands of telemetry channels. Fixed limit checks miss slow drifts and
broken relationships between channels, and they never say *why*. sathealth combines an
ensemble of detectors with an explainability layer, so each alert says which channel, when
the problem started, which dependency broke and what normal would have looked like.

> **Status: Phase 1 (data foundation) complete.** See [docs/ROADMAP.md](docs/ROADMAP.md).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
make install     # uv sync + git hooks
make download    # NASA SMAP/MSL, 165 files / 173 MB, every file SHA-256 verified
make prepare     # validate → split → scale → data/processed/telemanom-<fingerprint>/
make check       # ruff + mypy --strict + unit tests
make test-data   # integration tests against the real dataset
```

Use prepared data from Python:

```python
from pathlib import Path
from sathealth.pipeline.prepare import load_prepared

channel = load_prepared(Path("data/processed/telemanom-84ff983f088d"), "P-1")
X = channel.windows("train", size=100)  # (N, 100, 25) zero-copy view
y = channel.test_window_labels(size=100)  # (N_test,) bool
```

Every command accepts `--config` and `--set key.sub=value`, for example
`sathealth prepare --set dataset.spacecraft=[SMAP] --set preprocess.scaler=standard`.
`sathealth info` prints the resolved config.

## What Phase 1 guarantees

| Guarantee | How |
|---|---|
| **Integrity**: no corrupt data reaches a model | Pinned mirror commit; size and SHA-256 manifest; atomic `.part` writes; retries; `prepare` re-verifies first ([ADR 0001](docs/adr/0001-dataset-source.md)) |
| **Correct labels** | Explicit duplicate policy (`P-2`), sorting, bounds checks ([ADR 0002](docs/adr/0002-label-cleaning.md)) |
| **No leakage** | Chronological split, then the scaler is fit on train rows only, then windows are built per split ([ADR 0003](docs/adr/0003-train-only-robust-scaling.md)) |
| **Known data quality** | Per-channel ERROR/WARNING/INFO checks, published in [docs/DATA_CARD.md](docs/DATA_CARD.md) |
| **Reproducibility** | Output directory keyed by a config fingerprint; `manifest.json` records config, git commit, raw digest and scaler parameters |
| **Engineering quality** | `mypy --strict`, ruff, 95% branch coverage, CI on Python 3.12 and 3.13, plus a real-data CI job |

## What we found in the data

- The original NASA archive is offline (HTTP 403). A naive parallel download **silently
  truncated 12 of 164 files**.
- `P-2` has two conflicting label rows and is dropped, leaving **80 channels (53 SMAP, 27 MSL)
  and 103 anomaly events**.
- **16 channels are constant in train**. For example `M-6` is −1 throughout train and reaches
  258 in test.
- **15 channels show distribution shift**: unlabelled test points outside the train range.

Full details are in [docs/DATA_CARD.md](docs/DATA_CARD.md).

## Repository layout

```
configs/           YAML configs (default.yaml)
docs/              architecture, roadmap, data card, ADRs
scripts/           maintainer tools (manifest builder)
src/sathealth/
  config.py        typed configuration
  cli.py           `sathealth` command
  data/            download, manifests, Telemanom loader, schema
  features/        validation, scaling, windowing
  pipeline/        prepare run, data card
  utils/           hashing, seeding
tests/             unit tests + integration/ (real data, `-m data`)
```

Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Data citation

> K. Hundman, V. Constantinou, C. Laporte, I. Colwell, T. Soderstrom. *Detecting Spacecraft
> Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.* KDD 2018.
> [arXiv:1802.04431](https://arxiv.org/abs/1802.04431)
