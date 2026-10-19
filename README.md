# sathealth: explainable satellite health monitoring

AI-based satellite health monitoring with early anomaly detection and explainable alerts.
Built for the **Odisha Seva First Innovation Challenge**.

Satellites downlink thousands of telemetry channels. Fixed limit checks miss slow drifts and
broken relationships between channels, and they never say *why*. sathealth combines an
ensemble of detectors with an explainability layer, so each alert says which channel, when
the problem started, which dependency broke and what normal would have looked like.

> **Status: Phase 2 (evaluation harness + baselines) complete.** See
> [docs/ROADMAP.md](docs/ROADMAP.md) and the results in [docs/BASELINES.md](docs/BASELINES.md).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
make install     # uv sync + git hooks
make download    # NASA SMAP/MSL, 165 files / 173 MB, every file SHA-256 verified
make prepare     # validate → split → scale → data/processed/telemanom-<fingerprint>/
make leaderboard # evaluate all baselines → docs/BASELINES.md (~20 s)
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
| **Honest evaluation** | Thresholds never see labels; composite F1 headline; bootstrap CIs; random-score control ([ADR 0004](docs/adr/0004-evaluation-protocol.md)) |
| **Engineering quality** | `mypy --strict`, ruff, >90% branch coverage, CI on Python 3.12 and 3.13, plus a real-data CI job |

## Baseline results (Phase 2)

All 80 channels, thresholds fit without labels ([protocol](docs/adr/0004-evaluation-protocol.md)):

| Detector | Composite F1 [95% CI] | Event F1 | Point-adjusted F1 ⚠ |
|---|---|---:|---:|
| Limit check (today's practice) | 0.458 [0.30, 0.58] | 0.141 | 0.509 |
| Robust z-score | 0.395 [0.26, 0.52] | 0.202 | 0.506 |
| Isolation Forest | 0.372 [0.22, 0.50] | 0.054 | 0.589 |
| Random scores | 0.210 [0.11, 0.29] | 0.060 | **0.897** |

Random scores beat every real detector on point-adjusted F1, which is why we don't
headline it. The real baselines' confidence intervals overlap: this is the band Phase 3 has
to beat. Details are in [docs/BASELINES.md](docs/BASELINES.md).

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
  evaluation/      metrics, label-free thresholds, aggregation + bootstrap CIs
  detectors/       limit check, robust z-score, Isolation Forest, random
  pipeline/        prepare, evaluate, data card, leaderboard
  utils/           hashing, seeding
tests/             unit tests + integration/ (real data, `-m data`)
```

Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Data citation

> K. Hundman, V. Constantinou, C. Laporte, I. Colwell, T. Soderstrom. *Detecting Spacecraft
> Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.* KDD 2018.
> [arXiv:1802.04431](https://arxiv.org/abs/1802.04431)
