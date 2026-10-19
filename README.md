<p align="center">
  <img src="docs/assets/banner.svg" alt="sathealth: a satellite streams telemetry; an explainable AI flags the stretch that drifts from nominal behaviour" width="100%">
</p>

# sathealth: explainable satellite health monitoring

AI-based satellite health monitoring with early anomaly detection and explainable alerts.
Built for the **Odisha Seva First Innovation Challenge**.

Satellites downlink thousands of telemetry channels. Fixed limit checks miss slow drifts and
broken relationships between channels, and they never say *why*. sathealth combines an
ensemble of detectors with an explainability layer, so each alert says which channel, when
the problem started, which dependency broke and what normal would have looked like.

> **Status: Phase 3 (deep detectors) complete.** See [docs/ROADMAP.md](docs/ROADMAP.md) and
> the results in [docs/BASELINES.md](docs/BASELINES.md).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
make install     # uv sync + git hooks
make download    # NASA SMAP/MSL, 165 files / 173 MB, every file SHA-256 verified
make prepare     # validate → split → scale → data/processed/telemanom-<fingerprint>/
make evaluate-baselines  # classical baselines (~20 s)
make leaderboard # every detector incl. deep models → docs/BASELINES.md (~1 h on 4 cores)
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

Deep models train one network per channel on CPU. `sathealth evaluate lstm --jobs 7` spreads
channels over 7 processes; results are bit-identical for any `--jobs` value. To also log runs
to MLflow: `uv sync --extra tracking`, then add `--set tracking.mlflow_uri=file:./mlruns`.

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

## Results (Phases 2 and 3)

All 80 channels, thresholds fit without labels ([protocol](docs/adr/0004-evaluation-protocol.md)):

| Detector | Composite F1 [95% CI] | AUC-PR | Events found (of 103) | Point-adjusted F1 ⚠ |
|---|---|---:|---:|---:|
| **Transformer autoencoder** | **0.524** [0.33, 0.68] | 0.425 | 75 | 0.620 |
| Limit check (today's practice) | 0.458 [0.30, 0.58] | 0.268 | 47 | 0.509 |
| **LSTM forecaster** | 0.444 [0.32, 0.57] | **0.441** | **91** | 0.558 |
| Robust z-score | 0.395 [0.26, 0.52] | 0.312 | 60 | 0.506 |
| Isolation Forest | 0.372 [0.22, 0.50] | 0.323 | 65 | 0.589 |
| Random scores | 0.210 [0.11, 0.29] | 0.124 | 36 | **0.897** |

What Phase 3 showed ([ADR 0005](docs/adr/0005-deep-detectors.md)):

- **The deep models rank anomalies significantly better.** In a paired bootstrap against the
  best baseline, AUC-PR rises by +0.118 [+0.058, +0.181] (LSTM) and +0.102 [+0.039, +0.168]
  (Transformer). The LSTM finds **91 of 103** anomalies; the limit check finds 47.
- **At a fixed threshold the gain is not yet significant.** The Transformer leads on composite
  F1 (+0.112 on SMAP), but the paired interval [−0.002, +0.246] touches zero. A single static
  threshold turns good scores into runs of false alarms. Fixing that is Phase 4's job.
- **Random scores beat every real detector on point-adjusted F1**, which is why we don't
  headline it.

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
scripts/           maintainer tools (manifest builder, README banner)
src/sathealth/
  config.py        typed configuration
  cli.py           `sathealth` command
  data/            download, manifests, Telemanom loader, schema
  features/        validation, scaling, windowing
  evaluation/      metrics, label-free thresholds, aggregation + bootstrap CIs
  detectors/       limit check, robust z-score, Isolation Forest, random, LSTM, Transformer
  models/          PyTorch networks, training loop, device and seeding, error smoothing
  pipeline/        prepare, evaluate, data card, leaderboard, MLflow tracking
  utils/           hashing, seeding, provenance
tests/             unit tests + integration/ (real data, `-m data`)
```

Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Data citation

> K. Hundman, V. Constantinou, C. Laporte, I. Colwell, T. Soderstrom. *Detecting Spacecraft
> Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.* KDD 2018.
> [arXiv:1802.04431](https://arxiv.org/abs/1802.04431)
