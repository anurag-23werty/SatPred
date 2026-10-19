# ADR 0005: Deep detectors (LSTM forecaster, Transformer autoencoder)

- **Status:** accepted
- **Date:** 2026-10-19

## Context

Phase 2 left four baselines whose composite-F1 confidence intervals overlap (0.37–0.46). Two
of their weaknesses are structural: a limit check or z-score looks at one step at a time, and
the Isolation Forest saturates outside its training range. Phase 3 adds learned models of
*normal dynamics* and must answer one question honestly: do they rank anomalies better, and
does that survive a label-free threshold?

## Decision

| Aspect | Choice | Why |
|---|---|---|
| Models | **LSTM next-step forecaster** (Telemanom, Hundman et al. 2018) and a **bottlenecked Transformer autoencoder** | Two complementary questions: "is this reading what the recent past predicts?" and "does this window look nominal?" |
| Inputs | All channel features: the scaled value *and* the one-hot commands | Commands explain many nominal level changes |
| Transformer bottleneck | Mean-pooled 16-d code; the decoder sees only the code and positions | Without it, attention learns the identity and reconstructs anomalies perfectly |
| Not TranAD | We use TranAD's encoder idea but not its adversarial two-phase training | Simpler, and the bottleneck alone already prevents identity learning |
| Score | Absolute error of the value column(s), causal **EWMA** (span 30) | Raw errors are spiky. The EWMA uses only past steps, so the score is available live |
| Causality | The score at step *t* uses only steps ≤ *t*. Steps before the first full window use the prefix that exists (pinned by tests) | It must work on a live stream, and a whole test split is never scored at once in operations |
| Training | One model per channel; Adam; gradient clipping; early stopping on the **validation tail** (patience 5); best weights restored | Validation is nominal and already used for thresholds, so labels are never touched |
| Hyperparameters | Set **a priori** (window 64, LSTM 2×80, Transformer d=32, 2+2 layers) and never tuned on test labels | Tuning on test labels would invalidate every number in the leaderboard |
| Reproducibility | `seed_torch`: seeded init and shuffling, deterministic kernels, **1 CPU thread per channel**; parallelism comes from `--jobs` worker processes over channels | Results are bit-identical for any `--jobs` value (tested), and thread count cannot change floating-point reduction order |
| Tracking | Run directory is the source of truth (`summary.json`, `per_channel.json`, `training.json` curves, scores). Optional **MLflow** logging (`tracking.mlflow_uri`, extra `tracking`) | No server is needed to reproduce a result; MLflow is for comparing many runs |
| Comparison | **Paired** bootstrap over channels vs the best baseline, on composite F1 and AUC-PR | Detectors are scored on the same channels; unpaired intervals hide real differences |
| Dependency | PyTorch, CPU wheels from the PyTorch index on Linux | The default Linux wheel pulls ~2 GB of CUDA libraries into CI |

## Results (all 80 channels)

| | LSTM | Transformer | Best baseline |
|---|---:|---:|---:|
| Composite F1 | 0.444 | **0.524** | 0.458 (limit) |
| AUC-PR (threshold-free) | **0.441** | 0.425 | 0.323 (Isolation Forest) |
| Events detected (of 103) | **91** | 75 | 65 (Isolation Forest) |
| Oracle event F1 † | 0.899 | **0.905** | 0.757 (z-score) |

Paired 95% CIs against the best baseline:

- **AUC-PR is better with significance** for both models on all channels (LSTM +0.118
  [+0.058, +0.181]; Transformer +0.102 [+0.039, +0.168]) and on SMAP. On MSL (27 channels)
  the difference is positive but not significant.
- **Composite F1 is not significantly better.** The Transformer leads by +0.065 overall and
  +0.112 on SMAP, but the SMAP interval [−0.002, +0.246] just touches zero.

## Consequences

- The Phase 3 exit criterion as written in the roadmap (composite F1 with non-overlapping CIs
  on one spacecraft) is **not met**. We report this rather than tune on test labels. What the
  deep models clearly improve is *score quality*: they rank anomalies better and find 91 of
  103 events.
- The bottleneck is the **operating point**, not the detector. A single static 0.999 quantile
  per channel produces runs of alarm steps (LSTM: 2.9 false events per 1k steps), which costs
  point precision. That is exactly Phase 4's scope: dynamic thresholds (Telemanom NDT, EVT),
  event merging and fusion of the two complementary detectors.
- Deep runs cost 12 minutes (LSTM) and 30 minutes (Transformer) on 7 Apple M2 cores. CI
  therefore pins the findings on two channels (`tests/integration/test_real_deep.py`) instead
  of re-running the whole evaluation.
