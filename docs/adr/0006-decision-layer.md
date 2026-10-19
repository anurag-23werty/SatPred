# ADR 0006: Decision layer (calibration, fusion, alarms, events)

- **Status:** accepted
- **Date:** 2026-10-19
- **Pre-registered:** the primary configuration and the alarm budget below were fixed
  *before* any Phase 4 result was computed. Ablations are all reported, not selected.

## Context

Phase 3 showed that the deep detectors rank anomalies significantly better than every
baseline (AUC-PR +0.10 to +0.12, paired 95% CI above zero), yet their composite F1 is not
significantly better. The cause is the operating point: one static 0.999 quantile per
channel and per detector. Three specific problems:

1. **Uncalibrated scores.** A score of 0.8 means different things for different channels and
   detectors, so scores cannot be fused and one rule cannot fit all channels.
2. **A noisy threshold.** The 0.999 quantile of ~500 autocorrelated validation steps is
   essentially the validation maximum, so it varies a lot from channel to channel.
3. **Alarm flapping.** A score hovering around the threshold produces many short alarm runs,
   which inflates false events and costs point precision.

## Decision

The decision layer runs on the scores that `sathealth evaluate` already saved, so it needs no
retraining. Each step is label-free.

| Step | Choice | Why |
|---|---|---|
| Calibration | **Peaks-over-threshold EVT** per channel and detector: empirical tail probabilities below the validation 0.95 quantile `u`; above `u`, a Generalised Pareto tail fit by probability-weighted moments (Hosking & Wallis 1987), shape clamped to [0, 0.5]¹ | Turns any score into a tail probability `p` comparable across channels and detectors, and extrapolates beyond the largest validation score, where the empirical CDF has no resolution (with ~500 rows it cannot go below p ≈ 0.002) |
| Surprise | `s = −log10 p`, capped at 12 | Additive and readable: s = 3 means "once in 1,000 nominal steps" |
| Fusion | **Mean surprise** of the LSTM and Transformer (geometric mean of tail probabilities) | Both detectors must be fairly surprised. Their errors are partly independent (forecast vs reconstruction), so agreement suppresses single-model false alarms |
| Alarm rule | **Hysteresis**: enter alarm at s ≥ 3, leave when s < 2 | One noisy excursion around a single level no longer splits into many alarms |
| Event builder | Merge alarm runs separated by ≤ 10 steps; keep runs of any length | Operators should see one event per incident |
| Severity | Peak surprise ≥ 6 critical, ≥ 4.5 warning, otherwise watch | Ordered by how improbable the episode is under nominal behaviour |
| Anomaly type | `point` if the event contains a raw value outside the training range (±5% margin), else `contextual` | Matches the label taxonomy of the dataset and is checked against it |
| Budget | **≤ 1.0 false events per 1,000 steps** on all channels | About today's limit check (1.03), so the system never alarms more than current practice |

¹ *Amendment, made before any real-data run:* the first draft allowed shapes down to −0.5.
A synthetic unit test showed that on Gaussian scores the fit then estimates a negative shape,
which asserts a hard upper endpoint and makes far-tail surprise overconfident (7.2 instead of
the true 5.5 at 4.5σ). Overconfidence means false alarms, so negative shapes are clamped to
0 (an exponential tail), which errs on the conservative side.

Exit criterion (from the roadmap): composite F1 significantly above the best Phase 2
baseline (paired 95% CI above zero) on all channels or on one spacecraft, while staying
within the budget.

### Ablations (reported in full)

Each ablation changes one step of the primary configuration, or reproduces a reference:
single detectors with EVT, max and min fusion, no hysteresis, no merging, the decision layer
applied to the robust z-score baseline, and the original Telemanom non-parametric dynamic
threshold with pruning (Hundman et al. 2018) on the LSTM. The Telemanom method sets its
threshold from windows of *test* errors, so it is a batch method, not a live one; it is
included because it is the standard reference on this dataset.

We also report a **global oracle**: the best single surprise level for all channels, chosen
with test labels. Unlike the per-channel oracle of Phase 2 it cannot overfit to individual
channels, so it is a meaningful upper bound for the calibrated pipeline.

## Results

Full tables: [DECISION.md](../DECISION.md). Composite F1, with the paired difference to the
best Phase 2 baseline (limit check):

| Pipeline | Live? | All channels | SMAP | False events / 1k |
|---|---|---|---|---:|
| **Primary (pre-registered)** | yes | 0.459, Δ +0.001 [−0.080, +0.087] | 0.502, Δ +0.028 [−0.086, +0.146] | 1.30 ✗ |
| Telemanom reference (pre-registered ablation) | batch | 0.578, Δ +0.119 [−0.005, +0.263] | 0.641, Δ +0.166 [+0.029, +0.317] | 0.43 |
| Fused + dynamic threshold (pre-registered ablation) | batch | 0.583, Δ +0.125 [−0.010, +0.270] | 0.650, Δ +0.175 [+0.033, +0.326] | 0.33 |
| Streaming recalibration (**post-hoc**) | yes | 0.531, Δ +0.073 [+0.003, +0.149] | 0.623, Δ +0.148 [+0.047, +0.279] | 0.23 |

### Verdict

**The pre-registered exit criterion is not met.** The primary pipeline ties the limit check
on composite F1 and exceeds the alarm budget (1.30 > 1.0 false events per 1k steps).

What else the evidence says, from strongest to weakest:

1. **The cause is a validation-to-test shift, and it is visible without labels.** Surprise
   ≥ 3 should fire on 0.1% of nominal steps. On test steps labelled nominal it fires 20% of
   the time on average for the LSTM (median 1.8%, 46 of 80 channels above 1%), 7.7% for the
   Transformer and 15.8% even for the robust z-score (24 channels above 1%). So the shift is
   in the data, not only in the models. The share of test steps in alarm (21%) shows the
   same without any labels. The LSTM is worst because its validation errors are optimistic:
   the same validation tail chose its best epoch.
2. **Methods that adapt to the test stream win, consistently.** Both pre-registered dynamic-
   threshold ablations beat the limit check on SMAP (one-sided p ≈ 0.007–0.008). With 11
   variants this does not survive a Bonferroni correction (p < 0.0023 needed), so on its own
   it is suggestive. An out-of-sample check agrees: picking the best pre-registered variant
   using MSL labels only selects the Telemanom reference, which beats the limit check on the
   held-out SMAP channels (+0.166 [+0.029, +0.317]). The reverse direction is not significant
   (MSL has only 27 channels).
3. **Streaming recalibration is the best *live* option, but post-hoc.** It was designed after
   the pre-registered results and revised once (see below). It beats the limit check on all
   channels and on SMAP, within budget, with the highest event F1 of any variant (0.49).
   Because of how it was found, these p-values (0.021 and 0.002) are not confirmatory.

### Post-hoc iteration, recorded

Streaming recalibration (SPOT/DSPOT-style, Siffer et al., KDD 2017) was added because of
finding 1. Its first version used only the most recent non-alarm test scores as reference.
On the z-score detector that made things worse: on channel F-3 the reference shrank to a
quiet stretch, ordinary fluctuations went into alarm and stayed excluded from the reference
(alarm share 0% → 45%, visible without labels). The final version always keeps the
validation scores in the reference, so recalibration can loosen what counts as nominal but
never make it stricter than validation.

### Other outputs

- **Events:** 816 alarm events from the primary pipeline, each with span, peak, severity
  (243 critical, 199 warning, 374 watch), type and per-detector evidence, saved for Phase 5.
- **Anomaly type:** the point vs contextual rule agrees with the labels on 80% of 79 detected
  anomalies, against 68% for always answering the majority class.
- **Global oracle:** the best single surprise level for every channel (3.75, chosen with
  labels) reaches composite F1 0.502. The adaptive variants (dynamic thresholds, streaming
  recalibration) exceed this bound because their threshold changes over time, which a
  single level cannot.

## Consequences

- The default config keeps the pre-registered pipeline so this verdict stays reproducible.
  `configs/live.yaml` enables streaming recalibration for Phases 5 and 6, clearly labelled as
  not yet confirmed.
- **Confirmation moves to Phase 7:** fix streaming recalibration and the dynamic-threshold
  pipeline *now*, then test them once on an independent dataset (ESA-ADB or OPS-SAT) under
  this exact protocol.
- Phase 3 follow-up: early stopping and calibration should not share one validation tail.
  Splitting it would remove the LSTM's optimistic validation errors, but needs a retrain.
- Batch methods have a hidden decision latency of up to one window (2,100 steps), which the
  delay metric does not count. This is a second reason to prefer the live pipeline.
