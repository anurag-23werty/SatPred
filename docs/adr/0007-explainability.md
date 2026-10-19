# ADR 0007: Explainability (onset, counterfactual, attribution, grounded narrative)

- **Status:** accepted
- **Date:** 2026-10-19
- **Pre-registered:** methods, parameters, evaluation protocol and hypotheses below were fixed
  *before* any Phase 5 result was computed.

## Context

An alarm alone tells an operator *that* something is wrong. The challenge asks for
explainable AI: each alert should say **where** (which input drove it), **when** (when the
deviation really started), **what normal would have looked like**, and **how sure** the
system is, in plain sentences that never invent anything.

SMAP/MSL channels are univariate telemetry plus one-hot command flags (24 for SMAP, 54 for
MSL). "Where" therefore means: the channel's own recent history, or a specific command, and
which time steps. The channel names and units are anonymised by NASA, so explanations use the
dataset's own units.

Explanations attach to the events of the live decision pipeline (`configs/live.yaml`, ADR 0006).

## Decision

| Component | Method | Why |
|---|---|---|
| Model access | Retrain the channel's LSTM deterministically and **verify** it reproduces the saved Phase 3 validation scores (max abs. difference < 1e-4) before using it; cache the weights | Phase 3 did not store weights. Training is bit-reproducible, so an exact rebuild is possible, and the check proves we explain the model that raised the alarm |
| Onset | Walk back from the alarm start while the fused surprise stays ≥ 1 (p ≤ 0.1), at most 500 steps | The alarm fires at surprise 3; the deviation usually begins earlier and below that level. Label-free and causal up to the alarm |
| Counterfactual | Autoregressive LSTM rollout from the step before the onset, feeding back its own predicted value and the **commands actually sent**, for up to 250 steps. 90% band from horizon-wise residual quantiles of the same rollouts on the validation tail | "What the signal would have done, given what the spacecraft was told to do." A one-step forecast follows the anomaly once it starts; a rollout does not |
| Attribution | Integrated Gradients (Sundararajan et al. 2017) of the squared one-step forecast error at the event peak, with respect to the 64 × D input window; all-zeros baseline (training median value, no commands); 64 steps, trapezoidal rule; completeness gap reported | Axiomatic, works on any differentiable model, and the completeness axiom gives a built-in check |
| Summaries | Feature importance = Σ over time of \|IG\|; lag importance = Σ over features of \|IG\| | Answers "which input" and "which moment" separately |
| Narrative | Template sentences, each tied to the computed facts it states; a grounding checker verifies every number in every sentence appears in its facts | The narrator can only restate computed values; any LLM rewording must pass the same check or is discarded |
| Optional LLM | Rewords the template brief; never used in evaluation | Readability for operators without risking invented content |

## Evaluation protocol and hypotheses

**E1 Onset (all 80 channels, real labels).** For every alarm event overlapping a labelled
anomaly, compare |onset − labelled start| with |alarm start − labelled start|.
*H1: the onset estimate is closer to the labelled start than the alarm start, on the median.*

**E2 Attribution faithfulness (synthetic injections on real channels).** Channels are fixed
by rule: the first 5 SMAP and first 5 MSL channels in manifest order whose training data
contains at least one command firing. Per channel, 20 label-free test windows (no labelled
anomaly within the window or its target step), seed 0:

- *Temporal:* add +3 (scaled units) to the value at a random lag. Hit@3 = the injected lag
  is among the 3 lags with the largest importance. Random baseline 3/64.
- *Command:* switch on a command that is off, at a random lag. Counted only if it changes
  the forecast by ≥ 0.05 (otherwise the model ignores it and attribution *should* not point to
  it; the share counted is reported). Hit@3 = that command among the top 3 features.
  Random baseline 3/D.
- *Deletion:* reset the top 5% of attributed inputs to the baseline and compare the error
  reduction with resetting a random 5%.

*H2: hit@3 is above its random baseline (95% CI over injections) for both kinds, and
deleting top-attributed inputs reduces the error more than deleting random ones.*

**E3 Counterfactual calibration.** From 20 random starts per E2 channel in label-free test
stretches, roll out 100 steps and measure how often the true value falls inside the 90% band.
*H3: coverage between 80% and 95%* (it is calibrated on validation, so the shift found in
ADR 0006 may lower it; that would be reported, not tuned away).

**E4 Grounding.** Every sentence of every generated narrative passes the grounding checker.
*H4: 100%.* This is a hard requirement.

## Results

Full report: [EXPLAINABILITY.md](../EXPLAINABILITY.md). 199 events of the live pipeline on
70 channels were explained. **All 70 rebuilt LSTMs reproduced their saved Phase 3
validation scores**, so every explanation describes the model that raised the alarm.

| | Result | Verdict |
|---|---|---|
| H1 onset | Median error 33 vs 36 steps (71 detected anomalies); onset closer in 45%, tied in 48%; 95% CI of the median improvement [0, 2] steps | supported, but the gain is negligible |
| H2 attribution | Temporal hit@3 22% (chance 5%), command hit@3 38% on 8 (chance 9%); deletion test failed (−767 vs −1.8) | **not supported** |
| H3 band | 72% coverage (7 of 10 channels at 83–97%; A-5 28%, C-2 15%, D-16 42%) | **not supported** |
| H4 grounding | 1,373 of 1,373 sentences grounded | supported |

### What the failures mean (post-hoc, not confirmatory)

1. **The explanations exposed a modelling shortcut.** Only 13% of injected value spikes and
   4% of injected commands move the LSTM's forecast at all: the model has a short memory
   and nearly ignores commands. Where an injection does move the forecast, IG puts it in the
   top 3 in 65% of cases [46%, 81%]. So the narratives that credit "the channel's own recent
   values" are *faithful*: that really is what the model uses. The Phase 3 design assumed
   commands would explain nominal level changes; the attribution shows the model did not
   learn that.
2. **The deletion test was mis-specified, by us.** Ranking inputs by |IG| also removes
   inputs that lower the error, so resetting them raised it. Ranked by signed IG, deletion
   lowers the error in 64% of windows vs 45% for random inputs. Zero-baseline deletion is
   also out of distribution (Hooker et al. 2019). A future protocol should pre-register the
   signed ranking and a median.
3. **Band coverage fails where the data shifts.** The three failing channels are those whose
   test period differs from validation, the shift found in ADR 0006 (A-5 was already flagged
   by the Phase 1 data card). Median per-channel coverage is 87%.

## Consequences

- Narratives ship as they are: their grounding is verified (H4), and their attribution
  sentence was shown to be faithful to the model (post-hoc 1).
- The onset sentence is kept, but it should not be sold as "earlier warning": on this data
  it rarely moves more than a few steps before the alarm.
- The counterfactual band needs the same adaptivity as the alarms (ADR 0006): recalibrate
  it on recent nominal data. Until then, a channel whose alarm share is high (label-free
  signal) should present its band as less reliable.
- A command-aware model (e.g. one trained with command dropout, or one that predicts the
  response to each command) is a Phase 7 item. Attribution then becomes the test of whether
  it worked.
- The optional LLM rewording (`explain.llm`) can only add readability: its output is shown
  only when every number in it is grounded in the facts.
