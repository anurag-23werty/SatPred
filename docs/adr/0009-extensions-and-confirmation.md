# ADR 0009: Independent confirmation and operational extensions

- **Status:** accepted
- **Date:** 2026-10-20
- **Pre-registered:** the confirmation (A) and priming (B) protocols, hypotheses and
  decision rules below were fixed *before* any Phase 7 result was computed.

## Context

Phases 4–6 left open questions that only new evidence or new features can answer:

1. The live adaptive pipeline and the Telemanom dynamic threshold beat the limit check on
   SMAP, but the live variant was found post hoc (ADR 0006). It needs a one-shot test on an
   **independent** spacecraft.
2. 10% of live alerts are a start-up transient: test series begin without history (ADR 0008).
3. Channels with a permanent level shift stay in alarm until an operator re-baselines
   (ADRs 0006, 0008).
4. The validation-to-test shift is detectable without labels (ADR 0006), but nothing in the
   service watches for it.

## A. Confirmation on OPS-SAT-AD (pre-registered)

**Data.** OPSSAT-AD (Ruszczak et al., *Scientific Data* 2025; Zenodo 15108715, CC-BY-4.0):
telemetry of ESA's OPS-SAT cubesat, 2,123 labelled segments from 9 channels, with the
benchmark's own train/test split. Another spacecraft, operator and sampling; never used in
Phases 1–6. Files are pinned by MD5 like the NASA data.

**Fixed structure decisions** (made from the data layout, before any detector was run):

- Segments are short (median 48–239 points) and the benchmark's split interleaves train and
  test segments in time. A window never spans two segments: deep models train on windows
  cut *inside* nominal training segments, and every segment is scored on its own.
- Training uses the *nominal* training segments only (the curated-nominal assumption of
  Phases 1–6). Validation = the chronologically last 20% of those points, by segment.
- Robust scaling fitted on nominal training values, per channel.
- Channels where a deep model has no complete training window (`window + 1 = 65` points
  inside one nominal segment) are excluded from every comparison: CADC0886 and CADC0890.
  That leaves 7 channels and 523 of 529 test segments.
- The decision layer runs on each channel's test segments concatenated in time order.

**Pipelines, unchanged from Phases 2–4:** detectors with the Phase 3 hyperparameters
(window 64, LSTM 2×80, Transformer d=32 …), the decision configurations of ADR 0006, and the
limit check and robust z-score baselines (Isolation Forest is excluded: its 32-step
features do not fit within short segments).

**Units.** Each test segment is one event (anomalous segments are the labelled events).
Paired bootstrap over **test segments** (1,000 resamples, seed 42), because 7 channels are
too few to resample.

**Hypotheses (one-shot):**

- **C1.** The live pipeline (`configs/live.yaml`: EVT, streaming recalibration, mean fusion
  of LSTM and Transformer, hysteresis 3/2, merge 10) has a higher composite F1 than the best
  baseline (paired 95% CI above zero).
- **C2.** The same holds for the Telemanom reference (raw LSTM errors, dynamic threshold,
  pruning 0.13).

Also reported, not tested: the primary static pipeline, segment-level F1 (a segment is
flagged if any step alarms), segment AUC-PR (segment score = max surprise), and
false-alarm segments.

## B. Start-up priming on SMAP/MSL (pre-registered)

In operations the stream is continuous. **Priming** scores every split with the latest
nominal history in front of it: validation is scored after the last 256 training steps (it
follows them in time), test after the last 256 validation steps. Only the scores of the
split itself are kept. No training changes, and the models are bit-identical.

- **P1.** For the live pipeline, alarm events starting in the first 64 test steps drop by at
  least 50% (from 19).
- **P2.** Non-inferiority: the live pipeline's composite F1 with priming is at most 0.03
  below the unprimed one (paired bootstrap over channels, lower 95% bound > −0.03).

## C–E. Operational extensions (engineering, no hypotheses)

- **C. Operator re-baselining.** After an operator confirms a permanent shift, the channel's
  calibration reference is replaced by its most recent scores, alarm included, and the
  alarm is cleared: "this is the new normal". A service endpoint and a console button,
  audited in the store.
- **D. Drift monitor.** Label-free, per channel: the share of recent steps in alarm against
  the designed rate, and the population stability index (PSI) of recent values against
  training. A channel is flagged as drifting when either exceeds a fixed threshold
  (alarm share > 10× design, or PSI > 0.25, the conventional "significant shift").
- **E. Edge feasibility.** Measured once on an Apple M2 CPU (not shipped as code): PyTorch
  dynamic int8 quantization of the LSTM shrinks weights 3.4× (348 kB → 101 kB) with a
  maximum output difference of 7×10⁻⁴, but runs 2.3× *slower* than float32 on this CPU
  (1.0 → 2.3 ms per window), and the `torch.ao` API is deprecated in favour of `torchao`.
  Int8 pays off on integer accelerators; it must be evaluated on the target onboard
  hardware through ONNX or torchao, not on a laptop.

## Deferred, with reasons

- **Graph Deviation Network:** needs time-aligned multivariate telemetry. SMAP/MSL channels
  are separate series of different lengths and OPS-SAT segments are univariate; ESA-ADB
  (3.8 GB per mission) is the natural dataset.
- **Command-aware forecaster** (ADR 0007): new model research, evaluated with the Phase 5
  injection test.
- **Adaptive counterfactual band** (ADR 0007): recalibrate the band like the alarms.

## Results

### A. Confirmation on OPS-SAT-AD: both hypotheses fail

Full table: [CONFIRMATION.md](../CONFIRMATION.md). 7 channels, 523 test segments (110
anomalous); the best baseline is the robust z-score (composite F1 0.382).

| | Pipeline | Composite F1 | Δ vs z-score [paired 95% CI over segments] | Verdict |
|---|---|---:|---|---|
| C1 | Live pipeline (streaming recalibration, LSTM + Transformer) | 0.128 | −0.254 [−0.364, −0.128] | **not supported**: significantly *worse* |
| C2 | Telemanom reference (LSTM, dynamic threshold) | 0.397 | +0.016 [−0.079, +0.135] | **not supported**: a tie |

No pipeline is significantly better than the z-score. LSTM-based pipelines tie it (LSTM with
EVT +0.047 [−0.048, +0.147]); every pipeline involving the Transformer is significantly worse.
**The Phase 4 advantage on SMAP does not transfer to this spacecraft.**

Likely reasons (post hoc, not tested): OPS-SAT segments are short (median 48–239 steps)
against a 64-step window, so the window-reconstructing Transformer and the cross-detector
fusion rarely get the context they were designed for; the forecasting LSTM degrades more
gracefully.

**Bug found and fixed during the run, with no effect on the hypotheses.** The first run
was started with `--config configs/live.yaml`, which silently gave *every* decision variant
the live pipeline's streaming recalibration, so the "static" rows were adaptive. The
variants are now built from the pre-registered defaults whatever config is passed (tested).
C1 and C2 are unaffected by construction (C1 *is* the adaptive pipeline; C2 uses raw
scores), and the rerun reproduced both to the last digit; the first run's file is kept as
`runs/confirmation/results-first-run.json`.

**A secondary metric is confounded.** Segment AUC-PR ranks segments by their highest score,
which grows with segment length: the random control scores 0.399, the highest of all. It is
reported with that warning and not interpreted.

### B. Start-up priming: both hypotheses fail

[PRIMING.md](../PRIMING.md): events starting in the first 64 test steps went **up**, 19 → 32
(P1 not supported), and composite F1 fell 0.531 → 0.489, Δ −0.042 [−0.104, +0.012] (P2 not
supported). Post hoc: NASA's test series do not continue the validation series, so priming
glues two recordings together. The 17 channels that gained start-up alarms have a median
jump of 0.55 validation standard deviations at the join, against 0.17 for the other 63.
Priming is right for a truly continuous stream (operations), but this dataset cannot show
it; `configs/primed.yaml` is kept for that case and is not the default.

### C. Re-baselining

Works as designed (tested on a synthetic permanent shift: the alarm stops after the operator
accepts the new level; audited in a new `rebaselines` table, with the store's first schema
migration, v1 → v2, upgrading existing databases in place).

### D. Drift monitor: redesigned once, then informative but saturated on this data

The first design (PSI > 0.25 or alarm share > 10× design) flagged 78 of 80 channels and was
replaced (engineering iteration, not a hypothesis): the flag now fires when recent values
differ from training more than any training window of the same length does. On a full
SMAP/MSL replay it flags 52 channels, **persistently** (median 100% of checks once flagged),
so this is real: most test periods differ in distribution from their training periods, the
same shift ADR 0006 found through the alarm rate. The monitor is correct here, but on this
dataset it mostly confirms what was already known; its value is on a live mission where
periods are contiguous.

### Also found

Adding the `prime_steps` setting at first changed every published run fingerprint (all
Phase 2–6 results became unaddressable). Settings added later are now left out of
fingerprints at their neutral value, and a test pins the published fingerprints.

## Consequences

- The project's claims are narrowed accordingly: on SMAP, the deep detectors rank anomalies
  better and adaptive thresholds help; **on an independent spacecraft neither held**, and a
  robust z-score with a validation-quantile threshold was as good as any pipeline here.
  Every README statement about Phase 4 now points to this result.
- For a new mission, start with the simple baseline, run the pipelines in shadow mode, and
  adopt the deep pipeline only if it wins there, using operator verdicts (Phase 6) as labels.
