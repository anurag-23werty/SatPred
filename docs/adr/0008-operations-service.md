# ADR 0008: Operations service (live decisions, health, feedback, console)

- **Status:** accepted
- **Date:** 2026-10-19

## Context

Phases 1–5 produce detectors, a decision layer and explanations as batch jobs. An operator
needs the opposite: telemetry arriving step by step, alerts the moment they fire, an
explanation on request, a view of overall health, and a way to say "this alert was right"
or "this was a false alarm". The exit criterion is a demo: replayed stream → explained
alert → operator verdict stored.

## Decision

| Concern | Choice | Why |
|---|---|---|
| Live decisions | `StreamingDecider`: the Phase 4 decision layer (EVT calibration, streaming recalibration, fusion, hysteresis, event merging) re-implemented **one step at a time**, with a test that it reproduces the batch `decide` events and surprises **exactly** | Two implementations of the same rule drift apart silently. Exact parity means every number in DECISION.md describes the live system |
| What is replayed | The detector scores saved by `sathealth evaluate`, one test step per tick. They are causal by construction (each uses only past steps); calibration, alarms, events, health and explanations are computed live | Running two deep models per step on a laptop for 80 channels is possible but adds nothing to the demo; being explicit about it is what matters |
| Event lifecycle | `open` when the alarm fires (the early alert), `closed` once no alarm for more than `merge_gap` steps; an event shorter than `min_duration` is never announced | Operators see the alert at once; the merged, final event is identical to the batch one |
| Explanations | On request, computed with data **up to the current replay step only** (onset, counterfactual horizon and attribution never look ahead), then cached | A live system cannot know the future; a cached result keeps the console fast |
| Health index | Per channel: `100 × (1 − severity-weighted share of the last 500 steps in alarm)`, weights watch 0.25, warning 0.5, critical 1. Per subsystem: the **minimum** over its channels (weakest link), plus the count of alerting channels | Explainable in one sentence, bounded, and it moves only when alarms do |
| Subsystems | NASA anonymised the channels; the channel family letter (A, B, …) is the only grouping available, so it stands in for the subsystem | Honest about the data; with a real spacecraft the mapping comes from the telemetry database |
| Remaining useful life | **Not claimed.** SMAP/MSL has no run-to-failure data, so no RUL model can be validated on it. Instead: *time until the health index crosses 50 at its current trend*, a linear trend with a residual-bootstrap 80% interval, labelled as an extrapolation | A number that cannot be validated must not look like a prediction |
| Storage | SQLite in WAL mode via the standard library; schema version table; events, verdicts, explanation cache, replay runs | Zero setup, transactional, enough for one ground station; the repository layer is the only place that touches SQL |
| API | FastAPI: health and readiness probes, channels, telemetry window, events, explanation, verdicts, feedback summary, health index, replay control, Prometheus-style metrics, and a **server-sent events** stream of alert updates | Typed request/response models, OpenAPI docs for free, push updates without polling |
| Console | A single static page (HTML, CSS, vanilla JavaScript, inline SVG charts) served by the same process, not Streamlit | One process to start, no extra dependency, real push updates over SSE, works without internet at a demo venue |
| Feedback | Verdict per event (`confirmed`, `false_alarm`, `unsure`) with operator and note; summary of operator-confirmed precision by severity; exportable for retraining (Phase 7) | Operator feedback is the only label source a live mission has |

## Verification

- Online/offline parity on synthetic data (unit test) and on real channels (integration).
- End-to-end demo test through the HTTP API: start a replay, receive an alert over the
  stream, fetch its explanation, post a verdict, read it back from the database.
- Replay throughput and API latency measured by `sathealth demo` and reported below.

## Results

Measured on an Apple M2 laptop with the live profile (`configs/live.yaml`), 80 channels:

| Check | Result |
|---|---|
| Live = batch | `StreamingDecider` reproduces the batch decision run **exactly** on all 80 channels: 199 of 199 events identical field by field, maximum surprise difference 0 (also pinned in CI on the z-score pipeline) |
| Exit criterion | `sathealth demo`: replay → critical alert on A-8 → grounded explanation → verdict stored, in ≈ 7 s end to end; the same flow is tested through the HTTP API |
| Replay throughput | ≈ 39,000 channel-steps/s in the engine; ≈ 225 replay steps/s for all 80 channels through the running service (the default replay speed is 20) |
| API latency (80 channels, paused) | p95 under 2.5 ms for status, health, events, telemetry, metrics and cached explanations; an uncached explanation takes 1–3 s (model load, band calibration, Integrated Gradients) |
| Live explanations | Identical, sentence for sentence, to the batch explanations of Phase 5 for the same event (G-7) |

![Operator console](../assets/console.jpg)

### Findings

1. **Start-up transient.** Each test series starts without history, so the models' first
   64 steps are poorly informed: 19 of the 199 live-pipeline events (10%) start there, and 15
   of them overlap no labelled anomaly. In operations the stream is continuous; priming the
   models with the preceding nominal data would remove most of these. Phase 7 item. The
   demo picks its example after the warm-up and says so.
2. **The weakest-link health index exposes alarm lock-in.** A few channels (A-5, P-14, …)
   stay in alarm for thousands of steps: the permanent level shifts that ADR 0006 predicted
   streaming recalibration cannot absorb without an operator re-baselining. Their health is
   0, so their families show 0. The console's headline is therefore the number of healthy
   channels (70/80 at step 1,560 of the replay), with the lowest index shown beside it.
3. **Looking at the console found two bugs the tests did not:** unreadable health tiles and
   a speed control out of sync with the server. Both were fixed; the screenshot above is the
   fixed version.

## Consequences

- The service is the integration point for Phase 7: verdicts are exported at
  `/api/feedback/export` as labels for retraining.
- A real deployment would put the replay behind an ingest adapter (e.g. CCSDS frames from a
  ground station); everything downstream of `StreamingDecider.step` stays the same.
- Operator re-baselining (accepting a new nominal level after a confirmed permanent shift)
  is the natural next feature, and the fix for lock-in.
