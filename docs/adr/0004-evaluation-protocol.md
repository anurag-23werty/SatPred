# ADR 0004: Evaluation protocol for anomaly detectors

- **Status:** accepted
- **Date:** 2026-10-19

## Context

Published SMAP/MSL results are hard to compare and often optimistic. Three practices cause
most of this:

1. **Point-adjusted F1** counts a whole anomaly segment as detected if any single step in it
   is flagged. Kim et al. (AAAI 2022) showed that random scores do well under it.
2. **Thresholds tuned on test labels**, often reported as "best F1", measure how separable the
   scores are, not how a deployed detector would behave.
3. **No uncertainty.** With 80 channels and 103 events, differences of a few F1 points are
   usually noise.

We also do not know the sampling interval of the NASA data, so "false alarms per day" cannot
be computed honestly.

## Decision

| Aspect | Choice |
|---|---|
| Operating threshold | `val_quantile`: the 0.999 quantile of each detector's scores on the **nominal validation tail** of train, per channel. Test labels are never an input. Prediction is `score > threshold`. Set a priori, not tuned on test. |
| Headline metric | **Composite F1** (Garg et al. 2021): point-wise precision × event-wise recall. Rewards catching every event and penalises alarm floods. |
| Also reported | Event-wise F1 (Hundman et al. 2018), micro and macro; event precision and recall; FP events per 1k steps; median detection delay in steps; macro AUC-PR (threshold-free). |
| Comparison only | Point-adjusted F1, marked ⚠, next to a **random-score detector** that shows the inflation. |
| Upper bound | Per-channel oracle event F1, marked †, with the caveat below. |
| Uncertainty | 95% percentile bootstrap over **channels** (1,000 resamples, seeded). |
| Subsets | All channels; each spacecraft; without `constant_train`; without `constant_train` + `nominal_out_of_range` (both flags come from Phase 1). |
| Reproducibility | Each run is keyed by a fingerprint of the data, detector parameters and protocol. Val/test scores are saved for Phase 4 fusion. |

## What the baselines taught us (see docs/BASELINES.md)

- **Point-adjust inflation is real on this data.** Random scores reach point-adjusted F1
  **0.897**, higher than every real baseline (0.51–0.59), with event F1 of only 0.06.
- **One shifted channel can dominate micro metrics.** Channel A-5 produces **466 of the limit
  check's 517 false events (90%)**. Phase 1 had already flagged it as `nominal_out_of_range`.
  Once the flagged channels are excluded, the limit check's event F1 rises from 0.14 to
  0.35 and its FP rate drops from 1.03 to 0.04 per 1k steps, so we report macro F1 and the
  flagged-channel subsets.
- **No baseline is clearly best.** On composite F1 the limit check (0.458), z-score (0.395)
  and Isolation Forest (0.372) have overlapping 95% CIs. The learned detectors of Phase 3 must
  beat this band, not a single number.
- **Event precision is low for every detector** (0.03–0.12). Raw thresholded scores break into
  many short alarm runs. Turning scores into events (merging, pruning, minimum duration) is a
  separate stage, done in Phase 4.
- **The per-channel oracle is a loose bound.** Even random scores reach an oracle event F1 of
  0.74, because picking one of 200 thresholds per channel with labels overfits when a channel
  has only one or two events. Phase 4 will report a *global* oracle (one threshold rule for
  all channels) as the meaningful upper bound.
- **Isolation Forest has two structural weaknesses**, both pinned by tests. It isolates rare
  nominal events (a command-at-last-step feature caused false alarms and was removed), and it
  **saturates outside the training range** (+3 and +300 shifts score identically).

## Consequences

- Every later phase reports through the same harness, so improvements are comparable.
- The Phase 3 exit criterion is: beat the baseline band on composite F1, with
  non-overlapping CIs on at least one spacecraft.
