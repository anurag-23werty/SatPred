# ADR 0003: Robust scaling fit on chronological train rows only

- **Status:** accepted
- **Date:** 2026-10-19

## Context

- The train series have no labels, so we cannot rule out hidden anomalies in them.
- Many channels hit exactly ±1.0 in **test** but not in train. This suggests the source was
  min-max scaled with test-period ranges, and we must not add more leakage on top of that.
- 16 channels are constant (or nearly constant) in train. `M-6` is −1 for the whole of train
  but reaches 258 in test. Standardising by a spread of about 0 would turn the first nominal
  wiggle into a 10⁶σ "anomaly".
- Models need a validation set for early stopping and threshold calibration, taken from data
  they never trained on.

## Decision

1. **Split before anything else:** validation is the last 20% of each train series in time
   order. Random splits are not used, because they leak future context into training through
   overlapping windows. An optional `split_gap` drops rows between the two parts.
2. **Fit the scaler on the training rows only.** Validation and test are transformed with
   those parameters.
3. **Robust scaler:** `(x − median) / (IQR / 1.349)`. The 1.349 factor makes the scale equal
   to σ on Gaussian data, so scores read as σ-units, and a 1% contamination of the training
   data cannot inflate the scale (see `tests/test_scaling.py`).
4. **Degenerate spread falls back IQR → std → 1.0** and is flagged `degenerate` in the run
   manifest. Validation also raises a `constant_train` warning for these channels.
5. Only telemetry columns are scaled. One-hot command flags pass through unchanged.
6. Windows are built **after** splitting, as zero-copy views, so no window can contain rows
   from two splits.

## Consequences

Scores on `constant_train` channels need care in evaluation. Phase 2 will report metrics
both with and without them, so readers can see how much they affect the headline numbers.
