# ADR 0002: Label cleaning rules for `labeled_anomalies.csv`

- **Status:** accepted
- **Date:** 2026-10-19

## Context

Profiling the label file showed three problems:

| Finding | Evidence |
|---|---|
| Channel `P-2` is listed **twice** with different labels | rows 17 (`[[5350, 6575]]`) and 51 (`[[5300, 6420]]`), same series |
| Anomaly sequences are **not sorted** | `P-1`, `E-12`, `P-11` |
| The `class` column is not valid JSON | `[contextual, point]` with unquoted tokens |

## Decision

- **Duplicates:** the policy is configurable (`dataset.duplicate_policy`). The default is
  `drop`: two conflicting expert labels for one series mean the ground truth is unknown.
  `first` and `union` are available for sensitivity analysis. Each decision is logged and
  recorded in the run manifest.
- **Sorting:** sequences are sorted by start index. After sorting we checked that no
  sequence overlaps another, and none does.
- **Validation:** the parser rejects a sequence/class count mismatch, missing columns and
  `start > end`. Building the point mask rejects segments that run past the series.
- Segments stay **inclusive** `[start, end]`, as published.

## Consequences

Default evaluation covers **80 channels: 53 SMAP and 27 MSL, with 103 anomaly events**. This
matches the widely used OmniAnomaly preprocessing, which also excludes P-2. Papers that
report "55 SMAP channels" count label rows, not unique channels. Comparisons with them need
this caveat.
