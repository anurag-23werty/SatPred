# Roadmap

Each phase ships as working, tested code. Every phase must pass `make check` and CI before
the next one starts.

| Phase | Scope | Exit criteria | Status |
|---|---|---|---|
| **1. Data foundation** | Verified download, label cleaning, quality validation, train-only scaling, windowing, reproducible prepare runs, data card, CI | 80 channels prepared from checksummed data; 90 tests; `mypy --strict` clean | ✅ done |
| **2. Evaluation harness + baselines** | Label-free thresholds; composite, event-wise and point-adjusted F1, detection delay, FP events per 1k steps, AUC-PR; bootstrap CIs; limit-check, robust z-score, Isolation Forest and random baselines; generated leaderboard ([BASELINES.md](BASELINES.md)) | Reproducible baseline scores on SMAP and MSL, with and without flagged channels ([ADR 0004](adr/0004-evaluation-protocol.md)) | ✅ done |
| **3. Deep detectors** | PyTorch LSTM forecaster (Telemanom-style) and bottlenecked Transformer autoencoder; causal EWMA scoring; deterministic per-channel training with parallel `--jobs`; MLflow tracking (optional); paired bootstrap comparisons ([ADR 0005](adr/0005-deep-detectors.md)) | Beat the baseline band (composite F1 ≈ 0.37–0.46) with non-overlapping 95% CIs on at least one spacecraft. **Outcome:** AUC-PR beats every baseline with paired significance (all channels and SMAP); 91/103 events found. Composite F1 0.524 (Transformer) is ahead but not significant, because static thresholds flood alarms. That gap moves to Phase 4 | ✅ done |
| **4. Decision layer** | ECDF score normalisation, EVT peaks-over-threshold, Telemanom non-parametric dynamic thresholding, ensemble fusion, event builder (fixes the low event precision found in Phase 2), global oracle bound, severity and anomaly type | Composite F1 significantly above the best baseline (paired 95% CI > 0) on at least one spacecraft, with false events per 1k steps kept within a configurable budget | next |
| **5. Explainability** | Integrated Gradients/SHAP channel attribution, temporal onset, counterfactual expected band, grounded narrator (template, optional LLM) | Explanation hit@3 reported; every narrative sentence traceable to a computed value | planned |
| **6. Health and operations** | Subsystem Health Index, RUL with uncertainty, FastAPI service, Streamlit operator console, feedback capture | Demo: replayed stream → explained alert → operator verdict stored | planned |
| **7. Extensions** | Graph Deviation Network, onboard int8 ONNX export with a size and latency report, ESA-ADB / OPS-SAT loaders, drift monitoring and retraining | Stretch goals for the final pitch | planned |
