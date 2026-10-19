# Roadmap

Each phase ships as working, tested code. Every phase must pass `make check` and CI before
the next one starts.

| Phase | Scope | Exit criteria | Status |
|---|---|---|---|
| **1. Data foundation** | Verified download, label cleaning, quality validation, train-only scaling, windowing, reproducible prepare runs, data card, CI | 80 channels prepared from checksummed data; 90 tests; `mypy --strict` clean | ✅ done |
| **2. Evaluation harness + baselines** | Event-wise precision/recall/F1 (point-adjusted F1 reported only for comparison, labelled as inflated), detection lead time, false alarms per day; limit checks and Isolation Forest baselines; per-channel results table | Reproducible baseline scores on SMAP and MSL, with and without `constant_train` channels | next |
| **3. Deep detectors** | PyTorch LSTM forecaster (Telemanom-style) and Transformer autoencoder (TranAD); MLflow experiment tracking; seeds and determinism | Both models beat the baselines on event-wise F1 on at least one spacecraft | planned |
| **4. Decision layer** | ECDF score normalisation, EVT peaks-over-threshold, Telemanom non-parametric dynamic thresholding, ensemble fusion, event builder, severity and anomaly type | False alarms per day kept within a configurable budget | planned |
| **5. Explainability** | Integrated Gradients/SHAP channel attribution, temporal onset, counterfactual expected band, grounded narrator (template, optional LLM) | Explanation hit@3 reported; every narrative sentence traceable to a computed value | planned |
| **6. Health and operations** | Subsystem Health Index, RUL with uncertainty, FastAPI service, Streamlit operator console, feedback capture | Demo: replayed stream → explained alert → operator verdict stored | planned |
| **7. Extensions** | Graph Deviation Network, onboard int8 ONNX export with a size and latency report, ESA-ADB / OPS-SAT loaders, drift monitoring and retraining | Stretch goals for the final pitch | planned |
