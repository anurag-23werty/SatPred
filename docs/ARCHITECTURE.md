# System architecture

The target system has six layers. **Phase 1 implements L2** (ingest and preprocessing) for
the NASA SMAP/MSL benchmark. The other layers arrive in the phases listed in
[ROADMAP.md](ROADMAP.md).

```mermaid
flowchart TB
    subgraph L1[L1 · Space segment]
        BUS[Satellite bus<br/>EPS · TCS · ADCS · OBC · COMMS · Payload] -->|HK data| EDGE[Onboard edge detector<br/>int8 1D-CNN AE]
    end
    subgraph L2[L2 · Ingest & preprocess · Phase 1]
        GS[Ground station<br/>CCSDS decode] --> BUSQ[Telemetry bus<br/>prototype: verified file replay]
        BUSQ --> PRE[Validation + preprocessing<br/>robust z-score, train-only fit]
        PRE --> CTX[Context features<br/>eclipse · mode · commands]
        CTX --> WIN[Windowing + store<br/>zero-copy views]
    end
    subgraph L3[L3 · Detection ensemble]
        M1[Statistical<br/>limits + Isolation Forest]
        M2[LSTM forecaster<br/>Telemanom-style]
        M3[Transformer AE<br/>TranAD]
        M4[Graph Deviation Network]
    end
    subgraph L4[L4 · Scoring & decision]
        NORM[ECDF normalisation] --> THR[EVT-POT + NDT threshold] --> FUSE[Fusion<br/>≥2 detectors] --> EVT[Event builder] --> SEV[Severity + type]
    end
    subgraph L5[L5 · Explain & prognose]
        XAI[XAI engine<br/>attribution · onset · broken edges ·<br/>counterfactual band · grounded narrator]
        HI[Health Index + RUL]
    end
    subgraph L6[L6 · Operate & learn]
        API[FastAPI] --> UI[Operator console] --> FB[Operator feedback] --> MLOPS[Learning loop<br/>MLflow · drift · retrain]
    end
    BUS -->|TM downlink| GS
    WIN --> M1 & M2 & M3 & M4
    M1 & M2 & M3 & M4 -->|per-channel scores| NORM
    M1 & M2 & M3 & M4 -. gradients · attention · graph .-> XAI
    SEV -->|anomaly events| XAI & HI
    XAI & HI -->|explained alert + health| API
    MLOPS -. retrained models .-> M1
    MLOPS -. uplink int8 weights .-> EDGE
```

## Phase 1 data flow (implemented)

```
sathealth download ──► data/raw/telemanom/          165 files, every one SHA-256 verified
                        │
sathealth prepare ─────►├─ integrity gate             re-verify checksums, refuse on mismatch
                        ├─ label cleaning             duplicate policy, sort, bounds (ADR 0002)
                        ├─ quality validation         ERROR / WARNING / INFO per channel
                        ├─ chronological split        train | val (last 20%) per channel
                        ├─ scaler fit on train only   robust median/IQR (ADR 0003)
                        └─► data/processed/telemanom-<fingerprint>/
                              manifest.json            config, git commit, raw digest, scalers
                              quality_report.json
                              channels/<id>.npz        float32 train/val/test + labels
                        │
load_prepared() ───────► PreparedChannel.windows()    zero-copy (N, w, D) views at load time
```

## Module map

| Module | Responsibility |
|---|---|
| `sathealth.config` | Typed, frozen, strict YAML config with `--set` overrides |
| `sathealth.data.download` | Manifest-driven, atomic, retrying, checksum-verified download |
| `sathealth.data.telemanom` | Label parsing and cleaning, channel loading |
| `sathealth.data.schema` | Domain types: `ChannelData`, `AnomalySegment`, label masks |
| `sathealth.features.validation` | Data quality checks with severities |
| `sathealth.features.scaling` | Train-only robust or standard scalers, JSON-serialisable |
| `sathealth.features.windowing` | Zero-copy windows, window labels, chronological split |
| `sathealth.pipeline.prepare` | Atomic, idempotent prepare run with a provenance manifest |
| `sathealth.pipeline.report` | Markdown data card generator |
| `sathealth.cli` | `sathealth download / verify / prepare / report / info` |
