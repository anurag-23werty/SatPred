"""Typed, validated configuration loaded from YAML with dotted-key overrides.

All tunables live in ``configs/*.yaml``. Unknown keys are rejected so a typo fails loudly
instead of silently falling back to a default::

    cfg = load_config("configs/default.yaml", overrides=["windows.size=128"])
"""

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Spacecraft = Literal["SMAP", "MSL"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DuplicatePolicy(StrEnum):
    """How to treat a channel listed more than once in the label file.

    ``drop`` removes it entirely (default). SMAP ``P-2`` has two rows with conflicting labels;
    dropping it leaves 53 SMAP + 27 MSL channels, matching the widely used OmniAnomaly
    preprocessing, which also excludes ``P-2``.
    ``first`` keeps the first row, ``union`` merges the anomaly segments of all rows.
    """

    DROP = "drop"
    FIRST = "first"
    UNION = "union"


class PathsConfig(_Strict):
    raw_dir: Path = Path("data/raw/telemanom")
    processed_dir: Path = Path("data/processed")
    runs_dir: Path = Path("runs/evaluations")
    decisions_dir: Path = Path("runs/decisions")
    models_dir: Path = Path("runs/models")
    explanations_dir: Path = Path("runs/explanations")


class DatasetConfig(_Strict):
    name: Literal["telemanom"] = "telemanom"
    spacecraft: tuple[Spacecraft, ...] = ("SMAP", "MSL")
    channels: tuple[str, ...] | None = Field(
        default=None, description="Optional subset of channel ids; None means all."
    )
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.DROP


class DownloadConfig(_Strict):
    manifest: str = Field(
        default="telemanom.json",
        description="Manifest file name inside sathealth/data/manifests, or a filesystem path.",
    )
    max_workers: int = Field(default=8, ge=1, le=32)
    retries: int = Field(default=4, ge=0, le=10)
    timeout_s: float = Field(default=60.0, gt=0)
    backoff_s: float = Field(default=1.0, ge=0)


class PreprocessConfig(_Strict):
    scaler: Literal["robust", "standard", "none"] = "robust"
    value_columns: tuple[int, ...] = Field(
        default=(0,),
        description="Columns holding continuous telemetry. The remaining columns are one-hot "
        "command flags and are passed through unscaled.",
    )
    val_fraction: float = Field(default=0.2, ge=0.0, lt=0.5)
    split_gap: int = Field(
        default=0, ge=0, description="Rows discarded between train and validation tails."
    )
    min_scale: float = Field(
        default=1e-6, gt=0, description="Spread below this is treated as a constant signal."
    )


class WindowConfig(_Strict):
    size: int = Field(default=100, ge=2)
    stride: int = Field(default=1, ge=1)
    label_rule: Literal["any", "last"] = "any"


class ValidationConfig(_Strict):
    fail_on_error: bool = True
    out_of_range_warn_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    max_anomaly_fraction: float = Field(default=0.3, gt=0.0, le=1.0)
    low_cardinality_max: int = Field(default=10, ge=1)


class EvaluationConfig(_Strict):
    """Protocol for scoring detectors; see docs/adr/0004-evaluation-protocol.md.

    Thresholds are derived from validation scores only (the nominal tail of train), never
    from test labels. ``oracle_thresholds`` is used solely to report an upper bound.
    """

    threshold: Literal["val_quantile", "val_sigma"] = "val_quantile"
    quantile: float = Field(default=0.999, gt=0.5, lt=1.0)
    sigma: float = Field(default=4.0, gt=0.0)
    min_threshold_rows: int = Field(
        default=50, ge=1, description="Validation shorter than this falls back to train scores."
    )
    early_tolerance: int = Field(
        default=0, ge=0, description="Steps before a labelled start that still count as a hit."
    )
    min_event_gap: int = Field(
        default=0, ge=0, description="Predicted runs separated by <= this many steps merge."
    )
    oracle_thresholds: int = Field(default=200, ge=10)
    bootstrap: int = Field(default=1000, ge=0, description="Resamples for 95% CIs; 0 disables.")
    exclude_subsets: tuple[tuple[str, ...], ...] = Field(
        default=(("constant_train",), ("constant_train", "nominal_out_of_range")),
        description="Extra result subsets, each excluding channels flagged by these Phase 1 "
        "quality checks.",
    )


class LimitConfig(_Strict):
    margin: float = Field(
        default=0.05, ge=0.0, description="Band widening as a fraction of the train range."
    )


class IsolationForestConfig(_Strict):
    window: int = Field(default=32, ge=2)
    n_estimators: int = Field(default=200, ge=10)
    max_samples: int = Field(default=256, ge=16)


class BaselinesConfig(_Strict):
    limit: LimitConfig = LimitConfig()
    iforest: IsolationForestConfig = IsolationForestConfig()


class TrainingConfig(_Strict):
    """Optimisation settings shared by the deep detectors (one model per channel)."""

    epochs: int = Field(default=35, ge=1)
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    patience: int = Field(
        default=5, ge=1, description="Epochs without validation improvement before stopping."
    )
    min_delta: float = Field(default=1e-4, ge=0, description="Smallest loss drop that counts.")
    grad_clip: float = Field(default=1.0, gt=0)
    device: Literal["cpu", "mps", "cuda", "auto"] = Field(
        default="cpu", description="cpu is the only bit-for-bit reproducible choice."
    )
    threads: int | None = Field(
        default=1,
        ge=1,
        description="torch CPU threads per channel. 1 keeps results identical on every "
        "machine (parallelise over channels with --jobs instead); None = torch default.",
    )


class LSTMConfig(_Strict):
    window: int = Field(default=64, ge=2, description="Steps of history per forecast.")
    hidden: int = Field(default=80, ge=1)
    layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    smoothing: int = Field(
        default=30, ge=1, description="EWMA span applied to prediction errors; 1 disables."
    )


class TransformerConfig(_Strict):
    window: int = Field(default=64, ge=2)
    d_model: int = Field(default=32, ge=4)
    heads: int = Field(default=4, ge=1)
    layers: int = Field(default=2, ge=1)
    feedforward: int = Field(default=64, ge=1)
    latent: int = Field(default=16, ge=1, description="Bottleneck size; prevents identity.")
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    smoothing: int = Field(default=30, ge=1)

    @model_validator(mode="after")
    def _check_heads(self) -> "TransformerConfig":
        if self.d_model % self.heads:
            raise ValueError("transformer.d_model must be divisible by transformer.heads")
        return self


class DeepConfig(_Strict):
    training: TrainingConfig = TrainingConfig()
    lstm: LSTMConfig = LSTMConfig()
    transformer: TransformerConfig = TransformerConfig()


class TrackingConfig(_Strict):
    mlflow_uri: str | None = Field(
        default=None,
        description="Log every evaluation to this MLflow tracking URI (e.g. file:./mlruns). "
        "Requires the optional `tracking` extra.",
    )
    experiment: str = "sathealth"


class DecisionConfig(_Strict):
    """Decision layer on saved detector scores; see docs/adr/0006-decision-layer.md.

    Every field is label-free. Defaults are the pre-registered primary configuration.
    """

    name: str = Field(default="fused", pattern=r"^[a-z0-9_+-]+$")
    inputs: tuple[str, ...] = Field(default=("lstm", "transformer"), min_length=1)
    calibration: Literal["evt", "ecdf", "none"] = "evt"
    tail_quantile: float = Field(
        default=0.95, gt=0.5, lt=1.0, description="POT threshold u on validation scores."
    )
    max_surprise: float = Field(default=12.0, gt=0, description="Cap on -log10 p.")
    recalibrate_every: int | None = Field(
        default=None,
        ge=1,
        description="Causal recalibration on recent non-alarm test scores every N steps "
        "(streaming EVT, post-hoc variant); None keeps the validation calibration.",
    )
    recalibrate_window: int = Field(
        default=2000, ge=50, description="Most recent nominal scores kept as reference."
    )
    fusion: Literal["mean", "max", "min"] = "mean"
    threshold: Literal["hysteresis", "ndt"] = "hysteresis"
    alarm_on: float = Field(default=3.0, gt=0, description="Enter alarm at surprise >= this.")
    alarm_off: float = Field(default=2.0, gt=0, description="Leave alarm below this.")
    merge_gap: int = Field(default=10, ge=0, description="Merge alarm runs this close.")
    min_duration: int = Field(default=1, ge=1, description="Drop shorter alarm runs.")
    prune: float | None = Field(
        default=None,
        ge=0,
        lt=1,
        description="Telemanom pruning: drop alarm runs whose peak is within this relative "
        "margin of the next lower peak (0.13 in Hundman et al. 2018). Batch, not live.",
    )
    ndt_window: int = Field(default=2100, ge=10, description="NDT window (steps).")
    ndt_step: int = Field(default=70, ge=1, description="NDT window stride (steps).")
    severity_warning: float = Field(default=4.5, gt=0)
    severity_critical: float = Field(default=6.0, gt=0)
    point_margin: float = Field(
        default=0.05, ge=0, description="Margin on the train range for the point-type rule."
    )
    alarm_budget: float = Field(
        default=1.0, gt=0, description="Allowed false events per 1,000 steps (reported)."
    )

    @model_validator(mode="after")
    def _check_levels(self) -> "DecisionConfig":
        if self.alarm_off > self.alarm_on:
            raise ValueError("decision.alarm_off must be <= decision.alarm_on")
        if self.severity_warning > self.severity_critical:
            raise ValueError("decision.severity_warning must be <= decision.severity_critical")
        return self


class ExplainConfig(_Strict):
    """Explanations of alarm events; see docs/adr/0007-explainability.md (pre-registered)."""

    onset_level: float = Field(default=1.0, gt=0, description="Surprise the onset walks back on.")
    onset_lookback: int = Field(default=500, ge=1)
    horizon: int = Field(default=250, ge=10, description="Counterfactual rollout length.")
    band_level: float = Field(default=0.9, gt=0.5, lt=1.0, description="Counterfactual band.")
    band_starts: int = Field(default=50, ge=5, description="Rollouts used to calibrate the band.")
    ig_steps: int = Field(default=64, ge=8, description="Integrated Gradients path steps.")
    verify_tolerance: float = Field(
        default=1e-4, gt=0, description="Max |rebuilt - saved| validation score difference."
    )
    llm: bool = Field(
        default=False,
        description="Also reword each narrative with Claude; kept only if it passes the "
        "grounding check. Needs the `llm` extra and Anthropic credentials.",
    )
    llm_model: str = "claude-opus-5-5"
    eval_channels_per_spacecraft: int = Field(default=5, ge=1)
    eval_injections: int = Field(default=20, ge=1)
    eval_spike: float = Field(default=3.0, gt=0, description="Injected value spike (scaled).")
    eval_min_effect: float = Field(
        default=0.05, ge=0, description="Command injections must move the forecast this much."
    )
    eval_deletion_fraction: float = Field(default=0.05, gt=0, lt=1)
    eval_coverage_horizon: int = Field(default=100, ge=10)


class ServiceConfig(_Strict):
    """Operations service; see docs/adr/0008-operations-service.md."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    db_path: Path = Path("runs/service/sathealth.db")
    channels: tuple[str, ...] | None = Field(
        default=None, description="Channels to replay; None means every prepared channel."
    )
    speed: float = Field(default=20.0, gt=0, description="Replayed steps per second.")
    tick_s: float = Field(default=0.1, gt=0, le=5, description="Replay scheduling interval.")
    health_window: int = Field(default=500, ge=10)
    health_sample_every: int = Field(default=50, ge=1)
    health_threshold: float = Field(default=50.0, gt=0, lt=100)
    show_labels: bool = Field(
        default=False,
        description="Expose NASA's anomaly labels in the telemetry API (evaluation and demos "
        "only: a live mission has no labels).",
    )


class Config(_Strict):
    seed: int = 42
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    paths: PathsConfig = PathsConfig()
    dataset: DatasetConfig = DatasetConfig()
    download: DownloadConfig = DownloadConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    windows: WindowConfig = WindowConfig()
    validation: ValidationConfig = ValidationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    baselines: BaselinesConfig = BaselinesConfig()
    deep: DeepConfig = DeepConfig()
    tracking: TrackingConfig = TrackingConfig()
    decision: DecisionConfig = DecisionConfig()
    explain: ExplainConfig = ExplainConfig()
    service: ServiceConfig = ServiceConfig()

    @model_validator(mode="after")
    def _check_columns(self) -> "Config":
        if len(set(self.preprocess.value_columns)) != len(self.preprocess.value_columns):
            raise ValueError("preprocess.value_columns contains duplicates")
        if any(c < 0 for c in self.preprocess.value_columns):
            raise ValueError("preprocess.value_columns must be non-negative")
        return self

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (paths as strings) for manifests and fingerprints."""
        return self.model_dump(mode="json")


def _apply_override(tree: dict[str, Any], dotted: str) -> None:
    if "=" not in dotted:
        raise ValueError(f"override must look like key.sub=value, got {dotted!r}")
    key, raw = dotted.split("=", 1)
    parts = key.strip().split(".")
    node = tree
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot override {key!r}: {part!r} is not a section")
        node = child
    node[parts[-1]] = yaml.safe_load(raw)  # "128" -> 128, "[SMAP]" -> ["SMAP"], "null" -> None


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load YAML (or defaults when ``path`` is None) and apply ``key.sub=value`` overrides."""
    tree: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}: top level must be a mapping")
        tree = loaded
    for item in overrides or []:
        _apply_override(tree, item)
    return Config.model_validate(tree)
