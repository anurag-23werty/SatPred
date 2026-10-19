"""Optional MLflow experiment tracking.

The run directory under ``runs/evaluations/`` is the source of truth and needs no server.
When ``tracking.mlflow_uri`` is set, each finished evaluation is also logged to MLflow, to
compare runs in its UI: detector parameters, headline metrics for every subset, and the run
directory's JSON files as artifacts. MLflow is an optional extra
(``uv sync --extra tracking``), imported only when tracking is switched on.
"""

import importlib
from pathlib import Path
from typing import Any

from sathealth.logging import get_logger

log = get_logger(__name__)

METRICS = ("composite_f1", "event_f1_macro", "average_precision_macro", "fp_events_per_1k_steps")


def _flatten(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def run_metrics(summary: dict[str, Any]) -> dict[str, float]:
    """``{"<subset>.<metric>": value}`` with MLflow-safe names (no spaces or ``+``)."""
    out: dict[str, float] = {}
    for subset, numbers in summary["subsets"].items():
        safe = subset.replace(" ", "_").replace("+", "_")
        out[f"{safe}.event_f1"] = float(numbers["event"]["f1"])
        for key in METRICS:
            out[f"{safe}.{key}"] = float(numbers[key])
    return out


def log_to_mlflow(summary: dict[str, Any], run_dir: Path, uri: str, experiment: str) -> str:
    """Log one evaluation as an MLflow run and return its run id."""
    try:
        mlflow = importlib.import_module("mlflow")
    except ImportError as exc:
        raise RuntimeError(
            "tracking.mlflow_uri is set but MLflow is not installed; run `uv sync --extra tracking`"
        ) from exc
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_dir.name) as run:
        mlflow.set_tags(
            {
                "detector": summary["detector"],
                "data_run": summary["data_run"],
                "git_commit": summary["git"]["commit"] or "unknown",
            }
        )
        mlflow.log_params(_flatten(summary["params"]))
        mlflow.log_metrics({**run_metrics(summary), "runtime_s": float(summary["runtime_s"])})
        for path in sorted(run_dir.glob("*.json")):
            mlflow.log_artifact(str(path))
        run_id: str = run.info.run_id
    log.info("logged to mlflow", uri=uri, experiment=experiment, run_id=run_id)
    return run_id
