import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from sathealth.pipeline.tracking import log_to_mlflow, run_metrics


def _summary() -> dict[str, Any]:
    numbers = {
        "event": {"f1": 0.5},
        "composite_f1": 0.6,
        "event_f1_macro": 0.4,
        "average_precision_macro": 0.3,
        "fp_events_per_1k_steps": 1.2,
    }
    return {
        "detector": "lstm",
        "data_run": "telemanom-abc",
        "git": {"commit": None},
        "params": {"window": 64, "training": {"epochs": 35}},
        "runtime_s": 12.0,
        "subsets": {"all": numbers, "excluding a+b": numbers},
    }


def test_run_metrics_names_are_mlflow_safe() -> None:
    metrics = run_metrics(_summary())
    assert metrics["all.composite_f1"] == 0.6
    assert metrics["excluding_a_b.event_f1"] == 0.5


class _FakeMlflow(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("mlflow")
        self.calls: dict[str, Any] = {"artifacts": []}

    def set_tracking_uri(self, uri: str) -> None:
        self.calls["uri"] = uri

    def set_experiment(self, name: str) -> None:
        self.calls["experiment"] = name

    def start_run(self, run_name: str) -> Any:
        self.calls["run_name"] = run_name
        run = types.SimpleNamespace(info=types.SimpleNamespace(run_id="r1"))

        class _Ctx:
            def __enter__(self) -> Any:
                return run

            def __exit__(self, *exc: object) -> None:
                return None

        return _Ctx()

    def set_tags(self, tags: dict[str, str]) -> None:
        self.calls["tags"] = tags

    def log_params(self, params: dict[str, Any]) -> None:
        self.calls["params"] = params

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.calls["metrics"] = metrics

    def log_artifact(self, path: str) -> None:
        self.calls["artifacts"].append(Path(path).name)


def test_log_to_mlflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    (tmp_path / "summary.json").write_text(json.dumps(_summary()))
    assert log_to_mlflow(_summary(), tmp_path, "file:./mlruns", "exp") == "r1"
    assert fake.calls["uri"] == "file:./mlruns"
    assert fake.calls["params"] == {"window": 64, "training.epochs": 35}
    assert fake.calls["tags"]["git_commit"] == "unknown"
    assert fake.calls["metrics"]["runtime_s"] == 12.0
    assert fake.calls["artifacts"] == ["summary.json"]


def test_missing_mlflow_is_a_clear_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)  # makes `import mlflow` raise ImportError
    with pytest.raises(RuntimeError, match="extra tracking"):
        log_to_mlflow(_summary(), tmp_path, "file:./mlruns", "exp")
