import json
from pathlib import Path

import numpy as np
import pytest

from sathealth.config import Config, DecisionConfig, load_config
from sathealth.data.download import Manifest
from sathealth.decision.events import outside_range
from sathealth.ops.streaming import StreamingDecider
from sathealth.pipeline.decide import decide
from sathealth.pipeline.evaluate import evaluate_detector, evaluation_dir_for
from sathealth.pipeline.prepare import load_prepared, prepare_dataset, run_dir_for


@pytest.fixture
def config(raw_root: Path, raw_manifest: Manifest, tmp_path: Path) -> Config:
    cfg = load_config(
        None,
        overrides=[
            f"paths.raw_dir={raw_root}",
            f"paths.processed_dir={tmp_path / 'processed'}",
            f"paths.runs_dir={tmp_path / 'runs'}",
            f"paths.decisions_dir={tmp_path / 'decisions'}",
            "evaluation.bootstrap=10",
            "decision.inputs=[zscore, limit]",
        ],
    )
    prepare_dataset(cfg, manifest=raw_manifest)
    for name in ("zscore", "limit"):
        evaluate_detector(cfg, name)
    return cfg


def _stream(cfg: Config, chan: str) -> tuple[list[dict[str, object]], np.ndarray]:
    d = cfg.decision
    channel = load_prepared(run_dir_for(cfg), chan)
    scores = {k: np.load(evaluation_dir_for(cfg, k) / "scores" / f"{chan}.npz") for k in d.inputs}
    val = {k: v["val"].astype(np.float64) for k, v in scores.items()}
    test = {k: v["test"].astype(np.float64) for k, v in scores.items()}
    train = channel.train[:, [0]]
    oor = outside_range(channel.test[:, [0]], train.min(0), train.max(0), d.point_margin)
    decider = StreamingDecider(d, val)
    events = []
    for t in range(channel.test.shape[0]):
        result = decider.step({k: test[k][t] for k in d.inputs}, bool(oor[t]))
        if result.closed:
            events.append(result.closed.to_dict())
    if (last := decider.finish()) is not None:
        events.append(last.to_dict())
    return events, decider.fused


@pytest.mark.parametrize("every", [None, 50])
def test_streaming_reproduces_batch_decisions_exactly(config: Config, every: int | None) -> None:
    cfg = config.model_copy(
        update={
            "decision": config.decision.model_copy(
                update={"recalibrate_every": every, "recalibrate_window": 200, "merge_gap": 5}
            )
        }
    )
    run = decide(cfg).run_dir
    for chan in ("A-1", "C-1", "M-1"):
        events, fused = _stream(cfg, chan)
        assert events == json.loads((run / "events" / f"{chan}.json").read_text())
        with np.load(run / "scores" / f"{chan}.npz") as npz:
            np.testing.assert_array_equal(fused.astype(np.float32), npz["surprise"])


def _decider(**kw: object) -> StreamingDecider:
    d = DecisionConfig(inputs=("a",), calibration="none", **kw)  # type: ignore[arg-type]
    return StreamingDecider(d, {"a": np.zeros(10)})


def test_event_lifecycle_with_merge_gap() -> None:
    sd = _decider(alarm_on=3, alarm_off=3, merge_gap=2)
    scores = [0, 5, 5, 0, 0, 6, 0, 0, 0, 0]
    results = [sd.step({"a": s}) for s in scores]
    opened = [r.step for r in results if r.opened]
    closed = [(r.step, r.closed) for r in results if r.closed]
    assert opened == [1], "one alert, at the first alarm step"
    assert len(closed) == 1
    step, event = closed[0]
    assert step == 8, "closed once a gap longer than merge_gap is certain"
    assert (event.start, event.end, event.peak_step) == (1, 5, 5)
    assert sd.finish() is None


def test_min_duration_suppresses_short_blips_and_finish_closes() -> None:
    sd = _decider(alarm_on=3, alarm_off=3, merge_gap=0, min_duration=3)
    results = [sd.step({"a": s}) for s in [5, 0, 5, 5, 5, 5]]
    assert [r.step for r in results if r.opened] == [4]
    assert not any(r.closed for r in results)
    final = sd.finish()
    assert final is not None
    assert (final.start, final.end) == (2, 5)


def test_streaming_rejects_batch_only_rules() -> None:
    with pytest.raises(ValueError, match="hysteresis"):
        _decider(threshold="ndt")
    with pytest.raises(ValueError, match="validation scores"):
        StreamingDecider(DecisionConfig(inputs=("a", "b")), {"a": np.zeros(5)})


def test_rebaseline_ends_an_alarm_on_a_permanent_shift() -> None:
    d = DecisionConfig(inputs=("a",), recalibrate_every=50, recalibrate_window=200, merge_gap=0)
    rng = np.random.default_rng(0)
    decider = StreamingDecider(d, {"a": rng.normal(size=500)})
    for v in rng.normal(size=100):
        decider.step({"a": float(v)})
    shifted = rng.normal(loc=8.0, size=300)
    results = [decider.step({"a": float(v)}) for v in shifted[:150]]
    assert results[-1].alarm, "a permanent level shift stays in alarm (ADR 0006)"
    closed = decider.rebaseline(100)
    assert closed is not None
    assert closed.start == 100
    after = [decider.step({"a": float(v)}) for v in shifted[150:]]
    assert sum(r.alarm for r in after) <= 3, "the new level is the new normal"
    with pytest.raises(ValueError, match="50 steps"):
        StreamingDecider(d, {"a": np.zeros(10)}).rebaseline(100)
