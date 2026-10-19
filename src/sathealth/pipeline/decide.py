"""``decide``: run the decision layer on saved detector scores (no retraining).

Per channel: calibrate each input detector's scores on its validation scores (EVT) -> fuse
the surprises -> alarm rule -> event builder -> metrics against labels. Labels are only used
to score the result and for the global oracle, which is reported but never an operating point.

Output (one directory per decision config + input fingerprint, written atomically)::

    runs/decisions/<name>-<fingerprint>/
        summary.json         same schema as an evaluation summary, plus decision diagnostics
        per_channel.json     counts, metrics and calibration parameters per channel
        events/<chan>.json   operator-facing events: span, peak, severity, type, evidence
        scores/<chan>.npz    fused test surprise and alarm mask, input to Phase 5
"""

import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sathealth.config import Config, DecisionConfig
from sathealth.data.schema import BoolArray, FloatArray
from sathealth.decision.adaptive import adaptive_surprises
from sathealth.decision.alarms import drop_short, hysteresis, merge_gaps, ndt_mask, prune
from sathealth.decision.calibration import TailCalibrator
from sathealth.decision.events import Event, build_events, outside_range
from sathealth.decision.fusion import fuse
from sathealth.evaluation.aggregate import ChannelResult, summarize_subsets
from sathealth.evaluation.metrics import (
    EventCounts,
    PointCounts,
    evaluate_predictions,
    event_counts,
    f1,
    oracle_event_f1,
    point_counts,
)
from sathealth.logging import get_logger
from sathealth.pipeline.evaluate import SUMMARY_FILE, channel_flags, evaluation_dir_for
from sathealth.pipeline.prepare import load_prepared, load_run_manifest, run_dir_for
from sathealth.utils.hashing import fingerprint
from sathealth.utils.provenance import provenance

log = get_logger(__name__)

GLOBAL_ORACLE_LEVELS = np.round(np.arange(0.5, 12.01, 0.25), 2)


@dataclass(frozen=True)
class DecisionResult:
    run_dir: Path
    name: str
    summary: dict[str, Any]
    reused: bool


def decision_fingerprint(config: Config) -> str:
    d = config.decision
    return fingerprint(
        {
            "inputs": {name: evaluation_dir_for(config, name).name for name in d.inputs},
            "decision": d.model_dump(mode="json", exclude={"name", "alarm_budget"}),
            "evaluation": config.evaluation.model_dump(mode="json"),
        }
    )


def decision_dir_for(config: Config) -> Path:
    return config.paths.decisions_dir / f"{config.decision.name}-{decision_fingerprint(config)}"


def describe(d: DecisionConfig) -> str:
    inputs = " + ".join(d.inputs)
    fusion = f", {d.fusion} fusion" if len(d.inputs) > 1 else ""
    rule = (
        f"hysteresis {d.alarm_on:g}/{d.alarm_off:g}"
        if d.threshold == "hysteresis"
        else "Telemanom dynamic threshold"
    )
    extras = (
        (f", recalibrated every {d.recalibrate_every}" if d.recalibrate_every else "")
        + (f", pruning {d.prune:g}" if d.prune is not None else "")
        + (f", merge ≤ {d.merge_gap}" if d.merge_gap else "")
    )
    return f"Decision layer: {d.calibration.upper()} on {inputs}{fusion}, {rule}{extras}"


def _load_scores(config: Config, detector: str, chan_id: str) -> tuple[FloatArray, FloatArray]:
    path = evaluation_dir_for(config, detector) / "scores" / f"{chan_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"no saved scores for {detector!r} under the current config; "
            f"run `sathealth evaluate {detector}`"
        )
    with np.load(path) as npz:
        return npz["val"].astype(np.float64), npz["test"].astype(np.float64)


def alarm_mask(surprise: FloatArray, d: DecisionConfig) -> BoolArray:
    """The label-free alarm rule: threshold, optional pruning, then event clean-up."""
    if d.threshold == "hysteresis":
        mask = hysteresis(surprise, d.alarm_on, d.alarm_off)
    else:
        mask = ndt_mask(surprise, d.ndt_window, d.ndt_step)
    if d.prune is not None:
        mask = prune(mask, surprise, d.prune)
    return drop_short(merge_gaps(mask, d.merge_gap), d.min_duration)


@dataclass(frozen=True)
class _Channel:
    result: ChannelResult
    surprise: FloatArray
    mask: BoolArray
    events: list[Event]
    calibration: dict[str, Any]
    nominal_alarm_fraction: float  # label-negative test steps in alarm (diagnostic)
    kinds: list[tuple[str, str]]  # (labelled kind, predicted kind) per detected segment
    labels: BoolArray
    segments: list[tuple[int, int]]


def _decide_channel(config: Config, data_dir: Path, chan_id: str, meta: dict[str, Any]) -> _Channel:
    d, ecfg = config.decision, config.evaluation
    channel = load_prepared(data_dir, chan_id)
    vals, tests, calibration = [], [], {}
    for name in d.inputs:
        val, test = _load_scores(config, name, chan_id)
        vals.append(val)
        tests.append(test)
        cal = TailCalibrator(d.calibration, d.tail_quantile, d.max_surprise).fit(val)
        calibration[name] = cal.to_dict()
    if d.recalibrate_every is None:
        scored = [
            TailCalibrator(d.calibration, d.tail_quantile, d.max_surprise).fit(v).surprise(t)
            for v, t in zip(vals, tests, strict=True)
        ]
    else:
        scored = adaptive_surprises(vals, tests, d, d.recalibrate_every)
    surprises = dict(zip(d.inputs, scored, strict=True))
    fused = fuse(scored, d.fusion)
    mask = alarm_mask(fused, d)

    cols = list(config.preprocess.value_columns)
    train_values = channel.train[:, cols]
    oor = outside_range(
        channel.test[:, cols], train_values.min(axis=0), train_values.max(axis=0), d.point_margin
    )
    events = build_events(
        mask, fused, surprises, oor, warning=d.severity_warning, critical=d.severity_critical
    )
    kinds: list[tuple[str, str]] = []
    for seg in channel.segments:
        hits = [e for e in events if e.start <= seg.end and e.end >= seg.start]
        if hits:
            top = max(hits, key=lambda e: e.peak_surprise)
            kinds.append((str(seg.kind), top.kind))

    segments = [(s.start, s.end) for s in channel.segments]
    metrics = evaluate_predictions(
        mask,
        channel.test_labels,
        fused,
        segments,
        early_tolerance=ecfg.early_tolerance,
        min_gap=ecfg.min_event_gap,
    )
    oracle, _ = oracle_event_f1(
        fused,
        segments,
        ecfg.oracle_thresholds,
        early_tolerance=ecfg.early_tolerance,
        min_gap=ecfg.min_event_gap,
    )
    result = ChannelResult(
        chan_id,
        meta["spacecraft"],
        metrics,
        oracle,
        d.alarm_on,
        d.calibration,
        channel_flags(meta),
    )
    return _Channel(
        result,
        fused,
        mask,
        events,
        calibration,
        float(mask[~channel.test_labels].mean()) if (~channel.test_labels).any() else 0.0,
        kinds,
        channel.test_labels,
        segments,
    )


def global_oracle(
    surprises: dict[str, FloatArray],
    labels: dict[str, BoolArray],
    segments: dict[str, list[tuple[int, int]]],
) -> dict[str, float]:
    """Best composite F1 over one surprise level shared by *all* channels (uses labels).

    One level for every channel cannot adapt to individual channels, so unlike the
    per-channel oracle this is a meaningful upper bound for a calibrated pipeline.
    """
    best = {"level": float("nan"), "composite_f1": 0.0}
    for level in GLOBAL_ORACLE_LEVELS:
        points, events = PointCounts(0, 0, 0, 0), EventCounts(0, 0, 0)
        for chan, s in surprises.items():
            pred = s >= level
            points += point_counts(pred, labels[chan])
            events += event_counts(pred, segments[chan])
        score = f1(points.precision, events.recall)
        if score > best["composite_f1"]:
            best = {"level": float(level), "composite_f1": score}
    return best


def _kind_agreement(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0}
    labelled = [a for a, _ in pairs]
    majority = max(set(labelled), key=labelled.count)
    return {
        "n": len(pairs),
        "accuracy": sum(a == b for a, b in pairs) / len(pairs),
        "majority_baseline": labelled.count(majority) / len(pairs),
        "confusion": {
            f"{a}->{b}": sum(1 for x, y in pairs if (x, y) == (a, b))
            for a in ("point", "contextual")
            for b in ("point", "contextual")
        },
    }


def decide(config: Config, *, force: bool = False) -> DecisionResult:
    d = config.decision
    out_dir = decision_dir_for(config)
    if (out_dir / SUMMARY_FILE).is_file() and not force:
        log.info("reusing decision run", name=d.name, run_dir=out_dir)
        summary = json.loads((out_dir / SUMMARY_FILE).read_text())
        return DecisionResult(out_dir, d.name, summary, reused=True)

    data_dir = run_dir_for(config)
    if not (data_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"no prepared data at {data_dir}; run `sathealth prepare`")
    channels_meta: dict[str, Any] = load_run_manifest(data_dir)["channels"]

    config.paths.decisions_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{d.name}-", dir=config.paths.decisions_dir))
    try:
        (tmp / "events").mkdir()
        (tmp / "scores").mkdir()
        started = time.perf_counter()
        results: list[ChannelResult] = []
        per_channel: list[dict[str, Any]] = []
        kinds: list[tuple[str, str]] = []
        surprises: dict[str, FloatArray] = {}
        labels: dict[str, BoolArray] = {}
        segments: dict[str, list[tuple[int, int]]] = {}
        severities = {"watch": 0, "warning": 0, "critical": 0}
        nominal_alarm: list[float] = []
        alarm_fraction: list[float] = []
        for chan_id, meta in channels_meta.items():
            ch = _decide_channel(config, data_dir, chan_id, meta)
            results.append(ch.result)
            per_channel.append(
                {
                    **ch.result.to_dict(),
                    "calibration": ch.calibration,
                    "alarm_fraction": float(ch.mask.mean()),
                    "nominal_alarm_fraction": ch.nominal_alarm_fraction,
                    "n_alarm_events": len(ch.events),
                }
            )
            kinds += ch.kinds
            nominal_alarm.append(ch.nominal_alarm_fraction)
            alarm_fraction.append(float(ch.mask.mean()))
            for event in ch.events:
                severities[event.severity] += 1
            surprises[chan_id] = ch.surprise
            labels[chan_id] = ch.labels
            segments[chan_id] = ch.segments
            (tmp / "events" / f"{chan_id}.json").write_text(
                json.dumps([e.to_dict() for e in ch.events], indent=2)
            )
            np.savez_compressed(
                tmp / "scores" / f"{chan_id}.npz",
                surprise=ch.surprise.astype(np.float32),
                alarm=ch.mask,
            )

        subsets = summarize_subsets(
            results,
            exclude_subsets=config.evaluation.exclude_subsets,
            bootstrap=config.evaluation.bootstrap,
            seed=config.seed,
        )
        overall = subsets["all"]
        summary = {
            "detector": d.name,
            "kind": "decision",
            "description": describe(d),
            "params": d.model_dump(mode="json"),
            "inputs": {name: evaluation_dir_for(config, name).name for name in d.inputs},
            "evaluation": config.evaluation.model_dump(mode="json"),
            "data_run": data_dir.name,
            **provenance(),
            "runtime_s": round(time.perf_counter() - started, 3),
            "jobs": 1,
            "subsets": subsets,
            "within_budget": overall["fp_events_per_1k_steps"] <= d.alarm_budget,
            "alarm_budget": d.alarm_budget,
            # label-free: share of test steps in alarm (a calibration that transfers keeps
            # this near the designed rate plus the true anomaly share)
            "alarm_fraction_mean": float(np.mean(alarm_fraction)),
            # uses labels, diagnostic only: alarms on steps labelled nominal
            "nominal_alarm_fraction_mean": float(np.mean(nominal_alarm)),
            "global_oracle": global_oracle(surprises, labels, segments),
            "severity_counts": severities,
            "kind_agreement": _kind_agreement(kinds),
        }
        (tmp / "per_channel.json").write_text(json.dumps(per_channel, indent=2))
        (tmp / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp.rename(out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    log.info(
        "decision finished",
        name=d.name,
        composite_f1=round(overall["composite_f1"], 3),
        fp_events_per_1k=round(overall["fp_events_per_1k_steps"], 3),
        within_budget=summary["within_budget"],
        run_dir=out_dir,
    )
    return DecisionResult(out_dir, d.name, summary, reused=False)
