"""``explain``: turn every alarm event of a decision run into an explanation.

Per channel with events: rebuild the LSTM and verify it reproduces the saved validation
scores -> calibrate the counterfactual band on validation rollouts -> per event: onset,
counterfactual, Integrated Gradients at the peak, grounded narrative.

Output::

    runs/explanations/<decision run>/
        index.json               one row per event: channel, span, severity, headline
        <chan>/event-<k>.json    every computed value, the facts and the sentences
        <chan>/event-<k>.md      the narrative for an operator
        <chan>/event-<k>.npz     counterfactual arrays (actual, expected, band) for plots
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from sathealth.config import Config
from sathealth.data.schema import FloatArray
from sathealth.detectors.lstm import LSTMDetector
from sathealth.detectors.registry import build_detector
from sathealth.explain.attribution import Attribution, forecast_error_fn, integrated_gradients
from sathealth.explain.counterfactual import HorizonBand, calibrate_band, rollout
from sathealth.explain.llm import default_client as llm_default_client
from sathealth.explain.llm import reword
from sathealth.explain.narrative import narrate, ungrounded
from sathealth.explain.onset import estimate_onset
from sathealth.logging import get_logger
from sathealth.pipeline.decide import decision_dir_for
from sathealth.pipeline.evaluate import evaluation_dir_for
from sathealth.pipeline.prepare import PreparedChannel, load_prepared, run_dir_for
from sathealth.utils.parallel import parallel_map

log = get_logger(__name__)

TRAIN_FALLBACK_ROWS = 1500


def model_path(config: Config, chan_id: str) -> Path:
    return config.paths.models_dir / evaluation_dir_for(config, "lstm").name / f"{chan_id}.pt"


def fitted_lstm(config: Config, channel: PreparedChannel) -> LSTMDetector:
    """The channel's Phase 3 LSTM: loaded from cache, or retrained and verified.

    Training is bit-reproducible, so a retrained model must reproduce the validation scores
    saved by ``sathealth evaluate lstm``. If it does not, we would be explaining a different
    model from the one that raised the alarm, so this raises instead.
    """
    det = build_detector("lstm", config)
    if not isinstance(det, LSTMDetector):  # pragma: no cover - registry invariant
        raise TypeError("registry returned a non-LSTM detector for 'lstm'")
    path = model_path(config, channel.chan_id)
    if path.is_file():
        return det.load(path)
    saved = evaluation_dir_for(config, "lstm") / "scores" / f"{channel.chan_id}.npz"
    if not saved.is_file():
        raise FileNotFoundError("no saved LSTM scores; run `sathealth evaluate lstm` first")
    det.fit(channel.train, channel.val)
    with np.load(saved) as npz:
        reference = npz["val"].astype(np.float64)
    rebuilt = det.score(channel.val).astype(np.float32).astype(np.float64)
    gap = float(np.max(np.abs(rebuilt - reference))) if reference.size else 0.0
    if gap > config.explain.verify_tolerance:
        raise RuntimeError(
            f"{channel.chan_id}: rebuilt LSTM differs from the evaluated one (max gap {gap:.2e})"
        )
    det.save(path)
    return det


def channel_band(
    config: Config, det: LSTMDetector, channel: PreparedChannel, rng: np.random.Generator
) -> HorizonBand:
    """Counterfactual band calibrated on validation rollouts (train tail if val is short)."""
    e = config.explain
    tail = (1 - e.band_level) / 2
    for series, source in (
        (channel.val, "val"),
        (channel.train[-TRAIN_FALLBACK_ROWS:], "train"),
    ):
        try:
            return calibrate_band(
                det.model,
                series,
                det.value_columns,
                det.window,
                e.horizon,
                e.band_starts,
                rng,
                source=source,
                quantiles=(tail, 1 - tail),
            )
        except ValueError:
            continue
    raise ValueError(f"{channel.chan_id}: too little nominal data to calibrate a band")


def _unscale(values: FloatArray, channel: PreparedChannel) -> FloatArray:
    p = channel.scaler
    result: FloatArray = values * np.asarray(p.scale) + np.asarray(p.center)
    return result


def feature_name(column: int, value_columns: Sequence[int]) -> str:
    if column in value_columns:
        return "the channel's own recent values"
    return f"command {column}"


def attribute_peak(
    config: Config, det: LSTMDetector, channel: PreparedChannel, step: int
) -> Attribution:
    """IG of the squared forecast error at ``step`` w.r.t. the input window before it."""
    window = channel.test[max(0, step - det.window) : step]
    actual = channel.test[step, list(det.value_columns)]
    f = forecast_error_fn(det.model, actual)
    return integrated_gradients(f, window, steps=config.explain.ig_steps)


def explain_event(
    config: Config,
    det: LSTMDetector,
    band: HorizonBand,
    channel: PreparedChannel,
    event: dict[str, Any],
    surprise: FloatArray,
    *,
    llm_client: Any = None,
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    e = config.explain
    cols = list(det.value_columns)
    onset = estimate_onset(
        surprise, event["start"], level=e.onset_level, max_lookback=e.onset_lookback
    )
    explanation: dict[str, Any] = {
        "chan_id": channel.chan_id,
        "spacecraft": channel.spacecraft,
        "event": event,
        "onset": onset,
        "model_check": {"verified_against": "saved validation scores"},
    }
    arrays: dict[str, FloatArray] = {}

    horizon = min(e.horizon, channel.test.shape[0] - onset)
    if onset > 0 and horizon > 0:
        history = channel.test[max(0, onset - det.window) : onset]
        future = channel.test[onset : onset + horizon]
        expected = rollout(det.model, history[None], future[None], det.value_columns, det.window)
        expected = expected[0]
        low_off, high_off = band.at(horizon)
        lower, upper = expected + low_off, expected + high_off
        actual = future[:, cols]
        span = slice(0, min(event["end"], onset + horizon - 1) - onset + 1)
        outside = ((actual < lower) | (actual > upper)).any(axis=1)[span]
        deviation = np.abs(actual - expected)[span].max(axis=1)
        worst = int(np.argmax(deviation))
        units = {
            "actual": _unscale(actual, channel),
            "expected": _unscale(expected, channel),
            "lower": _unscale(lower, channel),
            "upper": _unscale(upper, channel),
        }
        explanation["counterfactual"] = {
            "start": onset,
            "horizon": int(span.stop),
            "band_level": e.band_level,
            "band_source": band.source,
            "expected_low": float(units["lower"][span].min()),
            "expected_high": float(units["upper"][span].max()),
            "actual_extreme": float(units["actual"][span][worst, 0]),
            "outside_fraction": float(outside.mean()),
        }
        arrays |= units

    if event["peak_step"] > 0:
        attr = attribute_peak(config, det, channel, event["peak_step"])
        importance = attr.feature_importance()
        total = float(importance.sum()) or 1.0
        explanation["attribution"] = {
            "method": "integrated_gradients",
            "target": "squared one-step forecast error at the peak",
            "completeness_gap": attr.completeness_gap,
            "top_features": [
                {
                    "column": c,
                    "name": feature_name(c, cols),
                    "share": float(importance[c] / total),
                }
                for c in attr.top_features(3)
            ],
            "top_lags": attr.top_lags(3),
        }
        arrays["lag_importance"] = attr.lag_importance()

    sentences, facts = narrate(explanation)
    explanation["facts"] = facts
    explanation["sentences"] = [{"text": s.text, "facts": list(s.facts)} for s in sentences]
    explanation["grounding_violations"] = ungrounded(sentences, facts)
    if e.llm:
        brief = reword(sentences, facts, model=e.llm_model, client=llm_client)
        if brief is not None:
            explanation["llm_brief"] = {
                "text": brief.text if brief.grounded else None,
                "grounded": brief.grounded,
                "invented_numbers": list(brief.invented),
                "model": brief.model,
            }
    return explanation, arrays


def to_markdown(explanation: dict[str, Any]) -> str:
    head = explanation["sentences"][0]["text"]
    body = "\n".join(f"- {s['text']}" for s in explanation["sentences"][1:])
    brief = (explanation.get("llm_brief") or {}).get("text")
    extra = f"\n**Summary** (reworded by an LLM, grounding-checked): {brief}\n" if brief else ""
    return f"### {head}\n\n{body}\n{extra}"


@dataclass(frozen=True)
class _ChannelJob:
    chan_id: str
    events: list[dict[str, Any]]


def _explain_channel(
    config: Config, decision_dir: Path, out_dir: Path, job: _ChannelJob
) -> list[dict[str, Any]]:
    """Explain one channel's events; top-level so worker processes can run it."""
    channel = load_prepared(run_dir_for(config), job.chan_id)
    det = fitted_lstm(config, channel)
    rng = np.random.default_rng([config.seed, int.from_bytes(job.chan_id.encode(), "little")])
    band = channel_band(config, det, channel, rng)
    with np.load(decision_dir / "scores" / f"{job.chan_id}.npz") as npz:
        surprise = npz["surprise"].astype(np.float64)
    llm_client = llm_default_client() if config.explain.llm else None
    rows = []
    chan_dir = out_dir / job.chan_id
    chan_dir.mkdir(parents=True, exist_ok=True)
    for k, event in enumerate(job.events):
        explanation, arrays = explain_event(
            config, det, band, channel, event, surprise, llm_client=llm_client
        )
        stem = chan_dir / f"event-{k}"
        stem.with_suffix(".json").write_text(json.dumps(explanation, indent=2))
        stem.with_suffix(".md").write_text(to_markdown(explanation))
        if arrays:
            np.savez_compressed(stem.with_suffix(".npz"), **arrays)  # type: ignore[arg-type]
        rows.append(
            {
                "chan_id": job.chan_id,
                "event": k,
                "start": event["start"],
                "end": event["end"],
                "severity": event["severity"],
                "kind": event["kind"],
                "peak_surprise": event["peak_surprise"],
                "headline": explanation["sentences"][0]["text"],
                "grounded": not explanation["grounding_violations"],
            }
        )
    return rows


def explain_run(config: Config, *, channels: Sequence[str] | None = None, jobs: int = 1) -> Path:
    """Explain every event of the configured decision run (optionally only ``channels``)."""
    decision_dir = decision_dir_for(config)
    if not (decision_dir / "summary.json").is_file():
        raise FileNotFoundError(f"no decision run at {decision_dir}; run `sathealth decide`")
    out_dir = config.paths.explanations_dir / decision_dir.name
    work = []
    for path in sorted((decision_dir / "events").glob("*.json")):
        events = json.loads(path.read_text())
        if events and (channels is None or path.stem in channels):
            work.append(_ChannelJob(path.stem, events))
    index = [
        row
        for rows in parallel_map(
            partial(_explain_channel, config, decision_dir, out_dir), work, jobs
        )
        for row in rows
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    log.info(
        "explanations written",
        events=len(index),
        channels=len(work),
        grounded=sum(r["grounded"] for r in index),
        out_dir=out_dir,
    )
    return out_dir
