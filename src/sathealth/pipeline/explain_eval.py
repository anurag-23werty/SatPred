"""Evaluate explanations under the pre-registered protocol of ADR 0007.

* E1 onset: is the estimated onset closer to the labelled start than the alarm start?
* E2 attribution faithfulness: do Integrated Gradients find injected causes (hit@3), and does
  deleting the top-attributed inputs reduce the error more than deleting random ones?
* E3 counterfactual calibration: how often does the 90% band contain the true value?
* E4 grounding: does every narrative sentence pass the grounding check?
"""

import json
import math
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from sathealth.config import Config
from sathealth.data.schema import BoolArray
from sathealth.explain.attribution import (
    deletion_effect,
    forecast_error_fn,
    integrated_gradients,
)
from sathealth.explain.counterfactual import rollout
from sathealth.explain.onset import estimate_onset
from sathealth.pipeline.decide import decision_dir_for
from sathealth.pipeline.explain import channel_band, fitted_lstm
from sathealth.pipeline.prepare import load_prepared, load_run_manifest, run_dir_for
from sathealth.utils.parallel import parallel_map


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _bootstrap_ci(
    values: Sequence[float] | np.ndarray, stat: Any, n: int, seed: int
) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, x.size, size=(n, x.size))
    boot = np.array([stat(x[i]) for i in idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


# --- E1 ----------------------------------------------------------------------------------


def evaluate_onset(config: Config) -> dict[str, Any]:
    """Pair each labelled anomaly with the first alarm event overlapping it."""
    e = config.explain
    decision_dir = decision_dir_for(config)
    data_dir = run_dir_for(config)
    rows = []
    for chan_id in load_run_manifest(data_dir)["channels"]:
        events = json.loads((decision_dir / "events" / f"{chan_id}.json").read_text())
        if not events:
            continue
        with np.load(decision_dir / "scores" / f"{chan_id}.npz") as npz:
            surprise = npz["surprise"].astype(np.float64)
        for seg in load_prepared(data_dir, chan_id).segments:
            hits = [ev for ev in events if ev["start"] <= seg.end and ev["end"] >= seg.start]
            if not hits:
                continue
            first = min(hits, key=lambda ev: ev["start"])
            onset = estimate_onset(
                surprise, first["start"], level=e.onset_level, max_lookback=e.onset_lookback
            )
            rows.append(
                {
                    "chan_id": chan_id,
                    "label_start": seg.start,
                    "alarm_error": first["start"] - seg.start,
                    "onset_error": onset - seg.start,
                }
            )
    alarm_abs = np.abs([r["alarm_error"] for r in rows])
    onset_abs = np.abs([r["onset_error"] for r in rows])
    diff = alarm_abs - onset_abs  # > 0: onset is closer
    return {
        "n": len(rows),
        "median_abs_alarm_error": float(np.median(alarm_abs)) if rows else None,
        "median_abs_onset_error": float(np.median(onset_abs)) if rows else None,
        "share_onset_closer": float(np.mean(diff > 0)) if rows else None,
        "share_tied": float(np.mean(diff == 0)) if rows else None,
        "median_improvement_ci": (
            _bootstrap_ci(diff, np.median, config.evaluation.bootstrap, config.seed)
            if len(rows) > 1
            else None
        ),
        "median_signed_onset_error": float(np.median([r["onset_error"] for r in rows]))
        if rows
        else None,
        "rows": rows,
    }


# --- E2 + E3 -----------------------------------------------------------------------------


def evaluation_channels(config: Config) -> list[str]:
    """Fixed rule: first N SMAP and first N MSL channels whose training data has a command."""
    data_dir = run_dir_for(config)
    picked: dict[str, list[str]] = {"SMAP": [], "MSL": []}
    n = config.explain.eval_channels_per_spacecraft
    value_cols = set(config.preprocess.value_columns)
    for chan_id, meta in load_run_manifest(data_dir)["channels"].items():
        craft = meta["spacecraft"]
        if craft not in picked or len(picked[craft]) >= n:
            continue
        train = load_prepared(data_dir, chan_id).train
        commands = [c for c in range(train.shape[1]) if c not in value_cols]
        if commands and train[:, commands].any():
            picked[craft].append(chan_id)
    return picked["SMAP"] + picked["MSL"]


def _clean_starts(labels: BoolArray, before: int, after: int) -> NDArray[np.int64]:
    """Steps s with no labelled anomaly in [s - before, s + after]."""
    bad = np.convolve(labels.astype(int), np.ones(before + after + 1, dtype=int), mode="full")
    # bad[s + after] counts labels in [s - before, s + after]
    s = np.arange(before, labels.size - after, dtype=np.int64)
    clean: NDArray[np.int64] = s[bad[s + after] == 0]
    return clean


def _evaluate_channel(config: Config, item: tuple[int, str]) -> dict[str, Any]:
    index, chan_id = item
    e = config.explain
    channel = load_prepared(run_dir_for(config), chan_id)
    det = fitted_lstm(config, channel)
    model, w, cols = det.model, det.window, list(det.value_columns)
    rng = np.random.default_rng([0, index])
    post_hoc_rng = np.random.default_rng([1, index])  # keeps the pre-registered draws intact
    test, labels = channel.test, channel.test_labels
    n_features = test.shape[1]
    commands = [c for c in range(n_features) if c not in cols]

    peaks = _clean_starts(labels, w, 0)
    temporal, command, ignored, deletion = [], [], 0, []
    temporal_effect: list[float] = []  # post-hoc: how much each spike moved the forecast
    deletion_positive = []  # post-hoc: deletion ranked by signed attribution
    for p in rng.choice(peaks, size=min(e.eval_injections, peaks.size), replace=False):
        actual = test[p, cols]
        f = forecast_error_fn(model, actual)
        # temporal: value spike at a random lag
        lag = int(rng.integers(1, w + 1))
        window = test[p - w : p].copy()
        window[w - lag, cols[0]] += e.eval_spike
        attr = integrated_gradients(f, window, steps=e.ig_steps)
        temporal.append(lag in attr.top_lags(3))
        deletion.append(deletion_effect(f, window, attr, e.eval_deletion_fraction, rng))
        with torch.no_grad():
            clean = torch.as_tensor(np.stack([test[p - w : p], window]), dtype=torch.float32)
            moved = model(clean)[:, -1, :].double().numpy()
        temporal_effect.append(float(np.abs(moved[1] - moved[0]).max()))
        deletion_positive.append(
            deletion_effect(
                f, window, attr, e.eval_deletion_fraction, post_hoc_rng, rank="positive"
            )
        )
        # command: switch on a command that is off
        if commands:
            lag = int(rng.integers(1, w + 1))
            base = test[p - w : p].copy()
            off = [c for c in commands if base[w - lag, c] == 0]
            if off:
                c = int(rng.choice(off))
                injected = base.copy()
                injected[w - lag, c] = 1.0
                with torch.no_grad():
                    pair = torch.as_tensor(np.stack([base, injected]), dtype=torch.float32)
                    pred = model(pair)[:, -1, :].double().numpy()
                if np.abs(pred[1] - pred[0]).max() >= e.eval_min_effect:
                    attr = integrated_gradients(f, injected, steps=e.ig_steps)
                    command.append(c in attr.top_features(3))
                else:
                    ignored += 1

    # E3: coverage of the counterfactual band on label-free test stretches
    band = channel_band(config, det, channel, np.random.default_rng([config.seed, index]))
    h = e.eval_coverage_horizon
    starts = _clean_starts(labels, w, h - 1)  # history [s-w, s), future [s, s+h)
    covered: list[float] = []
    if starts.size:
        chosen = rng.choice(starts, size=min(e.eval_injections, starts.size), replace=False)
        hist = np.stack([test[s - w : s] for s in chosen])
        fut = np.stack([test[s : s + h] for s in chosen])
        expected = rollout(model, hist, fut, det.value_columns, w)
        low, high = band.at(h)
        inside = (fut[:, :, cols] >= expected + low) & (fut[:, :, cols] <= expected + high)
        covered = inside.all(axis=2).mean(axis=1).tolist()
    return {
        "chan_id": chan_id,
        "spacecraft": channel.spacecraft,
        "n_features": n_features,
        "temporal": temporal,
        "command": command,
        "command_ignored": ignored,
        "deletion": deletion,
        "temporal_effect": temporal_effect,
        "deletion_positive": deletion_positive,
        "coverage": covered,
        "band_source": band.source,
    }


def evaluate_attribution_and_band(config: Config, jobs: int = 1) -> dict[str, Any]:
    chans = evaluation_channels(config)
    per = list(parallel_map(partial(_evaluate_channel, config), list(enumerate(chans)), jobs))
    temporal = [hit for r in per for hit in r["temporal"]]
    command = [hit for r in per for hit in r["command"]]
    ignored = sum(r["command_ignored"] for r in per)
    deletion = np.array([d for r in per for d in r["deletion"]])
    coverage = [c for r in per for c in r["coverage"]]
    effects = np.array([x for r in per for x in r["temporal_effect"]])
    hits = np.array(temporal, dtype=bool)
    moved = effects >= config.explain.eval_min_effect
    positive = np.array([d for r in per for d in r["deletion_positive"]])
    window = config.deep.lstm.window
    command_baseline = float(np.mean([3 / r["n_features"] for r in per if r["command"]] or [0]))
    diff = deletion[:, 0] - deletion[:, 1]
    return {
        "channels": chans,
        "temporal": {
            "n": len(temporal),
            "hit_at_3": float(np.mean(temporal)),
            "ci95": wilson(sum(temporal), len(temporal)),
            "random_baseline": 3 / window,
        },
        "command": {
            "n": len(command),
            "ignored_by_model": ignored,
            "hit_at_3": float(np.mean(command)) if command else None,
            "ci95": wilson(sum(command), len(command)),
            "random_baseline": command_baseline,
        },
        "deletion": {
            "n": int(deletion.shape[0]),
            "top_mean": float(deletion[:, 0].mean()),
            "random_mean": float(deletion[:, 1].mean()),
            "difference_ci95": _bootstrap_ci(
                diff, np.mean, config.evaluation.bootstrap, config.seed
            ),
        },
        "post_hoc": {
            "temporal_moved_share": float(moved.mean()),
            "temporal_moved_hit_at_3": float(hits[moved].mean()) if moved.any() else None,
            "temporal_moved_ci95": wilson(int(hits[moved].sum()), int(moved.sum())),
            "temporal_moved_n": int(moved.sum()),
            "command_moved_share": len(command) / max(len(command) + ignored, 1),
            "deletion_abs_share_reducing": float(np.mean(deletion[:, 0] > 0)),
            "deletion_positive_share_reducing": float(np.mean(positive[:, 0] > 0)),
            "deletion_random_share_reducing": float(np.mean(positive[:, 1] > 0)),
            "coverage_median_channel": float(
                np.median([np.mean(r["coverage"]) for r in per if r["coverage"]])
            ),
        },
        "coverage": {
            "n": len(coverage),
            "mean": float(np.mean(coverage)) if coverage else None,
            "per_channel": {
                r["chan_id"]: float(np.mean(r["coverage"])) for r in per if r["coverage"]
            },
            "band_sources": {r["chan_id"]: r["band_source"] for r in per},
        },
    }


# --- E4 ----------------------------------------------------------------------------------


def evaluate_grounding(explanations_dir: Path) -> dict[str, Any]:
    sentences = violations = narratives = 0
    for path in explanations_dir.glob("*/event-*.json"):
        e = json.loads(path.read_text())
        narratives += 1
        sentences += len(e["sentences"])
        violations += len(e["grounding_violations"])
    return {
        "narratives": narratives,
        "sentences": sentences,
        "violations": violations,
        "grounded_share": 1 - violations / sentences if sentences else None,
    }
