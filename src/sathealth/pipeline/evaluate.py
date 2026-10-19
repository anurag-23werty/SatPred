"""``evaluate``: score one detector on every prepared channel under the fixed protocol.

Per channel: fit on train (validation available for early stopping) -> fit the threshold on
validation scores -> score test -> metrics against labels. Labels are only touched in the last
step and in the oracle upper bound, which is reported but never used as an operating point.

Channels are independent, so ``jobs > 1`` evaluates them in parallel worker processes;
every channel is seeded on its own, so the results do not depend on ``jobs``.

Output (one directory per detector + settings fingerprint, written atomically)::

    runs/evaluations/<detector>-<fingerprint>/
        summary.json        subsets (all, per spacecraft, excluding flagged channels)
        per_channel.json    every count and metric per channel
        training.json       per-channel training curves (deep detectors only)
        scores/<chan>.npz   val/test scores + threshold, input to Phase 4 fusion
"""

import json
import multiprocessing
import shutil
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sathealth.config import Config
from sathealth.data.schema import FloatArray
from sathealth.detectors.registry import build_detector
from sathealth.evaluation.aggregate import ChannelResult, summarize_subsets
from sathealth.evaluation.metrics import evaluate_predictions, oracle_event_f1
from sathealth.evaluation.thresholds import fit_threshold
from sathealth.logging import get_logger
from sathealth.pipeline.prepare import (
    data_fingerprint,
    load_prepared,
    load_run_manifest,
    run_dir_for,
)
from sathealth.pipeline.tracking import log_to_mlflow
from sathealth.utils.hashing import fingerprint
from sathealth.utils.provenance import provenance

log = get_logger(__name__)

SUMMARY_FILE = "summary.json"
TRAINING_FILE = "training.json"


@dataclass(frozen=True)
class EvaluationResult:
    run_dir: Path
    detector: str
    summary: dict[str, Any]
    reused: bool


def evaluation_fingerprint(config: Config, detector: str) -> str:
    return fingerprint(
        {
            "data": data_fingerprint(config),
            "detector": detector,
            "params": build_detector(detector, config).params(),
            "evaluation": config.evaluation.model_dump(mode="json"),
        }
    )


def evaluation_dir_for(config: Config, detector: str) -> Path:
    return config.paths.runs_dir / f"{detector}-{evaluation_fingerprint(config, detector)}"


def _channel_flags(channel_meta: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted({i["check"] for i in channel_meta.get("issues", []) if i["severity"] != "info"})
    )


@dataclass(frozen=True)
class _ChannelOutcome:
    result: ChannelResult
    val_scores: FloatArray
    test_scores: FloatArray
    seconds: float
    diagnostics: dict[str, Any]


def _evaluate_channel(
    config: Config, detector: str, data_dir: Path, chan_id: str, meta: dict[str, Any]
) -> _ChannelOutcome:
    """Fit, threshold and score one channel. Top-level so worker processes can run it."""
    ecfg = config.evaluation
    channel = load_prepared(data_dir, chan_id)
    t0 = time.perf_counter()
    model = build_detector(detector, config).fit(channel.train, channel.val)
    val_scores = model.score(channel.val)
    train_scores = (
        model.score(channel.train) if val_scores.size < ecfg.min_threshold_rows else np.zeros(0)
    )
    threshold = fit_threshold(
        val_scores,
        train_scores,
        method=ecfg.threshold,
        quantile=ecfg.quantile,
        sigma=ecfg.sigma,
        min_rows=ecfg.min_threshold_rows,
    )
    test_scores = model.score(channel.test)
    seconds = time.perf_counter() - t0

    segments = [(s.start, s.end) for s in channel.segments]
    metrics = evaluate_predictions(
        test_scores > threshold.value,
        channel.test_labels,
        test_scores,
        segments,
        early_tolerance=ecfg.early_tolerance,
        min_gap=ecfg.min_event_gap,
    )
    oracle, _ = oracle_event_f1(
        test_scores,
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
        threshold.value,
        threshold.source,
        _channel_flags(meta),
    )
    return _ChannelOutcome(result, val_scores, test_scores, seconds, model.diagnostics())


def _run_channels(
    config: Config, detector: str, data_dir: Path, channels: dict[str, Any], jobs: int
) -> Iterator[_ChannelOutcome]:
    """Outcomes in manifest order. Each channel is seeded on its own, so ``jobs`` never
    changes the numbers, only the wall time."""
    if jobs == 1:
        for chan_id, meta in channels.items():
            yield _evaluate_channel(config, detector, data_dir, chan_id, meta)
        return
    ctx = multiprocessing.get_context("spawn")  # fork is unsafe with torch's thread pools
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
        futures = [
            pool.submit(_evaluate_channel, config, detector, data_dir, chan_id, meta)
            for chan_id, meta in channels.items()
        ]
        for future in futures:
            yield future.result()


def evaluate_detector(
    config: Config, detector: str, *, force: bool = False, jobs: int = 1
) -> EvaluationResult:
    """Evaluate ``detector`` on every prepared channel, using ``jobs`` worker processes."""
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    out_dir = evaluation_dir_for(config, detector)
    if (out_dir / SUMMARY_FILE).is_file() and not force:
        log.info("reusing evaluation", detector=detector, run_dir=out_dir)
        return EvaluationResult(
            out_dir, detector, json.loads((out_dir / SUMMARY_FILE).read_text()), reused=True
        )

    data_dir = run_dir_for(config)
    if not (data_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"no prepared data at {data_dir}; run `sathealth prepare`")
    channels_meta: dict[str, Any] = load_run_manifest(data_dir)["channels"]
    ecfg = config.evaluation

    config.paths.runs_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{detector}-", dir=config.paths.runs_dir))
    try:
        (tmp / "scores").mkdir()
        results: list[ChannelResult] = []
        timings: dict[str, float] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        started = time.perf_counter()
        outcomes = _run_channels(config, detector, data_dir, channels_meta, jobs)
        for done, outcome in enumerate(outcomes, start=1):
            chan_id = outcome.result.chan_id
            log.info(
                "channel evaluated",
                detector=detector,
                channel=chan_id,
                progress=f"{done}/{len(channels_meta)}",
                event_f1=round(outcome.result.metrics.events.f1, 3),
            )
            results.append(outcome.result)
            timings[chan_id] = outcome.seconds
            if outcome.diagnostics:
                diagnostics[chan_id] = outcome.diagnostics
            np.savez_compressed(
                tmp / "scores" / f"{chan_id}.npz",
                val=outcome.val_scores.astype(np.float32),
                test=outcome.test_scores.astype(np.float32),
                threshold=np.float64(outcome.result.threshold),
            )

        subsets = summarize_subsets(
            results,
            exclude_subsets=ecfg.exclude_subsets,
            bootstrap=ecfg.bootstrap,
            seed=config.seed,
        )
        summary = {
            "detector": detector,
            "params": build_detector(detector, config).params(),
            "evaluation": ecfg.model_dump(mode="json"),
            "data_run": data_dir.name,
            **provenance(),
            "runtime_s": round(time.perf_counter() - started, 3),
            "compute_s": round(sum(timings.values()), 3),
            "jobs": jobs,
            "runtime_per_channel_s": {k: round(v, 4) for k, v in timings.items()},
            "subsets": subsets,
        }
        (tmp / "per_channel.json").write_text(json.dumps([r.to_dict() for r in results], indent=2))
        if diagnostics:
            (tmp / TRAINING_FILE).write_text(json.dumps(diagnostics, indent=2))
        (tmp / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp.rename(out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if config.tracking.mlflow_uri:
        log_to_mlflow(summary, out_dir, config.tracking.mlflow_uri, config.tracking.experiment)
    overall = subsets["all"]
    log.info(
        "evaluation finished",
        detector=detector,
        channels=overall["n_channels"],
        event_f1=round(overall["event"]["f1"], 3),
        composite_f1=round(overall["composite_f1"], 3),
        run_dir=out_dir,
    )
    return EvaluationResult(out_dir, detector, summary, reused=False)
