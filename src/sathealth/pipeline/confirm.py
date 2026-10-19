"""``confirm``: the pre-registered one-shot test on OPS-SAT-AD (ADR 0009, part A).

Per channel: fit every detector on the nominal training segments (deep models on windows
inside segments), score validation and test segment by segment, then apply the Phase 2
baseline thresholds and every Phase 4 decision variant unchanged. Each test segment is one
unit: anomalous segments are the events, and the paired bootstrap resamples segments.

Nothing here is tuned on OPS-SAT: detector settings come from the Phase 3 config and the
decision variants from :data:`sathealth.pipeline.ablation.VARIANTS`.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from sathealth.config import Config, DecisionConfig
from sathealth.data.download import download_dataset, load_manifest
from sathealth.data.opssat import OpsSatChannel, load_segments, prepare_channel
from sathealth.data.schema import BoolArray, FloatArray
from sathealth.detectors.deep import DeepDetector
from sathealth.detectors.registry import build_detector
from sathealth.evaluation.aggregate import ChannelResult, paired_composite_delta, summarize
from sathealth.evaluation.metrics import average_precision, evaluate_predictions, f1
from sathealth.evaluation.thresholds import fit_threshold
from sathealth.logging import get_logger
from sathealth.pipeline.ablation import VARIANTS, variant_config
from sathealth.pipeline.decide import decide_arrays
from sathealth.utils.parallel import parallel_map
from sathealth.utils.provenance import provenance

log = get_logger(__name__)

DETECTORS = ("lstm", "transformer", "limit", "zscore", "random")
BASELINES = ("limit", "zscore")
HYPOTHESES = {"C1": "fused-adaptive", "C2": "telemanom-lstm"}


@dataclass(frozen=True)
class ChannelScores:
    chan_id: str
    val: dict[str, FloatArray]
    test: dict[str, FloatArray]
    segments: list[tuple[int, int]]  # inclusive index ranges of test segments
    anomalous: list[bool]

    @property
    def labels(self) -> BoolArray:
        mask = np.zeros(self.segments[-1][1] + 1, dtype=np.bool_)
        for (start, end), bad in zip(self.segments, self.anomalous, strict=True):
            mask[start : end + 1] = bad
        return mask


def trainable(channel: OpsSatChannel, config: Config) -> bool:
    """A deep model needs at least one window that fits inside one nominal segment."""
    need = max(config.deep.lstm.window + 1, config.deep.transformer.window)
    return any(seg.shape[0] >= need for seg in channel.train)


def score_channel(config: Config, channel: OpsSatChannel) -> ChannelScores:
    """Fit every detector and score validation and test segments one by one."""
    val, test = {}, {}
    for name in DETECTORS:
        det = build_detector(name, config)
        if isinstance(det, DeepDetector):
            det.fit_segments(channel.train, channel.val)
        else:
            det.fit(np.concatenate(channel.train), np.concatenate(channel.val))
        val[name] = np.concatenate([det.score(s) for s in channel.val])
        test[name] = np.concatenate([det.score(s.values) for s in channel.test])
    segments, start = [], 0
    for seg in channel.test:
        segments.append((start, start + seg.length - 1))
        start += seg.length
    return ChannelScores(channel.chan_id, val, test, segments, [s.anomaly for s in channel.test])


def segment_rows(pred: BoolArray, scores: ChannelScores) -> FloatArray:
    """Per test segment: (point TP, point FP, event TP, event FN), the composite-F1 counts."""
    rows = []
    for (start, end), bad in zip(scores.segments, scores.anomalous, strict=True):
        hits = int(pred[start : end + 1].sum())
        rows.append(
            (hits if bad else 0, 0 if bad else hits, int(bad and hits > 0), int(bad and hits == 0))
        )
    return np.asarray(rows, dtype=np.float64)


def _evaluate(
    name: str,
    preds: dict[str, BoolArray],
    ranks: dict[str, FloatArray],
    chans: list[ChannelScores],
) -> dict[str, Any]:
    results, rows, seg_scores, seg_labels = [], [], [], []
    for ch in chans:
        pred, score = preds[ch.chan_id], ranks[ch.chan_id]
        metrics = evaluate_predictions(
            pred,
            ch.labels,
            score,
            [s for s, bad in zip(ch.segments, ch.anomalous, strict=True) if bad],
        )
        results.append(ChannelResult(ch.chan_id, "OPS-SAT", metrics, 0.0, 0.0, "val"))
        rows.append(segment_rows(pred, ch))
        seg_scores += [float(score[a : b + 1].max()) for a, b in ch.segments]
        seg_labels += ch.anomalous
    counts = np.concatenate(rows)
    flagged = (counts[:, 0] + counts[:, 1]) > 0
    bad = np.asarray(seg_labels, dtype=bool)
    tp, fp, fn = (
        int((flagged & bad).sum()),
        int((flagged & ~bad).sum()),
        int((~flagged & bad).sum()),
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    summary = summarize(results, bootstrap=0)
    return {
        "name": name,
        "composite_f1": summary["composite_f1"],
        "point_precision": summary["point"]["precision"],
        "segment": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
        },
        "segment_auc_pr": average_precision(np.asarray(seg_scores), bad),
        "alarm_share": float(np.mean(np.concatenate([preds[c.chan_id] for c in chans]))),
        "_rows": counts,
    }


def run_confirmation(config: Config, *, jobs: int = 1, download: bool = True) -> dict[str, Any]:
    c = config.confirmation
    manifest = load_manifest(c.manifest)
    if download:
        report = download_dataset(c.raw_dir, manifest, max_workers=1)
        if not report.ok:
            raise RuntimeError("OPS-SAT-AD download or verification failed")
    segments = load_segments(c.raw_dir / "segments.csv")
    channels = {
        name: prepare_channel(name, segs, c.val_fraction) for name, segs in segments.items()
    }
    included = [ch for ch in channels.values() if trainable(ch, config)]
    excluded = sorted(set(channels) - {ch.chan_id for ch in included})
    log.info("confirmation channels", included=len(included), excluded=excluded)
    chans = list(parallel_map(partial(score_channel, config), included, jobs))
    scores_dir = c.out_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    for ch in chans:
        arrays = {
            **{f"val_{k}": v for k, v in ch.val.items()},
            **{f"test_{k}": v for k, v in ch.test.items()},
            "segments": np.asarray(ch.segments),
            "anomalous": np.asarray(ch.anomalous),
        }
        # numpy's stubs type **kwargs of savez as its `allow_pickle` flag
        np.savez_compressed(scores_dir / f"{ch.chan_id}.npz", **arrays)  # type: ignore[arg-type]

    rows: list[dict[str, Any]] = []
    for name in (*BASELINES, "random"):
        preds, ranks = {}, {}
        for ch in chans:
            e = config.evaluation
            thr = fit_threshold(
                ch.val[name],
                np.zeros(0),
                method=e.threshold,
                quantile=e.quantile,
                sigma=e.sigma,
                min_rows=e.min_threshold_rows,
            )
            preds[ch.chan_id], ranks[ch.chan_id] = ch.test[name] > thr.value, ch.test[name]
        rows.append(
            {
                **_evaluate(name, preds, ranks, chans),
                "kind": "baseline" if name != "random" else "control",
            }
        )
    # Variants are defined relative to the pre-registered primary pipeline (ADR 0006), never
    # to whatever decision settings the caller's config carries (e.g. configs/live.yaml).
    registered = config.model_copy(update={"decision": DecisionConfig()})
    for variant in VARIANTS:
        d = variant_config(registered, variant).decision
        preds, ranks = {}, {}
        for ch in chans:
            _, fused, mask = decide_arrays(
                d, [ch.val[i] for i in d.inputs], [ch.test[i] for i in d.inputs]
            )
            preds[ch.chan_id], ranks[ch.chan_id] = mask, fused
        rows.append(
            {
                **_evaluate(variant.name, preds, ranks, chans),
                "kind": "post-hoc" if variant.post_hoc else "pre-registered",
                "change": variant.change,
            }
        )

    by_name = {r["name"]: r for r in rows}
    best = max(BASELINES, key=lambda b: by_name[b]["composite_f1"])
    for r in rows:
        paired = paired_composite_delta(
            r["_rows"],
            by_name[best]["_rows"],
            bootstrap=config.evaluation.bootstrap,
            seed=config.seed,
        )
        r |= {
            "delta_vs_best_baseline": paired["delta"],
            "ci95": paired["ci95"],
            "p_not_better": paired["p_not_better"],
        }
    verdicts = {
        h: {
            "pipeline": name,
            "delta": by_name[name]["delta_vs_best_baseline"],
            "ci95": by_name[name]["ci95"],
            "supported": by_name[name]["ci95"] is not None and by_name[name]["ci95"][0] > 0,
        }
        for h, name in HYPOTHESES.items()
    }
    result = {
        "dataset": manifest.name,
        "channels": [ch.chan_id for ch in chans],
        "excluded_channels": excluded,
        "test_segments": sum(len(ch.segments) for ch in chans),
        "anomalous_test_segments": sum(sum(ch.anomalous) for ch in chans),
        "best_baseline": best,
        "verdicts": verdicts,
        "rows": [{k: v for k, v in r.items() if k != "_rows"} for r in rows],
        **provenance(),
    }
    c.out_dir.mkdir(parents=True, exist_ok=True)
    (c.out_dir / "results.json").write_text(json.dumps(result, indent=2))
    return result


def _ci(ci: Any) -> str:
    return "" if ci is None else f" [{ci[0]:+.3f}, {ci[1]:+.3f}]"


def render_confirmation(result: dict[str, Any]) -> str:
    v = result["verdicts"]
    word = {True: "**supported**", False: "**not supported**"}
    lines = [
        "# Independent confirmation: OPS-SAT-AD",
        "",
        f"_Generated by `sathealth confirm` on {datetime.now(UTC):%Y-%m-%d}. Do not edit by hand. "
        "Protocol and hypotheses were pre-registered in "
        "[ADR 0009](adr/0009-extensions-and-confirmation.md) before this run._",
        "",
        f"Data: ESA OPS-SAT cubesat, {len(result['channels'])} channels "
        f"({', '.join(result['channels'])}); excluded by the pre-registered rule: "
        f"{', '.join(result['excluded_channels']) or 'none'}. "
        f"{result['test_segments']} test segments, {result['anomalous_test_segments']} anomalous. "
        "Every setting comes unchanged from Phases 2–4; nothing was tuned on this data.",
        "",
        "## Verdicts",
        "",
        "| | Hypothesis | Δ composite F1 vs best baseline "
        f"({result['best_baseline']}) [paired 95% CI over segments] | Verdict |",
        "|---|---|---|---|",
        *[
            f"| {h} | {text} | {v[h]['delta']:+.3f}{_ci(v[h]['ci95'])} "
            f"| {word[v[h]['supported']]} |"
            for h, text in (
                ("C1", "The live pipeline beats the best baseline"),
                ("C2", "The Telemanom reference beats the best baseline"),
            )
        ],
        "",
        "## All pipelines",
        "",
        "Composite F1 is the headline metric of ADR 0004 with segments as events. A segment counts "
        "as flagged if any of its steps alarms; segment AUC-PR ranks segments by their highest "
        "score. **Caution:** the highest score of a longer segment is a maximum over more steps, "
        "so segment AUC-PR is confounded by segment length; the random control's AUC-PR shows "
        "how much.",
        "",
        "| Pipeline | Kind | Composite F1 | Δ vs best baseline [95% CI] | Segment P / R / F1 "
        "| Segment AUC-PR | False-alarm segments | Alarm share |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    for r in sorted(result["rows"], key=lambda r: -r["composite_f1"]):
        s = r["segment"]
        lines.append(
            f"| {r['name']} | {r['kind']} | {r['composite_f1']:.3f} | "
            f"{r['delta_vs_best_baseline']:+.3f}{_ci(r['ci95'])} | "
            f"{s['precision']:.2f} / {s['recall']:.2f} / {s['f1']:.2f} | "
            f"{r['segment_auc_pr']:.3f} | {s['fp']} | {r['alarm_share']:.1%} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_confirmation(config: Config, out: Path, *, jobs: int = 1, download: bool = True) -> Path:
    result = run_confirmation(config, jobs=jobs, download=download)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_confirmation(result))
    return out
