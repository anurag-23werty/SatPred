"""Render evaluation summaries as a Markdown leaderboard (``docs/BASELINES.md``).

Besides the per-subset tables, every non-baseline detector is compared with the best
baseline of each subset, on composite F1 and AUC-PR, using a paired bootstrap over channels
(:mod:`sathealth.evaluation.aggregate`).
"""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sathealth.config import Config
from sathealth.detectors.registry import BASELINE_NAMES
from sathealth.evaluation.aggregate import (
    composite_counts,
    paired_composite_delta,
    paired_mean_delta,
)
from sathealth.pipeline.decide import decision_dir_for
from sathealth.pipeline.evaluate import SUMMARY_FILE, evaluation_dir_for

COMPARISON_SUBSETS = ("all", "SMAP", "MSL")
COMPARISON_METRICS = {
    "composite_f1": "composite_f1",
    "average_precision": "average_precision_macro",
}
COMPARISON_LABELS = {"composite_f1": "Composite F1", "average_precision": "AUC-PR"}

_DESCRIPTIONS = {
    "random": "Uniform random scores: the floor every detector must clear",
    "limit": "Static limits = train range ± 5%: today's operational practice",
    "zscore": "Robust z-score of each step against the train median/IQR",
    "iforest": "Isolation Forest on 32-step window features",
    "lstm": "LSTM next-step forecaster (Telemanom-style), EWMA-smoothed error",
    "transformer": "Transformer autoencoder with a 16-d bottleneck, EWMA-smoothed error",
}


def run_dir(config: Config, name: str) -> Path:
    """Where ``name`` lives: the configured decision pipeline or a detector evaluation."""
    if name == config.decision.name:
        return decision_dir_for(config)
    return evaluation_dir_for(config, name)


def load_summaries(config: Config, detectors: Sequence[str]) -> list[dict[str, Any]]:
    """Summaries evaluated under the *current* config; missing runs are an error."""
    summaries = []
    for name in detectors:
        path = run_dir(config, name) / SUMMARY_FILE
        if not path.is_file():
            command = "decide" if name == config.decision.name else f"evaluate {name}"
            raise FileNotFoundError(f"no results for {name!r}; run `sathealth {command}`")
        summaries.append(json.loads(path.read_text()))
    return summaries


def _subset(per_channel: list[dict[str, Any]], subset: str) -> list[dict[str, Any]]:
    chans = [c for c in per_channel if subset in ("all", c["spacecraft"])]
    return sorted(chans, key=lambda c: c["chan_id"])


def _paired_metric(
    metric: str, a: list[dict[str, Any]], b: list[dict[str, Any]], bootstrap: int, seed: int
) -> dict[str, Any]:
    if metric == "composite_f1":
        return paired_composite_delta(
            composite_counts(a), composite_counts(b), bootstrap=bootstrap, seed=seed
        )
    values = [np.array([c["average_precision"] for c in rows]) for rows in (a, b)]
    return paired_mean_delta(*values, bootstrap=bootstrap, seed=seed)


def compare_with_baselines(
    config: Config, summaries: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Paired difference of each non-baseline detector vs the best baseline, per metric.

    Composite F1 measures the detector *at its operating threshold*; AUC-PR measures how well
    its scores rank anomalies independently of any threshold. The best baseline is chosen
    separately for each metric and subset.
    """
    real = set(BASELINE_NAMES) - {"random"}  # the random control is a floor, not a rival
    baselines = [s for s in summaries if s["detector"] in real]
    if not baselines:
        return []
    per_channel = {
        s["detector"]: json.loads((run_dir(config, s["detector"]) / "per_channel.json").read_text())
        for s in summaries
    }
    rows = []
    for candidate in summaries:
        if candidate["detector"] in BASELINE_NAMES:
            continue
        for subset in COMPARISON_SUBSETS:
            if subset not in candidate["subsets"]:
                continue
            for metric, key in COMPARISON_METRICS.items():
                best = max(baselines, key=lambda s: s["subsets"][subset][key])
                paired = _paired_metric(
                    metric,
                    _subset(per_channel[candidate["detector"]], subset),
                    _subset(per_channel[best["detector"]], subset),
                    config.evaluation.bootstrap,
                    config.seed,
                )
                ci_a = candidate["subsets"][subset].get("ci95", {}).get(metric)
                ci_b = best["subsets"][subset].get("ci95", {}).get(metric)
                rows.append(
                    {
                        "detector": candidate["detector"],
                        "subset": subset,
                        "metric": metric,
                        "value": candidate["subsets"][subset][key],
                        "baseline": best["detector"],
                        "baseline_value": best["subsets"][subset][key],
                        **paired,
                        "marginal_cis_separate": (
                            None if ci_a is None or ci_b is None else ci_a[0] > ci_b[1]
                        ),
                    }
                )
    return rows


def _comparison_lines(rows: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Versus the best baseline",
        "",
        "Paired bootstrap over channels: both detectors are resampled on the *same* channels, "
        'which is the right test for two detectors scored on identical data. "Better" '
        "means the 95% CI of the difference lies above zero. Composite F1 judges the "
        "detector at its label-free threshold; AUC-PR judges how well its scores rank "
        'anomalies, whatever the threshold. "Separate CIs" is the stricter check that the '
        "two unpaired composite-F1 intervals do not even overlap.",
        "",
        "| Detector | Subset | Metric | Value | Best baseline | Δ [paired 95% CI] "
        "| Better? | Separate CIs? |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    for r in rows:
        ci = r["ci95"]
        better = "—" if ci is None else ("**yes**" if ci[0] > 0 else "no")
        separate = {None: "—", True: "yes", False: "no"}[r["marginal_cis_separate"]]
        interval = "" if ci is None else f" [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        lines.append(
            f"| {r['detector']} | {r['subset']} | {COMPARISON_LABELS[r['metric']]} "
            f"| {r['value']:.3f} | {r['baseline']} ({r['baseline_value']:.3f}) "
            f"| {r['delta']:+.3f}{interval} | {better} | {separate} |"
        )
    return lines


def _wall_time(s: dict[str, Any]) -> str:
    seconds = s["runtime_s"]
    text = f"{seconds / 60:.0f} min" if seconds >= 120 else f"{seconds:.1f} s"
    jobs = s.get("jobs", 1)
    return f"{text} ({jobs} parallel jobs)" if jobs > 1 else text


def _ci(s: dict[str, Any], key: str) -> str:
    ci = s.get("ci95", {}).get(key)
    return f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""


def _row(name: str, s: dict[str, Any]) -> str:
    delay = s["delay_median_steps"]
    return (
        f"| {name} | **{s['composite_f1']:.3f}**{_ci(s, 'composite_f1')} "
        f"| {s['event']['f1']:.3f}{_ci(s, 'event_f1')} | {s['event_f1_macro']:.3f} "
        f"| {s['event']['precision']:.2f} | {s['event']['recall']:.2f} "
        f"| {s['fp_events_per_1k_steps']:.2f} | {'—' if delay is None else f'{delay:g}'} "
        f"| {s['average_precision_macro']:.3f} | {s['point_adjusted_f1']:.3f} "
        f"| {s['oracle_event_f1_macro']:.3f} |"
    )


_HEADER = [
    "| Detector | Composite F1 [95% CI] | Event F1 (micro) [95% CI] | Event F1 (macro) "
    "| Event P | Event R | FP events / 1k steps | Median delay (steps) | AUC-PR "
    "| Point-adjusted F1 ⚠ | Oracle event F1 † |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]


def render_leaderboard(
    summaries: Sequence[dict[str, Any]], comparisons: Sequence[dict[str, Any]] = ()
) -> str:
    if not summaries:
        raise ValueError("no summaries to render")
    ranked = sorted(summaries, key=lambda s: -s["subsets"]["all"]["composite_f1"])
    first = ranked[0]
    ev = first["evaluation"]
    random_oracle = next(
        (s["subsets"]["all"]["oracle_event_f1_macro"] for s in ranked if s["detector"] == "random"),
        None,
    )
    lines = [
        "# Detector results: NASA SMAP & MSL",
        "",
        f"_Generated by `sathealth leaderboard` on {datetime.now(UTC):%Y-%m-%d} from data run "
        f"`{first['data_run']}`. Do not edit by hand; regenerate with `make leaderboard`._",
        "",
        "## Protocol",
        "",
        f"- **Thresholds never see labels.** Each detector's threshold is the "
        f"{ev['quantile']} quantile of its scores on the nominal validation tail "
        f"(`{ev['threshold']}`). Test labels are used only to score the predictions.",
        "- **Composite F1 is the headline** (Garg et al. 2021): point-wise precision combined "
        "with event-wise recall. It rewards finding every anomaly and penalises flooding "
        "operators with alarm steps.",
        "- **Event F1** follows the dataset's own paper (Hundman et al. 2018). Micro "
        "averaging sums counts over channels; macro averaging takes the mean of per-channel F1 "
        "and is robust to one noisy channel.",
        "- ⚠ **Point-adjusted F1 is inflated** and shown only to compare with papers. "
        "Random scores score highly on it (Kim et al., AAAI 2022).",
        "- † **Oracle** picks the best of many thresholds per channel *using* test labels. "
        "It is never an operating point, and it is a **loose** bound: selecting per channel "
        "overfits to channels with only one or two events"
        + (f" (random scores reach {random_oracle:.2f})." if random_oracle is not None else "."),
        "- Delays and false-alarm rates are in **time steps**: NASA did not publish the "
        "sampling interval, so we do not convert them to minutes or days.",
        f"- 95% CIs come from {ev['bootstrap']:,} bootstrap resamples of channels.",
        "- Deep detectors train one model per channel on train only, early-stopped on the "
        "validation tail, with fixed seeds; their hyperparameters were set a priori, never "
        "tuned on test labels.",
        "",
        "## Detectors",
        "",
        "| Detector | What it is | Wall time, all channels |",
        "|---|---|---:|",
    ]
    lines += [
        f"| {s['detector']} | {s.get('description') or _DESCRIPTIONS.get(s['detector'], '')} "
        f"| {_wall_time(s)} |"
        for s in ranked
    ]
    if comparisons:
        lines += _comparison_lines(comparisons)

    for subset in first["subsets"]:
        n = first["subsets"][subset]
        lines += [
            "",
            f"## {subset.capitalize() if subset == 'all' else subset}"
            f" ({n['n_channels']} channels, {n['n_events']} events)",
            "",
            *_HEADER,
        ]
        lines += [
            _row(s["detector"], s["subsets"][subset]) for s in ranked if subset in s["subsets"]
        ]
    lines.append("")
    return "\n".join(lines)


def write_leaderboard(config: Config, detectors: Sequence[str], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries(config, detectors)
    out.write_text(render_leaderboard(summaries, compare_with_baselines(config, summaries)))
    return out
