"""Run every decision-layer variant and write ``docs/DECISION.md``.

The primary configuration and the pre-registered ablations were fixed before any Phase 4
result (docs/adr/0006-decision-layer.md); every row is reported, none is selected. Variants
added after seeing the pre-registered results are marked *post-hoc* and never replace the
pre-registered verdict.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sathealth.config import Config, DecisionConfig
from sathealth.detectors.registry import BASELINE_NAMES
from sathealth.evaluation.aggregate import composite_counts, paired_composite_delta
from sathealth.pipeline.decide import decide
from sathealth.pipeline.evaluate import SUMMARY_FILE, evaluation_dir_for

SUBSETS = ("all", "SMAP", "MSL")


@dataclass(frozen=True)
class Variant:
    name: str
    change: str
    overrides: dict[str, Any]
    post_hoc: bool = False


VARIANTS: tuple[Variant, ...] = (
    Variant("fused", "**primary** (pre-registered)", {}),
    Variant("lstm-evt", "LSTM only", {"inputs": ("lstm",)}),
    Variant("transformer-evt", "Transformer only", {"inputs": ("transformer",)}),
    Variant("fused-max", "max fusion (any detector)", {"fusion": "max"}),
    Variant("fused-min", "min fusion (all detectors)", {"fusion": "min"}),
    Variant("fused-no-hysteresis", "no hysteresis (leave at 3)", {"alarm_off": 3.0}),
    Variant("fused-no-merge", "no event merging", {"merge_gap": 0}),
    Variant("zscore-evt", "decision layer on the z-score baseline", {"inputs": ("zscore",)}),
    Variant(
        "telemanom-lstm",
        "Telemanom reference: raw LSTM errors, dynamic threshold, pruning 0.13 (batch)",
        {
            "inputs": ("lstm",),
            "calibration": "none",
            "threshold": "ndt",
            "prune": 0.13,
            "merge_gap": 0,
        },
    ),
    Variant(
        "fused-ndt",
        "dynamic threshold + pruning 0.13 on the fused surprise (batch)",
        {"threshold": "ndt", "prune": 0.13},
    ),
    Variant(
        "fused-adaptive",
        "streaming recalibration every 100 steps: validation + last 2,000 non-alarm scores",
        {"recalibrate_every": 100},
        post_hoc=True,
    ),
)


def variant_config(config: Config, variant: Variant) -> Config:
    """``config`` with the variant's decision settings, validated like any user config."""
    decision = DecisionConfig.model_validate(
        {**config.decision.model_dump(), "name": variant.name, **variant.overrides}
    )
    return config.model_copy(update={"decision": decision})


def _subset_rows(rows: list[dict[str, Any]], subset: str) -> list[dict[str, Any]]:
    chosen = [r for r in rows if subset in ("all", r["spacecraft"])]
    return sorted(chosen, key=lambda r: r["chan_id"])


def _load(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads((run_dir / SUMMARY_FILE).read_text())
    return summary, json.loads((run_dir / "per_channel.json").read_text())


def run_ablation(config: Config, *, force: bool = False) -> list[dict[str, Any]]:
    """Run (or reuse) every variant; return one row per variant and subset."""
    baselines = {
        name: _load(evaluation_dir_for(config, name))
        for name in BASELINE_NAMES
        if name != "random" and (evaluation_dir_for(config, name) / SUMMARY_FILE).is_file()
    }
    if not baselines:
        raise FileNotFoundError("no baseline evaluations; run `sathealth evaluate baselines`")
    rows = []
    for variant in VARIANTS:
        result = decide(variant_config(config, variant), force=force)
        per_channel = json.loads((result.run_dir / "per_channel.json").read_text())
        for subset in SUBSETS:
            best = max(baselines, key=lambda b: baselines[b][0]["subsets"][subset]["composite_f1"])
            paired = paired_composite_delta(
                composite_counts(_subset_rows(per_channel, subset)),
                composite_counts(_subset_rows(baselines[best][1], subset)),
                bootstrap=config.evaluation.bootstrap,
                seed=config.seed,
            )
            rows.append(
                {
                    "variant": variant,
                    "subset": subset,
                    "summary": result.summary,
                    "numbers": result.summary["subsets"][subset],
                    "baseline": best,
                    "baseline_composite_f1": baselines[best][0]["subsets"][subset]["composite_f1"],
                    **paired,
                }
            )
    return rows


def _fmt_ci(ci: Sequence[float] | None) -> str:
    return "" if ci is None else f" [{ci[0]:+.3f}, {ci[1]:+.3f}]"


def render_ablation(rows: Sequence[dict[str, Any]], config: Config) -> str:
    d = config.decision
    primary = next(r for r in rows if r["variant"].name == "fused" and r["subset"] == "all")
    ps = primary["summary"]
    verdict_rows = [r for r in rows if r["variant"].name == "fused"]
    passed = [
        r["subset"]
        for r in verdict_rows
        if r["ci95"] is not None and r["ci95"][0] > 0 and ps["within_budget"]
    ]
    lines = [
        "# Decision layer results: NASA SMAP & MSL",
        "",
        f"_Generated by `sathealth ablation` on {datetime.now(UTC):%Y-%m-%d}. Do not edit by "
        "hand; regenerate with `make ablation`. Design and pre-registration: "
        "[ADR 0006](adr/0006-decision-layer.md)._",
        "",
        "## Pre-registered verdict",
        "",
        f"Primary pipeline: {ps['description']}.",
        "",
        f"- Exit criterion: composite F1 significantly above the best baseline (paired 95% CI "
        f"above zero) on all channels or one spacecraft, within **≤ {d.alarm_budget:g} false "
        "events per 1,000 steps**.",
        f"- False events per 1,000 steps: **{primary['numbers']['fp_events_per_1k_steps']:.2f}** "
        f"({'within' if ps['within_budget'] else 'over'} budget).",
    ]
    for r in verdict_rows:
        lines.append(
            f"- {r['subset']}: composite F1 {r['numbers']['composite_f1']:.3f} vs "
            f"{r['baseline']} {r['baseline_composite_f1']:.3f}, "
            f"Δ {r['delta']:+.3f}{_fmt_ci(r['ci95'])}"
        )
    lines += [
        "",
        f"**Verdict: {'met on ' + ', '.join(passed) if passed else 'not met'}.**",
        "",
        "## All variants",
        "",
        "Δ is the paired difference in composite F1 against the best Phase 2 baseline of the "
        "same subset. *Post-hoc* variants were added after the pre-registered results and "
        "cannot change the verdict above.",
    ]
    header = [
        "",
        "| Variant | What changes | Composite F1 [95% CI] | Δ vs best baseline [paired 95% CI] "
        "| Event F1 | Event P | Event R | FP events / 1k | Within budget | Alarm share |",
        "|---|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for subset in SUBSETS:
        subset_rows = [r for r in rows if r["subset"] == subset]
        lines += ["", f"### {subset.capitalize() if subset == 'all' else subset}", *header]
        for r in subset_rows:
            n, v = r["numbers"], r["variant"]
            ci = n.get("ci95", {}).get("composite_f1")
            ci_text = "" if ci is None else f" [{ci[0]:.2f}, {ci[1]:.2f}]"
            budget = "yes" if n["fp_events_per_1k_steps"] <= d.alarm_budget else "no"
            tag = " *(post-hoc)*" if v.post_hoc else ""
            lines.append(
                f"| {v.name}{tag} | {v.change} | {n['composite_f1']:.3f}{ci_text} "
                f"| {r['delta']:+.3f}{_fmt_ci(r['ci95'])} | {n['event']['f1']:.3f} "
                f"| {n['event']['precision']:.2f} | {n['event']['recall']:.2f} "
                f"| {n['fp_events_per_1k_steps']:.2f} | {budget} "
                f"| {r['summary']['alarm_fraction_mean']:.1%} |"
            )

    oracle = ps["global_oracle"]
    kinds = ps["kind_agreement"]
    sev = ps["severity_counts"]
    lines += [
        "",
        "## Diagnostics of the primary pipeline",
        "",
        f"- **Calibration transfer.** Steps labelled nominal are in alarm "
        f"{ps['nominal_alarm_fraction_mean']:.1%} of the time on average (alarm share over all "
        f"test steps: {ps['alarm_fraction_mean']:.1%}). The alarm share is label-free: a "
        "calibration that transferred from validation to test would keep it near the share of "
        "anomalous steps.",
        f"- **Global oracle †.** The best single surprise level for all channels, chosen with "
        f"test labels, is {oracle['level']:g}, reaching composite F1 "
        f"{oracle['composite_f1']:.3f}. One level for every channel cannot overfit individual "
        "channels, so this is a meaningful upper bound for this calibrated pipeline.",
        f"- **Anomaly type.** On {kinds.get('n', 0)} detected labelled anomalies, the point vs "
        f"contextual rule agrees with the labels {kinds.get('accuracy', 0):.0%} of the time "
        f"(always answering the majority class: {kinds.get('majority_baseline', 0):.0%}). "
        f"Confusion (label → predicted): {kinds.get('confusion', {})}.",
        f"- **Severity** of all {sum(sev.values())} alarm events: {sev['critical']} critical, "
        f"{sev['warning']} warning, {sev['watch']} watch.",
        "",
    ]
    return "\n".join(lines)


def write_ablation(config: Config, out: Path, *, force: bool = False) -> Path:
    rows = run_ablation(config, force=force)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_ablation(rows, config))
    return out
