"""Write ``docs/EXPLAINABILITY.md`` and its example figure.

The report states the pre-registered hypotheses of ADR 0007, the evaluation results, and one
explained event with its narrative and a counterfactual plot.

The figure is a static SVG for Markdown (no hover is possible inside an ``<img>``): one
value axis, the actual value and the counterfactual expectation as 2px lines (the second one
dashed, so identity never rests on colour alone), the 90% band as a light fill, a legend
plus direct end labels, recessive grid. Colours follow the validated reference palette
(series slots 1 and 2, the critical status colour for the alarm span, neutral ink for the
label span), with a dark variant selected under ``prefers-color-scheme``.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from sathealth.config import Config
from sathealth.data.schema import FloatArray
from sathealth.pipeline.decide import decision_dir_for
from sathealth.pipeline.explain import explain_run
from sathealth.pipeline.explain_eval import (
    evaluate_attribution_and_band,
    evaluate_grounding,
    evaluate_onset,
)
from sathealth.pipeline.prepare import load_prepared, run_dir_for

CONTEXT = 120  # steps shown before the onset
W, H = 880, 330
LEFT, RIGHT, TOP, BOTTOM = 60, 150, 54, 42

STYLE = """
  .surface{fill:#fcfcfb}.ink{fill:#0b0b0b}.muted{fill:#52514e}
  .grid{stroke:#e1e0d9}.axis{stroke:#c3c2b7}
  .actual{stroke:#2a78d6}.expected{stroke:#eb6834}.band{fill:#eb6834;opacity:.16}
  .alarm{fill:#d03b3b;opacity:.10}.alarm-edge{stroke:#d03b3b}
  .label{fill:#52514e;opacity:.14}.onset{stroke:#0b0b0b}
  @media (prefers-color-scheme: dark){
    .surface{fill:#1a1a19}.ink{fill:#f0efec}.muted{fill:#c3c2b7}
    .grid{stroke:#2c2c2a}.axis{stroke:#383835}
    .actual{stroke:#3987e5}.expected{stroke:#d95926}.band{fill:#d95926;opacity:.22}
    .alarm{opacity:.18}.label{fill:#c3c2b7;opacity:.16}.onset{stroke:#f0efec}
  }
"""


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1.0
    step = min((m * mag for m in (1, 2, 2.5, 5, 10)), key=lambda s: abs(s - raw))
    start = np.ceil(lo / step) * step
    return [float(v) for v in np.arange(start, hi + step / 2, step)]


def render_event_figure(
    steps: FloatArray,
    actual: FloatArray,
    cf_steps: FloatArray,
    expected: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    *,
    onset: int,
    alarm: tuple[int, int],
    labels: list[tuple[int, int]],
    title: str,
) -> str:
    x0, x1 = float(steps[0]), float(steps[-1])
    values = np.concatenate([actual, lower, upper])
    y0, y1 = float(values.min()), float(values.max())
    pad = (y1 - y0) * 0.08 or 1.0
    y0, y1 = y0 - pad, y1 + pad
    pw, ph = W - LEFT - RIGHT, H - TOP - BOTTOM

    def sx(v: float) -> float:
        return LEFT + (v - x0) / (x1 - x0 or 1) * pw

    def sy(v: float) -> float:
        return TOP + (y1 - v) / (y1 - y0) * ph

    def path(xs: FloatArray, ys: FloatArray) -> str:
        return " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(xs, ys, strict=True))

    def span(a: int, b: int, cls: str) -> str:
        a_, b_ = max(a, x0), min(b, x1)
        if a_ > b_:
            return ""
        return (
            f'<rect class="{cls}" x="{sx(a_):.1f}" y="{TOP}" '
            f'width="{max(sx(b_) - sx(a_), 1.5):.1f}" height="{ph}"/>'
        )

    grid = []
    for t in _ticks(y0, y1):
        y = sy(t)
        grid.append(f'<line class="grid" x1="{LEFT}" x2="{LEFT + pw}" y1="{y:.1f}" y2="{y:.1f}"/>')
        grid.append(
            f'<text class="muted" x="{LEFT - 8}" y="{y + 4:.1f}" text-anchor="end">{t:g}</text>'
        )
    for t in _ticks(x0, x1, 6):
        grid.append(
            f'<text class="muted" x="{sx(t):.1f}" y="{TOP + ph + 18}" '
            f'text-anchor="middle">{t:.0f}</text>'
        )
    band = path(cf_steps, upper) + " " + path(cf_steps[::-1], lower[::-1])
    end_actual, end_expected = sy(float(actual[-1])), sy(float(expected[-1]))
    if abs(end_actual - end_expected) < 14:  # keep direct labels apart
        end_expected += 14 if end_expected >= end_actual else -14
    label_spans = "".join(span(a, b, "label") for a, b in labels)
    first_label = next((a for a, _ in labels if x0 <= a <= x1), None)
    label_text = (
        ""
        if first_label is None
        else f'<text class="muted" x="{sx(first_label) + 4:.1f}" y="{TOP + ph - 6}">'
        "labelled start</text>"
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}"
     height="{H}" role="img" font-family="Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
     font-size="12" aria-label="{title}: actual value against the value expected from the
     commands actually sent, with its 90% band, the alarm span and the estimated onset">
  <style>{STYLE}</style>
  <rect class="surface" width="{W}" height="{H}" rx="8"/>
  <text class="ink" x="{LEFT}" y="22" font-size="14" font-weight="600">{title}</text>
  <g transform="translate({LEFT},36)">
    <line class="actual" x1="0" x2="22" y1="0" y2="0" stroke-width="2"/>
    <text class="muted" x="28" y="4">actual</text>
    <line class="expected" x1="90" x2="112" y1="0" y2="0" stroke-width="2"
          stroke-dasharray="5 3"/>
    <text class="muted" x="118" y="4">expected from the commands sent</text>
    <rect class="band" x="330" y="-6" width="22" height="12"/>
    <text class="muted" x="358" y="4">90% band</text>
    <rect class="alarm" x="430" y="-6" width="22" height="12"/>
    <text class="muted" x="458" y="4">alarm</text>
    <rect class="label" x="510" y="-6" width="22" height="12"/>
    <text class="muted" x="538" y="4">labelled anomaly</text>
  </g>
  {"".join(grid)}
  {label_spans}
  {span(alarm[0], alarm[1], "alarm")}
  <line class="axis" x1="{LEFT}" x2="{LEFT + pw}" y1="{TOP + ph}" y2="{TOP + ph}"/>
  <polygon class="band" points="{band}"/>
  <polyline class="actual" points="{path(steps, actual)}" fill="none" stroke-width="2"
            stroke-linejoin="round"/>
  <polyline class="expected" points="{path(cf_steps, expected)}" fill="none"
            stroke-width="2" stroke-dasharray="5 3" stroke-linejoin="round"/>
  <line class="onset" x1="{sx(onset):.1f}" x2="{sx(onset):.1f}" y1="{TOP}" y2="{TOP + ph}"
        stroke-width="1.5" stroke-dasharray="2 3"/>
  <text class="ink" x="{sx(onset) - 4:.1f}" y="{TOP + 12}" text-anchor="end">onset</text>
  {label_text}
  <text class="ink" x="{LEFT + pw + 8}" y="{end_actual + 4:.1f}">actual</text>
  <text class="ink" x="{LEFT + pw + 8}" y="{end_expected + 4:.1f}">expected</text>
  <text class="muted" x="{LEFT + pw / 2}" y="{H - 6}" text-anchor="middle">test step</text>
</svg>
"""


def pick_example(out_dir: Path, config: Config) -> dict[str, Any] | None:
    """Showcase rule: the highest-surprise *contextual* event that overlaps a labelled anomaly
    and has a counterfactual (labels only choose which event to illustrate)."""
    data_dir = run_dir_for(config)
    best: tuple[float, dict[str, Any]] | None = None
    for path in sorted(out_dir.glob("*/event-*.json")):
        e = json.loads(path.read_text())
        ev = e["event"]
        if ev["kind"] != "contextual" or "counterfactual" not in e:
            continue
        segments = load_prepared(data_dir, e["chan_id"]).segments
        if not any(s.start <= ev["end"] and s.end >= ev["start"] for s in segments):
            continue
        if best is None or ev["peak_surprise"] > best[0]:
            best = (ev["peak_surprise"], {**e, "_path": str(path)})
    return None if best is None else best[1]


def example_figure(config: Config, example: dict[str, Any]) -> str:
    channel = load_prepared(run_dir_for(config), example["chan_id"])
    with np.load(Path(example["_path"]).with_suffix(".npz")) as npz:
        arrays = {k: npz[k] for k in npz.files}
    p = channel.scaler
    col = config.preprocess.value_columns[0]
    onset, ev = example["onset"], example["event"]
    horizon = arrays["expected"].shape[0]
    start = max(0, onset - CONTEXT)
    stop = onset + horizon
    raw = channel.test[start:stop, col] * p.scale[0] + p.center[0]
    return render_event_figure(
        np.arange(start, stop, dtype=np.float64),
        raw,
        np.arange(onset, onset + horizon, dtype=np.float64),
        arrays["expected"][:, 0],
        arrays["lower"][:, 0],
        arrays["upper"][:, 0],
        onset=onset,
        alarm=(ev["start"], ev["end"]),
        labels=[(s.start, s.end) for s in channel.segments],
        title=f"Channel {example['chan_id']} ({example['spacecraft']}): "
        f"{ev['severity']} {ev['kind']} anomaly",
    )


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def render_report(
    config: Config,
    onset: dict[str, Any],
    attribution: dict[str, Any],
    grounding: dict[str, Any],
    example: dict[str, Any] | None,
    figure_path: str | None,
) -> str:
    t, c, d, cov = (
        attribution["temporal"],
        attribution["command"],
        attribution["deletion"],
        attribution["coverage"],
    )
    h1 = onset["median_abs_onset_error"] < onset["median_abs_alarm_error"]
    h2 = (
        t["ci95"][0] > t["random_baseline"]
        and c["n"] > 0
        and c["ci95"][0] > c["random_baseline"]
        and d["difference_ci95"][0] > 0
    )
    h3 = cov["mean"] is not None and 0.80 <= cov["mean"] <= 0.95
    h4 = grounding["violations"] == 0 and grounding["sentences"] > 0
    ok = {True: "**supported**", False: "**not supported**"}
    ci = onset["median_improvement_ci"] or (float("nan"), float("nan"))
    ph = attribution["post_hoc"]
    lines = [
        "# Explainability results: NASA SMAP & MSL",
        "",
        "_Generated by `sathealth explain-report`. Do not edit by hand. Protocol, parameters "
        "and hypotheses were pre-registered in [ADR 0007](adr/0007-explainability.md)._",
        "",
        f"Explained run: `{decision_dir_for(config).name}` ({grounding['narratives']} events).",
        "",
        "## Hypotheses",
        "",
        "| | Hypothesis | Result | Verdict |",
        "|---|---|---|---|",
        f"| H1 | The onset estimate is closer to the labelled start than the alarm start "
        f"(median) | median error {onset['median_abs_onset_error']:g} vs "
        f"{onset['median_abs_alarm_error']:g} steps over {onset['n']} detected anomalies; "
        f"onset closer in {_pct(onset['share_onset_closer'])}, tied in "
        f"{_pct(onset['share_tied'])}; 95% CI of the median improvement "
        f"[{ci[0]:g}, {ci[1]:g}] steps | {ok[h1]} |",
        f"| H2 | Attribution finds injected causes above chance, and deleting top-attributed "
        f"inputs beats deleting random ones | temporal hit@3 {_pct(t['hit_at_3'])} "
        f"(baseline {_pct(t['random_baseline'])}); command hit@3 {_pct(c['hit_at_3'])} "
        f"(baseline {_pct(c['random_baseline'])}); deletion {d['top_mean']:.2f} vs "
        f"{d['random_mean']:.2f} | {ok[h2]} |",
        f"| H3 | The 90% counterfactual band covers 80–95% of nominal test values | "
        f"{_pct(cov['mean'])} over {cov['n']} rollouts | {ok[h3]} |",
        f"| H4 | Every narrative sentence is grounded in computed facts | "
        f"{grounding['sentences'] - grounding['violations']} of {grounding['sentences']} "
        f"sentences | {ok[h4]} |",
        "",
        "## E1: onset",
        "",
        f"For each labelled anomaly, the first alarm event that overlaps it. Positive errors "
        f"are late. Median signed onset error: {onset['median_signed_onset_error']:g} steps. "
        f"95% CI of the median improvement (alarm error − onset error): "
        f"[{ci[0]:g}, {ci[1]:g}] steps.",
        "",
        "## E2: attribution faithfulness",
        "",
        f"Channels (fixed rule): {', '.join(attribution['channels'])}.",
        "",
        "| Injection | n | hit@3 [95% CI] | Random baseline |",
        "|---|---:|---|---:|",
        f"| Value spike at a random lag (is the lag in the top 3?) | {t['n']} | "
        f"{_pct(t['hit_at_3'])} [{_pct(t['ci95'][0])}, {_pct(t['ci95'][1])}] | "
        f"{_pct(t['random_baseline'])} |",
        f"| Spurious command (is the command in the top 3 inputs?) | {c['n']} | "
        f"{_pct(c['hit_at_3'])} [{_pct(c['ci95'][0])}, {_pct(c['ci95'][1])}] | "
        f"{_pct(c['random_baseline'])} |",
        "",
        f"{c['ignored_by_model']} command injections moved the forecast by less than "
        f"{config.explain.eval_min_effect:g} and were not counted: the model ignores those "
        "commands, so attribution should not point to them.",
        "",
        f"Deletion test ({d['n']} windows): resetting the top "
        f"{100 * config.explain.eval_deletion_fraction:g}% attributed inputs reduces the "
        f"error by {d['top_mean']:.2f} on average, a random "
        f"{100 * config.explain.eval_deletion_fraction:g}% by {d['random_mean']:.2f} "
        f"(difference 95% CI [{d['difference_ci95'][0]:.2f}, {d['difference_ci95'][1]:.2f}]).",
        "",
        "## E3: counterfactual band",
        "",
        "| Channel | Coverage of the 90% band | Band calibrated on |",
        "|---|---:|---|",
        *[
            f"| {chan} | {_pct(v)} | {cov['band_sources'][chan]} |"
            for chan, v in cov["per_channel"].items()
        ],
        "",
        "## Post-hoc diagnostics (not pre-registered)",
        "",
        "Added after the results above, to understand them. They do not change any verdict.",
        "",
        f"- **The LSTM has a short memory and nearly ignores commands.** Only "
        f"{_pct(ph['temporal_moved_share'])} of the injected value spikes and "
        f"{_pct(ph['command_moved_share'])} of the injected commands moved the forecast by "
        f"≥ {config.explain.eval_min_effect:g}. When a spike does move it, IG ranks its lag "
        f"in the top 3 in {_pct(ph['temporal_moved_hit_at_3'])} of cases "
        f"[{_pct(ph['temporal_moved_ci95'][0])}, {_pct(ph['temporal_moved_ci95'][1])}], "
        f"n = {ph['temporal_moved_n']}. Narratives that credit the channel's own recent "
        "values therefore describe the model faithfully: that is what it uses.",
        f"- **The deletion test was mis-specified.** Ranking by \\|IG\\| also removes "
        f"inputs that lower the error, so resetting them raised it (error reduced in "
        f"{_pct(ph['deletion_abs_share_reducing'])} of windows). Ranking by signed IG reduces "
        f"the error in {_pct(ph['deletion_positive_share_reducing'])} of windows, against "
        f"{_pct(ph['deletion_random_share_reducing'])} for random inputs. Zero-baseline "
        "deletion is also out of distribution (Hooker et al. 2019), so the means above are "
        "dominated by a few windows.",
        f"- **Band coverage fails on shifted channels.** Median per-channel coverage is "
        f"{_pct(ph['coverage_median_channel'])}; the mean is pulled down by a few channels "
        "whose test period differs from validation (the shift found in ADR 0006; A-5 was "
        "already flagged in the Phase 1 data card).",
        "",
        "## E4: grounding",
        "",
        f"{grounding['narratives']} narratives, {grounding['sentences']} sentences, "
        f"{grounding['violations']} grounding violations.",
    ]
    if example is not None:
        lines += [
            "",
            "## Example explanation",
            "",
            "The highest-surprise contextual event that overlaps a labelled anomaly (labels "
            "are used only to choose which event to show).",
            "",
        ]
        if figure_path:
            lines += [f"![Counterfactual for channel {example['chan_id']}]({figure_path})", ""]
        lines += [f"> {s['text']}" + "  " for s in example["sentences"]]
    lines.append("")
    return "\n".join(lines)


def write_report(config: Config, out: Path, *, jobs: int = 1) -> Path:
    out_dir = explain_run(config, jobs=jobs)
    onset = evaluate_onset(config)
    attribution = evaluate_attribution_and_band(config, jobs=jobs)
    grounding = evaluate_grounding(out_dir)
    example = pick_example(out_dir, config)
    figure_rel = None
    if example is not None:
        figure = out.parent / "assets" / "explanation-example.svg"
        figure.parent.mkdir(parents=True, exist_ok=True)
        figure.write_text(example_figure(config, example))
        figure_rel = figure.relative_to(out.parent).as_posix()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(config, onset, attribution, grounding, example, figure_rel))
    results = {
        "onset": {k: v for k, v in onset.items() if k != "rows"},
        "attribution": attribution,
        "grounding": grounding,
    }
    (out_dir / "evaluation.json").write_text(json.dumps(results, indent=2))
    return out
