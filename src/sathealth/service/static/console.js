// sathealth operator console: live alerts over server-sent events, health, explanations,
// operator verdicts. Plain DOM APIs only; data is always inserted as text, never as HTML.
"use strict";

const SVG = "http://www.w3.org/2000/svg";
const SEVERITY_ICON = { watch: "▲", warning: "◆", critical: "■" };
const state = { selected: null, status: null, timers: {} };

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const child of children) node.append(child);
  return node;
}

function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

function badge(severity) {
  return el("span", { class: `badge ${severity}`, text: `${SEVERITY_ICON[severity]} ${severity}` });
}

// --- charts -----------------------------------------------------------------------------

function niceTicks(lo, hi, n = 4) {
  const raw = (hi - lo) / Math.max(n - 1, 1) || 1;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).reduce((a, b) =>
    Math.abs(b - raw) < Math.abs(a - raw) ? b : a);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step / 2; v += step) ticks.push(+v.toFixed(6));
  return ticks;
}

/** Line chart with optional band and spans; hover shows a crosshair and the value. */
function lineChart(container, { x, series, band, spans = [], threshold, format = (v) => v.toFixed(3) }) {
  container.replaceChildren();
  if (!x.length) return;
  const W = 640, H = 170, L = 46, R = 12, T = 8, B = 22;
  const all = series.flatMap((s) => s.values).concat(band ? [...band.lower, ...band.upper] : []);
  if (threshold !== undefined) all.push(threshold);
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.08 || 1;
  lo -= pad; hi += pad;
  const x0 = x[0], x1 = x[x.length - 1] || x0 + 1;
  const sx = (v) => L + ((v - x0) / (x1 - x0 || 1)) * (W - L - R);
  const sy = (v) => T + ((hi - v) / (hi - lo)) * (H - T - B);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  const grid = svg("g", { class: "grid" });
  for (const t of niceTicks(lo, hi)) {
    grid.append(svg("line", { x1: L, x2: W - R, y1: sy(t), y2: sy(t) }));
    const label = svg("text", { x: L - 6, y: sy(t) + 3, "text-anchor": "end", class: "axis-label" });
    label.textContent = String(t);
    grid.append(label);
  }
  for (const t of niceTicks(x0, x1, 5)) {
    const label = svg("text", { x: sx(t), y: H - 6, "text-anchor": "middle", class: "axis-label" });
    label.textContent = String(Math.round(t));
    grid.append(label);
  }
  root.append(grid);
  for (const span of spans) {
    const a = Math.max(span.start, x0), b = Math.min(span.end, x1);
    if (a > b) continue;
    root.append(svg("rect", { x: sx(a), y: T, width: Math.max(sx(b) - sx(a), 1.5), height: H - T - B, class: span.cls }));
  }
  if (band) {
    const top = band.x.map((v, i) => `${sx(v)},${sy(band.upper[i])}`);
    const bottom = band.x.map((v, i) => `${sx(v)},${sy(band.lower[i])}`).reverse();
    root.append(svg("polygon", { points: top.concat(bottom).join(" "), class: "band" }));
  }
  if (threshold !== undefined) {
    root.append(svg("line", { x1: L, x2: W - R, y1: sy(threshold), y2: sy(threshold), class: "threshold" }));
  }
  for (const s of series) {
    const xs = s.x || x;
    root.append(svg("polyline", {
      points: s.values.map((v, i) => `${sx(xs[i])},${sy(v)}`).join(" "),
      fill: "none", "stroke-width": 2, "stroke-linejoin": "round", class: s.cls,
      ...(s.dashed ? { "stroke-dasharray": "5 3" } : {}),
    }));
  }
  const cross = svg("line", { y1: T, y2: H - B, class: "crosshair", visibility: "hidden" });
  root.append(cross);
  const hit = svg("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent" });
  const tip = $("tooltip");
  hit.addEventListener("mousemove", (ev) => {
    const box = root.getBoundingClientRect();
    const vx = x0 + ((ev.clientX - box.left) * (W / box.width) - L) / (W - L - R) * (x1 - x0);
    const i = Math.max(0, Math.min(x.length - 1, Math.round(vx - x0)));
    cross.setAttribute("x1", sx(x[i])); cross.setAttribute("x2", sx(x[i]));
    cross.setAttribute("visibility", "visible");
    const parts = [`step ${x[i]}`];
    for (const s of series) {
      const xs = s.x || x;
      const j = xs.indexOf(x[i]);
      if (j >= 0) parts.push(`${s.name}: ${format(s.values[j])}`);
    }
    tip.textContent = parts.join(" · ");
    tip.hidden = false;
    tip.style.left = `${ev.clientX + 12}px`;
    tip.style.top = `${ev.clientY + 12}px`;
  });
  hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); tip.hidden = true; });
  root.append(hit);
  container.append(root);
}

// --- health -----------------------------------------------------------------------------

async function refreshHealth() {
  const health = await api("/api/health");
  $("healthy").textContent = `${health.healthy_channels}/${health.total_channels}`;
  $("lowest").textContent = `lowest channel index ${health.overall.toFixed(0)}`;
  const tiles = health.subsystems.map((s) => {
    const trend = s.trend && s.trend.steps_to_threshold !== null
      ? `≈${Math.round(s.trend.steps_to_threshold)} steps to 50 (trend)`
      : "no downward trend";
    const alarm = `${s.in_alarm}/${s.channels} channels in alarm`;
    return el("div", { class: "tile", title: `weakest: ${s.weakest_channel}. ${s.trend ? s.trend.label : ""}` }, [
      el("b", { text: `Family ${s.subsystem}` }),
      el("span", { class: "value", text: s.health.toFixed(0) }),
      el("span", { class: "sub", text: alarm }),
      el("span", { class: "sub", text: trend }),
    ]);
  });
  $("subsystems").replaceChildren(...tiles);
  const drift = (await api("/api/drift")).filter((d) => d.drifting);
  $("drift").replaceChildren(
    el("b", { text: `${drift.length} channel${drift.length === 1 ? "" : "s"} drifting` }),
    el("span", { text: drift.length ? `: ${drift.map((d) => d.chan_id).join(", ")} (value shift or a sustained alarm; label-free)` : " (label-free monitor)" }),
  );
}

// --- alerts -----------------------------------------------------------------------------

async function refreshAlerts() {
  const params = new URLSearchParams();
  if ($("filter-status").value) params.set("status", $("filter-status").value);
  if ($("filter-severity").value) params.set("severity", $("filter-severity").value);
  const events = await api(`/api/events?${params}`);
  const items = events.map((e) => {
    const li = el("li", { class: `${e.status}${state.selected === e.id ? " selected" : ""}` }, [
      badge(e.severity),
      el("span", { text: `${e.chan_id} · ${e.kind}` }),
      el("span", { class: "verdict-tag", text: e.verdict ? e.verdict.replace("_", " ") : e.status }),
      el("span", { class: "meta", text: `steps ${e.start}–${e.end} · peak surprise ${e.peak_surprise.toFixed(1)}` }),
    ]);
    li.addEventListener("click", () => select(e.id));
    return li;
  });
  $("alert-list").replaceChildren(...items);
  const fb = await api("/api/feedback/summary");
  const p = fb.all.precision === null ? "–" : `${Math.round(fb.all.precision * 100)}%`;
  $("feedback").textContent =
    `Reviewed ${fb.all.reviewed} · confirmed ${fb.all.confirmed} · false alarms ${fb.all.false_alarm} · operator-confirmed precision ${p}`;
}

// --- detail -----------------------------------------------------------------------------

async function select(eventId) {
  state.selected = eventId;
  history.replaceState(null, "", `#event=${eventId}`);
  $("detail-empty").hidden = true;
  $("detail").hidden = false;
  const event = await api(`/api/events/${eventId}`);
  $("detail-title").textContent = `${event.chan_id} (${event.spacecraft}) · event ${event.id}`;
  await Promise.all([refreshTelemetry(event), refreshExplanation(eventId), refreshVerdicts(eventId)]);
  refreshAlerts();
}

async function refreshTelemetry(event) {
  const t = await api(`/api/channels/${encodeURIComponent(event.chan_id)}/telemetry?window=900`);
  const x = t.values.map((_, i) => t.start + i);
  const spans = [{ start: event.start, end: event.end, cls: "alarm-span" }];
  if (t.labels) {
    let s = null;
    t.labels.forEach((on, i) => {
      if (on && s === null) s = x[i];
      if ((!on || i === t.labels.length - 1) && s !== null) { spans.push({ start: s, end: x[i], cls: "label-span" }); s = null; }
    });
  }
  lineChart($("chart-value"), { x, series: [{ name: "value", values: t.values, cls: "line-1" }], spans });
  lineChart($("chart-surprise"), {
    x, series: [{ name: "surprise", values: t.surprise, cls: "line-1" }], spans,
    threshold: t.alarm_on, format: (v) => v.toFixed(2),
  });
}

async function refreshExplanation(eventId) {
  const box = $("explanation");
  box.replaceChildren(el("span", { class: "muted", text: "Computing explanation…" }));
  try {
    const e = await api(`/api/events/${eventId}/explanation`);
    const paragraphs = e.sentences.map((s) => el("p", { text: s.text }));
    if (e.status === "open") paragraphs.push(el("p", { class: "provisional", text: `Provisional: event still open (as of step ${e.as_of_step}).` }));
    box.replaceChildren(...paragraphs);
    const cf = e.series && e.series.expected;
    $("cf-figure").hidden = !cf;
    if (cf) {
      const start = e.counterfactual.start;
      const x = cf.map((_, i) => start + i);
      lineChart($("chart-cf"), {
        x,
        series: [
          { name: "actual", values: e.series.actual, cls: "line-1" },
          { name: "expected", values: e.series.expected, cls: "line-2", dashed: true },
        ],
        band: { x, lower: e.series.lower, upper: e.series.upper },
        spans: [{ start: e.event.start, end: e.event.end, cls: "alarm-span" }],
      });
    }
  } catch (err) {
    box.replaceChildren(el("span", { class: "muted", text: `Explanation unavailable: ${err.message}` }));
  }
}

async function refreshVerdicts(eventId) {
  const rows = await api(`/api/events/${eventId}/verdicts`);
  $("verdicts").replaceChildren(...rows.map((v) =>
    el("li", { text: `${v.created_at.slice(0, 19).replace("T", " ")} · ${v.operator}: ${v.verdict.replace("_", " ")}${v.note ? ` (${v.note})` : ""}` })));
}

$("verdict-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (state.selected === null) return;
  const verdict = ev.submitter ? ev.submitter.value : "unsure";
  await api(`/api/events/${state.selected}/verdict`, {
    method: "POST",
    body: JSON.stringify({ verdict, operator: $("operator").value, note: $("note").value }),
  });
  try { localStorage.setItem("sathealth.operator", $("operator").value); } catch (_) { /* private mode */ }
  $("note").value = "";
  await Promise.all([refreshVerdicts(state.selected), refreshAlerts()]);
});

$("rebaseline").addEventListener("click", async () => {
  if (state.selected === null) return;
  const operator = $("operator").value.trim();
  if (!operator) { $("rebaseline-status").textContent = "Enter your operator name first."; return; }
  const event = await api(`/api/events/${state.selected}`);
  const result = await api(`/api/channels/${encodeURIComponent(event.chan_id)}/rebaseline`, {
    method: "POST",
    body: JSON.stringify({ operator, note: $("note").value }),
  });
  $("rebaseline-status").textContent = `Re-baselined ${result.chan_id} at step ${result.step} on its last ${result.steps} steps.`;
  await Promise.all([refreshAlerts(), refreshHealth()]);
});

// --- replay controls and live stream -----------------------------------------------------

function showStatus(s) {
  state.status = s;
  $("step").textContent = `step ${s.step} / ${s.total}`;
  $("progress-fill").style.width = `${s.total ? (100 * s.step) / s.total : 0}%`;
  $("play").textContent = s.running ? "Pause" : s.finished ? "Finished" : "Resume";
  const speed = $("speed");
  if (![...speed.options].some((o) => +o.value === s.speed)) speed.append(el("option", { value: String(s.speed), text: `${s.speed} steps/s` }));
  speed.value = String(s.speed);
}

$("play").addEventListener("click", async () => {
  const s = state.status;
  showStatus(await api(s && s.running ? "/api/replay/pause" : "/api/replay/resume", { method: "POST" }));
});
$("restart").addEventListener("click", async () => {
  showStatus(await api("/api/replay", { method: "POST", body: JSON.stringify({ speed: +$("speed").value }) }));
  state.selected = null;
  $("detail").hidden = true; $("detail-empty").hidden = false;
  refreshAlerts(); refreshHealth();
});
$("speed").addEventListener("change", async () => {
  showStatus(await api("/api/replay/speed", { method: "PUT", body: JSON.stringify({ speed: +$("speed").value }) }));
});
$("filter-status").addEventListener("change", refreshAlerts);
$("filter-severity").addEventListener("change", refreshAlerts);

function throttle(fn, ms) {
  let pending = false;
  return () => {
    if (pending) return;
    pending = true;
    setTimeout(() => { pending = false; fn(); }, ms);
  };
}

const alertsSoon = throttle(refreshAlerts, 500);
const healthSoon = throttle(refreshHealth, 1000);

function connect() {
  const source = new EventSource("/api/stream");
  source.addEventListener("open", () => $("conn").classList.add("live"));
  source.addEventListener("error", () => $("conn").classList.remove("live"));
  source.addEventListener("status", (msg) => { showStatus(JSON.parse(msg.data)); healthSoon(); });
  source.addEventListener("alert", () => { alertsSoon(); healthSoon(); });
  source.addEventListener("verdict", alertsSoon);
  source.addEventListener("rebaseline", () => { alertsSoon(); healthSoon(); });
}

async function tickSelected() {
  if (state.selected === null || !state.status || !state.status.running) return;
  const event = await api(`/api/events/${state.selected}`);
  refreshTelemetry(event);
}

async function init() {
  try { $("operator").value = localStorage.getItem("sathealth.operator") || ""; } catch (_) { /* private mode */ }
  showStatus(await api("/api/status"));
  await Promise.all([refreshHealth(), refreshAlerts()]);
  connect();
  state.timers.selected = setInterval(tickSelected, 1500);
  const linked = /^#event=(\d+)$/.exec(window.location.hash);  // shareable alert links
  if (linked) select(Number(linked[1]));
}

init().catch((err) => { $("detail-empty").textContent = `Service not ready: ${err.message}`; });
