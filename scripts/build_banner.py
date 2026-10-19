"""Generate the README artwork: docs/assets/banner.svg and docs/assets/pipeline.svg.

Both are deterministic, self-contained SVGs (no scripts, no external requests). GitHub strips
JavaScript and CSS from READMEs but plays CSS animations inside SVG images, so the motion
lives in the files themselves; ``prefers-reduced-motion`` turns it off and leaves a complete
static picture.

The satellite photo (docs/assets/satellite.jpg) is embedded as a data URI, so the SVG is a
single self-contained file that GitHub renders without fetching anything. The left half adds
the title and a telemetry strip: a nominal trace, the stretch that drifts away from it, and
the anomaly score crossing a threshold fit without labels.

    uv run python scripts/build_banner.py   (or: make banner)
"""

import base64
import math
import random
from pathlib import Path

W, H = 1280, 480
ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets"
PHOTO = ASSETS / "satellite.jpg"
PHOTO_W, PHOTO_H = 944, 630  # pixel size of satellite.jpg
OUT = ASSETS / "banner.svg"
PIPELINE_OUT = ASSETS / "pipeline.svg"

# One 8-second loop: the trace draws in, the anomaly lights up when the trace reaches it,
# the alert band pulses; stars twinkle on their own clocks.
BANNER_STYLE = """
    .star { animation: twinkle 3.2s ease-in-out infinite; }
    .draw { stroke-dasharray: 1; animation: draw 8s ease-out infinite; }
    .reveal { animation: reveal 8s ease-out infinite; }
    .pulse { animation: pulse 1.6s ease-in-out infinite; }
    .live { animation: blink 1.2s steps(2, jump-none) infinite; }
    @keyframes twinkle { 0%, 100% { opacity: .2; } 50% { opacity: .95; } }
    @keyframes draw {
      0% { stroke-dashoffset: 1; opacity: 1; }
      55% { stroke-dashoffset: 0; }
      92% { stroke-dashoffset: 0; opacity: 1; }
      100% { stroke-dashoffset: 0; opacity: 0; }
    }
    @keyframes reveal {
      0%, 37% { opacity: 0; } 42%, 92% { opacity: 1; } 100% { opacity: 0; }
    }
    @keyframes pulse { 0%, 100% { opacity: .1; } 50% { opacity: .32; } }
    @keyframes blink { 0% { opacity: 1; } 100% { opacity: .15; } }
    @media (prefers-reduced-motion: reduce) {
      .star, .draw, .reveal, .pulse, .live { animation: none; }
    }
"""


def stars(rng: random.Random, n: int, max_x: float) -> str:
    out = []
    for _ in range(n):
        x, y = rng.uniform(0, max_x), rng.uniform(0, H)
        r = rng.choice([0.6, 0.8, 1.0, 1.0, 1.3, 1.7])
        op = rng.uniform(0.3, 0.9)
        delay, period = rng.uniform(0, 3), rng.uniform(2.4, 4.5)
        out.append(
            f'<circle class="star" cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="#fff" '
            f'opacity="{op:.2f}" style="animation-delay:-{delay:.2f}s;'
            f'animation-duration:{period:.2f}s"/>'
        )
    return "\n    ".join(out)


def telemetry(x0: float, y0: float, width: float, height: float) -> tuple[str, str, str, float]:
    """Nominal trace, the anomalous stretch, the smoothed score, and the onset x position."""
    rng = random.Random(7)
    n = 220
    onset, end = 150, 182
    pts, bad, score = [], [], []
    level = 0.0
    for i in range(n + 1):
        v = 0.55 * math.sin(i / 7.0) + 0.25 * math.sin(i / 2.9) + rng.gauss(0, 0.05)
        if onset <= i <= end:  # drift away from the nominal pattern, then recover
            k = (i - onset) / (end - onset)
            v += 1.25 * math.sin(math.pi * k) + 0.25 * math.sin(i * 1.7)
        target = 1.0 if onset <= i <= end else 0.08 + 0.05 * abs(math.sin(i / 5))
        level += (target - level) * 0.18
        x = x0 + i / n * width
        y = y0 + height * 0.40 - v * height * 0.20
        pts.append(f"{x:.1f},{y:.1f}")
        if onset - 1 <= i <= end + 1:
            bad.append(f"{x:.1f},{y:.1f}")
        score.append(f"{x:.1f},{y0 + height - level * height * 0.24:.1f}")
    return " ".join(pts), " ".join(bad), " ".join(score), x0 + onset / n * width


def build() -> str:
    rng = random.Random(42)
    photo = base64.b64encode(PHOTO.read_bytes()).decode("ascii")
    photo_w = PHOTO_W * H / PHOTO_H  # scale to full banner height
    photo_x = W - photo_w
    sx, sy, sw, sh = 64, 236, 470, 190  # telemetry strip
    trace, bad, score, onset_x = telemetry(sx, sy + 26, sw, sh - 40)
    band_w = 32 / 220 * sw
    thr_y = sy + 26 + (sh - 40) * (1 - 0.5 * 0.24)  # score level 0.5
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     width="{W}" height="{H}" role="img"
     aria-label="sathealth: a satellite in orbit above Earth; its telemetry is watched by an
     explainable AI that flags the stretch drifting away from nominal behaviour">
  <style>{BANNER_STYLE}</style>
  <defs>
    <linearGradient id="fade" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#030712"/>
      <stop offset=".22" stop-color="#030712" stop-opacity=".85"/>
      <stop offset=".5" stop-color="#030712" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="bottom" x1="0" y1="0" x2="0" y2="1">
      <stop offset=".75" stop-color="#030712" stop-opacity="0"/>
      <stop offset="1" stop-color="#030712" stop-opacity=".55"/>
    </linearGradient>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="2.5" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <rect width="{W}" height="{H}" fill="#030712"/>
  <g>
    {stars(rng, 90, photo_x + 120)}
  </g>
  <image x="{photo_x:.1f}" y="0" width="{photo_w:.1f}" height="{H}"
         preserveAspectRatio="xMidYMid slice"
         href="data:image/jpeg;base64,{photo}"/>
  <!-- the fade starts a few px left of the photo so its edge never shows as a seam -->
  <rect x="{photo_x - 4:.1f}" y="0" width="{photo_w + 4:.1f}" height="{H}" fill="url(#fade)"/>
  <rect width="{W}" height="{H}" fill="url(#bottom)"/>

  <!-- title -->
  <g font-family="Inter, 'Segoe UI', Helvetica, Arial, sans-serif">
    <text x="62" y="104" font-size="64" font-weight="800" fill="#f8fafc"
          letter-spacing="-1.5">sat<tspan fill="#38bdf8">health</tspan></text>
    <text x="65" y="146" font-size="21" fill="#e2e8f0">
      Explainable AI for satellite health monitoring</text>
    <text x="65" y="174" font-size="21" fill="#e2e8f0">and early anomaly detection</text>
    <text x="65" y="208" font-size="13" fill="#94a3b8" letter-spacing=".5">
      NASA SMAP / MSL · LSTM · Transformer · label-free thresholds</text>
  </g>

  <!-- telemetry strip -->
  <g font-family="'JetBrains Mono', Menlo, Consolas, monospace">
    <rect x="{sx - 16}" y="{sy}" width="{sw + 32}" height="{sh + 20}" rx="12"
          fill="#0b1530" fill-opacity=".72" stroke="#334a7a" stroke-opacity=".8"/>
    <text x="{sx}" y="{sy + 22}" font-size="12" fill="#94a3b8">
      TELEMETRY · nominal pattern learned by the model</text>
    <circle class="live" cx="{sx + sw - 30}" cy="{sy + 18}" r="4" fill="#ef4444"/>
    <text x="{sx + sw - 22}" y="{sy + 22}" font-size="11" fill="#fca5a5">LIVE</text>
    <rect class="pulse" x="{onset_x:.1f}" y="{sy + 30}" width="{band_w:.1f}"
          height="{sh - 22}" fill="#f97316" opacity=".14"/>
    <text class="reveal" x="{onset_x + band_w / 2:.1f}" y="{sy + 44}" font-size="11"
          fill="#fdba74" text-anchor="middle">⚠ early alert</text>
    <polyline class="draw" pathLength="1" points="{trace}" fill="none" stroke="#38bdf8"
              stroke-width="1.8" stroke-linejoin="round"/>
    <polyline class="reveal" points="{bad}" fill="none" stroke="#fb923c" stroke-width="2.4"
              stroke-linejoin="round" filter="url(#glow)"/>
    <line x1="{sx}" y1="{thr_y:.1f}" x2="{sx + sw}" y2="{thr_y:.1f}" stroke="#f87171"
          stroke-dasharray="5 5" stroke-opacity=".85"/>
    <text x="{sx}" y="{thr_y - 6:.1f}" font-size="11" fill="#f87171">
      threshold · fit without labels</text>
    <polyline class="draw" pathLength="1" points="{score}" fill="none" stroke="#a78bfa"
              stroke-width="1.8"/>
    <text x="{sx}" y="{sy + sh + 12}" font-size="11" fill="#a78bfa">anomaly score</text>
  </g>
</svg>
'''


PIPELINE_STAGES = (
    ("Telemetry", "NASA SMAP · MSL", "ESA OPS-SAT replay", "#38bdf8"),
    ("Detectors", "LSTM forecaster", "Transformer AE", "#a78bfa"),
    ("Decision", "EVT surprise · fusion", "hysteresis · events", "#fb923c"),
    ("Explanation", "onset · counterfactual", "attribution · grounded text", "#f472b6"),
    ("Operator", "live console", "verdicts · re-baseline", "#34d399"),
)


def pipeline() -> str:
    """The five stages as cards, with data pulses flowing along the arrows between them."""
    w, h, card_w, card_h, top = 1280, 210, 200, 120, 52
    gap = (w - 2 * 40 - len(PIPELINE_STAGES) * card_w) / (len(PIPELINE_STAGES) - 1)
    parts = []
    for i, (title, line1, line2, color) in enumerate(PIPELINE_STAGES):
        x = 40 + i * (card_w + gap)
        parts.append(
            f'<g><rect x="{x:.1f}" y="{top}" width="{card_w}" height="{card_h}" rx="14" '
            f'fill="#0b1530" stroke="{color}" stroke-width="2"/>'
            f'<rect class="glow" x="{x:.1f}" y="{top}" width="{card_w}" height="{card_h}" '
            f'rx="14" fill="none" stroke="{color}" stroke-width="6" opacity=".15" '
            f'style="animation-delay:{i * 0.5:.1f}s"/>'
            f'<circle cx="{x + 26:.1f}" cy="{top + 30}" r="12" fill="{color}"/>'
            f'<text x="{x + 26:.1f}" y="{top + 35}" text-anchor="middle" font-size="13" '
            f'font-weight="800" fill="#030712">{i + 1}</text>'
            f'<text x="{x + 46:.1f}" y="{top + 36}" font-size="18" font-weight="700" '
            f'fill="#f8fafc">{title}</text>'
            f'<text x="{x + 18:.1f}" y="{top + 72}" font-size="13" fill="#cbd5e1">{line1}</text>'
            f'<text x="{x + 18:.1f}" y="{top + 94}" font-size="13" fill="#94a3b8">{line2}</text>'
            "</g>"
        )
        if i < len(PIPELINE_STAGES) - 1:
            x0, x1, y = x + card_w + 6, x + card_w + gap - 6, top + card_h / 2
            nxt = PIPELINE_STAGES[i + 1][3]
            parts.append(
                f'<line x1="{x0:.1f}" y1="{y}" x2="{x1:.1f}" y2="{y}" stroke="#334a7a" '
                f'stroke-width="2"/><polygon points="{x1:.1f},{y} {x1 - 8:.1f},{y - 5} '
                f'{x1 - 8:.1f},{y + 5}" fill="#334a7a"/>'
            )
            for k in range(2):
                parts.append(
                    f'<circle class="flow" cx="{x0:.1f}" cy="{y}" r="4" fill="{nxt}" '
                    f'style="--d:{x1 - x0 - 8:.1f}px;animation-delay:{i * 0.4 + k * 0.9:.1f}s"/>'
                )
    body = "\n  ".join(parts)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}"
     height="{h}" role="img" font-family="Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
     aria-label="Pipeline: telemetry, detectors, decision layer, explanation, operator">
  <style>
    .flow {{ animation: flow 1.8s linear infinite; opacity: 0; }}
    .glow {{ animation: glow 2.5s ease-in-out infinite; }}
    @keyframes flow {{
      0% {{ transform: translateX(0); opacity: 0; }}
      15% {{ opacity: 1; }} 85% {{ opacity: 1; }}
      100% {{ transform: translateX(var(--d)); opacity: 0; }}
    }}
    @keyframes glow {{ 0%, 100% {{ opacity: .08; }} 50% {{ opacity: .35; }} }}
    @media (prefers-reduced-motion: reduce) {{
      .flow {{ animation: none; opacity: 0; }} .glow {{ animation: none; }}
    }}
  </style>
  <rect width="{w}" height="{h}" rx="16" fill="#030712"/>
  <text x="40" y="34" font-size="14" font-weight="600" fill="#94a3b8"
        letter-spacing="1.5">HOW AN ALERT IS BORN</text>
  {body}
  <text x="{w - 40}" y="{h - 14}" text-anchor="end" font-size="12" fill="#64748b">
    every step causal · every sentence grounded · every claim pre-registered</text>
</svg>
"""


if __name__ == "__main__":
    OUT.write_text(build())
    PIPELINE_OUT.write_text(pipeline())
    print(OUT, PIPELINE_OUT)
