"""Generate docs/assets/banner.svg, the README header (deterministic, no dependencies).

uv run python scripts/build_banner.py
"""

import math
import random
from pathlib import Path

W, H = 1280, 420
OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "banner.svg"


def stars(rng: random.Random, n: int) -> str:
    out = []
    for _ in range(n):
        x, y = rng.uniform(0, W), rng.uniform(0, H * 0.8)
        r = rng.choice([0.6, 0.8, 1.0, 1.0, 1.3, 1.8])
        op = rng.uniform(0.35, 0.95)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="#fff" opacity="{op:.2f}"/>')
    return "\n    ".join(out)


def telemetry(x0: float, y0: float, width: float, height: float) -> tuple[str, str, str]:
    """Nominal trace, the anomalous stretch, and the smoothed anomaly score below it."""
    rng = random.Random(7)
    n = 220
    onset, end = 150, 182
    pts, bad, score = [], [], []
    level = 0.0
    for i in range(n + 1):
        t = i / n
        v = 0.55 * math.sin(i / 7.0) + 0.25 * math.sin(i / 2.9) + rng.gauss(0, 0.05)
        if onset <= i <= end:  # drift away from the nominal pattern, then recover
            k = (i - onset) / (end - onset)
            v += 1.25 * math.sin(math.pi * k) + 0.25 * math.sin(i * 1.7)
        target = 1.0 if onset <= i <= end else 0.08 + 0.05 * abs(math.sin(i / 5))
        level += (target - level) * 0.18
        x = x0 + t * width
        y = y0 + height * 0.42 - v * height * 0.22
        pts.append(f"{x:.1f},{y:.1f}")
        if onset - 1 <= i <= end + 1:
            bad.append(f"{x:.1f},{y:.1f}")
        score.append(f"{x:.1f},{y0 + height - level * height * 0.22:.1f}")
    return " ".join(pts), " ".join(bad), " ".join(score)


def satellite() -> str:
    cells = []
    for side, x in (("l", -150), ("r", 40)):
        cells.append(
            f'<rect x="{x}" y="-26" width="110" height="52" rx="3" fill="url(#panel)" '
            'stroke="#9cc7ff" stroke-width="1.2"/>'
        )
        for i in range(1, 5):
            cells.append(
                f'<line x1="{x + 22 * i}" y1="-26" x2="{x + 22 * i}" y2="26" '
                'stroke="#9cc7ff" stroke-opacity=".45"/>'
            )
        cells.append(
            f'<line x1="{x}" y1="0" x2="{x + 110}" y2="0" stroke="#9cc7ff" stroke-opacity=".45"/>'
        )
        arm = (-40, -18) if side == "l" else (18, 40)
        cells.append(
            f'<line x1="{arm[0]}" y1="0" x2="{arm[1]}" y2="0" stroke="#cfd8e3" stroke-width="3"/>'
        )
    return "\n      ".join(cells)


def build() -> str:
    rng = random.Random(42)
    trace, bad, score = telemetry(770, 150, 440, 200)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     width="{W}" height="{H}"
     role="img" aria-label="sathealth: a satellite in orbit above Earth streams telemetry;
     an explainable AI flags the stretch that drifts away from nominal behaviour">
  <defs>
    <linearGradient id="space" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#030712"/>
      <stop offset=".65" stop-color="#0a1633"/>
      <stop offset="1" stop-color="#0f2350"/>
    </linearGradient>
    <radialGradient id="earth" cx=".42" cy=".28" r=".75">
      <stop offset="0" stop-color="#4fa3ff"/>
      <stop offset=".45" stop-color="#1d5fbf"/>
      <stop offset=".8" stop-color="#0b2a66"/>
      <stop offset="1" stop-color="#061536"/>
    </radialGradient>
    <radialGradient id="halo" cx=".5" cy=".5" r=".5">
      <stop offset=".86" stop-color="#5cc8ff" stop-opacity="0"/>
      <stop offset=".92" stop-color="#5cc8ff" stop-opacity=".55"/>
      <stop offset="1" stop-color="#5cc8ff" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="panel" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#1e3a8a"/>
      <stop offset=".5" stop-color="#2f5fd0"/>
      <stop offset="1" stop-color="#14286b"/>
    </linearGradient>
    <linearGradient id="foil" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#fde68a"/>
      <stop offset=".5" stop-color="#d4a017"/>
      <stop offset="1" stop-color="#8a6510"/>
    </linearGradient>
    <linearGradient id="card" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#0f1d3d" stop-opacity=".85"/>
      <stop offset="1" stop-color="#0a1430" stop-opacity=".85"/>
    </linearGradient>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="3" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <rect width="{W}" height="{H}" fill="url(#space)"/>
  <g>
    {stars(rng, 140)}
  </g>

  <!-- Earth and atmosphere -->
  <circle cx="250" cy="760" r="520" fill="url(#halo)"/>
  <circle cx="250" cy="760" r="470" fill="url(#earth)"/>

  <!-- orbit and downlink -->
  <ellipse cx="420" cy="300" rx="430" ry="120" fill="none" stroke="#7dd3fc"
           stroke-opacity=".35" stroke-width="1.5" stroke-dasharray="6 8"
           transform="rotate(-12 420 300)"/>
  <g fill="none" stroke="#7dd3fc" stroke-linecap="round" filter="url(#glow)">
    <path d="M 655 228 q 22 22 0 44" stroke-opacity=".8" stroke-width="2"/>
    <path d="M 672 215 q 36 35 0 70" stroke-opacity=".55" stroke-width="2"/>
    <path d="M 689 202 q 50 48 0 96" stroke-opacity=".3" stroke-width="2"/>
  </g>

  <!-- satellite -->
  <g transform="translate(540 225) rotate(-14)">
      {satellite()}
      <rect x="-18" y="-22" width="36" height="44" rx="4" fill="url(#foil)" stroke="#fef3c7"
            stroke-width="1"/>
      <line x1="0" y1="22" x2="0" y2="40" stroke="#cfd8e3" stroke-width="2"/>
      <path d="M -14 52 Q 0 34 14 52 Z" fill="#e5e7eb" stroke="#94a3b8"/>
      <circle cx="0" cy="-30" r="3" fill="#f87171" filter="url(#glow)"/>
  </g>

  <!-- title -->
  <g font-family="Inter, 'Segoe UI', Helvetica, Arial, sans-serif">
    <text x="64" y="92" font-size="58" font-weight="800" fill="#f8fafc"
          letter-spacing="-1">sat<tspan fill="#38bdf8">health</tspan></text>
    <text x="66" y="130" font-size="19" fill="#cbd5e1">
      Explainable AI for satellite health monitoring</text>
    <text x="66" y="156" font-size="19" fill="#cbd5e1">and early anomaly detection</text>
  </g>

  <!-- telemetry card -->
  <g font-family="'JetBrains Mono', Menlo, Consolas, monospace">
    <rect x="745" y="110" width="490" height="270" rx="14" fill="url(#card)" stroke="#334a7a"/>
    <text x="770" y="140" font-size="13" fill="#94a3b8">
      TELEMETRY · nominal pattern learned by LSTM</text>
    <rect x="1068" y="160" width="68" height="185" fill="#f97316" opacity=".12"/>
    <polyline points="{trace}" fill="none" stroke="#38bdf8" stroke-width="2"
              stroke-linejoin="round"/>
    <polyline points="{bad}" fill="none" stroke="#fb923c" stroke-width="2.6"
              stroke-linejoin="round" filter="url(#glow)"/>
    <line x1="770" y1="318" x2="1210" y2="318" stroke="#f87171" stroke-dasharray="5 5"
          stroke-opacity=".8"/>
    <polyline points="{score}" fill="none" stroke="#a78bfa" stroke-width="2"/>
    <text x="772" y="368" font-size="12" fill="#a78bfa">anomaly score</text>
    <text x="772" y="310" font-size="12" fill="#f87171">threshold · fit without labels</text>
    <text x="1102" y="176" font-size="12" fill="#fdba74" text-anchor="middle">⚠ early alert</text>
  </g>
</svg>
'''


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build())
    print(OUT)
