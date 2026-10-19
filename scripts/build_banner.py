"""Generate docs/assets/banner.svg, the README header (deterministic, no dependencies).

The satellite photo (docs/assets/satellite.jpg) is embedded as a data URI, so the SVG is a
single self-contained file that GitHub renders without fetching anything. The left half adds
the title and a telemetry strip: a nominal trace, the stretch that drifts away from it, and
the anomaly score crossing a threshold fit without labels.

    uv run python scripts/build_banner.py
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


def stars(rng: random.Random, n: int, max_x: float) -> str:
    out = []
    for _ in range(n):
        x, y = rng.uniform(0, max_x), rng.uniform(0, H)
        r = rng.choice([0.6, 0.8, 1.0, 1.0, 1.3, 1.7])
        op = rng.uniform(0.3, 0.9)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="#fff" opacity="{op:.2f}"/>')
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
    <rect x="{onset_x:.1f}" y="{sy + 30}" width="{band_w:.1f}" height="{sh - 22}"
          fill="#f97316" opacity=".14"/>
    <text x="{onset_x + band_w / 2:.1f}" y="{sy + 44}" font-size="11" fill="#fdba74"
          text-anchor="middle">⚠ early alert</text>
    <polyline points="{trace}" fill="none" stroke="#38bdf8" stroke-width="1.8"
              stroke-linejoin="round"/>
    <polyline points="{bad}" fill="none" stroke="#fb923c" stroke-width="2.4"
              stroke-linejoin="round" filter="url(#glow)"/>
    <line x1="{sx}" y1="{thr_y:.1f}" x2="{sx + sw}" y2="{thr_y:.1f}" stroke="#f87171"
          stroke-dasharray="5 5" stroke-opacity=".85"/>
    <text x="{sx}" y="{thr_y - 6:.1f}" font-size="11" fill="#f87171">
      threshold · fit without labels</text>
    <polyline points="{score}" fill="none" stroke="#a78bfa" stroke-width="1.8"/>
    <text x="{sx}" y="{sy + sh + 12}" font-size="11" fill="#a78bfa">anomaly score</text>
  </g>
</svg>
'''


if __name__ == "__main__":
    OUT.write_text(build())
    print(OUT)
