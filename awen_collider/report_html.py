"""Self-contained HTML run-report generator — pure Python, pure SVG.

Takes a RunReport JSON (as written by run_collider --json) and renders a
single static HTML page with SVG charts of the measured telemetry. No
JavaScript at all: the page is a plain document, so it renders
identically in browsers, preview panes, and print.

Standalone use:
    python -m awen_collider.report_html <run_report.json> <out.html>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

W, H = 1040, 200                      # chart viewBox
PL, PR, PT, PB = 78, 18, 14, 30       # padding: left/right/top/bottom
GOLD, CYAN, VIOLET, GREEN = "#e8b84b", "#39c6ff", "#9d7bff", "#4be08a"
DIM, LINE = "#5a6a92", "#1c2742"


def _svg_chart(ticks, series, fmt=None, mark=None, legend=None,
               mark_label="phase lock @", mark_text=None):
    """Render one chart as an SVG string.

    series: list of (values, color). fmt: y-label formatter.
    mark: x position of a vertical marker line, labeled
    "{mark_label} {mark}" — or mark_text verbatim if given.
    """
    fmt = fmt or (lambda v: f"{v:.3f}")
    ymin = min(min(ys) for ys, _ in series)
    ymax = max(max(ys) for ys, _ in series)
    pad = (ymax - ymin) * 0.12 or abs(ymax) * 0.1 or 1.0
    ymin, ymax = ymin - pad, ymax + pad
    xmin, xmax = ticks[0], ticks[-1]
    xs = lambda t: PL + (t - xmin) / (xmax - xmin) * (W - PL - PR)
    ys_ = lambda v: H - PB - (v - ymin) / (ymax - ymin) * (H - PT - PB)

    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
             f'font-family="Consolas,monospace" font-size="11">']
    # gridlines + y labels
    for i in range(5):
        v = ymin + (ymax - ymin) * i / 4
        y = ys_(v)
        parts.append(f'<line x1="{PL}" y1="{y:.1f}" x2="{W-PR}" y2="{y:.1f}" '
                     f'stroke="{LINE}" stroke-width="1"/>')
        parts.append(f'<text x="6" y="{y+3.5:.1f}" fill="{DIM}">{fmt(v)}</text>')
    # x labels
    for i in range(7):
        t = xmin + (xmax - xmin) * i / 6
        parts.append(f'<text x="{xs(t)-10:.1f}" y="{H-9}" fill="{DIM}">'
                     f'{round(t)}</text>')
    # phase-lock marker
    if mark is not None:
        x = xs(mark)
        parts.append(f'<line x1="{x:.1f}" y1="{PT}" x2="{x:.1f}" y2="{H-PB}" '
                     f'stroke="{GOLD}" stroke-opacity="0.45" '
                     f'stroke-dasharray="4 4"/>')
        label = mark_text if mark_text else f"{mark_label} {mark}"
        parts.append(f'<text x="{x+5:.1f}" y="{PT+11}" fill="{GOLD}">'
                     f'{label}</text>')
    # data
    for vals, color in series:
        pts = " ".join(f"{xs(t):.1f},{ys_(v):.1f}" for t, v in zip(ticks, vals))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     f'stroke-width="1.8"/>')
        for t, v in zip(ticks, vals):
            parts.append(f'<circle cx="{xs(t):.1f}" cy="{ys_(v):.1f}" r="2" '
                         f'fill="{color}"/>')
    if legend:
        lx = W - PR - 12 * max(len(s) for s, _ in legend) - 8
        for i, (label, color) in enumerate(legend):
            parts.append(f'<text x="{lx}" y="{PT + 13 + i*15}" '
                         f'fill="{color}">&#9632; {label}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _stat(k, v, color):
    return (f'<div class="stat"><div class="k">{k}</div>'
            f'<div class="v" style="color:{color}">{v}</div></div>')


def write_html_report(data: dict, out_path: Path) -> Path:
    hist = data["history"]
    ticks = [h["tick"] for h in hist]
    rift = [h["rift_mean"] for h in hist]
    lion = [h["lion_ratio"] for h in hist]
    parity = [math.log10(max(h["parity_drift"], 1e-18)) for h in hist]
    base = [math.log10(max(h["rift_baseline"], 1e-18)) for h in hist]

    tail = rift[len(rift) // 3:]
    rift_mu = sum(tail) / len(tail)
    lock_tick = next(
        (h["tick"] for h in hist
         if abs(h["rift_mean"] - rift_mu) / rift_mu < 0.005), None)

    gpu = "cuda" in data["backend"]
    backend_badge = ("GPU" if gpu else "CPU")
    mass = hist[-1].get("mass_index")
    stats = (
        _stat("PHASE LOCK", f"tick {lock_tick}" if lock_tick else "—", GOLD)
        + _stat("RIFT (locked mean)", f"{rift_mu:.6f}", VIOLET)
        + _stat("LION (final)", f"{lion[-1]:.4f}", GOLD)
        + (_stat("MASS INDEX (m=i)", f"{mass:.4f}", VIOLET)
           if mass is not None else "")
        + _stat("PARITY DRIFT", f"{hist[-1]['parity_drift']:.1e}", GREEN)
        + _stat("BASELINE", f"{hist[-1]['rift_baseline']:.1e}", GREEN)
        + _stat("MASS-GAP REF", f"{data['constants']['mass_gap']:.6f}", CYAN)
        + _stat("BACKEND", backend_badge, CYAN if gpu else GOLD)
    )

    c_rift = _svg_chart(ticks, [(rift, VIOLET)], mark=lock_tick)
    c_lion = _svg_chart(ticks, [(lion, GOLD)], mark=lock_tick)
    c_cons = _svg_chart(
        ticks, [(parity, GREEN), (base, CYAN)],
        fmt=lambda v: f"1e{round(v)}",
        legend=[("parity drift", GREEN), ("H-only baseline", CYAN)])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Awen Collider — Run Report</title>
<style>
  body{{background:#06080f;color:#c8d4f0;
       font:13px/1.5 "Cascadia Mono","Consolas",monospace;padding:20px;margin:0}}
  h1{{font-size:16px;color:{GOLD};letter-spacing:.08em;margin:0 0 2px}}
  .sub{{color:{DIM};font-size:11px;margin-bottom:18px}}
  .stats{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px}}
  .stat{{background:#0b1020;border:1px solid {LINE};border-radius:6px;
        padding:10px 14px;min-width:140px}}
  .stat .k{{color:{DIM};font-size:10px;letter-spacing:.12em}}
  .stat .v{{font-size:16px;margin-top:2px}}
  .chart{{background:#0b1020;border:1px solid {LINE};
         border-radius:6px;padding:12px;margin-bottom:14px}}
  .chart h2{{font-size:11px;color:{DIM};letter-spacing:.14em;margin:0 0 6px}}
  svg{{width:100%;height:auto;display:block}}
  .foot{{color:{DIM};font-size:10px;margin-top:14px}}
</style>
</head>
<body>
<h1>AWEN GRID DIGITAL COLLIDER — RUN REPORT</h1>
<div class="sub">{data['nodes']:,} nodes/ledger · {data['ticks']} ticks ·
{data['backend']} · toggle cycle complete: {data['toggle_cycle_complete']}</div>
<div class="stats">{stats}</div>
<div class="chart"><h2>ASSOCIATIVE RIFT &lang;|[R,I,probe]|&rang; — collision signal</h2>{c_rift}</div>
<div class="chart"><h2>LION RATIO median |ijk|/|w| — Real Ledger attractor</h2>{c_lion}</div>
<div class="chart"><h2>CONSERVATION — log10 parity drift &amp; H-only baseline</h2>{c_cons}</div>
<div class="foot">Numerical simulation record — computes geometry, does not
act on matter. Generated by awen_collider.report_html (static SVG, no
scripts). Live animated view: visualizer.html.</div>
</body>
</html>
"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m awen_collider.report_html "
              "<run_report.json> <out.html>")
        return 2
    data = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    out = write_html_report(data, Path(argv[1]))
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
