"""Cross-scale combined report — merges the 144k / 1M / 10M Level I runs
into one page: a scale-comparison table plus overlaid rift and lion
curves. Renders whichever of the three run JSONs exist; missing ones are
listed so the page is honest about coverage.

Usage:  python experiments/combine_reports.py
Output: logs/level1_combined_report.html
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from awen_collider.report_html import _svg_chart, GOLD, CYAN, VIOLET, DIM, LINE  # noqa: E402

RUNS = [
    ("144k", "level1_144k.json", GOLD),
    ("1M", "level1_1M.json", CYAN),
    ("10M", "level1_10M.json", VIOLET),
]


def main() -> int:
    logs = ROOT / "logs"
    loaded, missing = [], []
    for label, fname, color in RUNS:
        p = logs / fname
        if p.exists():
            loaded.append((label, json.loads(p.read_text(encoding="utf-8")),
                           color))
        else:
            missing.append(fname)
    if not loaded:
        print("No run JSONs found — run the Level I bats first.")
        return 1

    # cross-scale stat table
    head = ("<tr><th>beam</th><th>nodes</th><th>locked rift (mean±sd)</th>"
            "<th>locked lion (mean±sd)</th><th>mass idx</th>"
            "<th>parity</th><th>baseline</th><th>backend</th></tr>")
    trows = []
    for label, d, color in loaded:
        h = d["history"]
        locked = [r for r in h if r["tick"] >= 36]
        rift = [r["rift_mean"] for r in locked]
        lion = [r["lion_ratio"] for r in locked]
        last = h[-1]
        gpu = "cuda" in d["backend"]
        trows.append(
            f"<tr><td style='color:{color}'>{label}</td>"
            f"<td>{d['nodes']:,}</td>"
            f"<td>{st.mean(rift):.6f} ± {st.pstdev(rift):.6f}</td>"
            f"<td>{st.mean(lion):.4f} ± {st.pstdev(lion):.4f}</td>"
            f"<td>{last.get('mass_index', float('nan')):.4f}</td>"
            f"<td>{last['parity_drift']:.1e}</td>"
            f"<td>{last['rift_baseline']:.1e}</td>"
            f"<td>{'GPU' if gpu else 'CPU'}</td></tr>")

    # overlaid curves (shared tick axis — all runs use ticks 9..369 step 9)
    ticks = [r["tick"] for r in loaded[0][1]["history"]]
    rift_series = [([r["rift_mean"] for r in d["history"]], color)
                   for _, d, color in loaded]
    lion_series = [([r["lion_ratio"] for r in d["history"]], color)
                   for _, d, color in loaded]
    legend = [(label, color) for label, _, color in loaded]
    c_rift = _svg_chart(ticks, rift_series, legend=legend, mark=27,
                        mark_label="phase lock @")
    c_lion = _svg_chart(ticks, lion_series, legend=legend, mark=27,
                        mark_label="phase lock @")

    missing_note = (f"Not yet run: {', '.join(missing)}."
                    if missing else "All three scales present.")
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Level I — Cross-Scale Combined Report</title>
<style>
 body{{background:#06080f;color:#c8d4f0;font:13px/1.5 Consolas,monospace;
      padding:20px;margin:0}}
 h1{{font-size:16px;color:{GOLD};letter-spacing:.08em;margin:0 0 2px}}
 .sub{{color:{DIM};font-size:11px;margin-bottom:16px}}
 table{{border-collapse:collapse;margin-bottom:16px;width:100%}}
 th,td{{border:1px solid {LINE};padding:6px 10px;text-align:left;
       font-size:12px}}
 th{{color:{DIM};font-size:10px;letter-spacing:.12em}}
 .chart{{background:#0b1020;border:1px solid {LINE};border-radius:6px;
        padding:12px;margin-bottom:14px}}
 .chart h2{{font-size:11px;color:{DIM};letter-spacing:.14em;margin:0 0 6px}}
 svg{{width:100%;height:auto;display:block}}
 .foot{{color:{DIM};font-size:10px;margin-top:12px}}
</style></head><body>
<h1>LEVEL I — CROSS-SCALE COMBINED REPORT</h1>
<div class="sub">369 ticks · measured every 9 · identical directive
dynamics per scale · {missing_note}</div>
<table>{head}{''.join(trows)}</table>
<div class="chart"><h2>ASSOCIATIVE RIFT vs TICK — all scales overlaid</h2>{c_rift}</div>
<div class="chart"><h2>LION RATIO vs TICK — all scales overlaid</h2>{c_lion}</div>
<div class="foot">Overlapping curves across beam sizes are the
scale-invariance result: the attractor is a property of the dynamics,
not the node count. Numerical simulation record — computes geometry,
does not act on matter.</div>
</body></html>"""
    out = logs / "level1_combined_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"combined report -> {out}  ({len(loaded)} scale(s); {missing_note})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
