"""Level II — Observer-weight sweep: how do the attractors depend on the
rotation:fold mix?

PRE-REGISTERED PROTOCOL (exploratory mapping, stated before execution):

  Question       The directive mix (2.5:1.5)/4 = 62.5% rotation gives
                 lion ~ 11.55; QTE's historical iteration gave 0.5352.
                 Map the attractor as a function of the mix and see
                 whether both regimes live on one smooth curve — or
                 whether the map bifurcates somewhere between.
  Dynamics       psi_next = normalize( w * (q_obs psi q_obs^-1)
                                     + (1-w) * (F_vec (x) psi) )
                 with F_vec = [0, 0.480000038, 0, 0] fixed (eta tax
                 unchanged), q_obs the 120-deg triadic observer.
                 w is the rotation weight; the directive point is
                 w = 2.5/4 = 0.625.
  Grid           w in linspace(0.02, 0.98, 193).
  Settle/metric  100 ticks; MEDIAN |ijk|/|w| (lion) and mean |vec|
                 (mass index), measured on the final state.
  Beam           1,000,000 nodes, identical seed-1400 Haar beam per
                 point (node-count independence established at Level I).
  Hardware       both GPUs, interleaved 60/40.

This is a MAPPING experiment, not a claim test: the output is the curve
itself. No feature criterion is registered; conclusions about any
structure found must be confirmed by a follow-up targeted run.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from awen_collider import algebra as alg                        # noqa: E402
from awen_collider.constants import FOLD_LOCK                   # noqa: E402
from awen_collider.report_html import _svg_chart, GOLD, VIOLET, DIM, LINE  # noqa: E402

NODES = 1_000_000
SEED = 1400
TICKS = 100
W_GRID = np.linspace(0.02, 0.98, 193)
DIRECTIVE_W = 2.5 / 4.0


def worker(device_str: str, points: list[float], base: np.ndarray,
           q_obs_np: np.ndarray, out: list, tag: str):
    dev = torch.device(device_str)
    master = torch.tensor(base, device=dev, dtype=torch.float32)
    q_obs = torch.tensor(q_obs_np[None, :], device=dev, dtype=torch.float32)
    q_obs_inv = torch.tensor(alg.qinverse(q_obs_np[None, :]),
                             device=dev, dtype=torch.float32)
    fv = torch.zeros(1, 4, device=dev, dtype=torch.float32)
    fv[0, 1] = FOLD_LOCK
    for w in points:
        psi = master.clone()
        for _ in range(TICKS):
            rot = alg.hamilton(alg.hamilton(q_obs, psi, xp=torch),
                               q_obs_inv, xp=torch)
            fol = alg.hamilton(fv, psi, xp=torch)
            psi = alg.qnormalize(w * rot + (1.0 - w) * fol, xp=torch)
        real = psi[:, 0].abs()
        imag = torch.linalg.norm(psi[:, 1:], dim=1)
        lion = float(torch.median(imag / (real + 1e-8)).item())
        mass = float(imag.mean().item())
        out.append({"w": float(w), "lion": lion, "mass": mass,
                    "device": tag})
        print(f"  [{tag}] w={w:.3f}  lion={lion:.4f}  mass={mass:.4f}",
              flush=True)


def write_report(rows: list[dict], meta: dict, out: Path):
    rows = sorted(rows, key=lambda r: r["w"])
    x = [int(round(r["w"] * 1000)) for r in rows]
    lion = [r["lion"] for r in rows]
    mass = [r["mass"] for r in rows]
    import math
    log_lion = [math.log10(max(v, 1e-9)) for v in lion]
    c1 = _svg_chart(x, [(log_lion, GOLD)], mark=625,
                    mark_label="directive w = 0.625",
                    fmt=lambda v: f"1e{v:.1f}")
    c2 = _svg_chart(x, [(mass, VIOLET)], mark=625,
                    mark_label="directive w = 0.625",
                    fmt=lambda v: f"{v:.3f}")
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Level II — Observer-Weight Sweep</title>
<style>
 body{{background:#06080f;color:#c8d4f0;font:13px/1.5 Consolas,monospace;
      padding:20px;margin:0}}
 h1{{font-size:16px;color:{GOLD};letter-spacing:.08em;margin:0 0 2px}}
 .sub{{color:{DIM};font-size:11px;margin-bottom:16px}}
 .chart{{background:#0b1020;border:1px solid {LINE};border-radius:6px;
        padding:12px;margin-bottom:14px}}
 .chart h2{{font-size:11px;color:{DIM};letter-spacing:.14em;margin:0 0 6px}}
 svg{{width:100%;height:auto;display:block}}
 .foot{{color:{DIM};font-size:10px;margin-top:12px}}
</style></head><body>
<h1>LEVEL II — OBSERVER-WEIGHT SWEEP</h1>
<div class="sub">{meta['nodes']:,} nodes x {len(rows)} points x
{meta['ticks']} ticks · fold amplitude fixed at {FOLD_LOCK} ·
x axis: rotation weight w &times; 1000 · wall {meta['wall_s']:.0f}s</div>
<div class="chart"><h2>LION ATTRACTOR — log10 median |ijk|/|w| vs mix</h2>{c1}</div>
<div class="chart"><h2>MASS INDEX (m=i channel) — mean |vec| vs mix</h2>{c2}</div>
<div class="foot">Mapping experiment (pre-registered as exploratory).
Numerical simulation record — computes geometry, does not act on matter.</div>
</body></html>"""
    out.write_text(html, encoding="utf-8")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable — this experiment requires the GPUs.")
        return 1
    n_dev = torch.cuda.device_count()
    names = {i: torch.cuda.get_device_name(i) for i in range(n_dev)}
    fast = next((i for i, nm in names.items() if "4070" in nm), 0)
    slow = next((i for i in names if i != fast), fast)
    print(f"Level II observer-weight sweep: {NODES:,} nodes x "
          f"{len(W_GRID)} points x {TICKS} ticks")

    rng = np.random.default_rng(SEED)
    base = rng.standard_normal((NODES, 4)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    q_obs_np = alg.from_axis_angle(
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0), 2.0 * np.pi / 3.0)

    pts_fast = [w for i, w in enumerate(W_GRID) if i % 5 < 3]
    pts_slow = [w for i, w in enumerate(W_GRID) if i % 5 >= 3]
    if slow == fast:
        pts_fast, pts_slow = list(W_GRID), []

    rows: list[dict] = []
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(
        f"cuda:{fast}", pts_fast, base, q_obs_np, rows, "4070"))]
    if pts_slow:
        threads.append(threading.Thread(target=worker, args=(
            f"cuda:{slow}", pts_slow, base, q_obs_np, rows, "2080S")))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    rows_sorted = sorted(rows, key=lambda r: r["w"])
    i_dir = min(range(len(rows_sorted)),
                key=lambda i: abs(rows_sorted[i]["w"] - DIRECTIVE_W))
    meta = {"nodes": NODES, "ticks": TICKS, "seed": SEED, "wall_s": wall}
    logs = ROOT / "logs"
    (logs / "level2_weight_sweep.json").write_text(json.dumps(
        {"protocol": __doc__, "meta": meta, "results": rows_sorted},
        indent=2), encoding="utf-8")
    write_report(rows_sorted, meta, logs / "level2_report.html")

    print(f"\n  wall {wall:.0f}s | at directive w=0.625: "
          f"lion {rows_sorted[i_dir]['lion']:.4f} "
          f"(Level I cross-check: ~11.55)")
    print(f"  lion range across mixes: "
          f"{min(r['lion'] for r in rows_sorted):.4f} .. "
          f"{max(r['lion'] for r in rows_sorted):.4f}")
    print(f"  JSON -> {logs / 'level2_weight_sweep.json'}")
    print(f"  report -> {logs / 'level2_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
