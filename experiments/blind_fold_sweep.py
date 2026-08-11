"""Blind fold-amplitude sweep — the honest test of the QTE "0.48 resonance".

PRE-REGISTERED PROTOCOL (fixed before execution; do not tune after):

  Claim under test   QTE README: "Resonance Peak (20M sweep): 0.48 — PROVEN"
  Dynamics           QTE lion_hunt_step, reproduced exactly:
                       rotated  = q_obs (x) psi (x) q_obs^-1   (full sandwich)
                       folded   = [0, F, 0, 0] (x) psi         (Hamilton product)
                       psi_next = normalize(rotated + folded)
                     with the 120-deg triadic observer on (1,1,1)/sqrt(3).
  Metric             MEDIAN of |ijk|/|w|  (QTE's measure_lion_ratio, verbatim.
                     Median, not mean — near-equator nodes make the mean
                     diverge; the mean is logged for completeness only.)
  Grid               F in linspace(0.40, 0.60, 201) — 0.001 spacing,
                     includes F = 0.480 exactly.
  Settle             100 ticks per point (QTE README protocol: settle-iters 100).
  Initial beam       identical for every F: Haar-uniform on S^3
                     (standard normals, normalized), seed 1400, 10,000,000
                     nodes. Differences between points are due to F alone.
  Hardware           both GPUs, interleaved point assignment (RTX 4070 60%,
                     RTX 2080S 40%), so either card alone still covers the
                     full range.
  Success criterion  the 0.48 claim is SUPPORTED if L(F) shows a
                     distinguished local feature at F = 0.48 +/- 0.005 —
                     a local extremum, or |second difference| exceeding
                     5x the grid median. Otherwise the claim is NOT
                     REPRODUCED under this protocol.

Fixes relative to the pasted Phase-3 draft script: out-of-bounds tensor
indices (q_obs[22] etc.) removed; the missing right-multiplication by
q_obs^-1 restored; fold implemented as the Hamilton product (not an
additive constant); Haar-correct initialization (torch.rand samples only
the positive orthant); median metric per QTE's own definition.
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

from awen_collider import algebra as alg                      # noqa: E402
from awen_collider.report_html import _svg_chart, GOLD, DIM, LINE  # noqa: E402

NODES = 10_000_000
SEED = 1400
TICKS = 100
F_GRID = np.linspace(0.40, 0.60, 201)
SPLIT_MOD = 5          # of every 5 points: 3 to the 4070, 2 to the 2080S


def lion_hunt_step(psi, q_obs, q_obs_inv, fold_vec):
    """QTE iteration, exactly (torch tensors)."""
    rotated = alg.hamilton(alg.hamilton(q_obs, psi, xp=torch), q_obs_inv,
                           xp=torch)
    folded = alg.hamilton(fold_vec, psi, xp=torch)
    return alg.qnormalize(rotated + folded, xp=torch)


def worker(device_str: str, points: list[float], base: np.ndarray,
           q_obs_np: np.ndarray, out: list, tag: str):
    dev = torch.device(device_str)
    master = torch.tensor(base, device=dev, dtype=torch.float32)
    q_obs = torch.tensor(q_obs_np[None, :], device=dev, dtype=torch.float32)
    q_obs_inv = torch.tensor(alg.qinverse(q_obs_np[None, :]),
                             device=dev, dtype=torch.float32)
    for f in points:
        psi = master.clone()
        fv = torch.zeros(1, 4, device=dev, dtype=torch.float32)
        fv[0, 1] = float(f)
        for _ in range(TICKS):
            psi = lion_hunt_step(psi, q_obs, q_obs_inv, fv)
        real = psi[:, 0].abs()
        imag = torch.linalg.norm(psi[:, 1:], dim=1)
        ratios = imag / (real + 1e-8)
        med = float(torch.median(ratios).item())
        mean = float(torch.mean(ratios).item())
        out.append({"F": float(f), "L_median": med, "L_mean": mean,
                    "device": tag})
        print(f"  [{tag}] F={f:.3f}  L_med={med:.6f}", flush=True)


def analyze(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: r["F"])
    F = [r["F"] for r in rows]
    L = [r["L_median"] for r in rows]
    i_max = max(range(len(L)), key=lambda i: L[i])
    i_min = min(range(len(L)), key=lambda i: L[i])
    # curvature (second difference) feature detector
    d2 = [abs(L[i + 1] - 2 * L[i] + L[i - 1]) for i in range(1, len(L) - 1)]
    d2_med = sorted(d2)[len(d2) // 2]
    i48 = min(range(len(F)), key=lambda i: abs(F[i] - 0.48))
    window = [j for j in range(1, len(L) - 1) if abs(F[j] - 0.48) <= 0.005]
    d2_at_48 = max(d2[j - 1] for j in window) if window else 0.0
    local_extremum_at_48 = any(
        (L[j] > L[j - 1] and L[j] > L[j + 1]) or
        (L[j] < L[j - 1] and L[j] < L[j + 1]) for j in window)
    feature = local_extremum_at_48 or (
        d2_med > 0 and d2_at_48 > 5 * d2_med)
    return {
        "argmax_F": F[i_max], "L_at_argmax": L[i_max],
        "argmin_F": F[i_min], "L_at_argmin": L[i_min],
        "L_at_0.48": L[i48],
        "d2_median": d2_med, "d2_max_near_0.48": d2_at_48,
        "local_extremum_at_0.48": local_extremum_at_48,
        "feature_at_0.48": feature,
        "verdict": ("SUPPORTED: distinguished feature at F=0.48 under the "
                    "pre-registered criterion" if feature else
                    "NOT REPRODUCED: no distinguished feature at F=0.48 "
                    "under the pre-registered criterion"),
    }


def write_report(rows: list[dict], analysis: dict, meta: dict, out: Path):
    rows = sorted(rows, key=lambda r: r["F"])
    x = [int(round(r["F"] * 1000)) for r in rows]          # milli-F
    y = [r["L_median"] for r in rows]
    chart = _svg_chart(x, [(y, GOLD)], mark=480, mark_label="F = 0.480",
                       fmt=lambda v: f"{v:.4f}")
    tiles = "".join(
        f'<div class="stat"><div class="k">{k}</div>'
        f'<div class="v">{v}</div></div>'
        for k, v in [
            ("CLAIM UNDER TEST", "QTE resonance peak at F = 0.48"),
            ("VERDICT", analysis["verdict"].split(":")[0]),
            ("L AT 0.480", f"{analysis['L_at_0.48']:.6f}"),
            ("GLOBAL ARGMAX", f"F = {analysis['argmax_F']:.3f}"),
            ("CURVATURE @0.48 / MEDIAN",
             f"{analysis['d2_max_near_0.48']:.2e} / {analysis['d2_median']:.2e}"),
            ("NODES x POINTS x TICKS",
             f"{meta['nodes']:,} x {len(rows)} x {meta['ticks']}"),
        ])
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Blind Fold-Amplitude Sweep — Report</title>
<style>
 body{{background:#06080f;color:#c8d4f0;font:13px/1.5 Consolas,monospace;
      padding:20px;margin:0}}
 h1{{font-size:16px;color:{GOLD};letter-spacing:.08em;margin:0 0 2px}}
 .sub{{color:{DIM};font-size:11px;margin-bottom:16px}}
 .stats{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}}
 .stat{{background:#0b1020;border:1px solid {LINE};border-radius:6px;
       padding:10px 14px;min-width:150px}}
 .stat .k{{color:{DIM};font-size:10px;letter-spacing:.12em}}
 .stat .v{{font-size:14px;margin-top:2px;color:#c8d4f0}}
 .chart{{background:#0b1020;border:1px solid {LINE};border-radius:6px;
        padding:12px}}
 .chart h2{{font-size:11px;color:{DIM};letter-spacing:.14em;margin:0 0 6px}}
 svg{{width:100%;height:auto;display:block}}
 .foot{{color:{DIM};font-size:10px;margin-top:14px;white-space:pre-line}}
</style></head><body>
<h1>BLIND FOLD-AMPLITUDE SWEEP — RUN REPORT</h1>
<div class="sub">{analysis['verdict']}</div>
<div class="stats">{tiles}</div>
<div class="chart"><h2>L(F) = MEDIAN |ijk|/|w| after {meta['ticks']} ticks
 &mdash; x axis: F &times; 1000</h2>{chart}</div>
<div class="foot">Pre-registered protocol: QTE lion_hunt_step dynamics, median
metric, identical seed-{meta['seed']} Haar-uniform beam per point, criterion
fixed before execution (local extremum at 0.48&plusmn;0.005 or curvature &gt;5&times;
grid median). Numerical simulation record — computes geometry, does not act
on matter. Wall time {meta['wall_s']:.0f}s on {meta['devices']}.</div>
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
    print(f"Blind fold sweep: {NODES:,} nodes x {len(F_GRID)} points x "
          f"{TICKS} ticks")
    print(f"  fast worker cuda:{fast} [{names[fast]}]  "
          f"slow worker cuda:{slow} [{names[slow]}]")

    rng = np.random.default_rng(SEED)
    base = rng.standard_normal((NODES, 4)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    q_obs_np = alg.from_axis_angle(
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0), 2.0 * np.pi / 3.0)

    pts_fast = [f for i, f in enumerate(F_GRID) if i % SPLIT_MOD < 3]
    pts_slow = [f for i, f in enumerate(F_GRID) if i % SPLIT_MOD >= 3]
    if slow == fast:
        pts_fast, pts_slow = list(F_GRID), []

    rows: list[dict] = []
    t0 = time.perf_counter()
    threads = [
        threading.Thread(target=worker, args=(
            f"cuda:{fast}", pts_fast, base, q_obs_np, rows, "4070")),
    ]
    if pts_slow:
        threads.append(threading.Thread(target=worker, args=(
            f"cuda:{slow}", pts_slow, base, q_obs_np, rows, "2080S")))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    analysis = analyze(rows)
    meta = {"nodes": NODES, "ticks": TICKS, "seed": SEED,
            "wall_s": wall, "devices": f"{names[fast]} + {names[slow]}"}
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "blind_sweep.json").write_text(json.dumps(
        {"protocol": __doc__, "meta": meta, "analysis": analysis,
         "results": sorted(rows, key=lambda r: r["F"])}, indent=2),
        encoding="utf-8")
    write_report(rows, analysis, meta, logs / "blind_sweep_report.html")

    print(f"\n  wall time {wall:.0f}s")
    print(f"  L(0.48) = {analysis['L_at_0.48']:.6f} | global argmax at "
          f"F = {analysis['argmax_F']:.3f}")
    print(f"  {analysis['verdict']}")
    print(f"  JSON -> {logs / 'blind_sweep.json'}")
    print(f"  report -> {logs / 'blind_sweep_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
