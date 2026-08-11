"""Awen Grid Digital Collider — CLI entry point.

Usage:
    python -m awen_collider.run_collider [--nodes N] [--ticks T]
        [--collide-every K] [--seed S] [--cpu] [--audit] [--json PATH]

Examples:
    python -m awen_collider.run_collider --nodes 100000 --ticks 288 --audit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .audit import audit_json, render_report, run_audit
from .constants import (
    BASE13_C, BASE13_CLOCK_RATIO, FOLD_LOCK, LION_CONSTANT, MASS_GAP,
    ROTOR_RPM, TOGGLE_STRIDE,
)
from .engine import AwenGridCollider

def _base13(n: int) -> str:
    """Render an integer in base-13 notation (digits 0-9, A, B, C)."""
    digits = "0123456789ABC"
    if n == 0:
        return "0"
    out = ""
    while n:
        out = digits[n % 13] + out
        n //= 13
    return out


BANNER = r"""
=======================================================================
  AWEN GRID DIGITAL COLLIDER  -  S^3 x S^7 dual-ledger engine
-----------------------------------------------------------------------
  Real Ledger  : quaternion beam (S^3), observer-weighted QTE step
  Imag Ledger  : octonion beam (S^7), Cayley-Dickson echo rotation
  Collision    : associator rift [psi_R, psi_I, probe]
  This instrument computes geometry and audits claims.
  It does not accelerate, collide, or transform physical matter.
=======================================================================
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Awen Grid Digital Collider")
    ap.add_argument("--nodes", type=int, default=20_000)
    ap.add_argument("--ticks", type=int, default=96)
    ap.add_argument("--collide-every", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1400)
    ap.add_argument("--cpu", action="store_true", help="force CPU backend")
    ap.add_argument("--audit", action="store_true",
                    help="run the falsifiability audit first")
    ap.add_argument("--json", type=Path, default=None,
                    help="write full run report JSON here")
    ap.add_argument("--html", type=Path, default=None,
                    help="write a graphical HTML report of this run here")
    args = ap.parse_args(argv)

    print(BANNER)
    print(f"  fold lock      : {FOLD_LOCK}   (eta-taxed, directive 2)")
    print(f"  toggle stride  : {TOGGLE_STRIDE} in Z/24    (directive 3)")
    print(f"  observer       : 2.5r + 1.5i / 4.0          (directive 4)")
    print(f"  base-13 clock  : e^19.5 = {BASE13_C:.4e}  "
          f"(ratio {BASE13_CLOCK_RATIO:.6f}, directive 1)")
    print(f"  rotor sync     : {ROTOR_RPM:.0f} RPM")
    print(f"  mass-gap ref   : {MASS_GAP:.6f}")
    print()

    if args.audit:
        findings = run_audit()
        print(render_report(findings))
        audit_path = Path(__file__).resolve().parent.parent / "logs"
        audit_path.mkdir(exist_ok=True)
        (audit_path / "claims_audit.json").write_text(
            audit_json(findings), encoding="utf-8")
        print(f"  audit JSON -> {audit_path / 'claims_audit.json'}\n")

    collider = AwenGridCollider(
        nodes=args.nodes, seed=args.seed, force_cpu=args.cpu)
    print(f"  {collider.backend.describe()}")
    print(f"  beams initialized: {args.nodes} nodes/ledger\n")

    t0 = time.perf_counter()
    report = collider.run(ticks=args.ticks, collide_every=args.collide_every)
    dt = time.perf_counter() - t0

    print("  tick   ph24   rift        baseline    parity      lion      mass")
    print("  " + "-" * 70)
    for t in report.history:
        print(f"  {t.tick:5d}  {t.phase24:3d}   "
              f"{t.rift_mean:9.6f}   {t.rift_baseline:.2e}   "
              f"{t.parity_drift:.2e}   {t.lion_ratio:.4f}   "
              f"{t.mass_index:.4f}")

    final = report.final()
    print("  " + "-" * 70)
    print(f"  run time {dt:.2f}s | toggle cycle complete: "
          f"{report.toggle_cycle_complete} | final tick {args.ticks} "
          f"(base-13: {_base13(args.ticks)})")
    print(f"  final rift {final.rift_mean:.6f} vs mass-gap ref "
          f"{MASS_GAP:.6f} | lion {final.lion_ratio:.4f} "
          f"(QTE attractor {LION_CONSTANT})")
    print(f"  mass index (m=i channel) {final.mass_index:.4f} | "
          f"ledger parity drift {final.parity_drift:.2e} "
          "(composition-algebra conservation)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(report.to_json(), encoding="utf-8")
        print(f"  report JSON -> {args.json}")
    if args.html:
        import json as _json  # noqa: PLC0415
        from .report_html import write_html_report  # noqa: PLC0415
        out = write_html_report(_json.loads(report.to_json()), args.html)
        print(f"  report HTML -> {out}")
        # Config handoff for visualizer.html: inject THIS run's
        # configuration inline between sentinel comments (browsers block
        # file->file subresource scripts, so no external config file).
        import re  # noqa: PLC0415
        viz = Path(__file__).resolve().parent.parent / "visualizer.html"
        if viz.exists():
            cfg_js = ("/*AWEN_CFG*/window.AWEN_CFG=" + _json.dumps({
                "nodes": args.nodes, "ticks": args.ticks,
                "backend": report.backend,
                "final_rift": report.final().rift_mean,
                "final_lion": report.final().lion_ratio,
            }) + ";/*/AWEN_CFG*/")
            html_txt = viz.read_text(encoding="utf-8")
            patched = re.sub(r"/\*AWEN_CFG\*/.*?/\*/AWEN_CFG\*/", cfg_js,
                             html_txt, count=1, flags=re.S)
            if patched != html_txt:
                viz.write_text(patched, encoding="utf-8")
                print(f"  live config -> injected into {viz.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
