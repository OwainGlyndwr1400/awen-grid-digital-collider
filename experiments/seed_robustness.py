"""Seed-robustness ensemble — is the Level I attractor initial-condition
independent?

PROTOCOL: 10 seeds, 50,000 nodes each (node-count independence already
established), 60 ticks (lock at ~27, so the final measurement is deep in
the locked region), directive dynamics unchanged. The attractor is
seed-independent if the across-seed spread of locked rift/lion/mass is
comparable to the known within-run wobble (rift ~0.001, lion ~0.02).
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from awen_collider.engine import AwenGridCollider  # noqa: E402

SEEDS = [7, 42, 137, 233, 361, 963, 1400, 2026, 5040, 19491]


def main() -> int:
    rows = []
    for seed in SEEDS:
        c = AwenGridCollider(nodes=50_000, seed=seed)
        rep = c.run(ticks=60, collide_every=60)   # single locked measurement
        t = rep.final()
        rows.append({"seed": seed, "rift": t.rift_mean,
                     "lion": t.lion_ratio, "mass": t.mass_index})
        print(f"  seed {seed:>6}: rift {t.rift_mean:.6f}  "
              f"lion {t.lion_ratio:.4f}  mass {t.mass_index:.4f}")

    def spread(key):
        vals = [r[key] for r in rows]
        return st.mean(vals), st.pstdev(vals), max(vals) - min(vals)

    out = {}
    print()
    for key, wobble in [("rift", 0.0011), ("lion", 0.02), ("mass", 0.001)]:
        mu, sd, rng = spread(key)
        verdict = "seed-independent" if rng < 5 * wobble else "SEED-DEPENDENT"
        out[key] = {"mean": mu, "sd": sd, "range": rng, "verdict": verdict}
        print(f"  {key:>4}: mean {mu:.6f}  sd {sd:.2e}  range {rng:.2e}  "
              f"-> {verdict} (within-run wobble ~{wobble})")

    (ROOT / "logs" / "seed_robustness.json").write_text(json.dumps(
        {"protocol": __doc__, "seeds": SEEDS, "rows": rows,
         "summary": out}, indent=2), encoding="utf-8")
    print(f"\n  JSON -> {ROOT / 'logs' / 'seed_robustness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
