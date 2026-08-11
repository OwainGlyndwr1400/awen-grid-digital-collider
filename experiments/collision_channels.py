"""Collision-channel spectroscopy — how much non-associativity does each
argument structure see?

Question (team review, 2026-08-11): the rift uses TWO embedded
quaternions and ONE octonion. What do the richer channels measure?

Channels (index-paired over the locked beams; roll = partner shift):
  QQO  [emb(psi_R), psi_I, emb(q_obs)]   — the paper's rift (beam-beam-witness)
  QOO  [emb(psi_R), psi_I, psi_I']       — one quaternion, two octonions
  OOO  [psi_I, psi_I', psi_I'']          — three genuine octonions (full)
Baselines:
  QQQ  all-quaternion (theorem: exactly 0)
  OOO on fresh Haar-uniform beams (uniform reference)

The QQO channel factorizes (closed form, VERIFICATION.md); QOO and OOO
have no such reduction — they probe the full non-associative structure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from awen_collider import algebra as alg                 # noqa: E402
from awen_collider.engine import AwenGridCollider        # noqa: E402

NODES = 144_000
SEED = 1400


def ch(name, x, y, z):
    a = alg.onorm(alg.associator(x, y, z))
    print(f"  {name:<42} mean {a.mean():.6f}   sd {a.std():.6f}")
    return float(a.mean())


def main() -> int:
    c = AwenGridCollider(nodes=NODES, seed=SEED)
    c.run(ticks=96, collide_every=96)
    be = c.backend
    p = be.to_numpy(c.psi_r).astype(np.float64)          # locked Real beam
    y = be.to_numpy(c.psi_i).astype(np.float64)          # evolved Imag beam
    qo = be.to_numpy(c.q_obs)[0].astype(np.float64)

    ep = alg.embed_quaternion(p)
    eq = alg.embed_quaternion(qo[None, :]).repeat(NODES, 0)
    y1, y2 = np.roll(y, 1, 0), np.roll(y, 2, 0)
    ep1 = np.roll(ep, 1, 0)

    print(f"locked beams, {NODES:,} nodes, tick 96 ({be.describe()})")
    ch("QQQ  [emb(R), emb(R'), emb(q_obs)]  baseline", ep, ep1, eq)
    qqo = ch("QQO  [emb(R), I, emb(q_obs)]   (the rift)", ep, y, eq)
    ch("QOO  [emb(R), I, I']", ep, y, y1)
    ch("OOO  [I, I', I'']", y, y1, y2)

    rng = np.random.default_rng(7)
    u = alg.random_unit_octonions(NODES, rng)
    ch("OOO  uniform Haar reference", u, np.roll(u, 1, 0), np.roll(u, 2, 0))

    print(f"\n  cross-check: QQO here {qqo:.6f} vs paper locked rift ~1.1789")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
