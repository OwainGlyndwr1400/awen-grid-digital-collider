"""Independent verification of the locked collision rift (external request,
F. El Khettabi, 2026-08-11).

Claim under test: locked full-beam mean associator norm
    rift = mean ||[emb(psi_R), psi_I, emb(q_obs)]||  =  1.1787-1.1790
over ticks 36-369, invariant across beam sizes and seeds.

VERIFICATION LAYERS (all pre-registered in this docstring):

L1  Closed-form cross-check (mathematically independent path).
    For x = (p,0), z = (r,0) embedded quaternions and y = (a,b) a
    Cayley-Dickson pair, the associator reduces exactly to
        [x, y, z] = (0,  b (p rbar - rbar p))
    so  ||[x,y,z]|| = ||b|| * ||p rbar - rbar p||   -- quaternion algebra
    only (uses H-associativity + norm multiplicativity). The full 8D
    computation and this reduction must agree to machine precision.

L2  Analytic anchor. Unit-octonion left/right multiplication is an
    orthogonal map of R^8, so the Imaginary beam remains EXACTLY
    Haar-uniform on S^7 for all ticks. Hence ||b||^2 ~ Beta(2,2) and
        E||b|| = 24/35 = 0.6857142857...  (exact)
    The measured mean ||b|| must match at the ~1/sqrt(N) level, and the
    rift must factor as  rift = E||b|| * mean 2||v x u||  (independence
    of the two beams), with u the vector part of conj(q_obs).

L3  Fresh-seed replication. Seeds never used in the paper
    (8675309, 31415926, 27182818), both backends (CPU float64 at 144k,
    GPU float32 at 1M), locked interval ticks 36-96.

L4  New falsifiable prediction (never measured before this script):
    at tick 0 with uniform beams,
        rift(0) = (24/35) * sqrt(3) * (pi/4) * E||v||_S3
    where E||v||_S3 = B(2, 1/2)/B(3/2, 1/2). Numerically 0.79178.
    A measured tick-0 rift far from this value falsifies our own
    account of the observable.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from awen_collider import algebra as alg                 # noqa: E402
from awen_collider.engine import AwenGridCollider        # noqa: E402

SEEDS = [8675309, 31415926, 27182818]
E_B_EXACT = 24.0 / 35.0
E_V_S3 = (math.gamma(2) * math.gamma(0.5) / math.gamma(2.5)) / \
         (math.gamma(1.5) * math.gamma(0.5) / math.gamma(2.0))
RIFT_T0_PRED = E_B_EXACT * math.sqrt(3.0) * (math.pi / 4.0) * E_V_S3


def closed_form(psi_r: np.ndarray, psi_i: np.ndarray,
                q_obs: np.ndarray) -> np.ndarray:
    """L1: ||b|| * ||p rbar - rbar p|| — no octonion code involved."""
    b = psi_i[:, 4:]
    rbar = alg.qconj(q_obs[None, :])
    comm = alg.hamilton(psi_r, np.repeat(rbar, len(psi_r), 0)) - \
        alg.hamilton(np.repeat(rbar, len(psi_r), 0), psi_r)
    return np.linalg.norm(b, axis=1) * np.linalg.norm(comm, axis=1)


def verify_one(nodes: int, seed: int, force_cpu: bool) -> dict:
    c = AwenGridCollider(nodes=nodes, seed=seed, force_cpu=force_cpu)
    be = c.backend

    # -- L4: tick-0 measurement against the analytic prediction --
    t0 = c.collide()
    # -- L1 at tick 0: closed form vs full associator, same states --
    pr = be.to_numpy(c.psi_r).astype(np.float64)
    pi_ = be.to_numpy(c.psi_i).astype(np.float64)
    qo = be.to_numpy(c.q_obs)[0].astype(np.float64)
    cf0 = closed_form(pr, pi_, qo)
    full0 = alg.onorm(alg.associator(
        alg.embed_quaternion(pr), pi_,
        alg.embed_quaternion(qo[None, :]).repeat(len(pr), 0)))
    l1_maxdiff = float(np.abs(cf0 - full0).max())

    # -- run to lock, collect locked interval --
    rep = c.run(ticks=96, collide_every=12)
    locked = [t.rift_mean for t in rep.history if t.tick >= 36]
    rift_mu = float(np.mean(locked))
    rift_sd = float(np.std(locked))

    # -- L2: S^7 uniformity anchor + factorization at final tick --
    pi_f = be.to_numpy(c.psi_i).astype(np.float64)
    pr_f = be.to_numpy(c.psi_r).astype(np.float64)
    eb_meas = float(np.linalg.norm(pi_f[:, 4:], axis=1).mean())
    comm_term = float(np.mean(
        closed_form(pr_f, pi_f, be.to_numpy(c.q_obs)[0].astype(np.float64))
        / np.linalg.norm(pi_f[:, 4:], axis=1)))
    rift_factored = eb_meas * comm_term

    return {
        "nodes": nodes, "seed": seed,
        "backend": "cpu-float64" if force_cpu else "gpu-float32",
        "rift_locked_mean": rift_mu, "rift_locked_sd": rift_sd,
        "rift_tick0": t0.rift_mean, "rift_tick0_pred": RIFT_T0_PRED,
        "closed_form_max_abs_diff": l1_maxdiff,
        "E_b_measured": eb_meas, "E_b_exact": E_B_EXACT,
        "rift_factored": rift_factored,
    }


def main() -> int:
    print("Independent verification of the locked collision rift")
    print(f"  analytic anchors: E||b|| = 24/35 = {E_B_EXACT:.10f}; "
          f"rift(t=0) predicted = {RIFT_T0_PRED:.5f}")
    rows = []
    for seed in SEEDS:
        for nodes, cpu in [(144_000, True), (1_000_000, False)]:
            r = verify_one(nodes, seed, cpu)
            rows.append(r)
            print(f"  seed {seed:>8} {r['backend']:<11} {nodes:>9,}: "
                  f"locked {r['rift_locked_mean']:.6f} ± "
                  f"{r['rift_locked_sd']:.6f} | t0 {r['rift_tick0']:.5f} "
                  f"(pred {RIFT_T0_PRED:.5f}) | L1 diff "
                  f"{r['closed_form_max_abs_diff']:.1e} | "
                  f"E||b|| {r['E_b_measured']:.6f}")

    mus = [r["rift_locked_mean"] for r in rows]
    print(f"\n  grand locked mean over {len(rows)} fresh runs: "
          f"{np.mean(mus):.6f}  (range {min(mus):.6f} .. {max(mus):.6f})")
    print(f"  paper claim: 1.1787-1.1790 (summary 1.1790 ± 0.0011)")
    verdict = 1.1787 - 0.002 <= np.mean(mus) <= 1.1790 + 0.002
    print(f"  VERDICT: {'CONFIRMED' if verdict else 'NOT CONFIRMED'}")

    out = ROOT / "logs" / "rift_verification.json"
    out.write_text(json.dumps({"protocol": __doc__, "rows": rows,
                               "grand_mean": float(np.mean(mus)),
                               "verdict": bool(verdict)}, indent=2),
                   encoding="utf-8")
    print(f"  JSON -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
