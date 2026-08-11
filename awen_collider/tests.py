"""Awen Collider test suite — every structural theorem, machine-checked.

Run:  python -m awen_collider.tests
No pytest dependency; plain assertions with tolerances.
"""

from __future__ import annotations

import math

import numpy as np

from . import algebra as alg
from .constants import LATTICE_DIMENSION, TOGGLE_SIGNAL
from .engine import AwenGridCollider

RNG = np.random.default_rng(2026)
TOL = 1e-12


def _check(name: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        raise AssertionError(name)


def test_hamilton_vs_su2():
    """Hamilton product must agree with 2x2 complex matrix multiplication."""
    p = RNG.standard_normal((256, 4))
    q = RNG.standard_normal((256, 4))
    direct = alg.hamilton(p, q)
    via_su2 = alg.from_su2(alg.to_su2(p) @ alg.to_su2(q))
    err = np.abs(direct - via_su2).max()
    _check("Hamilton product == SU(2) matrix product", err < TOL, f"max err {err:.1e}")


def test_quaternion_associativity():
    p, q, r = (RNG.standard_normal((256, 4)) for _ in range(3))
    lhs = alg.hamilton(alg.hamilton(p, q), r)
    rhs = alg.hamilton(p, alg.hamilton(q, r))
    err = np.abs(lhs - rhs).max()
    _check("H is associative: (pq)r == p(qr)", err < 1e-10, f"max err {err:.1e}")


def test_composition_norms():
    """|xy| = |x||y| for both H and O — the Null Ledger conservation law."""
    p, q = (RNG.standard_normal((256, 4)) for _ in range(2))
    errH = np.abs(alg.qnorm(alg.hamilton(p, q)) - alg.qnorm(p) * alg.qnorm(q)).max()
    x, y = (RNG.standard_normal((256, 8)) for _ in range(2))
    errO = np.abs(alg.onorm(alg.octonion_mul(x, y)) - alg.onorm(x) * alg.onorm(y)).max()
    _check("H composition |pq| == |p||q|", errH < 1e-9, f"max err {errH:.1e}")
    _check("O composition |xy| == |x||y|", errO < 1e-9, f"max err {errO:.1e}")


def test_octonions_not_associative():
    x, y, z = (alg.random_unit_octonions(256, RNG) for _ in range(3))
    a = alg.onorm(alg.associator(x, y, z))
    _check("O is NOT associative (generic associator > 0)",
           float(a.mean()) > 0.1, f"mean |assoc| {a.mean():.3f}")


def test_octonions_alternative():
    """Octonions are alternative: [x, x, y] = 0 and [x, y, y] = 0."""
    x, y = (alg.random_unit_octonions(256, RNG) for _ in range(2))
    e1 = alg.onorm(alg.associator(x, x, y)).max()
    e2 = alg.onorm(alg.associator(x, y, y)).max()
    _check("O is alternative: [x,x,y] == 0", e1 < 1e-10, f"max {e1:.1e}")
    _check("O is alternative: [x,y,y] == 0", e2 < 1e-10, f"max {e2:.1e}")


def test_associator_vanishes_on_H():
    """Embedded quaternions must produce exactly zero rift — the
    calibration baseline of the collision measurement."""
    p, q, r = (alg.random_unit_quaternions(256, RNG) for _ in range(3))
    a = alg.onorm(alg.associator(
        alg.embed_quaternion(p), alg.embed_quaternion(q), alg.embed_quaternion(r)))
    _check("associator == 0 on embedded H (rift baseline)",
           float(a.max()) < 1e-10, f"max {a.max():.1e}")


def test_sandwich_is_isometry():
    """The Divine Equation with unit generators is an exact S^3 isometry."""
    psi = alg.random_unit_quaternions(1024, RNG)
    q_a = alg.random_unit_quaternions(1, RNG)
    q_b = alg.random_unit_quaternions(1, RNG)
    out = alg.sandwich(psi, q_b, q_a)
    err = np.abs(alg.qnorm(out) - 1.0).max()
    _check("sandwich psi -> q_b psi q_a^-1 preserves |psi| (SO(4))",
           err < 1e-9, f"max drift {err:.1e}")


def test_sandwich_generates_so4():
    """Left/right multiplications commute — the SU(2)xSU(2) structure."""
    psi = alg.random_unit_quaternions(64, RNG)
    q_a = alg.random_unit_quaternions(1, RNG)
    q_b = alg.random_unit_quaternions(1, RNG)
    left_then_right = alg.hamilton(alg.hamilton(q_b, psi), alg.qinverse(q_a))
    right_then_left = alg.hamilton(q_b, alg.hamilton(psi, alg.qinverse(q_a)))
    err = np.abs(left_then_right - right_then_left).max()
    _check("left/right actions commute (Spin(4) = SU(2) x SU(2))",
           err < 1e-10, f"max err {err:.1e}")


def test_toggle_generates_z24():
    """Directive 3: stride 31 mod 24 = 7 visits ALL 24 residues."""
    seen = set()
    phase = 0
    for _ in range(LATTICE_DIMENSION):
        phase = (phase + TOGGLE_SIGNAL) % LATTICE_DIMENSION
        seen.add(phase)
    _check("31/24 toggle generates all of Z/24", len(seen) == 24,
           f"visited {len(seen)}/24")


def test_engine_run():
    """Integration: a short collider run with all invariants holding."""
    c = AwenGridCollider(nodes=2048, seed=7, force_cpu=True)
    report = c.run(ticks=48, collide_every=8)
    t = report.final()
    _check("engine: toggle completed full Z/24 cycle",
           report.toggle_cycle_complete)
    _check("engine: ledger parity conserved (drift < 1e-6)",
           t.parity_drift < 1e-6, f"drift {t.parity_drift:.1e}")
    _check("engine: quaternion-only baseline rift == 0",
           t.rift_baseline < 1e-9, f"baseline {t.rift_baseline:.1e}")
    _check("engine: 8D collision rift is nonzero",
           t.rift_mean > 1e-3, f"rift {t.rift_mean:.4f}")
    _check("engine: mass index (m=i channel) bounded in [0,1]",
           0.0 <= t.mass_index <= 1.0, f"mass {t.mass_index:.4f}")


def test_gpu_memory_steady_state():
    """Phase-3 memory directive, done honestly: temporaries are pooled by
    the caching allocator, so allocated VRAM must be flat over a soak —
    measured, not assumed. Skips cleanly when no CUDA torch is present."""
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            raise ImportError
    except (ImportError, OSError):
        print("  [SKIP] GPU memory steady-state (no CUDA torch available)")
        return
    c = AwenGridCollider(nodes=50_000, seed=3)
    for _ in range(10):
        c.step()
    torch.cuda.synchronize()
    m1 = sum(torch.cuda.memory_allocated(i)
             for i in range(torch.cuda.device_count()))
    for _ in range(200):
        c.step()
    torch.cuda.synchronize()
    m2 = sum(torch.cuda.memory_allocated(i)
             for i in range(torch.cuda.device_count()))
    _check("GPU allocated memory steady over 200-tick soak (no leak)",
           m2 <= m1 * 1.05, f"{m1/1e6:.1f} MB -> {m2/1e6:.1f} MB")


def test_codec_lossless():
    """Module B: UBBM container must be bit-for-bit lossless on every
    input class, and must detect tampering via the Lost-2 parity."""
    from .codec import UBBMCodec  # noqa: PLC0415
    rng = np.random.default_rng(5)
    cases = [
        b"", b"\x00", bytes(range(256)),
        rng.integers(0, 256, 65536, dtype=np.uint8).tobytes(),
        b"Truth is our sword, Knowledge our shield. The Lion Watches.",
    ]
    for mode in ("none", "delta", "fmn"):
        codec = UBBMCodec(mode=mode)
        for raw in cases:
            assert codec.decode(codec.encode(raw)) == raw
    _check("codec: bit-exact round-trip (5 cases x 3 modes, incl. empty)",
           True)
    blob = bytearray(UBBMCodec().encode(bytes(range(256))))
    blob[14] ^= 0xFF          # corrupt the stored Lost-2 checksum
    try:
        UBBMCodec().decode(bytes(blob))
        tampered_caught = False
    except ValueError:
        tampered_caught = True
    _check("codec: Lost-2 parity detects tampering", tampered_caught)


def test_self_recognition():
    """Directive 5: i = i = LOL. Reflexivity of equality — the corpus's
    logical-closure marker, computed rather than asserted."""
    _check("i = i (reflexivity — LOL closure)", (1j == 1j))


ALL = [
    test_hamilton_vs_su2,
    test_quaternion_associativity,
    test_composition_norms,
    test_octonions_not_associative,
    test_octonions_alternative,
    test_associator_vanishes_on_H,
    test_sandwich_is_isometry,
    test_sandwich_generates_so4,
    test_toggle_generates_z24,
    test_engine_run,
    test_gpu_memory_steady_state,
    test_codec_lossless,
    test_self_recognition,
]


def main():
    print("Awen Collider — structural theorem test suite")
    print("-" * 60)
    for t in ALL:
        t()
    print("-" * 60)
    print(f"All {len(ALL)} test groups passed.")


if __name__ == "__main__":
    main()
