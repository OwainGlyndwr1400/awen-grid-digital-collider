"""Awen Collider — exact quaternion (H) and octonion (O) algebra, vectorized.

Companion module to QTE core/quaternion.py (same conventions: batched
(..., 4) tensors, [w, x, y, z] layout), extended with the 8D octonion
sector that the collider's Imaginary Ledger runs on.

Everything in this file is standard, textbook-verifiable algebra:

* Quaternions: 4D normed division algebra. Associative, non-commutative.
  Unit quaternions form the S^3 hypersphere. The sandwich map
      psi -> q_b * psi * q_a^{-1}
  with unit q_a, q_b realizes the GENERAL element of SO(4): every
  rotation of 4D space has this form (via Spin(4) = SU(2) x SU(2)).
  This is the rigorous content of the Divine Equation.

* Octonions: 8D normed division algebra built from quaternion pairs by
  the Cayley-Dickson construction. NON-associative (but alternative).
  The associator [x,y,z] = (xy)z - x(yz) measures the failure of
  associativity: identically zero on any quaternionic subalgebra,
  generically nonzero when the full 8D algebra participates. It is the
  collider's "Associative Rift" observable.

Both algebras are composition algebras: |xy| = |x||y|. That identity is
the real mathematical reason Null Ledger parity (norm conservation)
holds in the engine to machine precision.

Backend: pure NumPy by default. Every function accepts xp= (numpy or
torch) so the identical formulas run on the CUDA ledgers when PyTorch
is installed; nothing here depends on which backend is active.
"""

from __future__ import annotations

import numpy as np


def _cat(parts, xp):
    """Concatenate along the last axis for numpy or torch."""
    if hasattr(xp, "concatenate"):
        return xp.concatenate(parts, -1)
    return xp.cat(parts, -1)  # older torch


# ----------------------------------------------------------------------
# Quaternions H — arrays (..., 4) as [w, x, y, z]
# ----------------------------------------------------------------------

def hamilton(p, q, xp=np):
    """Hamilton product p*q (non-commutative), vectorized over leading axes."""
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return xp.stack(
        (
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ),
        -1,
    )


def qconj(q, xp=np):
    """q* = w - xi - yj - zk."""
    out = q * (-1.0)
    out[..., 0] = q[..., 0]
    return out


def qnorm2(q):
    return (q * q).sum(-1)


def qnorm(q, xp=np):
    return xp.sqrt(qnorm2(q))


def qinverse(q, xp=np):
    """q^{-1} = q* / |q|^2.  (conj/|q| is a classic bug — see spec v1 code.)"""
    return qconj(q, xp) / qnorm2(q)[..., None]


def qnormalize(q, xp=np, eps=1e-12):
    return q / (qnorm(q, xp)[..., None] + eps)


def sandwich(psi, q_b, q_a, xp=np):
    """The Divine Equation step: psi -> q_b * psi * q_a^{-1}.

    For unit q_a, q_b this is an exact SO(4) rotation of R^4 — an
    isometry of S^3, norm-preserving to machine precision.
    """
    return hamilton(hamilton(q_b, psi, xp), qinverse(q_a, xp), xp)


def from_axis_angle(axis, angle):
    """Unit quaternion rotating by `angle` (rad) about 3D `axis` (numpy)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    angle = np.asarray(angle, dtype=float)
    half = angle / 2.0
    s = np.sin(half)
    return np.concatenate([np.cos(half)[..., None], axis * s[..., None]], -1)


def to_su2(q):
    """Quaternion(s) as 2x2 complex matrices.

    q = w + xi + yj + zk  ->  [[w+xi, y+zi], [-y+zi, w-xi]]
    Hamilton product becomes matrix multiplication; |q|^2 = det.
    Used in tests as an INDEPENDENT implementation to cross-validate
    `hamilton`. No such trick exists for octonions: matrix algebras are
    associative, octonions are not — which is exactly why the spec-v1
    idea of `matmul`-ing the 8D ledger was unimplementable and the
    Cayley-Dickson product below is required.
    """
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (2, 2), dtype=complex)
    m[..., 0, 0] = w + 1j * x
    m[..., 0, 1] = y + 1j * z
    m[..., 1, 0] = -y + 1j * z
    m[..., 1, 1] = w - 1j * x
    return m


def from_su2(m):
    m = np.asarray(m)
    return np.stack(
        (m[..., 0, 0].real, m[..., 0, 0].imag,
         m[..., 0, 1].real, m[..., 0, 1].imag), -1
    )


def random_unit_quaternions(n, rng):
    """Uniform (Haar) random points on S^3."""
    return qnormalize(rng.standard_normal((n, 4)))


# ----------------------------------------------------------------------
# Octonions O — arrays (..., 8) as Cayley-Dickson pairs of quaternions
# ----------------------------------------------------------------------
# x = (a, b) means x = a + b*l with components [a0..a3, b0..b3].
# Product: (a, b)(c, d) = (a c - d* b,  d a + b c*)   (* = quat conjugate)
# The construction guarantees a genuine octonion algebra; no hand-typed
# multiplication table (a classic error source) is needed.

def octonion_mul(x, y, xp=np):
    """Octonion product via Cayley-Dickson, vectorized."""
    a, b = x[..., :4], x[..., 4:]
    c, d = y[..., :4], y[..., 4:]
    left = hamilton(a, c, xp) - hamilton(qconj(d, xp), b, xp)
    right = hamilton(d, a, xp) + hamilton(b, qconj(c, xp), xp)
    return _cat((left, right), xp)


def oconj(x, xp=np):
    out = x * (-1.0)
    out[..., 0] = x[..., 0]
    return out


def onorm2(x):
    return (x * x).sum(-1)


def onorm(x, xp=np):
    return xp.sqrt(onorm2(x))


def onormalize(x, xp=np, eps=1e-12):
    return x / (onorm(x, xp)[..., None] + eps)


def oinverse(x, xp=np):
    return oconj(x, xp) / onorm2(x)[..., None]


def associator(x, y, z, xp=np):
    """[x, y, z] = (x y) z - x (y z) — the Associative Rift observable.

    Zero (to machine precision) whenever x, y, z lie in a common
    quaternionic subalgebra; nonzero in general on the full 8D algebra.
    """
    lhs = octonion_mul(octonion_mul(x, y, xp), z, xp)
    rhs = octonion_mul(x, octonion_mul(y, z, xp), xp)
    return lhs - rhs


def embed_quaternion(q, xp=np):
    """Embed H into O as the first Cayley-Dickson slot (b = 0).

    Embedded states have identically zero associator with each other —
    the calibration baseline for the collision measurement.
    """
    return _cat((q, q * 0.0), xp)


def random_unit_octonions(n, rng):
    """Uniform random points on S^7."""
    return onormalize(rng.standard_normal((n, 8)))
