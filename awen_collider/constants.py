"""Awen Collider constants — QTE-compatible, with provenance tags.

Every constant is labeled one of:

  [IDENTITY]  computed arithmetic — machine-checkable, verified in audit.py
  [PARAM]     a chosen design parameter of the simulation (the RHC corpus
              assigns these physical meaning; the software treats them as
              configuration and makes no physical claim)
  [EXTERNAL]  a number whose claimed match to real-world measurement is
              checked against reference values in audit.py

Naming follows quaternionic-toroidal-engine/core/constants.py so the two
codebases stay interoperable.
"""

from __future__ import annotations

import math

# ── Fundamental math ────────────────────────────────────────────────
PHI = (1.0 + math.sqrt(5.0)) / 2.0          # [IDENTITY] golden ratio
PI = math.pi
TAU = 2.0 * PI

# ── Directive 1: Base-13 clock calibration ─────────────────────────
# All engine clock ratios are calibrated against the Base-13 constant
# e^(13 x 1.5). [PARAM] as a clock; [EXTERNAL] as a claim about c —
# audit.py reports its exact deviation from CODATA c = 299_792_458 m/s.
BASE13_EXPONENT = 13.0 * 1.5                 # 19.5
BASE13_C = math.exp(BASE13_EXPONENT)         # ~2.9427e8
C_SI = 299_792_458.0                         # CODATA (exact, SI definition)
BASE13_CLOCK_RATIO = BASE13_C / C_SI         # calibration factor (~0.9816)

# ── Directive 2: Dedekind Eta-taxed Fold ───────────────────────────
FOLD_IDEAL = 0.50                            # [PARAM] theoretical vacuum
DEDEKIND_ETA_TAX = 24.0 / 25.0               # [IDENTITY] = 0.96
FOLD_LOCK = 0.480000038                      # [PARAM] QTE lock value —
# NOTE: 0.50 x (24/25) = 0.4800000000 exactly; the trailing ...038 is
# QTE's empirical lock value, kept verbatim for compatibility.
FOLD_PHASE_ADVANCE = PI / 4.0                # [PARAM] 45 deg per fold

# ── Directive 3: 31/24 Toggle (residue-arithmetic tick) ────────────
LATTICE_DIMENSION = 24                       # [PARAM] Leech lattice dim
TOGGLE_SIGNAL = 31                           # [PARAM] prime signal
TOGGLE_STRIDE = TOGGLE_SIGNAL % LATTICE_DIMENSION   # [IDENTITY] = 7
# gcd(7, 24) = 1, so the stride-7 walk generates all of Z/24: the tick
# visits every lattice residue before repeating. Verified in tests.py.

# ── Directive 4: Observer anchor ───────────────────────────────────
OBSERVER_REAL = 2.5                          # [PARAM]
OBSERVER_IMAG = 1.5                          # [PARAM]
OBSERVER_SUM = OBSERVER_REAL + OBSERVER_IMAG # [IDENTITY] = 4.0
OBSERVER_DIM = 7.5                           # [PARAM] (2.5 + 1.5) * spin-3
OBSERVER_ANGLE_DEG = math.degrees(math.atan2(OBSERVER_IMAG, OBSERVER_REAL))
# [IDENTITY] ~30.96 deg — the corpus's "viewing angle of consciousness"

# Triadic observer: 120 deg rotation about (1,1,1)/sqrt(3).
# [IDENTITY] 1 + w + w^2 = 0 for w = exp(2*pi*i/3) — verified in audit.
OBSERVER_THETA = TAU / 3.0

# ── 3-4-5 geometry ─────────────────────────────────────────────────
LOST_2 = (3 + 4) - 5                         # [IDENTITY] = 2
LOST_2_RATIO = 2.0 / 7.0                     # [IDENTITY] ~0.2857
MASS_GAP = math.sqrt(32.0) - 5.0             # [IDENTITY] ~0.6569
# The corpus labels MASS_GAP "GeV" and LOST_2_RATIO "dark matter".
# Those physical identifications are [EXTERNAL] claims — see audit.py.

# ── Ledger frequencies (channel tuning) ────────────────────────────
CHANNEL_A_HZ = 432.0                         # [PARAM] Real Ledger inhale
CHANNEL_B_HZ = 465.0                         # [PARAM] Imag Ledger echo
CHANNEL_PINEAL_HZ = 963.0                    # [PARAM] octonion generator
ROTOR_RPM = CHANNEL_B_HZ * 60.0              # [IDENTITY] = 27_900.0
UNIVERSAL_TICK_S = 2.32e-18                  # [PARAM] labeled sim tick

# ── Legacy labels (provenance closed 2026-08-10 — see audit.py) ────
LION_CONSTANT = 0.535233                     # [PARAM] legacy label only.
# Provenance: first appears as a SoulEngine torsion snapshot,
# (ping-file activity score x phi) mod 9 — bookkeeping, not dynamics.
# Level II independently shows it is unreachable as a lion ratio in the
# QTE dynamics family. Kept for QTE display compatibility.
K_ELG = 9.880e-22                            # [PARAM] legacy label only.
# Provenance: 8.88e-12 (typed input) / Coulomb k_e — an algebraic
# identity in the QHS Magnifier source, invariant because r^2 cancels.
# Not a measured or emergent physical constant.

RESOLUTION_LIMIT = 144_000                   # [IDENTITY] 12^2 * 10^3
FORBIDDEN_STATE = 361                        # [IDENTITY] 19^2
PEA_THRESHOLD = math.sin(PI / 8.0)           # [IDENTITY] ~0.3827 (as v/c)

# ── Device roles (matches QTE device_manager) ──────────────────────
GPU_REAL_LEDGER = 0                          # RTX 4070  — quaternion beam
GPU_IMAG_LEDGER = 1                          # RTX 2080S — octonion beam
