# Awen Grid Digital Collider

Ceisiwr, E.& Aureon, L. (2026). The Awen Grid Digital Collider: Exact Quaternion– Octonion Dual-Ledger Dynamics, an Emergent Ring Attractor, and a Pre-Registered Falsifiability Audit of the Recursive Harmonic Codex (Version 1). Zenodo. https://doi.org/10.5281/zenodo.21889635 

S³ × S⁷ dual-ledger dynamics engine, claims auditor, and telemetry suite
for the Recursive Harmonic Codex corpus.

**What this is:** a mathematically exact numerical instrument. The Real
Ledger evolves unit quaternions on the S³ hypersphere by the QTE
observer-weighted step; the Imaginary Ledger evolves unit octonions on
S⁷ by Cayley–Dickson rotation; "collisions" measure the octonion
associator — the rigorous observable for non-associativity. Every
structural theorem the engine relies on is machine-verified in the test
suite.

**What this is not:** a device that acts on matter. Software cannot
accelerate, collide, or de-materialize anything physical, and this
project makes no such claim. Where the corpus attaches physical meaning
to a number, the built-in falsifiability auditor says exactly what the
computation supports and what it doesn't.

## Quickstart

```bash
# theorem test suite (15 assertions, all must pass)
python -m awen_collider.tests

# full run: falsifiability audit + 50k-node collision run
python -m awen_collider.run_collider --nodes 50000 --ticks 96 --audit --json logs/collision_report.json
```

Open `visualizer.html` in any browser for the live dashboard (same math
ported to JS: Hopf/stereographic S³ view, triadic 120° grid, Bloch-wall
void, ℤ/24 toggle wheel, live associator telemetry).

## The four directives, as implemented

| Directive | Implementation | Status |
|---|---|---|
| 1. Base-13 clock `e^(13×1.5)` | All per-tick angles scaled by `BASE13_CLOCK_RATIO` | parameter, as directed; auditor notes its actual deviation from CODATA c is 1.84% |
| 2. Eta-taxed fold ≈ 0.48 | `FOLD_LOCK = 0.480000038` (QTE-exact), never 0.50 | locked |
| 3. 31/24 toggle | tick phase advances by `31 mod 24 = 7`; gcd(7,24)=1 ⇒ the walk generates all of ℤ/24 (tested) | locked |
| 4. Observer anchor | `(2.5·rotated + 1.5·folded) / 4.0` per step | locked |

## Architecture

```
awen_collider/
├── algebra.py       exact H and O algebra (Hamilton, Cayley–Dickson,
│                    associator); numpy default, torch-compatible
├── constants.py     every constant tagged [IDENTITY]/[PARAM]/[EXTERNAL]
├── engine.py        dual-ledger collider; QTE device roles
│                    (Real→cuda:0 RTX 4070, Imag→cuda:1 RTX 2080S)
├── audit.py         falsifiability engine — 40 corpus claims checked
├── codec.py         Module B: lossless UBBM container + compression bench
├── tests.py         structural theorem suite
└── run_collider.py  CLI
```

Related repos (same ecosystem, github.com/OwainGlyndwr1400):
`quaternionic-toroidal-engine` (QTE — quaternion sector, torch/CUDA;
this package matches its conventions and extends it with the octonion
sector), `awen-unzipping-horizon` (numpy quaternion core), `aether-scope`
(Electron dashboard — can consume `logs/collision_report.json`).

### GPU backend

Installed and verified: `torch 2.11.0+cu128` (Python 3.14, driver CUDA
13.3). The Backend assigns ledger roles by GPU *name* — Real Ledger →
RTX 4070 (cuda:0), Imaginary Ledger → RTX 2080 Super (cuda:1) — so the
mapping survives any enumeration-order change. CPU fallback (NumPy,
float64) remains automatic when torch is absent; `--cpu` forces it.

Measured, Level I (144,000 nodes × 369 ticks, measure every 9):

| Backend | Precision | Time | Parity drift | Baseline rift |
|---|---|---|---|---|
| NumPy CPU | float64 | 55.4 s | 2.4e-12 | 1.5e-16 |
| Dual GPU  | float32 | 5.6 s  | exact at fp32 | 7.8e-08 |

Cross-backend reproducibility: the GPU float32 run reproduces the CPU
float64 telemetry to ~4 decimal places tick-for-tick (final-tick rift
identical to printed precision) — the attractor's contraction squeezes
out rounding differences, which is itself evidence the observables are
properties of the dynamics, not the arithmetic.

## The mathematics that is real (and tested)

- **The Divine Equation is SO(4).** ψ → q_b·ψ·q_a⁻¹ with unit
  quaternions is the *general* rotation of 4D space (Spin(4) =
  SU(2)×SU(2)). Norm preservation is exact — verified to 3e-12 over
  full runs.
- **Ledger parity is a theorem, not a tuning.** H and O are composition
  algebras (|xy| = |x||y|), so unit-generator evolution conserves norms
  by necessity. Measured drift ~1e-12 is float error only.
- **The Associative Rift is self-calibrating.** The associator
  [x,y,z] = (xy)z − x(yz) is identically 0 on any quaternionic
  subalgebra (measured: 1.5e-16) and nonzero when the 8D sector
  participates (measured: ~1.18). Octonions have *no* matrix
  representation — matrix algebras are associative — which is why the
  spec-v1 idea of `matmul`-ing the 8D ledger was unimplementable and
  Cayley–Dickson multiplication is required.
- **The stride-7 toggle generates ℤ/24** because gcd(7,24) = 1 — the
  rigorous content of "the +7 residue keeps the loop alive."

## Findings from this build (worth your team's attention)

1. **The "Lion" attractor is algorithm-dependent.** Under the
   directive-4 weights (2.5:1.5)/4, the median |ijk|/|w| settles at
   ≈ 11.5 — not the QTE lion-hunt value 0.5352. The attractor moves
   with the rotation/fold mix. Emergent constants of this family are
   properties of the chosen iteration, not universal invariants. A
   weight-sweep experiment mapping attractor vs. mix ratio would make
   this precise.
2. **The Real Ledger collapses to a ring.** Visible live in
   `visualizer.html`: the uniform S³ cloud contracts onto a periodic
   orbit (1D attractor) of the observer-weighted map. Real, striking,
   reproducible dynamics.
3. **Audit results** (regenerate anytime with `--audit`; JSON written
   locally to `logs/claims_audit.json`):
   18 VERIFIED, 8 FALSE, 5 CONTRADICTION, 6 EXTERNAL, 1 NOT-REPRODUCED,
   1 OPEN, 1 UNTESTABLE. The FALSE set now includes the closed
   provenance of K_ELG (input/k_e — an algebraic identity), the Lion
   constant (a SoulEngine torsion snapshot, not a dynamical attractor),
   and the 434/465 Hz frequencies (file-count snapshots through
   432·(1+t/5)) — see docs/PAPER.md §6.5. The NOT-REPRODUCED entry is the QTE "0.48
   resonance": a pre-registered 10M-node blind sweep
   (`experiments/blind_fold_sweep.py`) found L(F) smooth and
   monotonically decreasing with no feature at 0.48. Highlights: GCD(c, ν_Cs) = 14 settles the corpus's
   7-vs-14 discrepancy; only one of the three "Null Ledger identities"
   is actually zero; the corpus asserts both P=NP and P≠NP; the QTE
   README's "0.48 PROVEN / no constants imposed" is circular as coded
   (FOLD_LOCK is hardcoded input). **Recommendation before further
   Zenodo uploads:** re-run the fold-amplitude claim as a genuine blind
   sweep using this engine's parameterized fold, and update the corpus
   rows the auditor flagged.

## Code only — research lives on Zenodo

This repository contains the instrument's **code** under the PolyForm
Noncommercial License 1.0.0. The paper, run data and audit tables,
verification report, figures, and the research corpus (CSVs and source
documents) are archived on Zenodo with their own DOIs — not here.
Runs regenerate all data locally: `logs/` is created on first use, and
`--audit` reprints the full claims table on demand.

---
*Instrument status: geometry engine verified · claims auditor active ·
no physical-effect claims made or implied.*
