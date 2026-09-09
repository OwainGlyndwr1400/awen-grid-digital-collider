# Awen Grid Digital Collider

**Ten million independent points, each following the same local rule and none
aware of any other, self-organise into a single ring in under thirty ticks — and
the same instrument then audits the framework that predicted it, publishing eight
of its own prior claims as FALSE.**

The paper is deliberately two-edged. One edge is constructive: an exact,
GPU-accelerated instrument that evolves unit **quaternions** on S³ and unit
**octonions** on S⁷ in parallel, and in doing so surfaces a genuine, reproducible
emergent attractor. The other edge is corrective: a pre-registered falsifiability
audit applied uniformly to the Recursive Harmonic Codex corpus — including the
authors' own earlier publications.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21889635.svg)](https://doi.org/10.5281/zenodo.21889635)
[![Python](https://img.shields.io/badge/Python-3.11+-3776ab)](https://www.python.org/)
[![Verified](https://img.shields.io/badge/theorems-machine--verified%20to%201e--15-2ea043)](#the-mathematics-that-is-real-and-tested)
[![Licence](https://img.shields.io/badge/Licence-PolyForm%20NC%201.0.0-lightgrey)](LICENSE)

---

## The ring

At `t = 0` the rendered beam is a uniform cloud of thousands of independent
points. Between ticks 9 and 27 — a few seconds of wall time — they arrange
themselves into a single solid ring, with **no inter-particle communication of
any kind**. Every point follows the same local map; the ring is the global
attractor of that map, made visible.

The transition is quantitative, not impressionistic. Over exactly those ticks the
collision rift climbs `1.0868 → 1.1640 → 1.1781` and the lion ratio climbs
`6.11 → 10.75 → 11.49`. From tick ≈27 every observable holds flat for the
remaining 342 ticks. That relaxation time is the **phase-lock threshold**, and it
is set by the contraction rate of the rotation:fold mix — *not* by beam size.

Open `visualizer.html` in any browser and watch it happen live.

## Invariance — the reason to believe it

The locked state is a property of the dynamics, not of the computation. The same
observables survive a **200× range of beam sizes, ten random seeds, two
arithmetic precisions, two backends, and two independent operators**:

| Beam | Nodes | Locked rift (mean ± sd) | Locked lion | Mass index | Backend | Runtime |
|---|---|---|---|---|---|---|
| 50k | 50,000 | 1.17871 ± 0.00147 | 11.554 ± 0.054 | 0.99627 | GPU fp32 | < 2 s |
| 144k | 144,000 | 1.17899 ± 0.00106 | 11.554 ± 0.017 | 0.99627 | GPU fp32 / CPU fp64 | 5.4 s / 55.4 s |
| 1M | 1,000,000 | 1.17886 ± 0.00060 | 11.555 ± 0.009 | 0.99627 | GPU fp32 | 14.9 s |
| 10M | 10,000,000 | 1.17875 ± 0.00007 | 11.556 ± 0.003 | 0.99627 | GPU fp32 | 217.6 s |

The float32 GPU and float64 CPU runs reproduce each other's telemetry to about
four decimal places, tick for tick. Because the map is contractive, rounding
differences **shrink rather than amplify** — so cross-precision agreement is
itself evidence that these numbers are properties of the dynamics rather than of
the arithmetic. The mass index saturates at 0.99627 with an across-seed range of
5×10⁻⁵, the tightest invariant the instrument measures.

## The attractor landscape

Sweeping the rotation weight across 193 values at 10⁶ nodes maps the attractor as
a continuous function of the mix. The lion ratio rises smoothly from 2.97 at
`w = 0.500` (fold-dominated), through 7.52 at the directive point `w = 0.625`
— independently cross-checking the Level I result — to a genuine interior maximum
of **39.4 at w ≈ 0.865**, before collapsing to 3.2 as pure rotation takes over and
preserves the uniform distribution.

That interior peak sits far from any previously hypothesised value, and it is a
real feature of the landscape rather than an assumed one.

## The audit

`--audit` turns the instrument on the corpus that motivated it and grades 40
published claims:

```
18 VERIFIED · 8 FALSE · 5 CONTRADICTION · 6 EXTERNAL · 1 NOT-REPRODUCED · 1 OPEN · 1 UNTESTABLE
```

Two previously published constants of the framework do not survive:

- **The claimed fold-amplitude resonance at 0.48** is **not reproduced** under a
  pre-registered 10⁷-node blind sweep — `L(F)` is smooth and monotone, with no
  feature at 0.48.
- **The legacy "Lion constant" of 0.5352** is **unreachable** anywhere on the
  measured slice, and traces to a bookkeeping snapshot rather than a dynamical
  attractor.

Both retractions are propagated back to the affected repository
([quaternionic-toroidal-engine](https://github.com/OwainGlyndwr1400/quaternionic-toroidal-engine))
rather than left standing there. The framework's compression claims are likewise
corrected by measurement: a **−30 to −37%** delta pre-transform on correlated
telemetry, and **−0% beyond entropy** on arbitrary data.

> A framework that only ever confirms itself is a mirror. A framework that can
> survive its own instrument is research.

> **What this is not:** a device that acts on matter. Software cannot accelerate,
> collide, or de-materialise anything physical, and no such claim is made here.
> The instrument prints that statement in the banner of every run.

---

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

## What else the audit turned up

Beyond the two retracted constants covered above, the 40-item pass produced
several results worth reading before citing the corpus. Regenerate any of them
with `--audit`; JSON is written to `logs/claims_audit.json`.

**Provenance traced to source.** Three claims in the FALSE set were resolved not
by measurement but by finding where the number originally came from:

| Claim | Actual origin |
|---|---|
| `K_ELG` | `input / k_e` — an algebraic identity, closed by construction |
| Lion constant 0.5352 | A SoulEngine torsion snapshot, not a dynamical attractor |
| The 434 / 465 Hz frequencies | File-count snapshots passed through `432·(1 + t/5)` |

**A long-standing discrepancy settled.** `GCD(c, ν_Cs) = 14` resolves the
corpus's 7-versus-14 ambiguity in favour of 14.

**Internal contradictions surfaced.** Only one of the three asserted "Null Ledger
identities" is actually zero. And the corpus asserts both `P = NP` and `P ≠ NP`
in different volumes — a contradiction no measurement can resolve, flagged rather
than silently dropped.

**Compression claims corrected by measurement.** A delta pre-transform yields
**−30 to −37%** on correlated telemetry, and **−0% beyond entropy** on arbitrary
data. The gain is real, and it is specific to correlated data — it is not a
general-purpose compression result.

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
