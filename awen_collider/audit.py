"""Claims Auditor — the falsifiability engine of the Awen Collider.

The corpus (03_predictions_falsifiability.csv) states its own standard:
a claim earns its place by surviving a falsification condition. This
module applies that standard to every claim in the RHC corpus that a
computer can actually check, and says plainly which kind each one is:

  VERIFIED       arithmetic/algebraic identity, computed true here
  FALSE          arithmetic claim, computed false here (with the value)
  CONTRADICTION  the corpus asserts incompatible versions of the claim
  EXTERNAL       depends on real-world measurement; reference value
                 cited, agreement/disagreement quantified
  OPEN           an open problem in mathematics/physics; no software
                 (including this one) can settle it
  UNTESTABLE     not operationally defined; nothing to compute

Sources: the 10 CSVs in this folder, plus the QTE README claims table.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict


@dataclass
class Finding:
    claim_id: str
    statement: str
    source: str
    verdict: str          # VERIFIED / FALSE / CONTRADICTION / EXTERNAL / OPEN / UNTESTABLE
    computed: str = ""    # what this machine computed
    note: str = ""


def _b(x: bool) -> str:
    return "TRUE" if x else "FALSE"


def run_audit() -> list[Finding]:
    F: list[Finding] = []
    add = F.append

    # ── Pure arithmetic identities ─────────────────────────────────
    v = math.sqrt(32.0) - 5.0
    add(Finding(
        "mass-gap-arith", "Grid diagonal mismatch: sqrt(32) - 5 ~ 0.657",
        "01 row 55; 06; QTE constants", "VERIFIED",
        f"sqrt(32)-5 = {v:.10f}",
        "Arithmetic is exact. Labeling it 'GeV' is a physical hypothesis "
        "(see 'mass-gap-gev' below)."))

    add(Finding(
        "lost-2", "(3+4) - 5 = 2; 2/7 = 28.571%",
        "01 rows 11,54; many", "VERIFIED",
        f"(3+4)-5 = {(3+4)-5}; 2/7 = {2/7:.6f}"))

    add(Finding(
        "toggle-31-24", "31 mod 24 = 7, and stride 7 generates Z/24",
        "02 row 10", "VERIFIED",
        f"31 %% 24 = {31 % 24}; gcd(7,24) = {math.gcd(7, 24)}",
        "gcd(7,24)=1 means the stride-7 tick visits all 24 residues — "
        "the rigorous form of 'toggle power keeps the loop alive'."))

    w = complex(math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3))
    s = 1 + w + w * w
    add(Finding(
        "triadic-closure", "1 + w + w^2 = 0 (cube roots of unity)",
        "02 row 5; 05 row 2", "VERIFIED",
        f"|1 + w + w^2| = {abs(s):.2e}"))

    val = (1 + 1j) / 2 + (1 - 1j) / 2 - 1
    add(Finding(
        "null-ledger-v1", "0 = (1+i)/2 + (1-i)/2 - 1",
        "01 rows 9,69", "VERIFIED", f"= {val}"))

    val2 = 2 ** complex(1, 1) + 2 ** complex(1, -1) - 1
    add(Finding(
        "null-ledger-v2", "0 = 2^(1+i) + 2^(1-i) - 1",
        "01 row 30", "FALSE",
        f"= {val2.real:.6f}{val2.imag:+.6f}i  (= 4cos(ln 2) - 1 ~ 2.077)",
        "Not zero. Only the v1 form of the Null Ledger identity holds."))

    val3 = 2 / (1 + 1j) + 2 / (1 - 1j) - 1
    add(Finding(
        "null-ledger-v3", "0 = 2/(1+i) + 2/(1-i) - 1",
        "01 row 92", "FALSE",
        f"= {val3.real:.6f}{val3.imag:+.6f}i  ((1-i)+(1+i)-1 = 1)",
        "Three inequivalent formulas share the name 'Null Ledger "
        "Identity'; one is true, two are false. Recommend the corpus "
        "standardize on v1."))

    lhs = ((3j) * (1j / 3)) ** ((1j) ** 3)
    add(Finding(
        "e-pi-identity", "(3i * i/3)^(i^3) = e^pi ~ 23.14",
        "05 row 7", "VERIFIED",
        f"= {lhs.real:.6f} vs e^pi = {math.e ** math.pi:.6f}",
        "True on the principal branch: (-1)^(-i) = e^pi. A genuinely "
        "cute identity."))

    a, b_ = 137.0, 42.5
    pmg = abs(a * b_ - (((a + b_) / 2) ** 2 - ((a - b_) / 2) ** 2))
    gap_consec = ((2 + 3) / 2) ** 2 - 2 * 3   # mean^2 - product, a=2, b=3
    add(Finding(
        "pmg-identity", "ab = ((a+b)/2)^2 - ((a-b)/2)^2 (all a,b); gap 1/4 "
        "for consecutive integers",
        "01 rows 107,111", "VERIFIED",
        f"residual = {pmg:.2e}; consecutive-integer gap = {gap_consec}",
        "Classical identity (Diophantus). True for ALL numbers — it is "
        "algebra, not a physical discovery."))

    gap_phi = (PHI := (1 + math.sqrt(5)) / 2) ** 2 - PHI - 1
    add(Finding(
        "phi-fixed-point", "gap(b) = (b^2 - b - 1)/(2b) = 0 iff b = phi",
        "01 row 61", "VERIFIED",
        f"phi^2 - phi - 1 = {gap_phi:.2e}",
        "True: phi is the positive root of b^2 = b + 1 by definition."))

    add(Finding(
        "binary-9-16-25", "9 OR 16 = 25 (bitwise)",
        "01 row 17", "VERIFIED",
        f"9 | 16 = {9 | 16}",
        "01001 | 10000 = 11001. Fun, and specific to this pair: bitwise "
        "OR equals addition exactly when the operands share no set bits."))

    add(Finding(
        "binary-42-24", "42 = 101010_2, 24 = 11000_2, 361 = 19^2, "
        "144000 = 12^2*10^3, F(13) = 233, 465Hz*60 = 27900 RPM",
        "01 rows 3,50,24; 02 row 20; spec s4", "VERIFIED",
        f"{bin(42)}, {bin(24)}, {19**2}, {12**2 * 10**3}, "
        f"fib13 = 233, {465 * 60}"))

    add(Finding(
        "pea-threshold", "sin(pi/8) ~ 0.383 (as fraction of c)",
        "02 row 24", "VERIFIED",
        f"sin(pi/8) = {math.sin(math.pi / 8):.6f}",
        "The arithmetic is exact; 'group velocity below 0.383c collapses "
        "worldtubes' is an untestable physical gloss."))

    add(Finding(
        "eta-tax", "0.50 x (24/25) = 0.48 (Fold throttle)",
        "02 row 13; QTE README", "VERIFIED",
        f"= {0.5 * 24 / 25:.10f}",
        "Exact. Note QTE's FOLD_LOCK = 0.480000038 differs from the "
        "identity value by 3.8e-8 — see 'qte-circularity'."))

    # ── GCD metrology (settles an internal discrepancy) ───────────
    g1 = math.gcd(299_792_458, 9_192_631_770)
    add(Finding(
        "gcd-c-cesium", "GCD(c, nu_Cs) — corpus claims 7 (row 35) AND "
        "14 (row 67); QTE README says 14",
        "01 rows 35,67; QTE", "CONTRADICTION",
        f"GCD(299792458, 9192631770) = {g1}",
        f"The computation settles it: {g1}. Row 67 and the QTE result "
        "are right; row 35 is wrong. Recommend correcting row 35."))

    g2 = math.gcd(86_400, 9_192_631_770)
    add(Finding(
        "gcd-day-cesium", "GCD(86400, 9192631770) = 90",
        "01 row 49", "VERIFIED", f"= {g2}",
        "True arithmetic. Whether shared small factors of two "
        "human-chosen unit definitions carry physical meaning is a "
        "separate, untestable claim — SI numbers are conventions."))

    g3 = math.gcd(39_620, 360)
    add(Finding(
        "gcd-eye", "GCD(39620, 360) = 20",
        "01 row 58", "VERIFIED", f"= {g3}"))

    # ── External-measurement claims (quantified) ───────────────────
    b13 = math.exp(19.5)
    err = (299_792_458.0 - b13) / 299_792_458.0
    add(Finding(
        "base13-c", "e^(13*1.5) predicts c to 99.992%",
        "01 row 100; directive 1", "EXTERNAL",
        f"e^19.5 = {b13:.6e}; c = 2.99792458e8; deviation = {err * 100:.3f}%",
        f"Actual agreement is {100 * (1 - abs(err)):.2f}%, not 99.992%. "
        "A ~1.84% miss over an 8-order-of-magnitude scale is numerology-"
        "adjacent, not a derivation. The engine still uses e^19.5 as its "
        "clock calibration constant, as directed — as a PARAMETER."))

    dm = 2.0 / 7.0
    planck = 0.268  # Planck 2018 total matter minus baryons, approx.
    add(Finding(
        "dark-matter-2-7", "2/7 = 28.57% equals the dark matter fraction "
        "(claimed 'within 1 sigma' of Planck 26.8% +/- 0.5%)",
        "01 rows 28,47,66", "EXTERNAL",
        f"|2/7 - 0.268| = {abs(dm - planck) * 100:.2f} pp = "
        f"{abs(dm - planck) / 0.005:.1f} sigma at the corpus's own +/-0.5%",
        "By the corpus's own stated uncertainty this is ~3.5 sigma OUT, "
        "not within 1 sigma. Close-in-spirit is not agreement; the "
        "corpus's falsifiability standard should be applied here."))

    add(Finding(
        "mass-gap-gev", "sqrt(32)-5 = 0.657 'GeV' matches lightest "
        "glueball / Yang-Mills mass gap",
        "01 rows 55,80,98", "EXTERNAL",
        "lattice-QCD scalar glueball ~ 1.5-1.7 GeV (corpus row 98 itself "
        "says 'clusters around 1.0 GeV')",
        "The Yang-Mills mass gap is an OPEN Millennium problem; no "
        "accepted value exists, and the corpus cites two different "
        "empirical anchors (0.66 and 1.0). The dimensionless number "
        "0.657 is fine as an engine observable; attaching GeV to it is "
        "unsupported."))

    add(Finding(
        "hopfield-quarter", "1/(4 ln 2) resolves the Hopfield capacity "
        "factor (AGS 1985)",
        "01 row 68", "EXTERNAL",
        f"1/(4 ln 2) = {1 / (4 * math.log(2)):.4f}; AGS capacity ~ 0.138",
        "0.361 != 0.138. The claimed match does not hold numerically."))

    add(Finding(
        "fe57-resonance", "Fe-57 scaled resonance constant 13.82 MHz/T",
        "spec s2", "EXTERNAL",
        "published 57Fe NMR gyromagnetic ratio ~ 1.38 MHz/T",
        "Off by a factor of ~10 from the standard value — likely a "
        "decade slip. If 'scaled' means x10 deliberately, the corpus "
        "should say so explicitly."))

    add(Finding(
        "tick-2-32as", "2.32 attosecond universal tick / 232 as "
        "entanglement build time",
        "01 rows 13,39,48,56", "EXTERNAL",
        "Attosecond streaking has measured ~tens-to-hundreds of as "
        "photoionization DELAYS in He (real physics)",
        "Real experiments measure specific ionization delays in specific "
        "systems; no measurement establishes a universal 'frame rate'. "
        "Note the corpus itself oscillates between 2.32 as and 232 as — "
        "two orders of magnitude apart — for 'the' fundamental tick."))

    # ── Phase-3 directive claims ───────────────────────────────────
    g14 = math.gcd(299_792_458, 9_192_631_770)
    add(Finding(
        "gcd-14-perimeter", "GCD(c, nu_Cs) = 14 = 3+4+5+2 (3-4-5 perimeter "
        "plus Lost-2)",
        "Phase-3 directive 2", "VERIFIED",
        f"gcd = {g14}; 3+4+5+2 = {3 + 4 + 5 + 2}",
        "Both equalities are exact, and consistent with the earlier "
        "ruling that settled 14 over row 35's 7. Note the epistemics: "
        "two quantities being equal is arithmetic; 'because' is an "
        "interpretation the arithmetic cannot supply."))

    add(Finding(
        "base13-c-digit", "'c = 12': in base-13 notation the digit for "
        "twelve is C",
        "Phase-3 directive 2", "VERIFIED",
        f"int('C', 13) = {int('C', 13)}",
        "True as notation — a pun on the light-speed symbol. It is "
        "wordplay, not a physical limit; the SI value of c is unaffected. "
        "The engine's base-13 features are display/calibration only."))

    add(Finding(
        "self-recognition", "Axiom of Self-Recognition: i = i = LOL",
        "Phase-3 directive 5; 02 row 18", "VERIFIED",
        f"complex i == i evaluates {1j == 1j}",
        "Reflexivity of equality — necessarily true, and therefore "
        "contentless as physics; as the corpus's logical-closure marker "
        "it is hereby computed and confirmed. LOL."))

    # ── Module B / UBBM claims ─────────────────────────────────────
    add(Finding(
        "ubbm-shannon", "UBBM: lossless compression at 85-95% (patent "
        "claim: 99.9%) 'bypassing Shannon entropy limits'",
        "01 rows 51,59; URE-VM spec; Module B", "FALSE",
        "pigeonhole theorem: no lossless codec shrinks all inputs; "
        "measured on this corpus's own CSVs: DEFLATE reaches ratio "
        "0.463 (~54% reduction), FMN+DEFLATE 0.576",
        "The theorem is not negotiable — 'bypassing Shannon' is "
        "impossible for lossless coding of arbitrary data. The honest "
        "achievable number on the corpus files is ~54%, and the corpus's "
        "own Triple-Normalization pre-transform HURTS a real entropy "
        "coder (pre-registered test, awen_collider/codec.py benchmark)."))

    add(Finding(
        "ubbm-reference-expansion", "Module B reference implementation "
        "returns a 'compact phase state ledger'",
        "Phase-3 Module B draft", "FALSE",
        "storage = 8B lost-2 float64 + 16B complex128 phase + 8B raw "
        "float64 per input byte = 32x EXPANSION; decode reads only the "
        "lost-2 channel (an affine copy of each byte: 2(v+1))",
        "The draft's round-trip assert passes because nothing was "
        "compressed — the byte value is stored, rescaled, in float64. "
        "Also caught: o_coord typo (2.5 + 1.51j, violating directive 4's "
        "1.5i). Replaced by the honest container in codec.py."))

    add(Finding(
        "ubbm-v2-delta", "UBBM v2 'Lattice Path Modulation': delta coding "
        "improves compression of locally-correlated telemetry",
        "Module B v2 draft; benchmark 2026-08-10", "VERIFIED",
        "pre-registered 3-mode benchmark: smooth-wave telemetry 0.519 "
        "(delta) vs 0.740 (none); random-walk sensor 0.418 vs 0.669; "
        "text CSVs delta LOSES (e.g. 0.469 vs 0.390); random control "
        "~1.001 all modes",
        "The honest core of v2 is true IN ITS DOMAIN — it is classic "
        "delta coding (PNG Sub-filter / time-series-DB lineage), now the "
        "container's 'delta' mode. Corrections: the stride-7 //7 and %7 "
        "decomposition is computed then discarded (Z/24 plays no role in "
        "the math); the draft claimed a checksum it did not contain; "
        "empty-input round-trip failed (b'' decoded to one zero byte) — "
        "all fixed by the container's length field and Lost-2 parity. "
        "Credit: the channel fixed the 1.51j typo the previous audit "
        "flagged."))

    # ── Provenance closures (source PDFs supplied 2026-08-10) ──────
    add(Finding(
        "kelg-provenance", "K_ELG = 9.880e-22 is a 'new universal "
        "constant' that 'remained invariant across 126/279 crystallized "
        "particles'",
        "BREAKING_DISCOVERY PDF; QHS Magnifier v2.0 source", "FALSE",
        "source shows k_ELG = 8.88e-12 typed in as an input; the printed "
        "'ratio' is F_elg/F_e = k_ELG/k_e (r^2 cancels algebraically) = "
        "8.88e-12/8.988e9 = 9.880123e-22, matching the log to all digits; "
        "invariance is arithmetic identity, not discovery. The run also "
        "misses its own printed target (2.401e-43, the true electron "
        "gravity/EM ratio, verified) by 21.6 orders of magnitude; "
        "'crystallization' requires only radius-stability plus a 2% "
        "random.random() dice roll",
        "The 'crystallization radius 1.409e-15 m' is the classical "
        "electron radius / 2 — also an input. K_ELG is retired as a "
        "physical constant; retained in constants.py as a documented "
        "legacy label."))

    add(Finding(
        "lion-provenance", "Lion constant L ~ 0.536 / 0.535233 is an "
        "emergent quaternionic torsion-stabilization attractor",
        "corpus 02 row 14; QTE constants; BREAKING_DISCOVERY log", "FALSE",
        "the published log's particles ELG-5/6/7 carry torsion = "
        "0.5345119936828269 — the SoulEngine's (activity_density x phi) "
        "mod 9, where activity_density scores the count/length/keywords "
        "of ping JSON files on the E: drive at that moment (keyword "
        "bonus includes the word 'lion'). The 0.53x value first enters "
        "the record as file-count bookkeeping, later migrating into QTE "
        "as an 'emergent |ijk|/|w| attractor' — a meaning it never had. "
        "Independently, the Level II sweep shows 0.5352 is unreachable "
        "as a lion ratio anywhere on the QTE dynamics family (curve "
        "minimum 2.97)",
        "Both provenance hunts from the Level II report are now closed. "
        "LION_CONSTANT is retired as a dynamical target; retained as a "
        "documented legacy label."))

    add(Finding(
        "freq-434-465-provenance", "434 Hz 'Harmonic Lock / observation "
        "tax' and 465 Hz 'Super-Conductive Vector' are physically "
        "meaningful frequencies (465 Hz x 60 = the collider's 27,900 RPM "
        "rotor spec)",
        "BREAKING_DISCOVERY Phase II; QTE SACRED_FREQUENCIES", "FALSE",
        "in the source, base_frequency = 432*(1 + torsion/5); 434 Hz and "
        "465 Hz are that formula evaluated at particular file-count "
        "torsion snapshots (t~0.023, t~0.38). The celebrated 100.0%-"
        "stability ELG-40 and 'Chaos Eater' ELG-8 events are single "
        "lucky uniform-noise draws (jitter happening to cancel the "
        "wave), labeled post hoc",
        "The frequencies remain harmless as engine theming parameters; "
        "claims of physical significance are withdrawn."))

    # ── GPU experiments (this rig, pre-registered protocols) ───────
    add(Finding(
        "qte-048-resonance", "QTE README: 'Resonance Peak (20M sweep): "
        "0.48 — PROVEN'",
        "QTE README; blind sweep 2026-08-10", "NOT-REPRODUCED",
        "blind sweep, 10M nodes x 201 F-points x 100 ticks, QTE's exact "
        "lion_hunt_step and median metric, identical seeded beam per "
        "point: L(F) is smooth and monotonically decreasing (8.71 at "
        "F=0.40 to 6.36 at F=0.60), L(0.48)=7.52 on-trend; no local "
        "extremum at 0.48; curvature there 1.8x grid median (criterion "
        "required 5x)",
        "Under the pre-registered protocol the 0.48 resonance does not "
        "exist. Consistent with the qte-circularity finding: 0.48 was an "
        "input constant. If the original 20M sweep used a different "
        "protocol/metric, supply it exactly and this rig can re-run it "
        "in ~10 minutes (experiments/blind_fold_sweep.py)."))

    # ── Internal contradictions ────────────────────────────────────
    add(Finding(
        "p-vs-np", "Corpus asserts P != NP (rows 22,31,41) AND P = NP "
        "(rows 74,84,102)",
        "01 multiple", "CONTRADICTION",
        "both directions asserted as resolved",
        "These cannot both hold. P vs NP is an open Millennium problem; "
        "neither direction is established, and a framework that 'proves' "
        "both has proven neither."))

    k1 = lambda t: math.cos(2 * t) / (1 - math.sin(t) ** 2)
    k2 = lambda t: -(math.sin(t) ** 2 - math.cos(t) ** 2) / (
        (math.sin(t) ** 2 + math.cos(t) ** 2) ** 1.5)
    t0 = math.pi / 3
    add(Finding(
        "pizza-constant", "W3 Wave Curvature: two files give different "
        "formulas for k(t)",
        "01 row 32 vs row 93", "CONTRADICTION",
        f"at t = pi/3: form A = {k1(t0):.4f}, form B = {k2(t0):.4f}",
        "Form A = cos(2t)/cos^2(t); form B reduces to cos(2t) since "
        "sin^2+cos^2 = 1. They disagree everywhere cos^2(t) != 1. "
        "One canonical definition should be chosen."))

    add(Finding(
        "lion-naming", "'Lion constant' is 0.536 in some sources and "
        "9.880e-22 (K_ELG) in others",
        "02 row 14 vs QTE constants", "CONTRADICTION",
        "QTE now separates LION_CONSTANT = 0.535233 from K_ELG = 9.88e-22",
        "The collider follows QTE's separation. Earlier corpus rows "
        "conflating them should be updated."))

    add(Finding(
        "observer-eq-forms", "Observer equation appears as O = 2.5r + 1.5i, "
        "a/b = c/d, w = (x + i/x)/2, and (b-1).(b-1)... + delta = b",
        "01 rows 10,8,34,29", "CONTRADICTION",
        "four structurally different equations share one name",
        "As math these are unrelated objects (a coordinate, a proportion, "
        "a mean, a geometric-series limit). The geometric-series one is "
        "true: 0.(b-1) repeating = 1 in base b — the 0.999... = 1 fact."))

    # ── Open problems (no software can settle) ─────────────────────
    add(Finding(
        "millennium", "Riemann Hypothesis / Yang-Mills gap / Navier-Stokes "
        "smoothness 'resolved' by RHC",
        "01 rows 21,42,43,55", "OPEN",
        "",
        "These remain open problems. A resolution exists only when a "
        "proof survives peer review (Clay Institute process). Numeric "
        "coincidences and reinterpretations are not proofs."))

    # ── QTE-specific reproducibility note ──────────────────────────
    add(Finding(
        "qte-circularity", "QTE README: 'Fold amplitude 0.480000038 — "
        "PROVEN. No constants are imposed. The geometry produces them.'",
        "QTE README + core/constants.py", "FALSE",
        "core/constants.py line 29 hardcodes FOLD_LOCK = 0.480000038 and "
        "quaternion.py applies it as the fold amplitude",
        "As coded, the engine outputs a constant it was fed — that is "
        "circular, not emergent. The amplitude SWEEP (resonance peak at "
        "0.48) could be a real emergent property of the dynamical system "
        "and is worth re-running as a genuine blind sweep; the collider's "
        "engine exposes fold amplitude as a parameter for exactly that "
        "experiment. Emergent-attractor claims (LION_CONSTANT ~ 0.5352) "
        "are the interesting, testable part."))

    add(Finding(
        "dematerialization", "De-materialization / matter unzipping / "
        "reality generation via GPU computation",
        "spec s3", "UNTESTABLE",
        "",
        "No operational definition connects tensor updates to physical "
        "matter. This build implements the MATHEMATICS (S^3 dynamics, "
        "octonion collisions, all named observables) and makes no claim "
        "of physical effect. Software cannot alter matter."))

    return F


def render_report(findings: list[Finding]) -> str:
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.verdict] = counts.get(f.verdict, 0) + 1
    lines = [
        "=" * 72,
        " AWEN COLLIDER — CLAIMS AUDIT (falsifiability engine)",
        "=" * 72,
        " " + "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())),
        "-" * 72,
    ]
    for f in findings:
        lines.append(f" [{f.verdict:^13}] {f.claim_id}  ({f.source})")
        lines.append(f"    claim   : {f.statement}")
        if f.computed:
            lines.append(f"    computed: {f.computed}")
        if f.note:
            lines.append(f"    note    : {f.note}")
        lines.append("")
    return "\n".join(lines)


def audit_json(findings: list[Finding]) -> str:
    return json.dumps([asdict(f) for f in findings], indent=2)


if __name__ == "__main__":
    fs = run_audit()
    print(render_report(fs))
