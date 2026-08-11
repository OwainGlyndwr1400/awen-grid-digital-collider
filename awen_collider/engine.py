"""Awen Grid Digital Collider — dual-ledger S^3 x S^7 dynamics engine.

Real Ledger  (target cuda:0, RTX 4070):  N unit quaternions on S^3,
    evolved by the QTE observer-weighted step (rotate 120 deg triadic,
    fold at the Eta-taxed amplitude, combine 2.5:1.5, normalize).

Imaginary Ledger (target cuda:1, RTX 2080S):  N unit octonions on S^7,
    evolved by unit-octonion left/right multiplication (norm-preserving
    because O is a composition algebra).

Collision:  embed Real states into O and measure the associator
    [psi_R, psi_I, probe] — the Associative Rift. Zero iff the collision
    stays inside an associative (quaternionic) subalgebra; nonzero rift
    is the signature of genuine 8D participation.

Directives implemented:
  1. Base-13 clock: all per-tick angles scaled by BASE13_CLOCK_RATIO.
  2. Eta tax: fold amplitude locked at FOLD_LOCK ~ 0.48, never 0.50.
  3. 31/24 toggle: tick phase advances by 31 mod 24 = 7 in Z/24.
  4. Observer anchor: O = 2.5r + 1.5i, feedback divided by the 4.0 sum.

Backend: NumPy on CPU by default. If PyTorch with CUDA is installed the
ledgers are placed on their assigned GPUs automatically (QTE device
roles). Physics disclaimer: this is a numerical simulation and claims
auditor; it computes geometry, it does not act on matter.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from . import algebra as alg
from .constants import (
    BASE13_CLOCK_RATIO, CHANNEL_A_HZ, CHANNEL_B_HZ, CHANNEL_PINEAL_HZ,
    FOLD_LOCK, LATTICE_DIMENSION, LION_CONSTANT, MASS_GAP,
    OBSERVER_IMAG, OBSERVER_REAL, OBSERVER_SUM, OBSERVER_THETA,
    ROTOR_RPM, TOGGLE_SIGNAL, TOGGLE_STRIDE, GPU_REAL_LEDGER,
    GPU_IMAG_LEDGER,
)

_SQRT3 = math.sqrt(3.0)


# ----------------------------------------------------------------------
# Backend selection (numpy CPU / torch CUDA), QTE device roles
# ----------------------------------------------------------------------

class Backend:
    """Places the Real Ledger on cuda:0 and the Imaginary Ledger on
    cuda:1 when torch+CUDA is available; otherwise NumPy on CPU."""

    def __init__(self, force_cpu: bool = False):
        self.xp = np
        self.torch = None
        self.real_device = "cpu"
        self.imag_device = "cpu"
        self.real_name = "CPU"
        self.imag_name = "CPU"
        if force_cpu:
            return
        try:
            torch = self._import_torch()
            if not torch.cuda.is_available():
                import sys  # noqa: PLC0415
                print("  *** WARNING: torch imported but CUDA is NOT "
                      "available in this process — falling back to CPU.")
                print(f"  ***          interpreter: {sys.executable}")
                return
            if torch.cuda.is_available():
                self.torch = torch
                self.xp = torch
                n = torch.cuda.device_count()
                names = {i: torch.cuda.get_device_name(i) for i in range(n)}
                # Assign ledger roles by NAME: CUDA enumeration order is
                # not guaranteed to match nvidia-smi (this machine lists
                # the 2080S first there). Real Ledger -> RTX 4070 if
                # present; Imaginary Ledger -> the other card.
                real_idx = next(
                    (i for i, nm in names.items() if "4070" in nm),
                    GPU_REAL_LEDGER if GPU_REAL_LEDGER in names else 0)
                imag_idx = next(
                    (i for i in names if i != real_idx), real_idx)
                self.real_device = f"cuda:{real_idx}"
                self.imag_device = f"cuda:{imag_idx}"
                self.real_name = names[real_idx]
                self.imag_name = names[imag_idx]
        except (ImportError, OSError) as exc:
            # OSError covers Windows DLL-load edge cases (fresh install
            # still being scanned, wrong-profile user site, etc.)
            import sys  # noqa: PLC0415
            print(f"  *** WARNING: torch unavailable ({type(exc).__name__}: "
                  f"{exc}) — falling back to CPU.")
            print(f"  ***          interpreter: {sys.executable}")

    @staticmethod
    def _import_torch():
        """Import torch, bootstrapping past user-site resolution gaps.

        On this rig torch is installed in renfi's USER site-packages
        (the system site C:\\Python314 was not writable at install
        time). Elevated/admin windows and `-s`/PYTHONNOUSERSITE
        contexts run the same interpreter but skip user-site, so plain
        `import torch` fails there. If that happens, insert the known
        user-site path and retry — durable alternative: install torch
        system-wide once from an elevated terminal.
        """
        try:
            import torch  # noqa: PLC0415
            return torch
        except (ImportError, OSError):
            import sys as _sys  # noqa: PLC0415
            from pathlib import Path as _P  # noqa: PLC0415
            usersite = _P(r"C:\Users\renfi\AppData\Roaming\Python"
                          r"\Python314\site-packages")
            if usersite.is_dir() and str(usersite) not in _sys.path:
                _sys.path.insert(0, str(usersite))
                import torch  # noqa: PLC0415
                print("  [backend] torch found via user-site bootstrap "
                      f"({usersite})")
                return torch
            raise

    def ship(self, arr: np.ndarray, device: str):
        if self.torch is None:
            return arr
        return self.torch.tensor(arr, device=device, dtype=self.torch.float32)

    def to_numpy(self, arr) -> np.ndarray:
        if self.torch is not None and isinstance(arr, self.torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    def describe(self) -> str:
        if self.torch is None:
            return ("backend=numpy/CPU (install PyTorch+CUDA to place the "
                    "ledgers on the RTX 4070 / RTX 2080S)")
        return (f"backend=torch  Real Ledger={self.real_device} "
                f"[{self.real_name}]  Imaginary Ledger={self.imag_device} "
                f"[{self.imag_name}]")


# ----------------------------------------------------------------------
# Telemetry
# ----------------------------------------------------------------------

@dataclass
class Telemetry:
    tick: int
    phase24: int                 # position in Z/24 (directive 3)
    rift_mean: float             # mean |associator| — the collision signal
    rift_baseline: float         # same probe, quaternion-only — should be ~0
    parity_drift: float          # |1 - mean|psi_R|| + |1 - mean|psi_I||
    lion_ratio: float            # median |ijk|/|w| on the Real Ledger
    mass_index: float = 0.0      # Perceived Mass Index (m = i channel):
                                 # mean |vector part| of the unit Real beam
                                 # = mean sin(angle from the real axis), in [0,1]
    mass_gap_ref: float = MASS_GAP


@dataclass
class RunReport:
    nodes: int
    ticks: int
    backend: str
    toggle_cycle_complete: bool
    history: list[Telemetry] = field(default_factory=list)

    def final(self) -> Telemetry:
        return self.history[-1]

    def to_json(self) -> str:
        return json.dumps({
            "nodes": self.nodes,
            "ticks": self.ticks,
            "backend": self.backend,
            "toggle_cycle_complete": self.toggle_cycle_complete,
            "constants": {
                "fold_lock": FOLD_LOCK,
                "observer": [OBSERVER_REAL, OBSERVER_IMAG, OBSERVER_SUM],
                "toggle": [TOGGLE_SIGNAL, LATTICE_DIMENSION, TOGGLE_STRIDE],
                "base13_clock_ratio": BASE13_CLOCK_RATIO,
                "mass_gap": MASS_GAP,
                "lion_constant_qte": LION_CONSTANT,
                "rotor_rpm": ROTOR_RPM,
            },
            "history": [vars(t) for t in self.history],
        }, indent=2)


# ----------------------------------------------------------------------
# The collider
# ----------------------------------------------------------------------

class AwenGridCollider:
    def __init__(self, nodes: int = 20_000, seed: int = 1400,
                 force_cpu: bool = False):
        self.nodes = int(nodes)
        self.backend = Backend(force_cpu=force_cpu)
        rng = np.random.default_rng(seed)

        # Real Ledger: S^3 beam
        psi_r = alg.random_unit_quaternions(self.nodes, rng)
        # Imaginary Ledger: S^7 beam
        psi_i = alg.random_unit_octonions(self.nodes, rng)

        self.psi_r = self.backend.ship(psi_r, self.backend.real_device)
        self.psi_i = self.backend.ship(psi_i, self.backend.imag_device)

        # Triadic observer: 120 deg about (1,1,1)/sqrt(3)  [QTE Q_obs]
        q_obs = alg.from_axis_angle(
            np.array([1.0, 1.0, 1.0]) / _SQRT3, OBSERVER_THETA)
        self.q_obs = self.backend.ship(
            q_obs[None, :], self.backend.real_device)

        # Fold vector [0, FOLD_LOCK, 0, 0]  (directive 2)
        fv = np.zeros((1, 4)); fv[0, 1] = FOLD_LOCK
        self.fold_vec = self.backend.ship(fv, self.backend.real_device)

        # Channel generators. Angles are per-tick phase advances derived
        # from the channel ratios, calibrated by the Base-13 clock
        # (directive 1). These are simulation parameters: 432/465/963 Hz
        # enter as ratios against the rotor sync frequency.
        f_ref = ROTOR_RPM / 60.0            # 465.0 — superconductive sync
        ang_a = 2 * math.pi * (CHANNEL_A_HZ / f_ref) * BASE13_CLOCK_RATIO / 24
        ang_p = 2 * math.pi * (CHANNEL_PINEAL_HZ / f_ref) * BASE13_CLOCK_RATIO / 24

        # Octonion generators for the Imaginary Ledger: unit octonions
        # e cos + sin on two different imaginary axes (one inside the
        # quaternion subalgebra, one outside — the l-axis e4), so the
        # echo phase genuinely leaves the associative sector.
        g_l = np.zeros(8); g_l[0] = math.cos(ang_p); g_l[4] = math.sin(ang_p)
        g_r = np.zeros(8); g_r[0] = math.cos(ang_a); g_r[1] = math.sin(ang_a)
        self.gen_l = self.backend.ship(g_l[None, :], self.backend.imag_device)
        self.gen_r = self.backend.ship(g_r[None, :], self.backend.imag_device)

        # Directive 3: Z/24 toggle state
        self.phase24 = 0
        self.visited: set[int] = {0}
        self.tick = 0

    # -- per-tick dynamics ------------------------------------------------

    def _real_step(self):
        """QTE observer-weighted step (directives 2 and 4)."""
        xp = self.backend.xp
        rotated = alg.sandwich(self.psi_r, self.q_obs, self.q_obs, xp=xp)
        folded = alg.hamilton(self.fold_vec, self.psi_r, xp=xp)
        combined = (OBSERVER_REAL * rotated + OBSERVER_IMAG * folded) \
            / OBSERVER_SUM
        self.psi_r = alg.qnormalize(combined, xp=xp)

    def _imag_step(self):
        """Echo phase: norm-preserving octonion rotation on S^7."""
        xp = self.backend.xp
        z = alg.octonion_mul(self.gen_l, self.psi_i, xp=xp)
        z = alg.octonion_mul(z, alg.oinverse(self.gen_r, xp=xp), xp=xp)
        self.psi_i = alg.onormalize(z, xp=xp)

    def step(self):
        self.tick += 1
        # Directive 3: advance the toggle by the 31 mod 24 = 7 residue.
        self.phase24 = (self.phase24 + TOGGLE_SIGNAL) % LATTICE_DIMENSION
        self.visited.add(self.phase24)
        self._real_step()
        self._imag_step()

    # -- collision --------------------------------------------------------

    def collide(self, probe_scale: float = 1.0) -> Telemetry:
        """Tensor-tip intersection of the 4D and 8D sectors.

        rift      = mean |[embed(psi_R), psi_I, probe]|   (8D participates)
        baseline  = mean |[embed(psi_R), embed(psi_R'), probe_H]|
                    (all-quaternionic — provably zero; calibration)
        """
        xp = self.backend.xp
        emb = alg.embed_quaternion(self.psi_r, xp=xp)
        if self.backend.torch is not None and \
                str(self.psi_i.device) != str(emb.device):
            emb = emb.to(self.psi_i.device)

        # Probe: the triadic observer, embedded, tilted by probe_scale.
        probe_np = np.zeros((1, 8))
        probe_np[0, :4] = self.backend.to_numpy(self.q_obs)[0]
        probe = self.backend.ship(probe_np * probe_scale,
                                  self.backend.imag_device)

        rift = alg.associator(emb, self.psi_i, probe, xp=xp)
        rift_mag = self.backend.to_numpy(alg.onorm(rift, xp=xp))

        # Baseline: collide the Real beam against a shifted copy of
        # itself, entirely inside the quaternion subalgebra.
        if self.backend.torch is not None:
            emb_shift = self.backend.xp.roll(emb, 1, 0)
        else:
            emb_shift = np.roll(emb, 1, axis=0)
        base = alg.associator(emb, emb_shift, probe, xp=xp)
        base_mag = self.backend.to_numpy(alg.onorm(base, xp=xp))

        # Ledger parity (composition-algebra conservation check)
        nr = self.backend.to_numpy(alg.qnorm(self.psi_r, xp=xp))
        ni = self.backend.to_numpy(alg.onorm(self.psi_i, xp=xp))
        parity = abs(1.0 - float(nr.mean())) + abs(1.0 - float(ni.mean()))

        # QTE Lion ratio: median |ijk| / |w| on the Real Ledger
        pr = self.backend.to_numpy(self.psi_r)
        imag_mag = np.linalg.norm(pr[:, 1:], axis=1)
        lion = float(np.median(imag_mag / (np.abs(pr[:, 0]) + 1e-8)))
        # Perceived Mass Index (directive: m = i as imaginary impedance).
        # For unit quaternions |vec| = sin(theta from real axis), so this
        # is a bounded [0,1] geometric statistic of the beam.
        mass_index = float(imag_mag.mean())

        return Telemetry(
            tick=self.tick,
            phase24=self.phase24,
            rift_mean=float(rift_mag.mean()),
            rift_baseline=float(base_mag.mean()),
            parity_drift=parity,
            lion_ratio=lion,
            mass_index=mass_index,
        )

    # -- full run ---------------------------------------------------------

    def run(self, ticks: int = 96, collide_every: int = 8) -> RunReport:
        report = RunReport(
            nodes=self.nodes, ticks=ticks,
            backend=self.backend.describe(),
            toggle_cycle_complete=False,
        )
        for _ in range(ticks):
            self.step()
            if self.tick % collide_every == 0:
                report.history.append(self.collide())
        # Directive-3 verification: stride 7 must generate all of Z/24.
        report.toggle_cycle_complete = (
            len(self.visited) == LATTICE_DIMENSION)
        return report
