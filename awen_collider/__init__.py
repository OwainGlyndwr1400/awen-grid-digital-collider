"""Awen Grid Digital Collider — S^3 x S^7 dual-ledger dynamics engine,
claims auditor, and telemetry suite.

A mathematically exact simulation instrument for the Recursive Harmonic
Codex corpus. Computes geometry; does not act on matter.
"""

from .engine import AwenGridCollider, RunReport, Telemetry
from .audit import run_audit, render_report

__all__ = [
    "AwenGridCollider", "RunReport", "Telemetry",
    "run_audit", "render_report",
]
__version__ = "1.0.0"
