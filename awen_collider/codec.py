"""UBBM Metrological Codec — Module B (v2-integrated), implemented honestly.

What this is: a genuinely lossless, bit-for-bit reversible container
format with three selectable pre-transforms, a real entropy-coding
stage (DEFLATE), and the Lost-2 geometric identity used for what it
honestly is — an integrity checksum ("Null Ledger parity check").

Modes (the pre-transform applied before DEFLATE):

  "none"   no pre-transform. The safe general-purpose default.
  "delta"  byte-wise delta mod 256 (UBBM v2 "Lattice Path Modulation",
           which — stripped of narrative — is classic delta coding, the
           same family as PNG's Sub filter and time-series-DB deltas).
           Helps on locally-correlated numeric streams; hurts on text.
  "fmn"    v1's full Fold-Mirror-Normalize (2-bit rotation + nibble
           swap + delta). Kept for research fidelity; measured to HURT
           compression on all tested data classes.

What this is not: a way around Shannon. By the pigeonhole principle no
lossless codec can shrink all inputs; achievable ratio is a property of
the DATA's entropy. The benchmark below measures each mode per data
class — the honest way to choose.

v2 draft corrections (recorded in audit.py):
  * the stride-7 // and % decomposition was computed and discarded —
    the stored value is the plain delta; Z/24 plays no role in the math;
  * the draft had no checksum despite claiming one — this container's
    Lost-2 parity provides it;
  * empty-input round-trip bug (b"" decoded to b"\\x00") — fixed here by
    the container's explicit length field.
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np

MAGIC = b"UBBM1"
FLAG_FMN = 0x01
FLAG_DELTA = 0x02
MODES = ("none", "delta", "fmn")


# ── Invertible byte transforms ──────────────────────────────────────

def _fold(a: np.ndarray) -> np.ndarray:
    """Rotate each byte right by 2 bits (90 deg of the 8-bit circle)."""
    a = a.astype(np.uint16)
    return (((a >> 2) | (a << 6)) & 0xFF).astype(np.uint8)


def _unfold(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.uint16)
    return (((a << 2) | (a >> 6)) & 0xFF).astype(np.uint8)


def _mirror(a: np.ndarray) -> np.ndarray:
    """Swap high/low nibbles (exchange the two 4-bit registers)."""
    a = a.astype(np.uint16)
    return (((a >> 4) | (a << 4)) & 0xFF).astype(np.uint8)
    # mirror is its own inverse


def _delta(a: np.ndarray) -> np.ndarray:
    """Delta-encode mod 256 (first byte kept as the anchor)."""
    d = np.diff(a.astype(np.int16), prepend=np.int16(0)) % 256
    return d.astype(np.uint8)


def _undelta(d: np.ndarray) -> np.ndarray:
    return (np.cumsum(d.astype(np.int64)) % 256).astype(np.uint8)


def fmn(a: np.ndarray) -> np.ndarray:
    return _delta(_mirror(_fold(a)))


def fmn_inverse(a: np.ndarray) -> np.ndarray:
    return _unfold(_mirror(_undelta(a)))


# ── Lost-2 parity: the 3-4-5 identity as an integrity checksum ──────

def lost2_checksum(a: np.ndarray) -> int:
    """Sum of per-byte Lost-2 debt, mod 2^32.

    For byte v the scaled 3-4-5 triangle has legs 3(v+1), 4(v+1) and
    hypotenuse 5(v+1); the L1-L2 debt is exactly 2(v+1). Summed over the
    stream this is an honest (weak) checksum — the 'Null Ledger parity
    check' of the container. Integrity only; it stores no data.
    """
    n = int(a.size)
    return int((2 * int(a.astype(np.uint64).sum()) + 2 * n) % (1 << 32))


# ── Codec ───────────────────────────────────────────────────────────

class UBBMCodec:
    """Lossless container: [MAGIC][flags][orig_len u64][lost2 u32][DEFLATE].

    mode: "none" (default) | "delta" (v2, for correlated telemetry) |
    "fmn" (v1, research fidelity). The legacy use_fmn bool still works.
    """

    def __init__(self, mode: str = "none", level: int = 9,
                 use_fmn: bool | None = None):
        if use_fmn is not None:            # v1 back-compat
            mode = "fmn" if use_fmn else "none"
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.level = level

    def encode(self, raw: bytes) -> bytes:
        arr = np.frombuffer(raw, dtype=np.uint8)
        chk = lost2_checksum(arr)
        flags = 0
        payload_arr = arr
        if arr.size:
            if self.mode == "fmn":
                payload_arr, flags = fmn(arr), FLAG_FMN
            elif self.mode == "delta":
                payload_arr, flags = _delta(arr), FLAG_DELTA
        payload = zlib.compress(payload_arr.tobytes(), self.level)
        head = (MAGIC + bytes([flags])
                + len(raw).to_bytes(8, "little")
                + chk.to_bytes(4, "little"))
        return head + payload

    def decode(self, blob: bytes) -> bytes:
        if blob[:5] != MAGIC:
            raise ValueError("not a UBBM1 container")
        flags = blob[5]
        orig_len = int.from_bytes(blob[6:14], "little")
        chk_stored = int.from_bytes(blob[14:18], "little")
        arr = np.frombuffer(zlib.decompress(blob[18:]), dtype=np.uint8)
        if arr.size:
            if flags & FLAG_FMN:
                arr = fmn_inverse(arr)
            elif flags & FLAG_DELTA:
                arr = _undelta(arr)
        if arr.size != orig_len:
            raise ValueError("length mismatch — container corrupt")
        if lost2_checksum(arr) != chk_stored:
            raise ValueError("Null Ledger parity broken — checksum mismatch")
        return arr.tobytes()


# ── Measurement bench ───────────────────────────────────────────────

REFERENCE_BYTES_PER_BYTE = 8 + 16 + 8   # v1 draft: float64 + complex128 + float64


def synthetic_streams(n: int = 65536, seed: int = 1400) -> list[tuple[str, bytes]]:
    """Models of the telemetry classes the v2 draft targets."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    smooth = (128 + 100 * np.sin(t * 0.01)
              + rng.normal(0, 2, n)).clip(0, 255).astype(np.uint8)
    walk = (np.cumsum(rng.integers(-3, 4, n)) % 256).astype(np.uint8)
    rand = rng.integers(0, 256, n, dtype=np.uint8)
    return [("telemetry: smooth wave + noise", smooth.tobytes()),
            ("telemetry: random-walk sensor", walk.tobytes()),
            ("control: uniform random bytes", rand.tobytes())]


def benchmark(named: list[tuple[str, bytes]]) -> list[dict]:
    """Ratio (output/input, lower is better) per mode per stream.
    Every encode is round-trip verified before its ratio is reported."""
    rows = []
    codecs = {m: UBBMCodec(mode=m) for m in MODES}
    for name, raw in named:
        if not raw:
            continue
        row = {"name": name, "bytes": len(raw)}
        for m, c in codecs.items():
            blob = c.encode(raw)
            assert c.decode(blob) == raw
            row[m] = len(blob) / len(raw)
        rows.append(row)
    return rows


def main() -> int:
    codec = UBBMCodec()
    msg = b"Truth is our sword, Knowledge our shield. The Lion Watches."
    assert codec.decode(codec.encode(msg)) == msg
    print("UBBM codec self-test: lossless round-trip OK")

    named = [(p.name, p.read_bytes()) for p in sorted(
        Path(__file__).resolve().parent.parent.glob("*.csv"))]
    named += synthetic_streams()

    print("\nPre-registered predictions (fixed before measurement):")
    print("  delta HELPS on correlated numeric telemetry;")
    print("  delta HURTS or is neutral on text; all modes ~1.0 on random.")
    print(f"\n  {'stream':<44} {'size':>8}  {'none':>7}  {'delta':>7}  {'fmn':>7}")
    rows = benchmark(named)
    for r in rows:
        best = min(MODES, key=lambda m: r[m])
        marks = {m: ("*" if m == best else " ") for m in MODES}
        print(f"  {r['name'][:44]:<44} {r['bytes']:>8}  "
              f"{r['none']:>6.3f}{marks['none']} "
              f"{r['delta']:>6.3f}{marks['delta']} "
              f"{r['fmn']:>6.3f}{marks['fmn']}")
    print("  (* = best mode for that stream)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
