"""Phase 16 - Error-Correcting Codes (ECC) for the watermark payload.

This module is **new Phase 16 code**. It is a *separate, optional* layer that
sits **around** the existing bit-list payload pipeline:

    raw payload bits  --ECC.encode-->  encoded bits  --frozen Phase 6 embed()-->
    watermarked image ... channel ... --extract-->  recovered encoded bits
    --ECC.decode-->  corrected payload bits

It imports nothing from the embedding pipeline and modifies nothing in the
frozen Phase 6 baseline, the Phase 7 UI, the Phase 8 CNN or the Phase 15 CNN.
The frozen ``embed()`` still receives a plain ``list[int]`` of 0/1 - it never
sees the ECC layer.

Method selected: extended Hamming(8, 4) - SEC-DED
------------------------------------------------
* **Binary-native.** The payload pipeline is bit-oriented (``list[int]`` of
  0/1). A binary linear block code plugs in directly; a byte-symbol code such
  as Reed-Solomon would need an artificial bits<->symbols repacking layer for
  no benefit at these payload sizes.
* **Established and fully explainable.** Hamming codes are the canonical
  single-error-correcting linear block code. The *extended* form appends one
  overall-parity bit to the classic (7, 4) code, upgrading it to
  Single-Error-Correction + Double-Error-Detection (SEC-DED). Encoding is a
  GF(2) matrix product; decoding is a 3-bit syndrome lookup plus one parity
  check - no iterative decoder, no tables beyond a 7-entry column map.
* **Lightweight.** Pure NumPy, ~1 small generator matrix, zero new
  dependencies (the project pins its dependencies tightly and ships no ECC
  library).
* **Deterministic.** Linear algebra over GF(2); no randomness anywhere.
* **Real decode-failure signal.** The DED capability lets the decoder *flag* a
  block with two detected errors as uncorrectable instead of silently
  mis-correcting it - this is what makes a "decoding success/failure rate"
  meaningful (Phase 16 requirement).
* **Capacity fits.** Rate 1/2: raw payloads of 16 / 32 / 64 bits encode to
  32 / 64 / 128 bits, all within the frozen LL-subband's 128-bit native
  capacity on a 256x256 image, so every experiment arm runs on the frozen
  Phase 6 LL-only baseline with no subband changes.

The classic ``hamming_7_4`` (SEC only, rate 4/7) and a pass-through ``none``
codec are also exposed so the ECC method is genuinely configurable and the
"no ECC" arm shares one code path with the ECC arm.

Capacity rule (Phase 16 requirement)
------------------------------------
:func:`ECCCodec.encoded_length` and :func:`check_capacity` let a caller
compute the encoded length up front and refuse to embed when it exceeds the
available singular-value slots - with the exact numbers in the error message.
Encoded data is never truncated and message bits are never silently dropped;
a partial final data block is zero-padded (the pad count is tracked and the
pad positions are removed on decode, never counted as message).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SUPPORTED_METHODS",
    "BlockDecodeStats",
    "DecodeResult",
    "ECCCapacityError",
    "ECCCodec",
    "ECCConfig",
    "ECCPlan",
    "build_codec",
    "check_capacity",
]

SUPPORTED_METHODS: tuple[str, ...] = ("none", "hamming_7_4", "hamming_8_4")


# ---------------------------------------------------------------------------
# GF(2) Hamming(7, 4) core - systematic [ I4 | P ]
# ---------------------------------------------------------------------------

# Parity sub-matrix P (4 x 3): p = d @ P (mod 2)
#   p0 = d0 ^ d1 ^ d3
#   p1 = d0 ^ d2 ^ d3
#   p2 = d1 ^ d2 ^ d3
_P74 = np.array(
    [
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.uint8,
)

# Generator G (4 x 7): codeword = [d0 d1 d2 d3 p0 p1 p2]
_G74 = np.concatenate([np.eye(4, dtype=np.uint8), _P74], axis=1)

# Parity-check H (3 x 7): [ P^T | I3 ];  syndrome = H @ recv^T (mod 2)
_H74 = np.concatenate([_P74.T, np.eye(3, dtype=np.uint8)], axis=1)

# syndrome (as a 3-tuple) -> error position in the 7-bit codeword
_SYNDROME_TO_POS: dict[tuple[int, int, int], int] = {
    tuple(int(v) for v in _H74[:, j]): j for j in range(7)
}

# Extended generator G (4 x 8): appends the overall-parity bit (XOR of all 7).
_G84 = np.concatenate([_G74, (_G74.sum(axis=1, keepdims=True) % 2).astype(np.uint8)], axis=1)


# ---------------------------------------------------------------------------
# Configuration / plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ECCConfig:
    """Which ECC method to apply. ``method='none'`` is a pass-through."""

    method: str = "hamming_8_4"

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"unsupported ECC method {self.method!r}; allowed: {list(SUPPORTED_METHODS)}"
            )

    @property
    def data_bits_per_block(self) -> int:
        return 1 if self.method == "none" else 4

    @property
    def code_bits_per_block(self) -> int:
        return {"none": 1, "hamming_7_4": 7, "hamming_8_4": 8}[self.method]

    @property
    def can_detect_uncorrectable(self) -> bool:
        """Only the extended (8, 4) code has double-error *detection*."""
        return self.method == "hamming_8_4"

    @property
    def code_rate(self) -> float:
        return self.data_bits_per_block / self.code_bits_per_block


@dataclass(frozen=True)
class ECCPlan:
    """The exact block layout for encoding ``n_raw_bits`` with a method."""

    method: str
    n_raw_bits: int
    data_bits_per_block: int
    code_bits_per_block: int
    n_blocks: int
    n_pad_bits: int
    n_encoded_bits: int

    @property
    def redundancy_overhead(self) -> float:
        """``(encoded - raw) / raw`` - fraction of extra bits ECC adds."""
        if self.n_raw_bits == 0:
            return 0.0
        return (self.n_encoded_bits - self.n_raw_bits) / self.n_raw_bits

    @property
    def code_rate(self) -> float:
        if self.n_encoded_bits == 0:
            return 0.0
        return self.n_raw_bits / self.n_encoded_bits


@dataclass(frozen=True)
class BlockDecodeStats:
    """Per-payload accounting of what the block decoder did."""

    n_blocks: int
    corrected_blocks: int
    detected_uncorrectable_blocks: int


@dataclass(frozen=True)
class DecodeResult:
    """Result of decoding one recovered encoded payload."""

    bits: list[int]
    stats: BlockDecodeStats

    @property
    def success(self) -> bool:
        """True when no block was flagged as a detected-uncorrectable error.

        For ``hamming_7_4`` and ``none`` there is no detection capability, so
        this is always True - the honest statement is "the decoder produced an
        output", not "the output is guaranteed correct".
        """
        return self.stats.detected_uncorrectable_blocks == 0


# ---------------------------------------------------------------------------
# Capacity guard
# ---------------------------------------------------------------------------

class ECCCapacityError(ValueError):
    """Raised when an ECC-encoded payload would not fit the embedding capacity."""


def check_capacity(n_encoded_bits: int, capacity_bits: int, *, context: str = "") -> None:
    """Refuse to proceed when the encoded payload exceeds the available slots.

    Never truncates and never drops bits - it raises with the exact numbers so
    the caller can pick a smaller raw payload instead.
    """
    if n_encoded_bits > capacity_bits:
        where = f" ({context})" if context else ""
        raise ECCCapacityError(
            f"ECC-encoded payload needs {n_encoded_bits} embedded bits but only "
            f"{capacity_bits} are available{where}. Reduce the raw payload size so "
            f"the encoded payload fits; encoded data must not be truncated."
        )


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------

def _as_bit_array(bits: Sequence[int], *, name: str) -> np.ndarray:
    arr = np.asarray(list(bits), dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D bit sequence")
    if arr.size and not np.isin(arr, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return arr.astype(np.uint8)


class ECCCodec:
    """Encode/decode a bit-list payload with the configured Hamming variant."""

    def __init__(self, config: ECCConfig) -> None:
        self.config = config

    # -- planning / capacity ------------------------------------------------

    def plan(self, n_raw_bits: int) -> ECCPlan:
        if n_raw_bits <= 0:
            raise ValueError(f"n_raw_bits must be positive; got {n_raw_bits}")
        k = self.config.data_bits_per_block
        c = self.config.code_bits_per_block
        n_blocks = (n_raw_bits + k - 1) // k
        n_pad = n_blocks * k - n_raw_bits
        return ECCPlan(
            method=self.config.method,
            n_raw_bits=n_raw_bits,
            data_bits_per_block=k,
            code_bits_per_block=c,
            n_blocks=n_blocks,
            n_pad_bits=n_pad,
            n_encoded_bits=n_blocks * c,
        )

    def encoded_length(self, n_raw_bits: int) -> int:
        return self.plan(n_raw_bits).n_encoded_bits

    def redundancy_overhead(self, n_raw_bits: int) -> float:
        return self.plan(n_raw_bits).redundancy_overhead

    # -- encode -----------------------------------------------------------

    def encode(self, bits: Sequence[int]) -> list[int]:
        """Turn ``len(bits)`` raw payload bits into ``encoded_length`` bits."""
        raw = _as_bit_array(bits, name="payload bits")
        if raw.size == 0:
            raise ValueError("payload bits must be non-empty")
        if self.config.method == "none":
            return raw.astype(int).tolist()

        plan = self.plan(raw.size)
        padded = np.zeros(plan.n_blocks * plan.data_bits_per_block, dtype=np.uint8)
        padded[: raw.size] = raw
        blocks = padded.reshape(plan.n_blocks, plan.data_bits_per_block)

        generator = _G84 if self.config.method == "hamming_8_4" else _G74
        code = (blocks @ generator) % 2
        return code.reshape(-1).astype(int).tolist()

    # -- decode -----------------------------------------------------------

    def decode(self, bits: Sequence[int], n_raw_bits: int) -> DecodeResult:
        """Recover ``n_raw_bits`` payload bits from a recovered encoded payload.

        ``bits`` must have exactly the encoded length for ``n_raw_bits`` - a
        wrong length is a hard error (this is what catches an upstream
        truncation), never a silent slice.
        """
        recv = _as_bit_array(bits, name="recovered bits")
        plan = self.plan(n_raw_bits)
        if recv.size != plan.n_encoded_bits:
            raise ValueError(
                f"expected {plan.n_encoded_bits} encoded bits for a {n_raw_bits}-bit "
                f"payload ({self.config.method}); got {recv.size}"
            )

        if self.config.method == "none":
            return DecodeResult(
                bits=recv[:n_raw_bits].astype(int).tolist(),
                stats=BlockDecodeStats(plan.n_blocks, 0, 0),
            )

        blocks = recv.reshape(plan.n_blocks, plan.code_bits_per_block)
        data = np.empty((plan.n_blocks, 4), dtype=np.uint8)
        corrected = 0
        detected = 0

        for i, block in enumerate(blocks):
            code7 = block[:7].copy()
            syndrome = tuple(int(v) for v in (_H74 @ code7) % 2)
            syndrome_nonzero = any(syndrome)

            if self.config.method == "hamming_8_4":
                overall_parity = int(block.sum() % 2)
                if overall_parity == 1:
                    # odd overall parity => exactly one bit error somewhere
                    if syndrome_nonzero:
                        pos = _SYNDROME_TO_POS[syndrome]
                        code7[pos] ^= 1
                    # syndrome == 0 here => the error is the overall-parity bit;
                    # the 7 data/parity bits are intact.
                    corrected += 1
                elif syndrome_nonzero:
                    # even overall parity + non-zero syndrome => >=2 errors:
                    # detectable but NOT correctable. Leave the block untouched.
                    detected += 1
            else:  # hamming_7_4 - SEC only, no detection
                if syndrome_nonzero:
                    pos = _SYNDROME_TO_POS[syndrome]
                    code7[pos] ^= 1
                    corrected += 1

            data[i] = code7[:4]

        flat = data.reshape(-1)[:n_raw_bits].astype(int).tolist()
        return DecodeResult(
            bits=flat,
            stats=BlockDecodeStats(plan.n_blocks, corrected, detected),
        )


def build_codec(method: str) -> ECCCodec:
    """Convenience constructor from a method name."""
    return ECCCodec(ECCConfig(method=method))
