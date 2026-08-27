"""Reproducible binary payload generation for watermarking experiments.

This module deliberately produces *raw payload bits* only. Error correction is
introduced and evaluated separately in Phase 15 so its overhead is measurable.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable


SUPPORTED_BIT_LENGTHS = frozenset({8, 16, 32, 64, 128, 256, 512, 1024})


def validate_bit_length(bit_length: int) -> None:
    """Reject payload sizes outside the preregistered experimental sweep."""
    if bit_length not in SUPPORTED_BIT_LENGTHS:
        allowed = ", ".join(map(str, sorted(SUPPORTED_BIT_LENGTHS)))
        raise ValueError(f"Unsupported watermark length {bit_length}; allowed: {allowed}")


def _digest_stream(seed: bytes, bit_length: int) -> bytes:
    """Expand a seed deterministically with counter-prefixed SHA-256 digests."""
    required_bytes = (bit_length + 7) // 8
    output = bytearray()
    counter = 0
    while len(output) < required_bytes:
        output.extend(hashlib.sha256(counter.to_bytes(4, "big") + seed).digest())
        counter += 1
    return bytes(output[:required_bytes])


def bytes_to_bits(payload: bytes, bit_length: int | None = None) -> list[int]:
    """Convert bytes to most-significant-bit-first binary values."""
    bits = [int(bit) for byte in payload for bit in f"{byte:08b}"]
    return bits if bit_length is None else bits[:bit_length]


def bits_to_bytes(bits: Iterable[int]) -> bytes:
    """Pack binary values into bytes; the final byte is zero-padded on the right."""
    values = list(bits)
    if any(value not in (0, 1) for value in values):
        raise ValueError("Bits must contain only 0 or 1")
    padding = (-len(values)) % 8
    padded = values + [0] * padding
    return bytes(int("".join(map(str, padded[index : index + 8])), 2) for index in range(0, len(padded), 8))


def generate_from_uuid(identifier: str | uuid.UUID, bit_length: int) -> list[int]:
    """Generate a fixed-length binary payload from a canonical UUID and SHA-256."""
    validate_bit_length(bit_length)
    canonical = str(uuid.UUID(str(identifier))).encode("ascii")
    return bytes_to_bits(_digest_stream(canonical, bit_length), bit_length)


def generate_random(bit_length: int, identifier: uuid.UUID | None = None) -> tuple[str, list[int]]:
    """Generate a UUID4 identifier and its reproducible binary payload."""
    identifier = identifier or uuid.uuid4()
    return str(identifier), generate_from_uuid(identifier, bit_length)


def text_to_bits(text: str) -> list[int]:
    """Encode non-empty human-readable text as UTF-8 bits without ECC."""
    if not isinstance(text, str) or not text:
        raise ValueError("Text watermark must be a non-empty string")
    return bytes_to_bits(text.encode("utf-8"))


def generate_from_text(text: str, bit_length: int) -> list[int]:
    """Derive a fixed raw binary payload from UTF-8 text using SHA-256 expansion."""
    validate_bit_length(bit_length)
    encoded = text.encode("utf-8")
    if not encoded:
        raise ValueError("Text watermark must be a non-empty string")
    return bytes_to_bits(_digest_stream(encoded, bit_length), bit_length)
