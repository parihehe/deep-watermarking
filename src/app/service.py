"""Adapter between HTTP payloads and the frozen Phase 6 watermarking pipeline.

Responsibilities kept here (not in the API layer, not in the research code):
  * decode uploaded image bytes -> RGB uint8 ndarray
  * build a watermark payload from a chosen source
    (message / random / text / uuid / bits)
  * call the existing ``embed`` / ``extract_traditional`` / ``compute_residual``
  * call the existing metric helpers
  * encode result images back to base64 PNG data URIs for the browser

No watermarking mathematics lives in this file. The ``message`` source is a
thin, fully reversible UTF-8 <-> bits codec built on the existing
``watermark_generator`` byte helpers so the UI can show the recovered watermark
back to the user as plain text; the embedding rule itself is untouched.
"""

from __future__ import annotations

import base64
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from src.evaluation.metrics import quality_report, recovery_report
from src.watermark.dwt import SUPPORTED_WAVELETS
from src.watermark.embed import (
    BASELINE_ALPHAS,
    SUPPORTED_SUBBANDS,
    EmbedConfig,
    compute_residual,
    embed,
    extract_traditional,
    subband_capacity,
)
from src.watermark.watermark_generator import (
    SUPPORTED_BIT_LENGTHS,
    bits_to_bytes,
    bytes_to_bits,
    generate_from_text,
    generate_from_uuid,
    generate_random,
)

MAX_IMAGE_PIXELS = 2048 * 2048
PAYLOAD_SOURCES = ("message", "random", "text", "uuid", "bits")

# ``message`` payload layout: a 1-byte length header followed by the raw UTF-8
# bytes of the watermark text, then repeated as many (odd) times as the LL
# sub-band can hold. Recovery majority-votes the repeats before decoding.
#
# Repetition is the only robustness aid used here; it costs nothing in the
# frozen pipeline (it is just more payload bits) and it does not touch the
# embedding rule. It matters because the frozen non-blind decoder is only
# reliable on the well-separated *leading* singular values of LL: the trailing
# ones are close together and a fraction flip on the uint8 round trip (this is
# documented in ``src/watermark/embed.py``). Longer text therefore reaches into
# the noisy region and may come back with a few character errors — the honest
# ``status`` / ``char_accuracy`` fields report exactly how well it did, and this
# limitation is precisely what the Phase 8 CNN decoder is meant to remove.
MESSAGE_HEADER_BYTES = 1
MESSAGE_MAX_REPETITION = 9


class ServiceError(ValueError):
    """Raised for user-correctable problems; the API maps this to HTTP 400."""


@dataclass(frozen=True)
class EmbedRequest:
    alpha: float
    bit_length: int
    wavelet: str
    subband: str
    payload_source: str
    payload_text: str | None = None
    payload_uuid: str | None = None
    payload_bits: str | None = None
    extra_subbands: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Image <-> bytes
# ---------------------------------------------------------------------------

def decode_image(data: bytes) -> np.ndarray:
    """Decode arbitrary image bytes to an RGB uint8 array."""
    if not data:
        raise ServiceError("empty upload")
    array = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ServiceError("could not decode image (supported: PNG, JPEG, BMP, TIFF, WEBP)")
    h, w = bgr.shape[:2]
    if h * w > MAX_IMAGE_PIXELS:
        raise ServiceError(f"image too large: {w}x{h} (limit {MAX_IMAGE_PIXELS} pixels)")
    if min(h, w) < 16:
        raise ServiceError(f"image too small: {w}x{h}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_png_data_uri(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ServiceError("failed to encode result image")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

def build_payload(req: EmbedRequest) -> tuple[list[int], str]:
    """Return (bits, human-readable description of the payload source)."""
    n = req.bit_length
    if req.payload_source == "random":
        identifier, bits = generate_random(n)
        return bits, f"random UUID {identifier}"
    if req.payload_source == "uuid":
        raw = (req.payload_uuid or "").strip()
        if not raw:
            raise ServiceError("payload_uuid is required when payload_source='uuid'")
        try:
            canonical = str(uuid.UUID(raw))
        except ValueError as exc:
            raise ServiceError(f"invalid UUID: {raw!r}") from exc
        return generate_from_uuid(canonical, n), f"UUID {canonical}"
    if req.payload_source == "text":
        text = req.payload_text or ""
        if not text:
            raise ServiceError("payload_text is required when payload_source='text'")
        return generate_from_text(text, n), f"SHA-256 of text ({len(text)} chars)"
    if req.payload_source == "bits":
        raw = (req.payload_bits or "").strip().replace(" ", "")
        if not raw or any(c not in "01" for c in raw):
            raise ServiceError("payload_bits must be a non-empty string of 0/1")
        if len(raw) > n:
            raise ServiceError(f"payload_bits has {len(raw)} bits, more than bit_length={n}")
        bits = [int(c) for c in raw] + [0] * (n - len(raw))
        suffix = "" if len(raw) == n else f" (zero-padded from {len(raw)} to {n})"
        return bits, f"explicit bits{suffix}"
    raise ServiceError(f"unknown payload_source {req.payload_source!r}; allowed: {list(PAYLOAD_SOURCES)}")


# ---------------------------------------------------------------------------
# Reversible "message" codec (UTF-8 text <-> payload bits)
# ---------------------------------------------------------------------------

def message_base_bits(text: str) -> list[int]:
    """Encode watermark text as ``[1-byte length header] + UTF-8 body`` bits."""
    body = text.encode("utf-8")
    if len(body) > 0xFF:
        raise ServiceError("Please keep the watermark text under 255 bytes.")
    return bytes_to_bits(bytes([len(body)]) + body)


def choose_repetition(base_len: int, capacity_bits: int) -> int:
    """Largest odd repeat count (<= 9) whose repeated payload fits ``capacity_bits``."""
    for reps in range(min(MESSAGE_MAX_REPETITION, 9), 0, -2):
        if reps * base_len <= capacity_bits:
            return reps
    return 0


def decode_message(recovered_bits: Sequence[int], base_len: int, reps: int) -> str:
    """Majority-vote the repeated copies, then decode the length-prefixed body."""
    flat = list(recovered_bits)[: base_len * reps]
    grid = np.asarray(flat, dtype=int).reshape(reps, base_len)
    voted = (grid.mean(axis=0) >= 0.5).astype(int).tolist()
    data = bits_to_bytes(voted)
    if not data:
        return ""
    length = data[0]
    return data[1 : 1 + length].decode("utf-8", errors="replace")


def character_accuracy(original: str, recovered: str) -> float:
    """Fraction of character positions that agree (length differences count as wrong)."""
    span = max(len(original), len(recovered))
    if span == 0:
        return 1.0
    hits = sum(1 for i in range(span) if i < len(original) and i < len(recovered) and original[i] == recovered[i])
    return hits / span


# ---------------------------------------------------------------------------
# Main operation
# ---------------------------------------------------------------------------

def run_embed(image_bytes: bytes, req: EmbedRequest) -> dict:
    """Embed a watermark and return everything the UI needs, as a JSON-able dict."""
    if req.payload_source not in PAYLOAD_SOURCES:
        raise ServiceError(f"unknown payload_source {req.payload_source!r}")
    if req.wavelet not in SUPPORTED_WAVELETS:
        raise ServiceError(f"unsupported wavelet {req.wavelet!r}; allowed: {sorted(SUPPORTED_WAVELETS)}")

    image = decode_image(image_bytes)

    message_text: str | None = None
    message_base_len = 0
    message_reps = 0
    if req.payload_source == "message":
        message_text = (req.payload_text or "").strip()
        if not message_text:
            raise ServiceError("Please enter some watermark text.")
        base_bits = message_base_bits(message_text)
        message_base_len = len(base_bits)
        h, w = image.shape[:2]
        ll_capacity = min((h + 1) // 2, (w + 1) // 2)
        message_reps = choose_repetition(message_base_len, ll_capacity)
        if message_reps == 0:
            max_chars = max(1, ll_capacity // 8 - MESSAGE_HEADER_BYTES)
            raise ServiceError(
                "Your watermark text is too long for this image. "
                f"This image can hold about {max_chars} characters — "
                "use a shorter watermark or upload a larger image."
            )
        bits = (base_bits * message_reps)[: message_base_len * message_reps]
        config = EmbedConfig(
            wavelet=req.wavelet, subband="LL", alpha=req.alpha, bit_length=len(bits)
        )
        payload_desc = (
            f'watermark text ("{message_text}", {len(message_text)} chars, '
            f"{message_reps}x repetition in LL)"
        )
    else:
        try:
            config = EmbedConfig(
                wavelet=req.wavelet,
                subband=req.subband,
                extra_subbands=tuple(req.extra_subbands),
                alpha=req.alpha,
                bit_length=req.bit_length,
            )
        except ValueError as exc:
            raise ServiceError(str(exc)) from exc

        capacity = subband_capacity(image.shape[:2], config)
        if req.bit_length > capacity:
            raise ServiceError(
                f"payload of {req.bit_length} bits does not fit this image: the "
                f"{'+'.join(config.subband_order)} subband(s) of a "
                f"{image.shape[1]}x{image.shape[0]} image provide {capacity} slots. "
                f"Use a larger image, a smaller payload, or add overflow subbands."
            )

        bits, payload_desc = build_payload(req)

    capacity = subband_capacity(image.shape[:2], config)

    try:
        result = embed(image, bits, config)
    except ValueError as exc:
        raise ServiceError(str(exc)) from exc

    watermarked = result.watermarked_image
    residual = compute_residual(image, watermarked)
    recovered = extract_traditional(watermarked, image, config)

    quality = quality_report(image, watermarked)
    recovery = recovery_report(bits, recovered)

    if message_text is not None:
        recovered_text = decode_message(recovered, message_base_len, message_reps)
        char_acc = character_accuracy(message_text, recovered_text)
        if recovered_text == message_text:
            status = "recovered"
        elif char_acc >= 0.5:
            status = "partial"
        else:
            status = "failed"
    else:
        recovered_text = None
        char_acc = None
        if recovery["ber"] == 0.0:
            status = "recovered"
        elif recovery["bit_accuracy"] >= 0.75:
            status = "partial"
        else:
            status = "failed"

    headline = {
        "recovered": "Watermark successfully recovered",
        "partial": "Watermark partially recovered",
        "failed": "Watermark recovery failed",
    }[status]

    return {
        "config": {
            "alpha": config.alpha,
            "bit_length": config.bit_length,
            "wavelet": config.wavelet,
            "subband_order": list(config.subband_order),
            "start_sv_index": config.start_sv_index,
            "mode": config.mode,
        },
        "payload": {
            "source": req.payload_source,
            "description": payload_desc,
            "text": message_text,
            "bits": bits,
            "bit_string": "".join(map(str, bits)),
        },
        "image_info": {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "subband_capacity_bits": int(capacity),
        },
        "quality_metrics": {
            "psnr_db": _round(quality["psnr"]),
            "ssim": _round(quality["ssim"], 6),
            "mse": _round(quality["mse"], 6),
        },
        "recovery_metrics": {
            "note": "non-blind reference decoder (compares against the original image)",
            "ber": _round(recovery["ber"], 6),
            "bit_accuracy": _round(recovery["bit_accuracy"], 6),
            "nc": _round(recovery["nc"], 6),
            "recovered_bit_string": "".join(map(str, recovered)),
            "recovered_text": recovered_text,
            "text_match": (None if message_text is None else bool(recovered_text == message_text)),
            "char_accuracy": (None if char_acc is None else _round(char_acc, 4)),
        },
        "summary": {
            "mode": req.payload_source,
            "status": status,
            "watermark_recovered": status == "recovered",
            "headline": headline,
            "original_text": message_text,
            "recovered_text": recovered_text,
            "char_accuracy": (None if char_acc is None else _round(char_acc, 4)),
            "repetition": (message_reps if message_text is not None else None),
        },
        "images": {
            "original": encode_png_data_uri(image),
            "watermarked": encode_png_data_uri(watermarked),
            "residual_x15": encode_png_data_uri(residual),
        },
    }


def options() -> dict:
    """Static choices the UI needs to populate its controls."""
    return {
        "wavelets": sorted(SUPPORTED_WAVELETS),
        "subbands": list(SUPPORTED_SUBBANDS),
        "bit_lengths": sorted(SUPPORTED_BIT_LENGTHS),
        "baseline_alphas": list(BASELINE_ALPHAS),
        "payload_sources": list(PAYLOAD_SOURCES),
        "alpha_range": {"min": 0.001, "max": 0.1, "step": 0.001, "default": 0.010},
    }


def _round(value: float, ndigits: int = 4) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), ndigits)
