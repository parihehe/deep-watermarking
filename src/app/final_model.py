"""Phase 17B - the Phase 17 final integrated model, wired into the Phase 7 web app.

This is an **additive** layer. It imports:

* the Phase 7 service helpers (``src.app.service``) for image <-> bytes,
  base64 PNG encoding and the reversible "message" text codec;
* the frozen Phase 6 ``embed`` / ``extract_traditional`` / ``compute_residual``;
* the Phase 8 blind CNN (``src.evaluation.blind_extract.BlindExtractor``);
* the metric helpers (``src.evaluation.metrics``).

Nothing in Phase 6, Phase 7 or Phase 8 is modified. No watermarking
mathematics lives in this file - it only selects the frozen Phase 17
configuration, marshals inputs, and calls the existing functions.

Final model (from ``docs/phase17_final.md``)
-------------------------------------------
Haar wavelet, single-level **LL** subband, multiplicative embedding at
**fixed alpha = 0.020**, **64-bit** payload, **no ECC**, no adaptive
embedding.

* **Non-blind extraction** - the frozen reference decoder. **Requires the
  original image.**
* **Blind extraction** - the Phase 8 CNN checkpoint. **Requires only the
  watermarked image** (no original, no key, no embedding metadata).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.app import service
from src.app.service import ServiceError
from src.evaluation.metrics import quality_report, recovery_report
from src.evaluation.phase17_final import CandidateSpec
from src.watermark import id_registry
from src.watermark.embed import (
    EmbedConfig,
    compute_residual,
    embed,
    extract_traditional,
    subband_capacity,
)
from src.watermark.watermark_generator import SUPPORTED_BIT_LENGTHS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- the frozen Phase 17 selection -----------------------------------------
# alpha=0.005 was trialled (non-blind BER 0.094->0.028, "hello" non-blind
# 3/40->62/100) but reverted: the blind CNN retrained at 0.005 got WORSE
# (64-bit bit-acc 0.778->0.640, "hello" blind exact stayed 0/40) - a weaker
# embed signal hurts the reference-free blind decoder more than it helps it,
# even though it helps the non-blind decoder a lot. See docs/blind_extraction_diagnosis.md.
FINAL_MODEL_SPEC = CandidateSpec(
    name="phase17_final",
    wavelet="haar",
    subband="LL",
    alpha=0.020,
    bit_length=64,
    start_sv_index=0,
    mode="symmetric",
    ecc_method="none",
    blind_extractor="phase8",
    role="candidate",
)
FINAL_ALPHA = FINAL_MODEL_SPEC.alpha
FINAL_WAVELET = FINAL_MODEL_SPEC.wavelet
FINAL_SUBBAND = FINAL_MODEL_SPEC.subband
FINAL_BIT_LENGTH = FINAL_MODEL_SPEC.bit_length
FINAL_MODE = FINAL_MODEL_SPEC.mode

# Payload sizes the app recognises at all - the pre-registered experimental
# sweep from ``src.watermark.watermark_generator`` (8 .. 1024). This is the
# single source of truth; the previous ``(16, 32, 64)`` tuple silently
# disagreed with the watermark generator and the UI's size dropdown.
SUPPORTED_PAYLOAD_BITS: tuple[int, ...] = tuple(sorted(SUPPORTED_BIT_LENGTHS))

# Each blind decoder is one Phase 8 ``BlindCNNExtractor`` whose payload width is
# baked into its weights (``ExtractorConfig.bit_length``) and emitted as exactly
# that many per-bit logits. There is one checkpoint per supported width.
BLIND_MODEL_DIR = PROJECT_ROOT / "models" / "phase8_cnn"

# Widths a single-level, LL-only extractor can serve: the LL sub-band of a
# 256x256 image has 128 singular values, so <= 128 bits fit in LL alone. 256 /
# 512 need extra sub-bands (LL+HL+...), which the Phase 8 feature extractor - LL
# singular values only - cannot read; 1024 exceeds a 256x256 image's capacity.
BLIND_TRAINABLE_SIZES: tuple[int, ...] = (8, 16, 32, 64, 128)
BLIND_MAX_BITS = max(BLIND_TRAINABLE_SIZES)

# Legacy default: the pre-existing fixed 64-bit Phase 8 checkpoint, used as a
# fallback for the 64-bit width until a per-size ``blind_cnn_64bit_best.pt``
# exists. Kept as-is; never modified.
BLIND_CHECKPOINT = BLIND_MODEL_DIR / "phase8_cnn_best.pt"

# Reference width for callers that do not name one (and for the info route).
BLIND_BIT_LENGTH = 64

# Confidence at/above which a blind text decode is reported to the user, for
# the LEGACY (non-registry) header+repetition text path - i.e. `confidence =
# mean(|CNN sigmoid prob - 0.5|) * 2`, the raw per-bit margin on a single
# embedded copy. Calibrated (docs/blind_extraction_diagnosis.md S6) back when
# blind accuracy was ~0% end to end, so this threshold's only measured job was
# "does it suppress known-garbage output" (it does - trivially, since nothing
# was ever correct). Left unchanged: this path still exists and nothing in
# this session touched it or re-measured it.
RELIABLE_TEXT_CONFIDENCE = 0.85

# Confidence at/above which a REGISTRY-ID decode (`payload_kind ==
# "registry_id"`, src/watermark/id_registry.py) is reported to the user. This
# is a DIFFERENT metric from RELIABLE_TEXT_CONFIDENCE above - mean majority-
# vote agreement across the 4 ID-bit slots, quantized to multiples of 1/15 -
# so RELIABLE_TEXT_CONFIDENCE's value was never actually calibrated against
# it (it predates the registry-ID path entirely). Applying 0.85 to this
# metric passed only 6% of real decodes (12/200,
# results/phase18_id_registry/per_trial.json) - it was silently discarding
# nearly all now-correct output, not doing its job.
#
# Recalibrated from experiments/calibrate_registry_confidence.py (480 real
# trials: all 16 IDs x 30 held-out DIV2K images, seed 42, real blind CNN
# checkpoint - see results/phase18_id_registry/confidence_calibration.json):
# 0.78 is the lowest achievable confidence value (47/60) at/above which ZERO
# of the 480 trials decoded wrong (180/480 = 37.5% of decodes clear it), vs.
# 91.0% coverage at only 96.8% precision if the bar were dropped to 0.70.
# This is an empirical "no observed errors yet" cutoff, not a mathematical
# guarantee - re-run the calibration script if the underlying codec changes.
#
# IMPORTANT residual-risk caveat (docs/phase18_id_registry.md S5): across all
# real data gathered for this calibration (960 trials total, all 16 IDs), 5
# wrong decodes occurred at confidence >= 0.78 (~0.5%). None were ever shown
# to a user only because none happened to decode to one of the 5 currently
# REGISTERED ids - MessageRegistry's sparsity (5/14 usable slots filled)
# caught them, not this threshold. That clean "0 wrong shown" record is a
# property of which IDs are populated today, not a guarantee of the codec.
# See the MESSAGE_REGISTRY comment below for the required check before
# adding more messages.
RELIABLE_REGISTRY_ID_CONFIDENCE = 0.78

_PAYLOAD_SOURCES = ("message", "text", "uuid", "bits", "random")

# ``message`` payloads no longer embed text directly (see
# ``src/watermark/id_registry.py`` for why: raw text needs far more
# redundancy than the blind CNN's fixed 64-bit output can hold). Instead the
# text is registered here and only its 4-bit ID (repeated 15x, 60/64 bits) is
# embedded; blind extraction majority-votes the ID back and looks the text up
# again. Pre-populated with the project's standing test vocabulary so
# existing recordings/screenshots keep the same IDs; new text registers into
# the remaining free slots (14 usable, 2 degenerate excluded) on first use.
#
# REQUIRED CHECK BEFORE ADDING MORE MESSAGES (docs/phase18_id_registry.md S5):
# RELIABLE_REGISTRY_ID_CONFIDENCE (0.78, above) was calibrated with only these
# 5 of 14 usable IDs populated. Real data already shows a ~0.5% rate of wrong
# decodes at confidence >= 0.78 (results/phase18_id_registry/confidence_calibration.json,
# results/phase18_id_registry/id15_large_sample_check.json) that has never
# been shown to a user only because none happened to land on one of these 5
# registered ids. Populating more slots raises the chance a future wrong
# decode lands on a NOW-registered id and gets shown confidently and
# incorrectly. Before registering any message beyond this initial 5, re-run
# ``experiments/calibrate_registry_confidence.py`` against the intended
# larger registry and re-verify the threshold still holds at acceptable
# precision - do not assume today's clean "0 wrong shown" record continues
# automatically.
MESSAGE_REGISTRY = id_registry.MessageRegistry(
    ["hello", "hi", "owner-2026", "Vestigia", "日本語"]
)


class FinalModelError(ServiceError):
    """User-correctable problem in a final-model request (HTTP 400)."""


class CheckpointError(FinalModelError):
    """The blind CNN checkpoint is missing or unreadable (HTTP 503)."""


def _decode(data: bytes) -> np.ndarray:
    """Decode uploaded bytes, re-raising Phase 7's ``ServiceError`` as a
    ``FinalModelError`` so the final-model routes map it to a clean HTTP 400."""
    try:
        return service.decode_image(data)
    except ServiceError as exc:
        raise FinalModelError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Blind CNN extractors - one per payload width, lazily loaded and cached
# ---------------------------------------------------------------------------

# Test hook: when set, this single extractor is returned for EVERY width
# (used by tests to inject a stub without touching the filesystem).
_EXTRACTOR = None  # type: ignore[var-annotated]

# Production cache: width -> loaded BlindExtractor.
_EXTRACTORS: dict[int, object] = {}


def _checkpoint_for(bit_length: int) -> Path:
    """Filesystem path of the blind checkpoint that serves ``bit_length``-bit
    payloads. Per-size ``blind_cnn_<n>bit_best.pt`` wins; the legacy fixed
    64-bit ``phase8_cnn_best.pt`` is the fallback for width 64 only."""
    per_size = BLIND_MODEL_DIR / f"blind_cnn_{bit_length}bit_best.pt"
    if per_size.is_file():
        return per_size
    if bit_length == BLIND_BIT_LENGTH and BLIND_CHECKPOINT.is_file():
        return BLIND_CHECKPOINT
    return per_size  # does not exist -> the loader raises a clear CheckpointError


def blind_decoder_sizes() -> list[int]:
    """Payload widths that currently have a usable blind checkpoint on disk."""
    return [n for n in BLIND_TRAINABLE_SIZES if _checkpoint_for(n).is_file()]


def blind_extractor(bit_length: int = BLIND_BIT_LENGTH):
    """Return a cached ``BlindExtractor`` whose model width equals
    ``bit_length``.

    Raises ``CheckpointError`` (HTTP 503) if there is no checkpoint for that
    width, it cannot be loaded, or the checkpoint does not *declare* the
    requested width - a mismatched checkpoint is refused, never used anyway.
    """
    if _EXTRACTOR is not None:  # test hook - one injected model for any width
        return _EXTRACTOR
    if bit_length in _EXTRACTORS:
        return _EXTRACTORS[bit_length]

    path = _checkpoint_for(bit_length)
    if not path.is_file():
        raise CheckpointError(
            f"no blind decoder checkpoint for {bit_length}-bit payloads "
            f"(expected {path.name} under {BLIND_MODEL_DIR}). Run "
            f"experiments/train_blind_size_models.py to create the per-size checkpoints."
        )
    try:
        from src.evaluation.blind_extract import BlindExtractor

        ext = BlindExtractor.from_checkpoint(str(path))
    except Exception as exc:
        raise CheckpointError(f"could not load blind checkpoint {path.name}: {exc!r}") from exc

    declared = int(getattr(ext.config, "bit_length", -1))
    if declared != bit_length:
        raise CheckpointError(
            f"blind checkpoint {path.name} declares bit_length={declared}, but a "
            f"{bit_length}-bit decoder was requested - refusing to use a mismatched "
            f"checkpoint."
        )
    _EXTRACTORS[bit_length] = ext
    return ext


def reset_extractor_cache() -> None:
    """Test hook - drop cached extractors so the loader path runs again."""
    global _EXTRACTOR
    _EXTRACTOR = None
    _EXTRACTORS.clear()


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FinalEmbedRequest:
    payload_source: str = "message"
    payload_text: str | None = None
    payload_uuid: str | None = None
    payload_bits: str | None = None
    bit_length: int = FINAL_BIT_LENGTH  # ignored for payload_source == "message"


@dataclass(frozen=True)
class ExtractRequest:
    expected_text: str | None = None
    expected_bits: str | None = None
    bit_length: int = FINAL_BIT_LENGTH  # non-blind only
    # Blind only: the true width of the embedded payload and, for a text
    # message, the repetition count chosen at embed time. Supplied by the
    # caller (the page echoes the embed response's ``config.bit_length`` /
    # ``summary.repetition``) so the blind decoder inverts the *exact* payload
    # format that was embedded instead of guessing. ``None`` -> assume a
    # single-copy payload at the decoder's native width.
    payload_bit_length: int | None = None
    repetition: int | None = None
    # "registry_id" selects the registry-ID majority-vote decode path (see
    # id_registry.py); any other value (or None) uses the legacy
    # header+repetition text codec. Echoed back from the embed response's
    # payload.kind, same pattern as payload_bit_length / repetition.
    payload_kind: str | None = None


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------

def final_model_info() -> dict:
    return {
        "model": "phase17_final",
        "reference": "docs/phase17_final.md",
        "embedding": {
            "wavelet": FINAL_WAVELET,
            "subband": FINAL_SUBBAND,
            "mode": FINAL_MODE,
            "alpha": FINAL_ALPHA,
            "bit_length": FINAL_BIT_LENGTH,
            "ecc": "none",
            "adaptive": False,
        },
        "payload_sources": list(_PAYLOAD_SOURCES),
        "supported_payload_bits": list(SUPPORTED_PAYLOAD_BITS),
        "extraction": {
            "non_blind": {
                "decoder": "frozen DWT-SVD reference decoder",
                "requires_original_image": True,
            },
            "blind": {
                "decoder": "per-size Phase 8 blind CNN",
                "checkpoint": str(BLIND_CHECKPOINT.relative_to(PROJECT_ROOT)),
                "checkpoint_available": BLIND_CHECKPOINT.is_file(),
                "requires_original_image": False,
                "bit_length": BLIND_BIT_LENGTH,
                "trainable_sizes": list(BLIND_TRAINABLE_SIZES),
                "available_sizes": blind_decoder_sizes(),
                "unsupported_sizes": [s for s in SUPPORTED_PAYLOAD_BITS if s > BLIND_MAX_BITS],
            },
        },
    }


# ---------------------------------------------------------------------------
# Embed with the final model
# ---------------------------------------------------------------------------

def _final_config(bit_length: int) -> EmbedConfig:
    return EmbedConfig(
        wavelet=FINAL_WAVELET,
        subband=FINAL_SUBBAND,
        mode=FINAL_MODE,
        alpha=FINAL_ALPHA,
        bit_length=bit_length,
        start_sv_index=FINAL_MODEL_SPEC.start_sv_index,
    )


def run_final_embed(image_bytes: bytes, req: FinalEmbedRequest) -> dict:
    if req.payload_source not in _PAYLOAD_SOURCES:
        raise FinalModelError(
            f"unknown payload_source {req.payload_source!r}; allowed: {list(_PAYLOAD_SOURCES)}"
        )

    image = _decode(image_bytes)
    h, w = image.shape[:2]

    message_text: str | None = None
    message_id: int | None = None

    if req.payload_source == "message":
        message_text = (req.payload_text or "").strip()
        if not message_text:
            raise FinalModelError("Please enter some watermark text.")
        try:
            message_id = MESSAGE_REGISTRY.register(message_text)
        except id_registry.RegistryError as exc:
            raise FinalModelError(str(exc)) from exc
        bits = id_registry.encode_id_bits(message_id)
        ll_capacity = min((h + 1) // 2, (w + 1) // 2)
        if len(bits) > ll_capacity:
            raise FinalModelError(
                f"This image is too small to hold the watermark ID payload "
                f"({len(bits)} bits needed, this image's LL sub-band holds "
                f"{ll_capacity}). Use a larger image."
            )
        config = _final_config(len(bits))
        payload_desc = (
            f'watermark text ("{message_text}", registry ID {message_id}, '
            f"{id_registry.REPETITION}x repetition over {id_registry.ID_BITS} bits)"
        )
    else:
        if req.bit_length not in SUPPORTED_PAYLOAD_BITS:
            raise FinalModelError(
                f"unsupported payload size {req.bit_length}; the final model supports "
                f"{list(SUPPORTED_PAYLOAD_BITS)} bits (blind CNN extraction requires "
                f"{BLIND_BIT_LENGTH})."
            )
        config = _final_config(req.bit_length)
        capacity = subband_capacity(image.shape[:2], config)
        if req.bit_length > capacity:
            raise FinalModelError(
                f"payload of {req.bit_length} bits does not fit this image: the "
                f"{FINAL_SUBBAND} subband of a {w}x{h} image provides {capacity} slots. "
                "Use a larger image or a smaller payload."
            )
        embed_req = service.EmbedRequest(
            alpha=FINAL_ALPHA,
            bit_length=req.bit_length,
            wavelet=FINAL_WAVELET,
            subband=FINAL_SUBBAND,
            payload_source=req.payload_source,
            payload_text=req.payload_text,
            payload_uuid=req.payload_uuid,
            payload_bits=req.payload_bits,
        )
        try:
            bits, payload_desc = service.build_payload(embed_req)
        except ServiceError as exc:
            raise FinalModelError(str(exc)) from exc

    capacity = subband_capacity(image.shape[:2], config)

    try:
        result = embed(image, bits, config)
    except ValueError as exc:
        raise FinalModelError(str(exc)) from exc

    watermarked = result.watermarked_image
    residual = compute_residual(image, watermarked)
    recovered = extract_traditional(watermarked, image, config)

    quality = quality_report(image, watermarked)
    recovery = recovery_report(bits, recovered)

    if message_text is not None:
        recovered_id, _, _ = id_registry.decode_id_bits(recovered)
        recovered_text = MESSAGE_REGISTRY.message_for(recovered_id) or ""
        char_acc = service.character_accuracy(message_text, recovered_text)
        status = ("recovered" if recovered_text == message_text
                  else "partial" if char_acc >= 0.5 else "failed")
    else:
        recovered_text = None
        char_acc = None
        status = ("recovered" if recovery["ber"] == 0.0
                  else "partial" if recovery["bit_accuracy"] >= 0.75 else "failed")

    watermarked_uri = service.encode_png_data_uri(watermarked)
    return {
        "model": "phase17_final",
        "config": {
            "alpha": config.alpha,
            "bit_length": config.bit_length,
            "wavelet": config.wavelet,
            "subband": FINAL_SUBBAND,
            "mode": config.mode,
            "ecc": "none",
        },
        "payload": {
            "source": req.payload_source,
            "description": payload_desc,
            "text": message_text,
            "bit_string": "".join(map(str, bits)),
            # the values the blind decoder needs to invert this exact payload
            # format (see /api/final-model/extract/blind)
            "bit_length": config.bit_length,
            "repetition": (id_registry.REPETITION if message_text is not None else 1),
            "kind": ("registry_id" if message_text is not None else "raw"),
            "registry_id": message_id,
        },
        "image_info": {
            "width": int(w),
            "height": int(h),
            "subband_capacity_bits": int(capacity),
        },
        "quality_metrics": {
            "psnr_db": _round(quality["psnr"]),
            "ssim": _round(quality["ssim"], 6),
            "mse": _round(quality["mse"], 6),
        },
        "recovery_metrics": {
            "decoder": "non_blind reference (compares against the original image)",
            "requires_original_image": True,
            "ber": _round(recovery["ber"], 6),
            "bit_accuracy": _round(recovery["bit_accuracy"], 6),
            "nc": _round(recovery["nc"], 6),
            "recovered_bit_string": "".join(map(str, recovered)),
            "recovered_text": recovered_text,
            "text_match": (None if message_text is None else bool(recovered_text == message_text)),
            "char_accuracy": (None if char_acc is None else _round(char_acc, 4)),
            "status": status,
        },
        "images": {
            "original": service.encode_png_data_uri(image),
            "watermarked": watermarked_uri,
            "residual_x15": service.encode_png_data_uri(residual),
        },
        "download": {
            "filename": "watermarked.png",
            "data_uri": watermarked_uri,
        },
        # a blind decode is *attemptable* whenever the payload fits a trained
        # per-size decoder (<= 128 bits); whether it comes back reliably is a
        # separate question answered by /api/final-model/extract/blind.
        "blind_extractable": config.bit_length <= BLIND_MAX_BITS,
    }


# ---------------------------------------------------------------------------
# Extraction - shared helpers
# ---------------------------------------------------------------------------

def _parse_expected_bits(raw: str, n: int) -> list[int]:
    cleaned = (raw or "").strip().replace(" ", "")
    if not cleaned or any(c not in "01" for c in cleaned):
        raise FinalModelError("expected_bits must be a non-empty string of 0/1")
    if len(cleaned) > n:
        raise FinalModelError(f"expected_bits has {len(cleaned)} bits, more than {n}")
    return [int(c) for c in cleaned] + [0] * (n - len(cleaned))


def _decode_payload_message(
    bits: list[int], *, payload_bits: int | None = None, repetition: int | None = None
) -> str | None:
    """Decode ``bits`` as a watermark text message using the **exact same
    payload contract as embedding** (``service.message_base_bits`` /
    ``choose_repetition`` / ``decode_message_strict``): a ``[1-byte length]
    [UTF-8 body]`` block, repeated ``repetition`` times, majority-voted per
    position. Returns ``None`` if ``bits`` are not a well-formed message.

    ``payload_bits`` is the true embedded width (defaults to ``len(bits)``);
    ``repetition`` is the embed-time repeat count (defaults to 1). Any trailing
    padding bits past ``payload_bits`` are ignored, matching the embedder.
    """
    n = payload_bits if payload_bits is not None else len(bits)
    reps = repetition if repetition and repetition > 0 else 1
    if n <= 0 or n > len(bits) or n % reps != 0:
        return None
    base_len = n // reps
    return service.decode_message_strict(bits[:n], base_len, reps)


def _score(reference: list[int], recovered: list[int]) -> dict:
    rep = recovery_report(reference, recovered)
    return {
        "ber": _round(rep["ber"], 6),
        "bit_accuracy": _round(rep["bit_accuracy"], 6),
        "nc": _round(rep["nc"], 6),
    }


# ---------------------------------------------------------------------------
# Blind extraction - watermarked image ONLY
# ---------------------------------------------------------------------------

def _blind_unsupported(image: np.ndarray, requested_bits: int, reason: str) -> dict:
    """Structured 'this payload width has no blind decoder' response.

    Returned (HTTP 200) instead of running a differently-sized CNN on a
    payload it was not trained for. ``recovered`` is ``None`` so no bit string
    or text can be mistaken for a real result.
    """
    h, w = image.shape[:2]
    return {
        "extraction": "blind",
        "model": None,
        "checkpoint": None,
        "requires_original_image": False,
        "blind_supported": False,
        "requested_payload_bits": int(requested_bits),
        "available_blind_sizes": blind_decoder_sizes(),
        "reason": reason,
        "image_info": {"width": int(w), "height": int(h)},
        "recovered": None,
        "reference_scoring": None,
        "text_match": None,
        "char_accuracy": None,
        "decoding_status": reason,
    }


def run_blind_extract(watermarked_bytes: bytes, req: ExtractRequest) -> dict:
    """Recover the payload from a **watermarked image alone**.

    The original image is neither accepted nor used. The blind decoder is the
    per-size Phase 8 CNN whose declared width matches the payload:
    ``payload_bit_length`` selects the checkpoint (the smallest trained width
    that holds the payload), and the recovered bits go through the *identical*
    payload contract used at embed time (``service.message_base_bits`` /
    ``choose_repetition`` / ``decode_message_strict``), with ``repetition``
    supplied by the caller. Widths with no trained checkpoint are rejected
    explicitly (:func:`_blind_unsupported`) - a mismatched checkpoint is never
    used.
    """
    image = _decode(watermarked_bytes)

    requested = (
        int(req.payload_bit_length)
        if req.payload_bit_length is not None
        else BLIND_BIT_LENGTH
    )
    if requested <= 0:
        raise FinalModelError("payload_bit_length must be a positive integer")

    # A fixed-width payload (uuid / random / explicit bits) must be one of the
    # pre-registered sizes. A text message is length-prefixed and its width is
    # base_len x repetition (any multiple of 8), identified by a supplied
    # repetition and exempted from the sweep-size check.
    is_message = req.repetition is not None
    if not is_message and requested not in SUPPORTED_PAYLOAD_BITS:
        return _blind_unsupported(
            image, requested,
            f"{requested} is not a supported payload size; allowed: "
            f"{list(SUPPORTED_PAYLOAD_BITS)}.",
        )

    # Checkpoint selection: the smallest trained width that can hold the
    # payload. A sweep-size payload maps to its own exact checkpoint; a shorter
    # text message maps to the next size up and occupies that decoder's leading
    # positions (its output is sliced back to `requested` before decoding).
    candidates = [s for s in BLIND_TRAINABLE_SIZES if s >= requested]
    if not candidates:
        return _blind_unsupported(
            image, requested,
            f"no blind decoder is available for a {requested}-bit payload. Blind "
            f"decoders exist for {list(BLIND_TRAINABLE_SIZES)}-bit payloads (the "
            f"single-level LL sub-band holds 128 bits); recover wider payloads with "
            f"non-blind extraction, which needs the original image.",
        )
    model_width = min(candidates)

    reps = req.repetition if (req.repetition and req.repetition > 0) else 1
    # A registry-ID payload deliberately does not fill payload_bit_length evenly
    # (60 encoded bits + 4 unused padding bits within the 64-bit CNN output) -
    # only the legacy header+repetition text contract requires exact division.
    if req.payload_kind != "registry_id" and requested % reps != 0:
        raise FinalModelError(
            f"repetition {reps} does not evenly divide payload_bit_length {requested}"
        )

    extractor = blind_extractor(model_width)  # raises CheckpointError -> HTTP 503
    declared = int(getattr(extractor.config, "bit_length", model_width))

    probs_full = np.asarray(extractor.extract_proba(image), dtype=float)
    if probs_full.shape[0] < requested:
        raise CheckpointError(
            f"blind checkpoint for {model_width}-bit emitted {probs_full.shape[0]} "
            f"logits, fewer than the {requested}-bit payload needs."
        )
    # the payload occupies the leading `requested` positions (same index
    # convention as the embedder, which puts payload bit i into singular value i)
    probs = probs_full[:requested]
    bits = (probs > 0.5).astype(int).tolist()
    n = len(bits)
    is_registry_id = is_message and req.payload_kind == "registry_id"

    if is_registry_id:
        # confidence is the majority-vote agreement over the ID bits, not the
        # raw pre-vote CNN margin - it's the number that actually reflects how
        # decisive the repetition code's decode was.
        recovered_id, confidence, min_bit_confidence = id_registry.decode_id_bits(
            bits[: id_registry.ENCODED_BITS]
        )
    else:
        confidence = float(np.mean(np.abs(probs - 0.5)) * 2.0)
        min_bit_confidence = float(np.min(np.abs(probs - 0.5)) * 2.0)

    h, w = image.shape[:2]
    resized = (h, w) != (extractor.image_size, extractor.image_size)

    scored = None
    reliable = False
    if req.expected_bits is not None:
        reference = _parse_expected_bits(req.expected_bits, n)
        scored = _score(reference, bits)
        reliable = scored["bit_accuracy"] == 1.0

    if is_registry_id:
        decoded_text = MESSAGE_REGISTRY.message_for(recovered_id)
    else:
        decoded_text = _decode_payload_message(bits, payload_bits=n, repetition=reps)
    reliable_threshold = RELIABLE_REGISTRY_ID_CONFIDENCE if is_registry_id else RELIABLE_TEXT_CONFIDENCE
    if not reliable:
        reliable = decoded_text is not None and confidence >= reliable_threshold

    text_match = None
    char_accuracy = None
    if req.expected_text is not None:
        expected = req.expected_text.strip()
        got = decoded_text or ""
        text_match = bool(decoded_text is not None and got == expected)
        char_accuracy = _round(service.character_accuracy(expected, got), 4)

    if scored is not None:
        decoding_status = (
            f"scored against supplied reference bits: bit_accuracy={scored['bit_accuracy']}, "
            f"BER={scored['ber']}"
        )
    elif reliable:
        decoding_status = f"text recovered (confidence {round(confidence, 3)})"
    elif decoded_text is not None:
        decoding_status = (
            f"bits recovered; text decode not reliable at confidence {round(confidence, 3)} "
            f"(need >= {reliable_threshold})"
        )
    else:
        decoding_status = (
            f"bits recovered; not decodable as a watermark message "
            f"(confidence {round(confidence, 3)})"
        )

    ckpt_path = _checkpoint_for(model_width)
    return {
        "extraction": "blind",
        "model": f"blind_cnn_{model_width}bit",
        "checkpoint": ckpt_path.name,
        "checkpoint_bit_length": declared,      # the model's declared/validated width
        "selected_for_payload_bits": requested,
        "requires_original_image": False,
        "blind_supported": True,
        "decoder_native_bits": declared,
        "payload_format": {
            "bit_length": n,
            "repetition": reps,
            "base_len": n // reps,
            "contract": (
                f"{id_registry.ID_BITS}-bit registry ID x{id_registry.REPETITION}, "
                "majority-voted, looked up in MESSAGE_REGISTRY"
                if is_registry_id
                else "[1-byte length][UTF-8 body] x repetition, majority-voted"
            ),
        },
        "image_info": {"width": int(w), "height": int(h),
                       "resized_to_model_input": resized,
                       "model_input_size": int(extractor.image_size)},
        "recovered": {
            "bit_length": n,
            "bit_string": "".join(map(str, bits)),
            "confidence_mean": _round(confidence, 4),
            "confidence_min_bit": _round(min_bit_confidence, 4),
            "text": (decoded_text if reliable else None),
            "text_reliable": bool(reliable),
        },
        "reference_scoring": scored,          # None unless expected_bits supplied
        "text_match": text_match,             # None unless expected_text supplied
        "char_accuracy": char_accuracy,
        "decoding_status": decoding_status,
    }


# ---------------------------------------------------------------------------
# Non-blind extraction - watermarked + original both required
# ---------------------------------------------------------------------------

def run_nonblind_extract(
    watermarked_bytes: bytes, original_bytes: bytes, req: ExtractRequest
) -> dict:
    if not original_bytes:
        raise FinalModelError("non-blind extraction requires the original image.")
    watermarked = _decode(watermarked_bytes)
    original = _decode(original_bytes)
    if watermarked.shape != original.shape:
        raise FinalModelError(
            f"watermarked image {watermarked.shape[1]}x{watermarked.shape[0]} and original "
            f"{original.shape[1]}x{original.shape[0]} must have the same dimensions."
        )

    bit_length = req.bit_length or FINAL_BIT_LENGTH
    if bit_length not in SUPPORTED_PAYLOAD_BITS:
        raise FinalModelError(
            f"unsupported payload size {bit_length}; the final model supports "
            f"{list(SUPPORTED_PAYLOAD_BITS)} bits."
        )
    config = _final_config(bit_length)
    capacity = subband_capacity(original.shape[:2], config)
    if bit_length > capacity:
        raise FinalModelError(
            f"payload of {bit_length} bits does not fit: {FINAL_SUBBAND} of this image "
            f"provides {capacity} slots."
        )

    recovered = extract_traditional(watermarked, original, config)
    n = len(recovered)

    scored = None
    if req.expected_bits is not None:
        reference = _parse_expected_bits(req.expected_bits, n)
        scored = _score(reference, recovered)

    if req.payload_kind == "registry_id":
        recovered_id, _, _ = id_registry.decode_id_bits(recovered[: id_registry.ENCODED_BITS])
        decoded_text = MESSAGE_REGISTRY.message_for(recovered_id)
    else:
        decoded_text = _decode_payload_message(recovered, payload_bits=n, repetition=1)
    text_match = None
    char_accuracy = None
    if req.expected_text is not None:
        expected = req.expected_text.strip()
        got = decoded_text or ""
        text_match = bool(decoded_text is not None and got == expected)
        char_accuracy = _round(service.character_accuracy(expected, got), 4)

    return {
        "extraction": "non_blind",
        "decoder": "frozen DWT-SVD reference decoder",
        "requires_original_image": True,
        "config": {"alpha": config.alpha, "bit_length": config.bit_length,
                   "wavelet": config.wavelet, "subband": FINAL_SUBBAND},
        "recovered": {
            "bit_length": n,
            "bit_string": "".join(map(str, recovered)),
            "text": decoded_text,
        },
        "reference_scoring": scored,
        "text_match": text_match,
        "char_accuracy": char_accuracy,
    }


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), ndigits)
