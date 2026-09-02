"""Phase 12 - High-capacity, multi-level DWT-SVD watermark embedding.

This module is **new Phase 12 code**. It does not modify the frozen Phase 6
baseline (``embed.py`` / ``dwt.py`` / ``svd.py`` / ``watermark_generator.py``):
those files are byte-for-byte untouched. This module reuses their public,
unmodified primitives - ``decompose_2d`` / ``reconstruct_2d`` from ``dwt.py``
(called recursively for multi-level decomposition), ``decompose`` /
``reconstruct`` / ``SVDComponents`` / ``get_singular_values`` from ``svd.py``,
and ``generate_from_uuid`` / ``validate_bit_length`` / ``SUPPORTED_BIT_LENGTHS``
from ``watermark_generator.py`` for deterministic payloads.

Why capacity is limited, and why a new path is needed
-------------------------------------------------------
The frozen baseline embeds one bit per leading singular value of a DWT
subband. A single-level DWT of an ``N x N`` image gives four ``N/2 x N/2``
subbands (LL, LH, HL, HH), each with ``N/2`` singular values. For the
project's 256x256 processed images that is **128 bits per subband**. The
frozen baseline already supports spending more than one subband's worth via
``EmbedConfig.extra_subbands`` (LL, LL+HL, ... up to all four = 512 bits) -
that ceiling is real and is hit exactly at 512 bits with a single DWT level.

To go beyond 512 bits, the *number of coefficients available for embedding*
must grow, which single-level DWT cannot do (only four subbands exist,
however they are combined). **Multi-level DWT** recursively decomposes the LL
subband again: a 2-level decomposition of a 256x256 image gives three
128x128 detail subbands (LH1/HL1/HH1, untouched by the second decomposition)
plus four 64x64 subbands from decomposing LL1 (LL2/LH2/HL2/HH2). Each
subband still contributes only its own side length in singular values
(64 for a 64x64 subband), so going deeper adds capacity but each new level's
contribution shrinks geometrically. In closed form, for an ``N x N`` image
decomposed to ``L`` levels, using every detail subband at levels ``1..L-1``
and all four subbands at the final level ``L``:

    capacity(L) = 3 * sum_{l=1}^{L-1} (N / 2^l)  +  4 * (N / 2^L)

As ``L -> infinity`` this converges to ``capacity(inf) = 3*N`` (256x256 ->
**768 bits**), because ``sum_{l=1}^{inf} N/2^l = N``. This is a hard
theoretical ceiling for this project's watermarking scheme (one bit per
leading singular value, whole-subband SVD, 256x256 images) - **no number of
DWT levels can reach 1024 bits this way**, and going deep enough to approach
768 makes the smallest subbands (single-digit pixels wide) far too fragile to
be a usable embedding domain in practice. Phase 12's experiment
(``src/evaluation/phase12_capacity.py``) measures this directly instead of
assuming it: 128/256/512 bits succeed; 1024 bits is attempted at 1, 2 and 3
DWT levels and fails explicitly (capacity 512 / 640 / 704 < 1024 respectively),
which is recorded, not hidden or forced.

Embedding rule
---------------
Identical to the frozen baseline's own multiplicative rule, applied per
configured (level, subband) window:

    S'[i] = S[i] * (1 + alpha * (2*b[i] - 1))

Restrictions (documented, not silently ignored)
-------------------------------------------------
* An intermediate level's LL subband (level < ``dwt_levels``) is always
  decomposed further and is therefore *not* directly available for embedding
  - only the final level's LL, and any level's LH/HL/HH, are valid targets.
  Listing an intermediate LL in ``subband_plan`` raises ``ValueError``.
* Extraction is non-blind only, mirroring ``extract_traditional``'s decision
  rule (``bit = 1 if S_w[i] > S_o[i] else 0``) applied per window. No CNN
  work is in scope here (Phase 8's architecture is untouched).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from src.watermark.dwt import DWTCoefficients, decompose_2d, reconstruct_2d
from src.watermark.svd import SVDComponents, decompose, get_singular_values, reconstruct

__all__ = [
    "SUPPORTED_SUBBANDS",
    "CapacityEmbedConfig",
    "CapacityEmbedResult",
    "InsufficientCapacityError",
    "embed_capacity",
    "extract_capacity",
    "full_subband_plan",
    "plan_windows",
    "subband_side_length",
    "theoretical_capacity",
    "theoretical_capacity_limit",
    "total_capacity",
]

SUPPORTED_SUBBANDS = ("LL", "LH", "HL", "HH")


class InsufficientCapacityError(ValueError):
    """Raised when a config's subband_plan cannot hold the requested bit_length."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CapacityEmbedConfig:
    """Configuration for one Phase 12 multi-level, multi-subband embedding.

    Attributes
    ----------
    dwt_levels:
        Number of recursive DWT decomposition levels. 1 = single-level, same
        depth as the frozen baseline (but still a separate code path).
    subband_plan:
        Ordered ``((level, band), ...)`` describing which subbands carry the
        payload and in what order bits are assigned. ``level`` is 1-indexed
        (1 = first decomposition of the Y channel); ``band`` is one of
        ``LL``/``LH``/``HL``/``HH``. An intermediate level's ``LL`` (level <
        ``dwt_levels``) is invalid - it is always decomposed further.
    alpha, bit_length, start_sv_index:
        Same meaning as the frozen baseline's ``EmbedConfig``. ``start_sv_index``
        applies only to the first window in ``subband_plan``; every later
        window starts at singular-value index 0 (identical convention to
        ``embed.py``'s multi-subband overflow).
    """

    wavelet: str = "haar"
    mode: str = "symmetric"
    dwt_levels: int = 1
    subband_plan: tuple[tuple[int, str], ...] = ((1, "LL"),)
    alpha: float = 0.02
    bit_length: int = 128
    start_sv_index: int = 0

    def __post_init__(self) -> None:
        if self.wavelet != "haar":
            raise ValueError(f"CapacityEmbedConfig only supports wavelet='haar' (got {self.wavelet!r})")
        if self.mode not in ("symmetric", "periodization"):
            raise ValueError(f"unsupported DWT mode {self.mode!r}")
        if self.dwt_levels < 1:
            raise ValueError(f"dwt_levels must be >= 1; got {self.dwt_levels}")
        if not self.subband_plan:
            raise ValueError("subband_plan must be non-empty")
        seen: set[tuple[int, str]] = set()
        for level, band in self.subband_plan:
            if level < 1 or level > self.dwt_levels:
                raise ValueError(f"subband_plan entry level={level} out of range [1, {self.dwt_levels}]")
            if band not in SUPPORTED_SUBBANDS:
                raise ValueError(f"unsupported band {band!r}; allowed: {list(SUPPORTED_SUBBANDS)}")
            if band == "LL" and level < self.dwt_levels:
                raise ValueError(
                    f"LL at level {level} < dwt_levels ({self.dwt_levels}) is decomposed further "
                    "and is not directly available for embedding; use the final level's LL "
                    "or a detail band (LH/HL/HH) at this level"
                )
            key = (level, band)
            if key in seen:
                raise ValueError(f"subband_plan has a duplicate entry: {key}")
            seen.add(key)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha}")
        if self.bit_length <= 0:
            raise ValueError(f"bit_length must be positive; got {self.bit_length}")
        if self.start_sv_index < 0:
            raise ValueError(f"start_sv_index must be >= 0; got {self.start_sv_index}")


def full_subband_plan(dwt_levels: int) -> tuple[tuple[int, str], ...]:
    """The maximum-capacity plan for ``dwt_levels`` levels: every detail
    subband at levels ``1..dwt_levels-1``, plus all four subbands at the
    final level - i.e. the plan ``theoretical_capacity`` assumes."""
    if dwt_levels < 1:
        raise ValueError(f"dwt_levels must be >= 1; got {dwt_levels}")
    plan: list[tuple[int, str]] = []
    for level in range(1, dwt_levels):
        plan.extend((level, band) for band in ("LH", "HL", "HH"))
    plan.extend((dwt_levels, band) for band in SUPPORTED_SUBBANDS)
    return tuple(plan)


def subband_side_length(image_size: int, level: int) -> int:
    """Side length (= singular-value count, since subbands are square) of a
    subband at the given decomposition ``level`` for an ``image_size x image_size``
    image. Raises if the halving does not divide evenly at every level up to it."""
    size = image_size
    for _ in range(level):
        if size % 2 != 0:
            raise ValueError(
                f"image_size={image_size} does not evenly halve down to level {level} "
                f"(stuck at size {size})"
            )
        size //= 2
    return size


def theoretical_capacity(image_size: int, dwt_levels: int) -> int:
    """Max bits embeddable with the ``full_subband_plan`` at ``dwt_levels`` levels."""
    total = 0
    for level in range(1, dwt_levels):
        side = subband_side_length(image_size, level)
        total += 3 * side
    total += 4 * subband_side_length(image_size, dwt_levels)
    return total


def theoretical_capacity_limit(image_size: int) -> int:
    """The supremum of ``theoretical_capacity`` as ``dwt_levels -> infinity``:
    ``3 * image_size`` (since ``sum_{l=1}^{inf} image_size / 2^l = image_size``).
    A hard ceiling independent of how many DWT levels are used."""
    return 3 * image_size


def plan_windows(
    config: CapacityEmbedConfig, image_size: int
) -> list[tuple[int, str, int, int]]:
    """Split ``config.bit_length`` across ``config.subband_plan``.

    Returns a list of ``(level, band, start_index, n_bits)`` windows covering
    exactly ``bit_length`` bits, or raises :class:`InsufficientCapacityError`
    with the exact available count if the plan does not provide enough room.
    """
    windows: list[tuple[int, str, int, int]] = []
    remaining = config.bit_length
    capacity = 0
    for position, (level, band) in enumerate(config.subband_plan):
        side = subband_side_length(image_size, level)
        start = config.start_sv_index if position == 0 else 0
        available = max(0, side - start)
        capacity += available
        if remaining <= 0:
            continue
        take = min(remaining, available)
        if take > 0:
            windows.append((level, band, start, take))
            remaining -= take
    if remaining > 0:
        raise InsufficientCapacityError(
            f"cannot embed {config.bit_length} bits: subband_plan {config.subband_plan} "
            f"(dwt_levels={config.dwt_levels}) provides only {capacity} singular-value slots "
            f"(start_sv_index={config.start_sv_index}) for a {image_size}x{image_size} image"
        )
    return windows


def total_capacity(config: CapacityEmbedConfig, image_size: int) -> int:
    """Total singular-value slots ``config.subband_plan`` provides, regardless
    of ``bit_length`` (does not raise even if less than ``bit_length``)."""
    capacity = 0
    for position, (level, band) in enumerate(config.subband_plan):
        side = subband_side_length(image_size, level)
        start = config.start_sv_index if position == 0 else 0
        capacity += max(0, side - start)
    return capacity


# ---------------------------------------------------------------------------
# Colour-space helpers (self-contained; mirrors the frozen baseline's proven
# RGB<->YCrCb round trip, same pattern already used in Phase 11)
# ---------------------------------------------------------------------------

def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an RGB image (H x W x 3); got shape {image.shape}")
    if image.dtype == np.uint8:
        return image
    return np.clip(np.round(np.asarray(image, dtype=np.float64)), 0, 255).astype(np.uint8)


def _rgb_to_ycrcb(image_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ycrcb = cv2.cvtColor(image_u8, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float64)
    cr = ycrcb[:, :, 1].copy()
    cb = ycrcb[:, :, 2].copy()
    return y, cr, cb


def _ycrcb_to_rgb(y: np.ndarray, cr_u8: np.ndarray, cb_u8: np.ndarray) -> np.ndarray:
    y_u8 = np.clip(np.round(y), 0, 255).astype(np.uint8)
    ycrcb = np.stack([y_u8, cr_u8.astype(np.uint8), cb_u8.astype(np.uint8)], axis=2)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


# ---------------------------------------------------------------------------
# Multi-level DWT (built on the frozen single-level dwt.py, called recursively)
# ---------------------------------------------------------------------------

def _decompose_multilevel(y: np.ndarray, wavelet: str, mode: str, levels: int) -> list[DWTCoefficients]:
    """Recursively decompose ``y`` ``levels`` times; ``result[i].ll`` is what
    gets decomposed again at ``result[i+1]`` for every ``i < levels - 1``."""
    out: list[DWTCoefficients] = []
    current = y
    for _ in range(levels):
        c = decompose_2d(current, wavelet=wavelet, mode=mode)
        out.append(c)
        current = c.ll
    return out


def _reconstruct_multilevel(levels: list[DWTCoefficients]) -> np.ndarray:
    """Invert ``_decompose_multilevel``, propagating each level's reconstruction
    up into its parent's ``ll`` field."""
    levels = list(levels)
    for i in range(len(levels) - 1, 0, -1):
        recon = reconstruct_2d(levels[i])
        levels[i - 1] = replace(levels[i - 1], ll=recon)
    return reconstruct_2d(levels[0])


def _get_band(levels: list[DWTCoefficients], level: int, band: str) -> np.ndarray:
    coeffs = levels[level - 1]
    return {"LL": coeffs.ll, "LH": coeffs.lh, "HL": coeffs.hl, "HH": coeffs.hh}[band]


def _set_band(levels: list[DWTCoefficients], level: int, band: str, new_band: np.ndarray) -> None:
    coeffs = levels[level - 1]
    kwargs = {"ll": coeffs.ll, "lh": coeffs.lh, "hl": coeffs.hl, "hh": coeffs.hh}
    kwargs[band.lower()] = new_band
    levels[level - 1] = replace(coeffs, **kwargs)


def _modulate(components: SVDComponents, bits: list[int], alpha: float, start: int) -> np.ndarray:
    """Frozen baseline's multiplicative rule, applied to one window."""
    s_modified = components.S.copy()
    signs = np.array([2 * b - 1 for b in bits], dtype=np.float64)
    s_modified[start:start + len(bits)] *= 1.0 + alpha * signs
    return s_modified


# ---------------------------------------------------------------------------
# Embedding / extraction
# ---------------------------------------------------------------------------

@dataclass
class CapacityEmbedResult:
    watermarked_image: np.ndarray
    original_image: np.ndarray
    bits: list[int]
    config: CapacityEmbedConfig
    windows: list[tuple[int, str, int, int]]
    original_sv: dict[tuple[int, str], np.ndarray]
    watermarked_sv: dict[tuple[int, str], np.ndarray]


def embed_capacity(image: np.ndarray, bits: list[int], config: CapacityEmbedConfig) -> CapacityEmbedResult:
    """Embed ``bits`` using the frozen multiplicative rule across a
    configurable multi-level, multi-subband plan. Raises
    :class:`InsufficientCapacityError` (a ``ValueError`` subclass) if
    ``config.subband_plan`` cannot hold ``len(bits)`` - never silently drops
    or truncates the payload."""
    rgb = _to_uint8_rgb(image)
    if any(b not in (0, 1) for b in bits):
        raise ValueError("bits must contain only 0 and 1")
    if len(bits) != config.bit_length:
        raise ValueError(f"expected {config.bit_length} bits from config; got {len(bits)}")

    y, cr, cb = _rgb_to_ycrcb(rgb)
    image_size = y.shape[0]
    if y.shape[0] != y.shape[1]:
        raise ValueError(f"capacity_embed requires a square image; got {y.shape}")

    windows = plan_windows(config, image_size)  # may raise InsufficientCapacityError
    levels = _decompose_multilevel(y, config.wavelet, config.mode, config.dwt_levels)

    original_sv: dict[tuple[int, str], np.ndarray] = {}
    watermarked_sv: dict[tuple[int, str], np.ndarray] = {}
    offset = 0
    for level, band, start, n_bits in windows:
        arr = _get_band(levels, level, band)
        svd_band = decompose(arr)
        original_sv[(level, band)] = svd_band.S.copy()
        window_bits = bits[offset:offset + n_bits]
        s_mod = _modulate(svd_band, window_bits, config.alpha, start)
        watermarked_sv[(level, band)] = s_mod.copy()
        recon_band = reconstruct(
            SVDComponents(U=svd_band.U, S=s_mod, Vt=svd_band.Vt, source_shape=svd_band.source_shape)
        )
        _set_band(levels, level, band, recon_band)
        offset += n_bits

    y_modified = _reconstruct_multilevel(levels)
    watermarked = _ycrcb_to_rgb(y_modified, cr, cb)

    return CapacityEmbedResult(
        watermarked_image=watermarked,
        original_image=rgb,
        bits=list(bits),
        config=config,
        windows=windows,
        original_sv=original_sv,
        watermarked_sv=watermarked_sv,
    )


def _subband_singular_values(image: np.ndarray, config: CapacityEmbedConfig) -> tuple[list[DWTCoefficients], int]:
    y, _, _ = _rgb_to_ycrcb(_to_uint8_rgb(image))
    if y.shape[0] != y.shape[1]:
        raise ValueError(f"capacity_embed requires a square image; got {y.shape}")
    levels = _decompose_multilevel(y, config.wavelet, config.mode, config.dwt_levels)
    return levels, y.shape[0]


def extract_capacity(
    watermarked_image: np.ndarray, original_image: np.ndarray, config: CapacityEmbedConfig
) -> list[int]:
    """Non-blind decode: ``bit = 1 if S_w[i] > S_o[i] else 0`` per configured
    (level, band) window - the same decision rule as ``extract_traditional``,
    generalised across DWT levels."""
    levels_o, image_size = _subband_singular_values(original_image, config)
    levels_w, _ = _subband_singular_values(watermarked_image, config)
    windows = plan_windows(config, image_size)

    recovered: list[int] = []
    for level, band, start, n_bits in windows:
        s_o = get_singular_values(_get_band(levels_o, level, band))
        s_w = get_singular_values(_get_band(levels_w, level, band))
        recovered.extend(1 if s_w[start + i] > s_o[start + i] else 0 for i in range(n_bits))
    return recovered
