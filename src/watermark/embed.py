"""DWT-SVD baseline watermark embedding and traditional (non-blind) extraction.

Phase 6 — Frozen Baseline System
--------------------------------
This module implements the classic DWT-SVD watermarking pipeline in the form the
project uses as its permanent reproducible baseline. Every learned/adaptive model
introduced in later phases is compared against the numbers this module produces.

Baseline embedding rule (selected and frozen in Phase 6)
-------------------------------------------------------
Phase 0 deliberately left the bit-to-singular-value rule open. Phase 6 selects a
**multiplicative (relative) modulation** of the leading singular values of the
luminance-channel DWT subband(s):

    S'[i] = S[i] * (1 + alpha * (2 * b[i] - 1))

    b[i] = 1  ->  S'[i] = S[i] * (1 + alpha)
    b[i] = 0  ->  S'[i] = S[i] * (1 - alpha)

Rationale for choosing multiplicative over the generic additive candidate
``S'[i] = S[i] + alpha * f(b[i])``:

* The singular values of a subband span several orders of magnitude
  (~1e4 down to ~1e0 for a 256x256 luminance image). A single additive ``alpha``
  is simultaneously invisible for the small values and destructive for the large
  ones. A relative step keeps the perceptual perturbation proportional and keeps
  every modified value above the uint8 quantization floor for realistic ``alpha``.
* The roadmap's own strength sweep (alpha in {0.005, 0.010, 0.015}, i.e.
  0.5%-1.5%) only makes sense as a relative quantity.
* The decision statistic for blind decoding (Phase 7) becomes scale-free:
  ``sign(S_w[i] / S_o[i] - 1)``.

Pipeline
--------
Embedding:
  1. RGB uint8 -> YCrCb; keep Cr, Cb untouched, operate on Y (float64).
  2. Single-level 2D DWT of Y (default wavelet: haar, mode: symmetric).
  3. For each subband in the configured order, SVD it, modulate as many leading
     singular values as there are remaining payload bits, reconstruct the subband.
  4. Inverse DWT -> modified Y; re-clip to [0, 255] uint8; merge with the
     original Cr, Cb; convert back to RGB.

Traditional (non-blind) extraction:
  1. DWT + SVD of the watermarked image's Y subband(s) -> S_w.
  2. DWT + SVD of the original image's Y subband(s)      -> S_o.
  3. bit[i] = 1 if S_w[i] > S_o[i] else 0.

The blind CNN extractor (Phase 7) replaces steps 2-3 with a learned model that
does not need the original image.

Payload capacity (measured, documented)
---------------------------------------
A single-level DWT of an ``N x N`` image gives subbands of size ``N/2 x N/2``,
each with ``N/2`` singular values. For the processed dataset (``N = 256``) that is
**128 bits per subband**. The frozen baseline uses the LL subband only, so its
capacity is **128 bits**. Payloads of 256 bits (and, in later phases, up to
1024) are reached by extending the subband order — e.g. ``("LL", "HL")`` gives
256 bits, ``("LL", "HL", "LH", "HH")`` gives 512. Going beyond 512 on a 256x256
image requires a multi-level DWT and is left to Phase 12 (high-capacity study).
``embed`` raises ``ValueError`` with the exact available count when a payload does
not fit the configured subband order.

Other known baseline limitations:
  * ``alpha`` is fixed, not content-aware (Phase 10 addresses this).
  * Extraction is non-blind here; robustness to attacks is not claimed until
    Phase 9 measures it.
  * Recovery on a clean image is exact for the well-separated leading singular
    values; the smallest trailing singular values of a subband are close together
    and a fraction of them can flip under the IDWT + uint8 round trip. The
    Phase 6 experiment reports the real per-payload BER.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.watermark.dwt import DWTCoefficients, decompose_2d, reconstruct_2d
from src.watermark.svd import SVDComponents, decompose, reconstruct

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_SUBBANDS = ("LL", "LH", "HL", "HH")
BASELINE_ALPHAS = (0.005, 0.010, 0.015)


@dataclass(frozen=True)
class EmbedConfig:
    """Configuration for one baseline embedding experiment.

    Attributes
    ----------
    wavelet:
        DWT wavelet kernel. Baseline: ``haar``.
    subband:
        Primary DWT subband that carries the watermark. Baseline: ``LL``.
    extra_subbands:
        Ordered additional subbands used only when the payload exceeds the
        primary subband's singular-value count. Baseline: empty.
    alpha:
        Relative embedding strength. Baseline sweep: 0.005, 0.010, 0.015.
    bit_length:
        Payload size in bits.
    start_sv_index:
        Index of the first singular value modified in the primary subband.
        Baseline: 0 (largest first). Applies to the primary subband only;
        overflow subbands always start at index 0.
    mode:
        DWT boundary mode. Baseline: ``symmetric``.
    """

    wavelet: str = "haar"
    subband: str = "LL"
    extra_subbands: tuple[str, ...] = ()
    alpha: float = 0.010
    bit_length: int = 64
    start_sv_index: int = 0
    mode: str = "symmetric"

    def __post_init__(self) -> None:
        for name in self.subband_order:
            if name not in SUPPORTED_SUBBANDS:
                raise ValueError(
                    f"Unsupported subband {name!r}; allowed: {list(SUPPORTED_SUBBANDS)}"
                )
        if len(set(self.subband_order)) != len(self.subband_order):
            raise ValueError(f"subband order has duplicates: {self.subband_order}")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha}")
        if self.bit_length <= 0:
            raise ValueError(f"bit_length must be positive; got {self.bit_length}")
        if self.start_sv_index < 0:
            raise ValueError(f"start_sv_index must be >= 0; got {self.start_sv_index}")

    @property
    def subband_order(self) -> tuple[str, ...]:
        return (self.subband, *self.extra_subbands)


# ---------------------------------------------------------------------------
# Colour-space helpers (consistent, loss-checked round trip)
# ---------------------------------------------------------------------------

def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an RGB image (H x W x 3); got shape {image.shape}")
    if image.dtype == np.uint8:
        return image
    return np.clip(np.round(np.asarray(image, dtype=np.float64)), 0, 255).astype(np.uint8)


def _rgb_to_ycrcb(image_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB uint8 -> (Y float64, Cr uint8, Cb uint8). OpenCV channel order is Y, Cr, Cb."""
    ycrcb = cv2.cvtColor(image_u8, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float64)
    cr = ycrcb[:, :, 1].copy()
    cb = ycrcb[:, :, 2].copy()
    return y, cr, cb


def _ycrcb_to_rgb(y: np.ndarray, cr_u8: np.ndarray, cb_u8: np.ndarray) -> np.ndarray:
    """(Y float64, Cr uint8, Cb uint8) -> RGB uint8, matching :func:`_rgb_to_ycrcb`."""
    y_u8 = np.clip(np.round(y), 0, 255).astype(np.uint8)
    ycrcb = np.stack([y_u8, cr_u8.astype(np.uint8), cb_u8.astype(np.uint8)], axis=2)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def _get_subband(coeffs: DWTCoefficients, subband: str) -> np.ndarray:
    return {"LL": coeffs.ll, "LH": coeffs.lh, "HL": coeffs.hl, "HH": coeffs.hh}[subband]


def _set_subband(coeffs: DWTCoefficients, subband: str, new_band: np.ndarray) -> DWTCoefficients:
    return DWTCoefficients(
        ll=new_band if subband == "LL" else coeffs.ll,
        lh=new_band if subband == "LH" else coeffs.lh,
        hl=new_band if subband == "HL" else coeffs.hl,
        hh=new_band if subband == "HH" else coeffs.hh,
        source_shape=coeffs.source_shape,
        wavelet=coeffs.wavelet,
        mode=coeffs.mode,
    )


def _plan_windows(config: EmbedConfig, sv_counts: dict[str, int]) -> list[tuple[str, int, int]]:
    """Split the payload across the configured subband order.

    Returns a list of (subband_name, start_index, n_bits) covering exactly
    ``config.bit_length`` bits, or raises ``ValueError`` if it does not fit.
    """
    windows: list[tuple[str, int, int]] = []
    remaining = config.bit_length
    total_capacity = 0
    for position, name in enumerate(config.subband_order):
        start = config.start_sv_index if position == 0 else 0
        available = max(0, sv_counts[name] - start)
        total_capacity += available
        if remaining <= 0:
            continue
        take = min(remaining, available)
        if take > 0:
            windows.append((name, start, take))
            remaining -= take
    if remaining > 0:
        raise ValueError(
            f"cannot embed {config.bit_length} bits: subband order "
            f"{config.subband_order} provides only {total_capacity} singular-value "
            f"slots (start_sv_index={config.start_sv_index})"
        )
    return windows


def _modulate(components: SVDComponents, bits: list[int], alpha: float, start: int) -> np.ndarray:
    """Return a modified singular-value vector using the frozen baseline rule."""
    s_modified = components.S.copy()
    signs = np.array([2 * b - 1 for b in bits], dtype=np.float64)
    s_modified[start:start + len(bits)] *= 1.0 + alpha * signs
    return s_modified


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

@dataclass
class EmbedResult:
    """Everything produced by one embedding operation.

    Attributes
    ----------
    watermarked_image / original_image:
        RGB uint8 arrays.
    bits:
        The embedded payload (list of 0/1).
    config:
        The :class:`EmbedConfig` used.
    windows:
        The (subband, start_index, n_bits) plan the payload was split into.
    original_sv / watermarked_sv:
        Per-subband singular-value vectors before and after modification
        (kept for the non-blind decoder and for diagnostics).
    """

    watermarked_image: np.ndarray
    original_image: np.ndarray
    bits: list[int]
    config: EmbedConfig
    windows: list[tuple[str, int, int]]
    original_sv: dict[str, np.ndarray] = field(default_factory=dict)
    watermarked_sv: dict[str, np.ndarray] = field(default_factory=dict)


def subband_capacity(image_shape: tuple[int, int], config: EmbedConfig) -> int:
    """Total bits that fit in the configured subband order for this image size."""
    h, w = image_shape[:2]
    sh, sw = (h + 1) // 2, (w + 1) // 2
    per_band = min(sh, sw)
    total = 0
    for position, _ in enumerate(config.subband_order):
        start = config.start_sv_index if position == 0 else 0
        total += max(0, per_band - start)
    return total


def embed(image: np.ndarray, bits: list[int], config: EmbedConfig) -> EmbedResult:
    """Embed ``bits`` into an RGB image with the frozen DWT-SVD baseline rule."""
    rgb = _to_uint8_rgb(image)
    if any(b not in (0, 1) for b in bits):
        raise ValueError("bits must contain only 0 and 1")
    if len(bits) != config.bit_length:
        raise ValueError(f"expected {config.bit_length} bits from config; got {len(bits)}")

    y, cr, cb = _rgb_to_ycrcb(rgb)
    coeffs = decompose_2d(y, wavelet=config.wavelet, mode=config.mode)

    sv_counts = {name: min(_get_subband(coeffs, name).shape) for name in config.subband_order}
    windows = _plan_windows(config, sv_counts)

    original_sv: dict[str, np.ndarray] = {}
    watermarked_sv: dict[str, np.ndarray] = {}
    offset = 0
    for name, start, n_bits in windows:
        band = _get_subband(coeffs, name)
        svd_band = decompose(band)
        original_sv[name] = svd_band.S.copy()
        window_bits = bits[offset:offset + n_bits]
        s_mod = _modulate(svd_band, window_bits, config.alpha, start)
        watermarked_sv[name] = s_mod.copy()
        modified_band = reconstruct(
            SVDComponents(U=svd_band.U, S=s_mod, Vt=svd_band.Vt, source_shape=svd_band.source_shape)
        )
        coeffs = _set_subband(coeffs, name, modified_band)
        offset += n_bits

    y_modified = reconstruct_2d(coeffs)
    watermarked = _ycrcb_to_rgb(y_modified, cr, cb)

    return EmbedResult(
        watermarked_image=watermarked,
        original_image=rgb,
        bits=list(bits),
        config=config,
        windows=windows,
        original_sv=original_sv,
        watermarked_sv=watermarked_sv,
    )


# ---------------------------------------------------------------------------
# Traditional (non-blind) extraction
# ---------------------------------------------------------------------------

def _subband_singular_values(image: np.ndarray, config: EmbedConfig) -> dict[str, np.ndarray]:
    y, _, _ = _rgb_to_ycrcb(_to_uint8_rgb(image))
    coeffs = decompose_2d(y, wavelet=config.wavelet, mode=config.mode)
    return {name: decompose(_get_subband(coeffs, name)).S for name in config.subband_order}


def extract_traditional(
    watermarked_image: np.ndarray, original_image: np.ndarray, config: EmbedConfig
) -> list[int]:
    """Non-blind decode: compare watermarked vs original subband singular values.

    Decision rule: ``bit[i] = 1 if S_w[i] > S_o[i] else 0``. This establishes the
    perfect-condition reference BER for the baseline; the blind CNN (Phase 7)
    replaces the dependence on the original image.
    """
    sv_counts_source = _subband_singular_values(original_image, config)
    sv_counts = {name: len(vec) for name, vec in sv_counts_source.items()}
    windows = _plan_windows(config, sv_counts)

    s_orig = sv_counts_source
    s_wm = _subband_singular_values(watermarked_image, config)

    recovered: list[int] = []
    for name, start, n_bits in windows:
        o = s_orig[name]
        w = s_wm[name]
        recovered.extend(1 if w[start + i] > o[start + i] else 0 for i in range(n_bits))
    return recovered


# ---------------------------------------------------------------------------
# Residual image (visualisation only)
# ---------------------------------------------------------------------------

def compute_residual(original: np.ndarray, watermarked: np.ndarray, amplify: float = 15.0) -> np.ndarray:
    """Amplified absolute-difference image so the (near-invisible) change is visible."""
    diff = np.abs(original.astype(np.float64) - watermarked.astype(np.float64)) * amplify
    return np.clip(np.round(diff), 0, 255).astype(np.uint8)
