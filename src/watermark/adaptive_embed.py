"""Phase 11 - Adaptive, content-aware DWT-SVD watermark embedding.

This module is **new Phase 11 code**. It does not modify, import-and-monkeypatch,
or otherwise change the frozen Phase 6 baseline (``src/watermark/embed.py``,
``dwt.py``, ``svd.py``, ``watermark_generator.py``): those files are byte-for-byte
untouched (verified by the Phase 9 frozen-file hashes, reused here). This module
only reuses their public, unmodified low-level primitives (``decompose_2d`` /
``reconstruct_2d`` from ``dwt.py``, ``decompose`` / ``reconstruct`` /
``SVDComponents`` from ``svd.py``).

Why a new embedding path instead of passing a smarter ``alpha`` into ``embed()``
------------------------------------------------------------------------------
The frozen baseline applies **one SVD to the whole LL subband** and modulates its
leading singular values. Each singular value's outer product ``u_i @ v_i.T``
spans the *entire* image - there is no way to make "singular value index 7"
correspond to a spatial region, so a single global ``alpha`` cannot be made
region-adaptive without changing what ``embed()`` computes (which would break
the Phase 6 freeze).

Phase 11 instead partitions the LL subband into a grid of non-overlapping
blocks and runs one small SVD per block, embedding exactly one bit per block in
that block's leading singular value with the frozen baseline's own multiplicative
rule, ``S'[0] = S[0] * (1 + alpha_block * (2*bit - 1))``. This makes a per-block
``alpha_block`` meaningful: it can now depend on that block's local image
content. With ``block_size=16`` and the project's 256x256 processed images, the
LL subband (128x128) yields an 8x8 = 64-block grid, so the default 64-bit
operating point used since Phase 8 needs no change.

Two ``alpha_mode`` values share this exact machinery so the comparison isolates
only the one variable under test:

* ``"fixed"``    - every block gets the same ``alpha`` (the classic baseline
  behaviour, reproduced inside the block-SVD topology).
* ``"adaptive"`` - each block's alpha is derived from a Sobel edge-energy score
  of that block (more texture/edges -> higher alpha, the HVS is less sensitive
  there; flat/smooth blocks -> lower alpha, to avoid visible banding), then
  *rescaled so the per-image mean alpha equals the configured* ``alpha``. This
  keeps the total embedding "energy budget" identical between the two modes -
  adaptive is a reallocation of the same budget, not simply a stronger or
  weaker watermark.

Restrictions (documented, not silently ignored)
-------------------------------------------------
* ``wavelet`` is fixed to ``"haar"``. The Sobel edge map is computed on the
  full-resolution Y channel and average-pooled 2x2 to line up with the LL grid;
  that exact 2x2 relationship only holds for a single-level Haar DWT. Wavelet
  choice is Phase 13's concern, not Phase 11's.
* ``bit_length`` must equal the number of blocks the image produces
  (``(ll_size // block_size) ** 2``); there is no multi-subband overflow here
  (that is Phase 12, high-capacity, out of scope).
* Extraction is non-blind only (compares against the original image's block
  singular values, mirroring ``extract_traditional``'s decision rule). The
  Phase 8 CNN was trained on the frozen whole-subband topology and is not
  retrained or repurposed here - no CNN architecture change, per scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from src.watermark.dwt import decompose_2d, reconstruct_2d
from src.watermark.svd import SVDComponents, decompose, get_singular_values, reconstruct

__all__ = [
    "AdaptiveEmbedConfig",
    "AdaptiveEmbedResult",
    "block_grid",
    "embed_adaptive",
    "expected_bit_length",
    "extract_adaptive",
    "texture_map",
]

ALPHA_MODES = ("fixed", "adaptive")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdaptiveEmbedConfig:
    """Configuration for one Phase 11 block-SVD embedding.

    Attributes
    ----------
    block_size:
        Side length of the (square) LL-subband blocks. Default 16, which
        divides the 128x128 LL subband of a 256x256 image into an 8x8 grid of
        64 blocks - one bit per block, matching the project's established
        64-bit operating point.
    bit_length:
        Payload size in bits. Must equal ``rows * cols`` of the block grid for
        the image the config is used on (checked at embed/extract time, where
        the image size is known).
    alpha_mode:
        ``"fixed"`` (every block same strength) or ``"adaptive"`` (content-aware,
        mean-preserving reallocation).
    alpha:
        The strength used directly in ``"fixed"`` mode, and the *per-image
        mean* strength preserved in ``"adaptive"`` mode. Same range and meaning
        as the frozen baseline's ``EmbedConfig.alpha``.
    alpha_low_mult / alpha_high_mult:
        In adaptive mode, a block with the lowest texture score in the image
        gets a raw weight of ``alpha_low_mult`` and the highest gets
        ``alpha_high_mult`` (linearly interpolated in between); weights are
        then rescaled so their mean equals 1, i.e. ``mean(alpha_block) == alpha``.
    texture_percentile_clip:
        ``(low, high)`` percentiles used to clip the raw per-block texture
        score before normalising, so a few outlier blocks (e.g. a single very
        sharp edge) do not compress every other block's weight toward one end.
    wavelet / mode:
        DWT parameters. ``wavelet`` is restricted to ``"haar"`` (see module
        docstring); ``mode`` is the DWT boundary mode, default ``"symmetric"``
        to match the frozen baseline.
    """

    block_size: int = 16
    bit_length: int = 64
    alpha_mode: str = "adaptive"
    alpha: float = 0.02
    alpha_low_mult: float = 0.4
    alpha_high_mult: float = 1.6
    texture_percentile_clip: tuple[float, float] = (5.0, 95.0)
    wavelet: str = "haar"
    mode: str = "symmetric"

    def __post_init__(self) -> None:
        if self.wavelet != "haar":
            raise ValueError(
                f"AdaptiveEmbedConfig only supports wavelet='haar' (got {self.wavelet!r}); "
                "wavelet choice is a Phase 13 concern"
            )
        if self.mode not in ("symmetric", "periodization"):
            raise ValueError(f"unsupported DWT mode {self.mode!r}")
        if self.block_size < 2:
            raise ValueError(f"block_size must be >= 2; got {self.block_size}")
        if self.bit_length <= 0:
            raise ValueError(f"bit_length must be positive; got {self.bit_length}")
        if self.alpha_mode not in ALPHA_MODES:
            raise ValueError(f"alpha_mode must be one of {ALPHA_MODES}; got {self.alpha_mode!r}")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha}")
        if self.alpha_low_mult <= 0 or self.alpha_high_mult <= 0:
            raise ValueError("alpha_low_mult and alpha_high_mult must be positive")
        if self.alpha_low_mult >= self.alpha_high_mult:
            raise ValueError(
                f"alpha_low_mult ({self.alpha_low_mult}) must be < alpha_high_mult "
                f"({self.alpha_high_mult})"
            )
        lo, hi = self.texture_percentile_clip
        if not 0.0 <= lo < hi <= 100.0:
            raise ValueError(f"texture_percentile_clip must satisfy 0 <= low < high <= 100; got {(lo, hi)}")


def expected_bit_length(image_size: int, block_size: int) -> int:
    """Number of blocks (= bits) an ``image_size x image_size`` haar LL subband
    yields at the given ``block_size``. Raises if it does not divide evenly."""
    ll_size = image_size // 2
    if ll_size % block_size != 0:
        raise ValueError(
            f"block_size={block_size} does not evenly divide the LL subband size "
            f"({ll_size} for a {image_size}x{image_size} image)"
        )
    grid = ll_size // block_size
    return grid * grid


# ---------------------------------------------------------------------------
# Colour-space helpers (self-contained; mirrors the frozen baseline's proven
# RGB<->YCrCb round trip so Phase 11 does not import embed.py's private API)
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
# Block grid
# ---------------------------------------------------------------------------

def block_grid(ll_shape: tuple[int, int], block_size: int) -> tuple[int, int]:
    """Return ``(rows, cols)`` of non-overlapping ``block_size`` blocks tiling
    ``ll_shape`` exactly. Raises ``ValueError`` if it does not divide evenly."""
    h, w = ll_shape
    if h % block_size != 0 or w % block_size != 0:
        raise ValueError(
            f"block_size={block_size} does not evenly divide LL subband shape {ll_shape}"
        )
    return h // block_size, w // block_size


# ---------------------------------------------------------------------------
# Texture / edge-complexity scoring
# ---------------------------------------------------------------------------

def texture_map(y_full_res: np.ndarray, rows: int, cols: int, block_size: int) -> np.ndarray:
    """Per-block Sobel edge-energy score, aligned to the LL block grid.

    The Sobel gradient magnitude is computed on the full-resolution Y channel,
    then 2x2-average-pooled (matching a single-level Haar DWT's implicit
    downsampling) so the resulting map has exactly the LL subband's shape
    before being block-averaged into a ``(rows, cols)`` score array. Higher
    score = more edges/texture in that block's source region.
    """
    y = np.asarray(y_full_res, dtype=np.float64)
    gx = cv2.Sobel(y, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    h, w = grad.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"texture_map requires an even-sized image; got {grad.shape}")
    grad_ll = 0.25 * (grad[0::2, 0::2] + grad[1::2, 0::2] + grad[0::2, 1::2] + grad[1::2, 1::2])

    expected = (rows * block_size, cols * block_size)
    if grad_ll.shape != expected:
        raise ValueError(
            f"downsampled edge map shape {grad_ll.shape} does not match the block "
            f"grid {expected}; image size / block_size mismatch"
        )

    scores = np.empty((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            block = grad_ll[r * block_size:(r + 1) * block_size, c * block_size:(c + 1) * block_size]
            scores[r, c] = float(block.mean())
    return scores


def _alpha_map(scores: np.ndarray, config: AdaptiveEmbedConfig) -> np.ndarray:
    """Per-block alpha, either constant (``fixed``) or content-aware and
    mean-preserving (``adaptive``, see :class:`AdaptiveEmbedConfig`)."""
    if config.alpha_mode == "fixed":
        return np.full(scores.shape, config.alpha, dtype=np.float64)

    lo_pct, hi_pct = config.texture_percentile_clip
    lo, hi = np.percentile(scores, [lo_pct, hi_pct])
    if hi <= lo:
        # Degenerate (near-uniform) texture: fall back to a constant alpha map,
        # equivalent to fixed mode for this one image.
        return np.full(scores.shape, config.alpha, dtype=np.float64)

    clipped = np.clip(scores, lo, hi)
    normalised = (clipped - lo) / (hi - lo)  # in [0, 1]
    weights = config.alpha_low_mult + normalised * (config.alpha_high_mult - config.alpha_low_mult)
    weights = weights / weights.mean()  # mean-preserving: mean(alpha_map) == config.alpha
    return config.alpha * weights


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveEmbedResult:
    """Everything produced by one Phase 11 embedding operation."""

    watermarked_image: np.ndarray
    original_image: np.ndarray
    bits: list[int]
    config: AdaptiveEmbedConfig
    grid: tuple[int, int]
    texture_scores: np.ndarray
    alpha_map: np.ndarray
    original_sv0: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    watermarked_sv0: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


def embed_adaptive(image: np.ndarray, bits: list[int], config: AdaptiveEmbedConfig) -> AdaptiveEmbedResult:
    """Embed ``bits`` using block-SVD modulation with fixed or adaptive per-block alpha."""
    rgb = _to_uint8_rgb(image)
    if any(b not in (0, 1) for b in bits):
        raise ValueError("bits must contain only 0 and 1")
    if len(bits) != config.bit_length:
        raise ValueError(f"expected {config.bit_length} bits from config; got {len(bits)}")

    y, cr, cb = _rgb_to_ycrcb(rgb)
    coeffs = decompose_2d(y, wavelet=config.wavelet, mode=config.mode)
    ll = coeffs.ll

    rows, cols = block_grid(ll.shape, config.block_size)
    if rows * cols != config.bit_length:
        raise ValueError(
            f"block grid produces {rows * cols} blocks ({rows}x{cols}) but "
            f"bit_length={config.bit_length}; they must match (no overflow in Phase 11)"
        )

    scores = texture_map(y, rows, cols, config.block_size)
    alphas = _alpha_map(scores, config)

    bs = config.block_size
    ll_mod = ll.copy()
    original_sv0 = np.zeros((rows, cols))
    watermarked_sv0 = np.zeros((rows, cols))

    bit_index = 0
    for r in range(rows):
        for c in range(cols):
            block = ll[r * bs:(r + 1) * bs, c * bs:(c + 1) * bs]
            svd_block = decompose(block)
            original_sv0[r, c] = svd_block.S[0]

            bit = bits[bit_index]
            bit_index += 1
            a = float(alphas[r, c])
            s_mod = svd_block.S.copy()
            s_mod[0] *= 1.0 + a * (2 * bit - 1)
            watermarked_sv0[r, c] = s_mod[0]

            recon_block = reconstruct(
                SVDComponents(U=svd_block.U, S=s_mod, Vt=svd_block.Vt, source_shape=svd_block.source_shape)
            )
            ll_mod[r * bs:(r + 1) * bs, c * bs:(c + 1) * bs] = recon_block

    y_modified = reconstruct_2d(replace(coeffs, ll=ll_mod))
    watermarked = _ycrcb_to_rgb(y_modified, cr, cb)

    return AdaptiveEmbedResult(
        watermarked_image=watermarked,
        original_image=rgb,
        bits=list(bits),
        config=config,
        grid=(rows, cols),
        texture_scores=scores,
        alpha_map=alphas,
        original_sv0=original_sv0,
        watermarked_sv0=watermarked_sv0,
    )


# ---------------------------------------------------------------------------
# Non-blind extraction
# ---------------------------------------------------------------------------

def _ll_subband(image: np.ndarray, config: AdaptiveEmbedConfig) -> np.ndarray:
    y, _, _ = _rgb_to_ycrcb(_to_uint8_rgb(image))
    return decompose_2d(y, wavelet=config.wavelet, mode=config.mode).ll


def extract_adaptive(
    watermarked_image: np.ndarray, original_image: np.ndarray, config: AdaptiveEmbedConfig
) -> list[int]:
    """Non-blind decode: compare watermarked vs original per-block leading
    singular value. Decision rule: ``bit = 1 if S_w[0] > S_o[0] else 0`` -
    the same rule ``extract_traditional`` uses, applied per block."""
    ll_w = _ll_subband(watermarked_image, config)
    ll_o = _ll_subband(original_image, config)
    if ll_w.shape != ll_o.shape:
        raise ValueError(f"shape mismatch between watermarked and original LL: {ll_w.shape} vs {ll_o.shape}")

    rows, cols = block_grid(ll_o.shape, config.block_size)
    if rows * cols != config.bit_length:
        raise ValueError(
            f"block grid produces {rows * cols} blocks ({rows}x{cols}) but "
            f"bit_length={config.bit_length}"
        )

    bs = config.block_size
    bits: list[int] = []
    for r in range(rows):
        for c in range(cols):
            sw0 = get_singular_values(ll_w[r * bs:(r + 1) * bs, c * bs:(c + 1) * bs])[0]
            so0 = get_singular_values(ll_o[r * bs:(r + 1) * bs, c * bs:(c + 1) * bs])[0]
            bits.append(1 if sw0 > so0 else 0)
    return bits
