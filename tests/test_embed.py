"""Tests for src/watermark/embed.py — Phase 6 frozen DWT-SVD baseline.

Behavioural tests run against the real held-out DIV2K test split
(``data/processed/div2k_256/test/``), which is what the baseline is designed for.
If that processed data is not present the data-dependent tests are skipped.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.evaluation.metrics import bit_error_rate, psnr
from src.watermark.embed import (
    BASELINE_ALPHAS,
    EmbedConfig,
    _rgb_to_ycrcb,
    _ycrcb_to_rgb,
    compute_residual,
    embed,
    extract_traditional,
    subband_capacity,
)

_TEST_SPLIT = Path(__file__).resolve().parents[1] / "data" / "processed" / "div2k_256" / "test"


def _real_images(limit: int = 8) -> list[np.ndarray]:
    if not _TEST_SPLIT.is_dir():
        return []
    images = []
    for path in sorted(_TEST_SPLIT.glob("*.png"))[:limit]:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is not None:
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return images


_IMAGES = _real_images()
_needs_data = pytest.mark.skipif(not _IMAGES, reason="processed DIV2K test split not available")


def _bits(n: int, seed: int = 1) -> list[int]:
    return np.random.default_rng(seed).integers(0, 2, size=n).tolist()


# --- colour round trip ---------------------------------------------------

@_needs_data
def test_ycrcb_round_trip_is_high_fidelity() -> None:
    """The RGB<->YCrCb round trip alone must not visibly degrade the image."""
    for image in _IMAGES:
        y, cr, cb = _rgb_to_ycrcb(image)
        restored = _ycrcb_to_rgb(y, cr, cb)
        assert psnr(image, restored) > 45.0


# --- config validation -------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"subband": "XX"},
    {"extra_subbands": ("LL",)},          # duplicate of primary
    {"alpha": 0.0},
    {"alpha": 1.0},
    {"alpha": -0.01},
    {"bit_length": 0},
    {"start_sv_index": -1},
])
def test_invalid_config_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        EmbedConfig(**kwargs)


# --- embedding behaviour ---------------------------------------------

@_needs_data
def test_embed_returns_uint8_same_shape_and_changes_image() -> None:
    image = _IMAGES[0]
    result = embed(image, _bits(32), EmbedConfig(alpha=0.01, bit_length=32))
    assert result.watermarked_image.shape == image.shape
    assert result.watermarked_image.dtype == np.uint8
    assert not np.array_equal(result.watermarked_image, image)


@_needs_data
def test_embed_is_visually_near_lossless() -> None:
    for image in _IMAGES:
        result = embed(image, _bits(64), EmbedConfig(alpha=0.015, bit_length=64))
        assert psnr(image, result.watermarked_image) > 40.0


def test_bit_length_mismatch_rejected() -> None:
    image = _IMAGES[0] if _IMAGES else np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        embed(image, _bits(32), EmbedConfig(bit_length=16))


def test_non_bit_payload_rejected() -> None:
    image = _IMAGES[0] if _IMAGES else np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        embed(image, [0, 1, 2, 0], EmbedConfig(bit_length=4))


def test_payload_exceeding_single_subband_capacity_rejected() -> None:
    image = _IMAGES[0] if _IMAGES else np.zeros((256, 256, 3), dtype=np.uint8)
    # 256x256 -> LL is 128x128 -> 128 singular values; 129 must not fit LL alone.
    with pytest.raises(ValueError, match="singular-value"):
        embed(image, _bits(129), EmbedConfig(bit_length=129))


def test_non_rgb_input_rejected() -> None:
    with pytest.raises(ValueError):
        embed(np.zeros((64, 64), dtype=np.uint8), _bits(8), EmbedConfig(bit_length=8))


# --- non-blind extraction --------------------------------------------

@_needs_data
@pytest.mark.parametrize("alpha", BASELINE_ALPHAS)
def test_non_blind_extraction_is_exact_for_small_payload(alpha: float) -> None:
    """An 8-bit payload uses only the dominant, well-separated singular values and
    is recovered without error on every un-attacked image (measured over the
    Phase 6 experiment: BER = 0.0000 at all three baseline alphas)."""
    cfg = EmbedConfig(alpha=alpha, bit_length=8)
    for image in _IMAGES:
        bits = _bits(8)
        result = embed(image, bits, cfg)
        recovered = extract_traditional(result.watermarked_image, image, cfg)
        assert bit_error_rate(bits, recovered) == 0.0


@_needs_data
@pytest.mark.parametrize(("n_bits", "max_mean_ber"), [(32, 0.05), (64, 0.10), (128, 0.20)])
def test_non_blind_extraction_ber_grows_with_payload(n_bits: int, max_mean_ber: float) -> None:
    """As the payload reaches deeper into the singular spectrum the trailing
    singular values become fragile through the IDWT + uint8 round trip, so BER
    rises with payload but stays far below chance. Thresholds bound the values
    measured in the Phase 6 experiment (results/phase6_baseline/)."""
    cfg = EmbedConfig(alpha=0.010, bit_length=n_bits)
    bers = []
    for image in _IMAGES:
        bits = _bits(n_bits)
        result = embed(image, bits, cfg)
        recovered = extract_traditional(result.watermarked_image, image, cfg)
        bers.append(bit_error_rate(bits, recovered))
    assert float(np.mean(bers)) < max_mean_ber


@_needs_data
def test_multi_subband_payload_256_bits() -> None:
    """256 bits does not fit the 128-value LL subband alone; the documented
    overflow order ("LL", "HL") provides 256 slots and stays above chance."""
    cfg = EmbedConfig(alpha=0.010, bit_length=256, extra_subbands=("HL",))
    bers = []
    for image in _IMAGES:
        bits = _bits(256)
        result = embed(image, bits, cfg)
        assert [name for name, _, _ in result.windows] == ["LL", "HL"]
        recovered = extract_traditional(result.watermarked_image, image, cfg)
        assert len(recovered) == 256
        bers.append(bit_error_rate(bits, recovered))
    assert float(np.mean(bers)) < 0.25


@_needs_data
def test_higher_alpha_lowers_psnr() -> None:
    image = _IMAGES[0]
    bits = _bits(64)
    weak = embed(image, bits, EmbedConfig(alpha=0.005, bit_length=64))
    strong = embed(image, bits, EmbedConfig(alpha=0.015, bit_length=64))
    assert psnr(image, weak.watermarked_image) > psnr(image, strong.watermarked_image)


# --- helpers ---------------------------------------------------------

def test_subband_capacity_matches_singular_value_count() -> None:
    assert subband_capacity((256, 256), EmbedConfig(bit_length=8)) == 128
    assert subband_capacity((128, 128), EmbedConfig(bit_length=8)) == 64
    assert subband_capacity((256, 256), EmbedConfig(bit_length=8, start_sv_index=100)) == 28
    assert subband_capacity((256, 256), EmbedConfig(bit_length=8, extra_subbands=("HL",))) == 256


@_needs_data
def test_residual_is_uint8_and_nonzero_where_image_changed() -> None:
    image = _IMAGES[0]
    result = embed(image, _bits(64), EmbedConfig(alpha=0.015, bit_length=64))
    residual = compute_residual(image, result.watermarked_image)
    assert residual.dtype == np.uint8
    assert residual.max() > 0
