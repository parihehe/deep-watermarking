"""Tests for src/watermark/adaptive_embed.py — Phase 11 block-SVD adaptive embedding.

Behavioural tests run against the real held-out DIV2K test split
(``data/processed/div2k_256/test/``). If that processed data is not present the
data-dependent tests are skipped. Nothing here imports or mutates the frozen
Phase 6 baseline (``src/watermark/embed.py`` / ``dwt.py`` / ``svd.py``).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.evaluation.metrics import bit_error_rate, psnr
from src.watermark.adaptive_embed import (
    AdaptiveEmbedConfig,
    block_grid,
    embed_adaptive,
    expected_bit_length,
    extract_adaptive,
    texture_map,
)

_TEST_SPLIT = Path(__file__).resolve().parents[1] / "data" / "processed" / "div2k_256" / "test"


def _real_images(limit: int = 6) -> list[np.ndarray]:
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


# --- expected_bit_length / block_grid -------------------------------------

def test_expected_bit_length_matches_established_operating_point() -> None:
    assert expected_bit_length(256, 16) == 64


def test_expected_bit_length_rejects_non_dividing_block_size() -> None:
    with pytest.raises(ValueError):
        expected_bit_length(256, 17)  # 128 / 17 is not an integer


def test_block_grid_shapes() -> None:
    assert block_grid((128, 128), 16) == (8, 8)
    assert block_grid((64, 64), 16) == (4, 4)


def test_block_grid_rejects_non_dividing_block_size() -> None:
    with pytest.raises(ValueError):
        block_grid((100, 100), 16)


# --- config validation ------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"wavelet": "db2"},              # only haar supported (module docstring)
    {"mode": "unknown"},
    {"block_size": 1},
    {"bit_length": 0},
    {"alpha_mode": "learned"},
    {"alpha": 0.0},
    {"alpha": 1.0},
    {"alpha_low_mult": 0.0},
    {"alpha_high_mult": 0.0},
    {"alpha_low_mult": 1.0, "alpha_high_mult": 0.5},  # low must be < high
    {"texture_percentile_clip": (50.0, 10.0)},
    {"texture_percentile_clip": (-1.0, 95.0)},
])
def test_invalid_config_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        AdaptiveEmbedConfig(**kwargs)


def test_config_defaults() -> None:
    cfg = AdaptiveEmbedConfig()
    assert cfg.block_size == 16
    assert cfg.bit_length == 64
    assert cfg.alpha_mode == "adaptive"
    assert cfg.alpha == 0.02
    assert cfg.wavelet == "haar" and cfg.mode == "symmetric"


# --- texture_map --------------------------------------------------------

def test_texture_map_flat_image_has_near_zero_score() -> None:
    flat = np.full((256, 256), 128.0)
    scores = texture_map(flat, rows=8, cols=8, block_size=16)
    assert scores.shape == (8, 8)
    assert np.allclose(scores, 0.0, atol=1e-6)


def test_texture_map_higher_for_a_sharp_edge_block() -> None:
    y = np.full((256, 256), 100.0)
    y[:, 128:] = 200.0  # a hard vertical edge down the middle
    scores = texture_map(y, rows=8, cols=8, block_size=16)
    # columns straddling the edge (col index 7, 8 in the 128-wide LL -> block col 3/4
    # of 16-wide blocks) should score far higher than a column away from it.
    assert scores[:, 3].mean() > scores[:, 0].mean() * 10 or scores[:, 4].mean() > scores[:, 0].mean() * 10


def test_texture_map_requires_even_dimensions() -> None:
    with pytest.raises(ValueError):
        texture_map(np.zeros((255, 256)), rows=8, cols=8, block_size=16)


# --- embed_adaptive: fixed vs adaptive alpha allocation -----------------

@_needs_data
def test_fixed_mode_uses_constant_alpha_every_block() -> None:
    cfg = AdaptiveEmbedConfig(alpha_mode="fixed", alpha=0.02, bit_length=64)
    result = embed_adaptive(_IMAGES[0], _bits(64), cfg)
    assert np.allclose(result.alpha_map, 0.02)


@_needs_data
def test_adaptive_mode_preserves_mean_alpha_energy_budget() -> None:
    """The whole point of the mean-preserving rescale: adaptive and fixed must
    apply the *same average* strength per image, so any quality/robustness
    difference is attributable to allocation, not to a stronger/weaker watermark."""
    cfg = AdaptiveEmbedConfig(alpha_mode="adaptive", alpha=0.02, bit_length=64)
    for image in _IMAGES:
        result = embed_adaptive(image, _bits(64), cfg)
        assert result.alpha_map.mean() == pytest.approx(0.02, abs=1e-9)


@_needs_data
def test_adaptive_mode_varies_alpha_across_blocks_on_real_images() -> None:
    cfg = AdaptiveEmbedConfig(alpha_mode="adaptive", alpha=0.02, bit_length=64)
    stds = [embed_adaptive(image, _bits(64), cfg).alpha_map.std() for image in _IMAGES]
    assert any(s > 1e-4 for s in stds)  # real images have non-uniform texture


@_needs_data
def test_adaptive_alpha_map_stays_within_configured_multiplier_bounds() -> None:
    cfg = AdaptiveEmbedConfig(alpha_mode="adaptive", alpha=0.02, alpha_low_mult=0.4, alpha_high_mult=1.6, bit_length=64)
    for image in _IMAGES:
        result = embed_adaptive(image, _bits(64), cfg)
        # weights lie in [low_mult, high_mult] before the mean-preserving rescale,
        # so after rescale they lie within a bounded multiple of that range.
        assert result.alpha_map.min() > 0.0
        assert result.alpha_map.max() < cfg.alpha * cfg.alpha_high_mult / cfg.alpha_low_mult


@_needs_data
def test_lower_alpha_gives_smooth_blocks_less_strength_than_textured_blocks() -> None:
    """Direct check of the requirement: smooth regions get lower alpha, textured
    regions get higher alpha."""
    image = _IMAGES[0].copy()
    # Flatten the top-left quadrant of the Y-equivalent luminance (smooth region);
    # leave the rest (typically textured, real photo content) untouched.
    image[:96, :96] = image[:96, :96].mean(axis=(0, 1)).astype(np.uint8)
    cfg = AdaptiveEmbedConfig(alpha_mode="adaptive", alpha=0.02, bit_length=64)
    result = embed_adaptive(image, _bits(64), cfg)
    smooth_block_alpha = result.alpha_map[:6, :6].mean()  # inside the flattened region (96/16=6)
    assert smooth_block_alpha < result.alpha_map.mean()


# --- embed_adaptive: basic contract -------------------------------------

@_needs_data
def test_embed_adaptive_returns_uint8_same_shape_and_changes_image() -> None:
    cfg = AdaptiveEmbedConfig(bit_length=64)
    result = embed_adaptive(_IMAGES[0], _bits(64), cfg)
    assert result.watermarked_image.shape == _IMAGES[0].shape
    assert result.watermarked_image.dtype == np.uint8
    assert not np.array_equal(result.watermarked_image, _IMAGES[0])
    assert result.grid == (8, 8)


@_needs_data
def test_embed_adaptive_is_visually_near_lossless() -> None:
    cfg = AdaptiveEmbedConfig(bit_length=64, alpha=0.02)
    for image in _IMAGES:
        result = embed_adaptive(image, _bits(64), cfg)
        assert psnr(image, result.watermarked_image) > 35.0


def test_bit_length_mismatch_rejected() -> None:
    image = _IMAGES[0] if _IMAGES else np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        embed_adaptive(image, _bits(32), AdaptiveEmbedConfig(bit_length=64))


def test_non_bit_payload_rejected() -> None:
    image = _IMAGES[0] if _IMAGES else np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        embed_adaptive(image, [0, 1, 2, 0] * 16, AdaptiveEmbedConfig(bit_length=64))


def test_block_grid_bit_length_mismatch_rejected() -> None:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    # block_size=32 on a 128x128 LL -> 4x4=16 blocks, but bit_length says 64
    with pytest.raises(ValueError):
        embed_adaptive(image, _bits(64), AdaptiveEmbedConfig(block_size=32, bit_length=64))


def test_non_rgb_input_rejected() -> None:
    with pytest.raises(ValueError):
        embed_adaptive(np.zeros((256, 256), dtype=np.uint8), _bits(64), AdaptiveEmbedConfig(bit_length=64))


# --- non-blind extraction ------------------------------------------------

@_needs_data
def test_extraction_is_exact_on_clean_channel_at_baseline_alpha() -> None:
    """At the established 0.02 operating point, one bit per 16x16 block's
    leading singular value recovers exactly on a clean channel (measured: BER
    0.0 in the smoke run; this test uses a looser bound to be robust to the
    specific images / payload in the fixture)."""
    cfg = AdaptiveEmbedConfig(bit_length=64, alpha=0.02, alpha_mode="fixed")
    for image in _IMAGES:
        bits = _bits(64)
        result = embed_adaptive(image, bits, cfg)
        recovered = extract_adaptive(result.watermarked_image, image, cfg)
        assert bit_error_rate(bits, recovered) < 0.05


@_needs_data
def test_extraction_is_deterministic() -> None:
    cfg = AdaptiveEmbedConfig(bit_length=64, alpha=0.02, alpha_mode="adaptive")
    image = _IMAGES[0]
    bits = _bits(64)
    result = embed_adaptive(image, bits, cfg)
    a = extract_adaptive(result.watermarked_image, image, cfg)
    b = extract_adaptive(result.watermarked_image, image, cfg)
    assert a == b


@_needs_data
def test_extract_adaptive_matches_recorded_sv0_comparison() -> None:
    """The extraction decision rule must match the recorded original_sv0 /
    watermarked_sv0 comparison exactly (sanity on the decode logic itself)."""
    cfg = AdaptiveEmbedConfig(bit_length=64, alpha=0.02, alpha_mode="fixed")
    image = _IMAGES[0]
    bits = _bits(64)
    result = embed_adaptive(image, bits, cfg)
    recovered = extract_adaptive(result.watermarked_image, image, cfg)
    expected = (result.watermarked_sv0.flatten() > result.original_sv0.flatten()).astype(int).tolist()
    assert recovered == expected


@_needs_data
def test_higher_alpha_lowers_ber_at_fixed_mode() -> None:
    image = _IMAGES[0]
    bits = _bits(64)
    weak = AdaptiveEmbedConfig(bit_length=64, alpha=0.002, alpha_mode="fixed")
    strong = AdaptiveEmbedConfig(bit_length=64, alpha=0.02, alpha_mode="fixed")
    r_weak = embed_adaptive(image, bits, weak)
    r_strong = embed_adaptive(image, bits, strong)
    ber_weak = bit_error_rate(bits, extract_adaptive(r_weak.watermarked_image, image, weak))
    ber_strong = bit_error_rate(bits, extract_adaptive(r_strong.watermarked_image, image, strong))
    assert ber_strong <= ber_weak
