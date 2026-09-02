"""Tests for src/watermark/capacity_embed.py — Phase 12 high-capacity, multi-level
DWT-SVD embedding.

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
from src.watermark.capacity_embed import (
    CapacityEmbedConfig,
    InsufficientCapacityError,
    embed_capacity,
    extract_capacity,
    full_subband_plan,
    plan_windows,
    subband_side_length,
    theoretical_capacity,
    theoretical_capacity_limit,
    total_capacity,
)
from src.watermark.embed import EmbedConfig, embed, extract_traditional

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


# --- subband_side_length / capacity arithmetic --------------------------

def test_subband_side_length_single_level() -> None:
    assert subband_side_length(256, 1) == 128
    assert subband_side_length(256, 2) == 64
    assert subband_side_length(256, 3) == 32


def test_subband_side_length_rejects_odd_halving() -> None:
    with pytest.raises(ValueError):
        subband_side_length(255, 1)


def test_full_subband_plan_level_1_is_all_four_bands() -> None:
    plan = full_subband_plan(1)
    assert plan == ((1, "LL"), (1, "LH"), (1, "HL"), (1, "HH"))


def test_full_subband_plan_level_2_excludes_intermediate_ll() -> None:
    plan = full_subband_plan(2)
    assert (1, "LL") not in plan
    assert (2, "LL") in plan
    assert plan == ((1, "LH"), (1, "HL"), (1, "HH"), (2, "LL"), (2, "LH"), (2, "HL"), (2, "HH"))


@pytest.mark.parametrize(("level", "expected"), [(1, 512), (2, 640), (3, 704), (4, 736)])
def test_theoretical_capacity_matches_closed_form(level: int, expected: int) -> None:
    assert theoretical_capacity(256, level) == expected


def test_theoretical_capacity_limit_is_three_n() -> None:
    assert theoretical_capacity_limit(256) == 768


def test_theoretical_capacity_approaches_limit_monotonically() -> None:
    values = [theoretical_capacity(256, level) for level in range(1, 8)]
    assert values == sorted(values)  # monotone increasing
    assert all(v < theoretical_capacity_limit(256) for v in values)


def test_theoretical_capacity_never_reaches_1024() -> None:
    """The central Phase 12 finding: no number of DWT levels reaches 1024 bits
    on a 256x256 image with this one-bit-per-singular-value scheme. 256 = 2^8,
    so level 8 (a 1x1 final subband) is the deepest level the image supports;
    even that degenerate extreme falls short of 1024."""
    for level in range(1, 9):
        assert theoretical_capacity(256, level) < 1024
    assert theoretical_capacity_limit(256) < 1024


def test_subband_side_length_rejects_levels_deeper_than_the_image_supports() -> None:
    """256 = 2^8 halves exactly 8 times to reach a 1x1 subband; a 9th level
    would require halving 1, which is impossible - a hard, explicit stop, not
    a silent/degenerate result."""
    assert subband_side_length(256, 8) == 1
    with pytest.raises(ValueError):
        subband_side_length(256, 9)


# --- config validation ----------------------------------------------------

def test_config_defaults() -> None:
    cfg = CapacityEmbedConfig()
    assert cfg.dwt_levels == 1
    assert cfg.subband_plan == ((1, "LL"),)
    assert cfg.bit_length == 128


@pytest.mark.parametrize("kwargs", [
    {"wavelet": "db2"},
    {"mode": "unknown"},
    {"dwt_levels": 0},
    {"subband_plan": ()},
    {"subband_plan": ((1, "XX"),)},
    {"subband_plan": ((2, "LL"),), "dwt_levels": 1},           # level out of range
    {"subband_plan": ((1, "LL"), (2, "LL")), "dwt_levels": 3},  # intermediate LL invalid
    {"subband_plan": ((1, "LL"), (1, "LL"))},                   # duplicate
    {"alpha": 0.0},
    {"alpha": 1.0},
    {"bit_length": 0},
    {"start_sv_index": -1},
])
def test_invalid_config_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CapacityEmbedConfig(**kwargs)


def test_intermediate_ll_at_final_level_is_valid() -> None:
    # LL at the FINAL level (level == dwt_levels) is a valid embedding target.
    cfg = CapacityEmbedConfig(dwt_levels=2, subband_plan=((2, "LL"),), bit_length=64)
    assert cfg.subband_plan == ((2, "LL"),)


# --- plan_windows / total_capacity ----------------------------------------

def test_plan_windows_single_subband() -> None:
    cfg = CapacityEmbedConfig(dwt_levels=1, subband_plan=((1, "LL"),), bit_length=100)
    windows = plan_windows(cfg, 256)
    assert windows == [(1, "LL", 0, 100)]


def test_plan_windows_splits_across_subbands_in_order() -> None:
    cfg = CapacityEmbedConfig(dwt_levels=1, subband_plan=((1, "LL"), (1, "HL")), bit_length=200)
    windows = plan_windows(cfg, 256)
    assert windows == [(1, "LL", 0, 128), (1, "HL", 0, 72)]


def test_plan_windows_raises_insufficient_capacity() -> None:
    cfg = CapacityEmbedConfig(dwt_levels=1, subband_plan=((1, "LL"),), bit_length=129)
    with pytest.raises(InsufficientCapacityError):
        plan_windows(cfg, 256)


def test_total_capacity_matches_manual_sum() -> None:
    cfg = CapacityEmbedConfig(dwt_levels=1, subband_plan=((1, "LL"), (1, "LH"), (1, "HL"), (1, "HH")), bit_length=1)
    assert total_capacity(cfg, 256) == 512


def test_total_capacity_two_level_matches_theoretical() -> None:
    cfg = CapacityEmbedConfig(dwt_levels=2, subband_plan=full_subband_plan(2), bit_length=1)
    assert total_capacity(cfg, 256) == theoretical_capacity(256, 2) == 640


# --- embed_capacity / extract_capacity: contract --------------------------

def test_bit_length_mismatch_rejected() -> None:
    image = _IMAGES[0] if _IMAGES else np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        embed_capacity(image, _bits(64), CapacityEmbedConfig(bit_length=128))


def test_non_bit_payload_rejected() -> None:
    image = _IMAGES[0] if _IMAGES else np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        embed_capacity(image, [0, 1, 2] * 43, CapacityEmbedConfig(bit_length=129))


def test_non_rgb_input_rejected() -> None:
    with pytest.raises(ValueError):
        embed_capacity(np.zeros((256, 256), dtype=np.uint8), _bits(128), CapacityEmbedConfig(bit_length=128))


def test_non_square_image_rejected() -> None:
    with pytest.raises(ValueError):
        embed_capacity(np.zeros((256, 128, 3), dtype=np.uint8), _bits(128), CapacityEmbedConfig(bit_length=128))


# --- insufficient-capacity handling (explicit, no silent forcing) ---------

@_needs_data
def test_embed_capacity_raises_on_insufficient_capacity_1024_single_level() -> None:
    cfg = CapacityEmbedConfig(
        dwt_levels=1, subband_plan=((1, "LL"), (1, "LH"), (1, "HL"), (1, "HH")), bit_length=1024
    )
    with pytest.raises(InsufficientCapacityError):
        embed_capacity(_IMAGES[0], _bits(1024), cfg)


@_needs_data
def test_embed_capacity_raises_on_insufficient_capacity_1024_two_level() -> None:
    cfg = CapacityEmbedConfig(dwt_levels=2, subband_plan=full_subband_plan(2), bit_length=1024)
    with pytest.raises(InsufficientCapacityError):
        embed_capacity(_IMAGES[0], _bits(1024), cfg)


@_needs_data
def test_embed_capacity_raises_on_insufficient_capacity_1024_three_level() -> None:
    cfg = CapacityEmbedConfig(dwt_levels=3, subband_plan=full_subband_plan(3), bit_length=1024)
    with pytest.raises(InsufficientCapacityError):
        embed_capacity(_IMAGES[0], _bits(1024), cfg)


# --- embed_capacity / extract_capacity: single-level behaviour ------------

@_needs_data
def test_embed_capacity_returns_uint8_same_shape_and_changes_image() -> None:
    cfg = CapacityEmbedConfig(dwt_levels=1, subband_plan=((1, "LL"),), bit_length=128)
    result = embed_capacity(_IMAGES[0], _bits(128), cfg)
    assert result.watermarked_image.shape == _IMAGES[0].shape
    assert result.watermarked_image.dtype == np.uint8
    assert not np.array_equal(result.watermarked_image, _IMAGES[0])


@_needs_data
def test_embed_capacity_is_visually_near_lossless() -> None:
    cfg = CapacityEmbedConfig(dwt_levels=1, subband_plan=((1, "LL"),), bit_length=128, alpha=0.02)
    for image in _IMAGES:
        result = embed_capacity(image, _bits(128), cfg)
        assert psnr(image, result.watermarked_image) > 30.0


@_needs_data
def test_128_bit_single_level_matches_frozen_baseline_exactly() -> None:
    """128 bits, single-level, LL-only must reproduce the frozen baseline's
    own embed()/extract_traditional() bit-for-bit (same formula, same domain)."""
    image = _IMAGES[0]
    bits = _bits(128)
    cap_cfg = CapacityEmbedConfig(dwt_levels=1, subband_plan=((1, "LL"),), bit_length=128, alpha=0.02)
    frozen_cfg = EmbedConfig(alpha=0.02, bit_length=128)

    cap_result = embed_capacity(image, bits, cap_cfg)
    frozen_result = embed(image, bits, frozen_cfg)
    assert np.array_equal(cap_result.watermarked_image, frozen_result.watermarked_image)

    cap_rec = extract_capacity(cap_result.watermarked_image, image, cap_cfg)
    frozen_rec = extract_traditional(frozen_result.watermarked_image, image, frozen_cfg)
    assert cap_rec == frozen_rec


@_needs_data
def test_512_bit_single_level_all_four_subbands() -> None:
    cfg = CapacityEmbedConfig(
        dwt_levels=1, subband_plan=((1, "LL"), (1, "LH"), (1, "HL"), (1, "HH")), bit_length=512, alpha=0.02
    )
    for image in _IMAGES[:3]:
        bits = _bits(512)
        result = embed_capacity(image, bits, cfg)
        recovered = extract_capacity(result.watermarked_image, image, cfg)
        assert len(recovered) == 512
        assert bit_error_rate(bits, recovered) < 0.5  # better than chance


# --- multi-level DWT behaviour ---------------------------------------------

@_needs_data
def test_two_level_embed_extract_round_trip() -> None:
    cfg = CapacityEmbedConfig(dwt_levels=2, subband_plan=full_subband_plan(2), bit_length=512, alpha=0.02)
    for image in _IMAGES[:3]:
        bits = _bits(512)
        result = embed_capacity(image, bits, cfg)
        recovered = extract_capacity(result.watermarked_image, image, cfg)
        assert len(recovered) == 512
        assert psnr(image, result.watermarked_image) > 30.0
        assert bit_error_rate(bits, recovered) < 0.5


@_needs_data
def test_three_level_embed_extract_round_trip() -> None:
    cfg = CapacityEmbedConfig(dwt_levels=3, subband_plan=full_subband_plan(3), bit_length=512, alpha=0.02)
    image = _IMAGES[0]
    bits = _bits(512)
    result = embed_capacity(image, bits, cfg)
    recovered = extract_capacity(result.watermarked_image, image, cfg)
    assert len(recovered) == 512
    assert psnr(image, result.watermarked_image) > 25.0


@_needs_data
def test_deeper_final_level_ll_is_directly_embeddable() -> None:
    """The deepest level's LL (unlike intermediate levels') is a valid,
    directly embeddable target."""
    cfg = CapacityEmbedConfig(dwt_levels=2, subband_plan=((2, "LL"),), bit_length=64, alpha=0.02)
    image = _IMAGES[0]
    bits = _bits(64)
    result = embed_capacity(image, bits, cfg)
    recovered = extract_capacity(result.watermarked_image, image, cfg)
    assert len(recovered) == 64


# --- determinism ------------------------------------------------------------

@_needs_data
def test_embed_capacity_is_deterministic() -> None:
    cfg = CapacityEmbedConfig(dwt_levels=2, subband_plan=full_subband_plan(2), bit_length=256, alpha=0.02)
    image = _IMAGES[0]
    bits = _bits(256)
    a = embed_capacity(image, bits, cfg)
    b = embed_capacity(image, bits, cfg)
    assert np.array_equal(a.watermarked_image, b.watermarked_image)


@_needs_data
def test_extract_capacity_is_deterministic() -> None:
    cfg = CapacityEmbedConfig(dwt_levels=1, subband_plan=((1, "LL"),), bit_length=128, alpha=0.02)
    image = _IMAGES[0]
    bits = _bits(128)
    result = embed_capacity(image, bits, cfg)
    a = extract_capacity(result.watermarked_image, image, cfg)
    b = extract_capacity(result.watermarked_image, image, cfg)
    assert a == b


# --- alpha sensitivity: monotonic for shallow payloads, NOT for full-depth --

@_needs_data
def test_higher_alpha_lowers_mean_ber_for_a_shallow_payload() -> None:
    """Sanity check on the modulation direction, using a shallow 16-bit
    payload (only the leading, well-separated singular values) - the regime
    the frozen Phase 6 baseline already validated as reliably monotonic.
    Averaged over the image fixture, since per-image BER has too much sampling
    noise to be monotonic on a single image (see the *_full_depth test below)."""
    weak = CapacityEmbedConfig(dwt_levels=1, subband_plan=((1, "LL"),), bit_length=16, alpha=0.001)
    strong = CapacityEmbedConfig(dwt_levels=1, subband_plan=((1, "LL"),), bit_length=16, alpha=0.003)
    weak_bers, strong_bers = [], []
    for i, image in enumerate(_IMAGES):
        bits = _bits(16, seed=200 + i)
        r_weak = embed_capacity(image, bits, weak)
        r_strong = embed_capacity(image, bits, strong)
        weak_bers.append(bit_error_rate(bits, extract_capacity(r_weak.watermarked_image, image, weak)))
        strong_bers.append(bit_error_rate(bits, extract_capacity(r_strong.watermarked_image, image, strong)))
    assert float(np.mean(strong_bers)) <= float(np.mean(weak_bers))


@_needs_data
def test_alpha_vs_ber_is_not_monotonic_at_full_ll_depth() -> None:
    """Documented Phase 12 finding (see docs/phase12_capacity.md): at full
    128-bit LL depth, where every singular value (including the fragile,
    near-degenerate trailing ones) is modified simultaneously, increasing
    alpha does NOT reliably reduce BER and can make it worse - a real,
    reproducible property of the frozen multiplicative rule at this payload
    depth, not a defect in the Phase 12 embedder (reproduces identically via
    the frozen ``embed()``/``extract_traditional()`` at the same operating
    point). This test asserts the *documented* direction (mean BER at
    alpha=0.02 is not lower than at alpha=0.005) so a future change that
    silently "fixes" this without updating the docs is caught."""
    weak = CapacityEmbedConfig(dwt_levels=1, subband_plan=((1, "LL"),), bit_length=128, alpha=0.005)
    strong = CapacityEmbedConfig(dwt_levels=1, subband_plan=((1, "LL"),), bit_length=128, alpha=0.02)
    weak_bers, strong_bers = [], []
    for i, image in enumerate(_IMAGES):
        bits = _bits(128, seed=300 + i)
        r_weak = embed_capacity(image, bits, weak)
        r_strong = embed_capacity(image, bits, strong)
        weak_bers.append(bit_error_rate(bits, extract_capacity(r_weak.watermarked_image, image, weak)))
        strong_bers.append(bit_error_rate(bits, extract_capacity(r_strong.watermarked_image, image, strong)))
    assert float(np.mean(strong_bers)) >= float(np.mean(weak_bers))
