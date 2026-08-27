"""Tests for src/evaluation/metrics.py — Phase 6 metrics module."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    bit_accuracy,
    bit_error_rate,
    mse,
    normalized_correlation,
    psnr,
    quality_report,
    recovery_report,
    ssim,
)


def _rng_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)


# --- image-quality metrics -------------------------------------------------

def test_identical_images_give_zero_mse_and_infinite_psnr() -> None:
    image = _rng_image()
    assert mse(image, image) == 0.0
    assert psnr(image, image) == float("inf")
    assert ssim(image, image) == pytest.approx(1.0)


def test_mse_and_psnr_are_consistent() -> None:
    image = _rng_image().astype(np.float64)
    noisy = image + 5.0
    assert mse(image, noisy) == pytest.approx(25.0)
    assert psnr(image, noisy) == pytest.approx(10.0 * np.log10(255.0**2 / 25.0))


def test_psnr_decreases_as_distortion_grows() -> None:
    image = _rng_image().astype(np.float64)
    assert psnr(image, image + 2.0) > psnr(image, image + 8.0)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        mse(np.zeros((4, 4)), np.zeros((4, 5)))


# --- watermark-recovery metrics ------------------------------------------

def test_perfect_recovery() -> None:
    bits = [0, 1, 1, 0, 1, 0, 0, 1]
    assert bit_error_rate(bits, bits) == 0.0
    assert bit_accuracy(bits, bits) == 1.0
    assert normalized_correlation(bits, bits) == pytest.approx(1.0)


def test_inverted_recovery() -> None:
    bits = [0, 1, 1, 0]
    inverted = [1, 0, 0, 1]
    assert bit_error_rate(bits, inverted) == 1.0
    assert normalized_correlation(bits, inverted) == pytest.approx(-1.0)


def test_half_wrong_is_chance_level() -> None:
    ref = [0, 0, 1, 1]
    rec = [0, 1, 1, 0]
    assert bit_error_rate(ref, rec) == pytest.approx(0.5)
    assert normalized_correlation(ref, rec) == pytest.approx(0.0)


def test_nc_equals_one_minus_two_ber() -> None:
    ref = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    rec = [1, 0, 0, 0, 1, 1, 1, 0, 1, 0]
    assert normalized_correlation(ref, rec) == pytest.approx(1.0 - 2.0 * bit_error_rate(ref, rec))


def test_recovery_metrics_reject_bad_input() -> None:
    with pytest.raises(ValueError):
        bit_error_rate([0, 1], [0, 1, 1])
    with pytest.raises(ValueError):
        bit_error_rate([0, 2], [0, 1])
    with pytest.raises(ValueError):
        bit_error_rate([], [])


def test_report_bundles_have_expected_keys() -> None:
    image = _rng_image()
    q = quality_report(image, image)
    assert set(q) == {"psnr", "ssim", "mse"}
    r = recovery_report([0, 1, 0], [0, 1, 1])
    assert set(r) == {"ber", "bit_accuracy", "nc"}
