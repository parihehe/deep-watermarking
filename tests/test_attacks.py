"""Tests for Phase 10 image attacks (``src/evaluation/attacks.py``).

Pure, data-free: every attack is exercised on a synthetic structured image. The
attacks module imports nothing from the watermarking pipeline, so these tests
cannot touch the frozen Phase 6 baseline / Phase 7 UI / Phase 8 CNN.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.evaluation.attacks import (
    ATTACKS,
    apply_attack,
    attack_names,
    center_crop,
    combined,
    gaussian_blur,
    gaussian_noise,
    identity,
    jpeg_compress,
    median_filter,
    resize_roundtrip,
    rotate,
    severity_of,
)


@pytest.fixture
def image() -> np.ndarray:
    """A 96x96 RGB image with a smooth gradient plus fine texture."""
    yy, xx = np.mgrid[0:96, 0:96]
    base = ((xx * 2 + yy) % 256).astype(np.float64)
    rng = np.random.default_rng(0)
    texture = rng.integers(0, 40, size=(96, 96)).astype(np.float64)
    chan = np.clip(base + texture, 0, 255)
    return np.stack([chan, np.roll(chan, 7, axis=0), np.roll(chan, 3, axis=1)], axis=2).astype(np.uint8)


def _laplacian_var(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


# ---------------------------------------------------------------------------
# generic contract
# ---------------------------------------------------------------------------

ALL_CALLS = [
    ("identity", {}),
    ("jpeg_compress", {"quality": 50}),
    ("gaussian_noise", {"sigma": 10, "seed": 1}),
    ("gaussian_blur", {"ksize": 5}),
    ("median_filter", {"ksize": 3}),
    ("resize_roundtrip", {"scale": 0.5}),
    ("rotate", {"degrees": 7}),
    ("center_crop", {"keep": 0.75}),
    ("combined", {"steps": [{"name": "jpeg_compress", "params": {"quality": 60}},
                            {"name": "gaussian_noise", "params": {"sigma": 4}}], "seed": 2}),
]


@pytest.mark.parametrize("name,params", ALL_CALLS)
def test_attack_output_contract(image: np.ndarray, name: str, params: dict) -> None:
    out = apply_attack(name, image, params)
    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.shape[:2] == image.shape[:2]        # all attacks return full frame at source size
    assert out.min() >= 0 and out.max() <= 255


def test_registry_is_complete() -> None:
    assert set(attack_names()) == {
        "identity", "jpeg_compress", "gaussian_noise", "gaussian_blur", "median_filter",
        "resize_roundtrip", "rotate", "center_crop", "combined",
    }
    for spec in ATTACKS.values():
        assert callable(spec.func)


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

def test_identity_is_exact_copy(image: np.ndarray) -> None:
    out = identity(image)
    assert np.array_equal(out, image)
    assert out is not image


# ---------------------------------------------------------------------------
# jpeg
# ---------------------------------------------------------------------------

def test_jpeg_lower_quality_is_more_distortion(image: np.ndarray) -> None:
    d_high = _mse(image, jpeg_compress(image, quality=95))
    d_low = _mse(image, jpeg_compress(image, quality=15))
    assert 0.0 <= d_high < d_low


@pytest.mark.parametrize("q", [0, -5, 101, 200])
def test_jpeg_quality_validation(image: np.ndarray, q: int) -> None:
    with pytest.raises(ValueError):
        jpeg_compress(image, quality=q)


# ---------------------------------------------------------------------------
# gaussian noise
# ---------------------------------------------------------------------------

def test_gaussian_noise_is_seed_deterministic(image: np.ndarray) -> None:
    a = gaussian_noise(image, sigma=12, seed=7)
    b = gaussian_noise(image, sigma=12, seed=7)
    c = gaussian_noise(image, sigma=12, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_gaussian_noise_sigma_zero_is_noop(image: np.ndarray) -> None:
    assert np.array_equal(gaussian_noise(image, sigma=0, seed=1), image)


def test_gaussian_noise_more_sigma_more_mse(image: np.ndarray) -> None:
    assert _mse(image, gaussian_noise(image, sigma=3, seed=1)) < \
           _mse(image, gaussian_noise(image, sigma=25, seed=1))


def test_gaussian_noise_rejects_negative_sigma(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        gaussian_noise(image, sigma=-1.0)


# ---------------------------------------------------------------------------
# blur / median
# ---------------------------------------------------------------------------

def test_gaussian_blur_reduces_high_frequency(image: np.ndarray) -> None:
    assert _laplacian_var(gaussian_blur(image, ksize=7)) < _laplacian_var(image)


def test_blur_even_ksize_is_coerced_odd(image: np.ndarray) -> None:
    # ksize 4 -> 5; must not raise and must smooth
    assert _laplacian_var(gaussian_blur(image, ksize=4)) < _laplacian_var(image)


def test_median_filter_reduces_high_frequency(image: np.ndarray) -> None:
    assert _laplacian_var(median_filter(image, ksize=5)) < _laplacian_var(image)


# ---------------------------------------------------------------------------
# resize round-trip
# ---------------------------------------------------------------------------

def test_resize_scale_one_is_noop(image: np.ndarray) -> None:
    assert np.array_equal(resize_roundtrip(image, scale=1.0), image)


def test_resize_smaller_scale_loses_more_detail(image: np.ndarray) -> None:
    mild = _laplacian_var(resize_roundtrip(image, scale=0.75))
    harsh = _laplacian_var(resize_roundtrip(image, scale=0.25))
    assert harsh < mild < _laplacian_var(image)


@pytest.mark.parametrize("scale", [0.0, -0.2, 1.5])
def test_resize_scale_validation(image: np.ndarray, scale: float) -> None:
    with pytest.raises(ValueError):
        resize_roundtrip(image, scale=scale)


# ---------------------------------------------------------------------------
# rotate
# ---------------------------------------------------------------------------

def test_rotate_zero_degrees_is_near_identity(image: np.ndarray) -> None:
    assert _mse(image, rotate(image, degrees=0)) < 1e-6


def test_rotate_changes_pixels(image: np.ndarray) -> None:
    assert _mse(image, rotate(image, degrees=15)) > 1.0


def test_rotate_unknown_border_raises(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        rotate(image, degrees=5, border="nope")


# ---------------------------------------------------------------------------
# center crop
# ---------------------------------------------------------------------------

def test_center_crop_keep_one_is_noop(image: np.ndarray) -> None:
    assert np.array_equal(center_crop(image, keep=1.0), image)


def test_center_crop_blacks_out_border(image: np.ndarray) -> None:
    out = center_crop(image, keep=0.5)
    assert out[0, 0].sum() == 0 and out[-1, -1].sum() == 0
    assert out[48, 48].sum() > 0                      # centre preserved


@pytest.mark.parametrize("keep", [0.0, -0.1, 1.2])
def test_center_crop_validation(image: np.ndarray, keep: float) -> None:
    with pytest.raises(ValueError):
        center_crop(image, keep=keep)


# ---------------------------------------------------------------------------
# combined
# ---------------------------------------------------------------------------

def test_combined_runs_steps_in_order(image: np.ndarray) -> None:
    steps_a = [{"name": "jpeg_compress", "params": {"quality": 40}},
               {"name": "gaussian_blur", "params": {"ksize": 5}}]
    steps_b = list(reversed(steps_a))
    out_a = combined(image, steps=steps_a, seed=1)
    out_b = combined(image, steps=steps_b, seed=1)
    assert not np.array_equal(out_a, out_b)          # order matters
    assert _mse(image, out_a) > 0


def test_combined_is_seed_deterministic(image: np.ndarray) -> None:
    steps = [{"name": "gaussian_noise", "params": {"sigma": 6}},
             {"name": "gaussian_noise", "params": {"sigma": 6}}]
    assert np.array_equal(combined(image, steps=steps, seed=3), combined(image, steps=steps, seed=3))
    assert not np.array_equal(combined(image, steps=steps, seed=3), combined(image, steps=steps, seed=4))


def test_combined_rejects_nested_combined(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        combined(image, steps=[{"name": "combined", "params": {"steps": []}}])


# ---------------------------------------------------------------------------
# dispatch / severity
# ---------------------------------------------------------------------------

def test_apply_attack_unknown_name_raises(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        apply_attack("teleport", image, {})


def test_apply_attack_injects_seed_only_for_randomised(image: np.ndarray) -> None:
    # randomised: seed changes the result
    assert not np.array_equal(
        apply_attack("gaussian_noise", image, {"sigma": 10}, seed=1),
        apply_attack("gaussian_noise", image, {"sigma": 10}, seed=2),
    )
    # non-randomised: passing seed is harmless and ignored
    assert np.array_equal(
        apply_attack("gaussian_blur", image, {"ksize": 3}, seed=1),
        apply_attack("gaussian_blur", image, {"ksize": 3}, seed=2),
    )


def test_severity_of() -> None:
    assert severity_of("jpeg_compress", {"quality": 50}) == 50.0
    assert severity_of("rotate", {"degrees": 5}) == 5.0
    assert severity_of("identity", {}) is None
    assert severity_of("combined", {"steps": []}) is None
    with pytest.raises(ValueError):
        severity_of("nope", {})
