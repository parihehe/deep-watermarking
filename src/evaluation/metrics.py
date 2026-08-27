"""Image-quality and watermark-recovery metrics for the DWT-SVD study.

All functions are pure and dependency-light so they can be reused unchanged by
the baseline runner (Phase 6), the CNN evaluation (Phase 8+), the attack study
(Phase 9) and the API (Phase 23).

Image-quality metrics operate on uint8 or float RGB/grayscale arrays of equal
shape. Watermark-recovery metrics operate on equal-length sequences of 0/1 ints.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from skimage.metrics import structural_similarity as _ssim

# ---------------------------------------------------------------------------
# Image-quality metrics
# ---------------------------------------------------------------------------

def _as_float(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("image metrics require a numeric array")
    return array.astype(np.float64, copy=False)


def mse(reference: np.ndarray, test: np.ndarray) -> float:
    """Mean squared error between two equally shaped images."""
    ref = _as_float(reference)
    tst = _as_float(test)
    if ref.shape != tst.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {tst.shape}")
    return float(np.mean((ref - tst) ** 2))


def psnr(reference: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio in decibels.

    Returns ``inf`` when the two images are identical.
    """
    error = mse(reference, test)
    if error == 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / error))


def ssim(reference: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    """Structural similarity index. Uses a per-channel mean for colour images."""
    ref = _as_float(reference)
    tst = _as_float(test)
    if ref.shape != tst.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {tst.shape}")
    channel_axis = 2 if ref.ndim == 3 else None
    return float(_ssim(ref, tst, data_range=data_range, channel_axis=channel_axis))


# ---------------------------------------------------------------------------
# Watermark-recovery metrics
# ---------------------------------------------------------------------------

def _as_bit_array(bits: Sequence[int]) -> np.ndarray:
    array = np.asarray(list(bits), dtype=int)
    if array.ndim != 1:
        raise ValueError("bit sequence must be one-dimensional")
    if array.size == 0:
        raise ValueError("bit sequence must be non-empty")
    if not np.isin(array, (0, 1)).all():
        raise ValueError("bit sequence must contain only 0 and 1")
    return array


def _check_pair(reference: Sequence[int], recovered: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    ref = _as_bit_array(reference)
    rec = _as_bit_array(recovered)
    if ref.shape != rec.shape:
        raise ValueError(f"bit-length mismatch: {ref.size} vs {rec.size}")
    return ref, rec


def bit_error_rate(reference: Sequence[int], recovered: Sequence[int]) -> float:
    """Fraction of bits that differ (0.0 = perfect, 0.5 = chance, 1.0 = inverted)."""
    ref, rec = _check_pair(reference, recovered)
    return float(np.mean(ref != rec))


def bit_accuracy(reference: Sequence[int], recovered: Sequence[int]) -> float:
    """Fraction of bits recovered correctly. Equals ``1 - bit_error_rate``."""
    return 1.0 - bit_error_rate(reference, recovered)


def normalized_correlation(reference: Sequence[int], recovered: Sequence[int]) -> float:
    """Normalized correlation of the two bit strings mapped to bipolar ``{-1, +1}``.

    ``NC = <w_ref, w_rec> / (||w_ref|| * ||w_rec||)``. For non-degenerate bipolar
    vectors this reduces to ``1 - 2 * BER`` and lies in ``[-1, 1]``.
    """
    ref, rec = _check_pair(reference, recovered)
    w_ref = 2 * ref.astype(np.float64) - 1
    w_rec = 2 * rec.astype(np.float64) - 1
    denom = np.linalg.norm(w_ref) * np.linalg.norm(w_rec)
    if denom == 0.0:
        return 0.0
    return float(np.dot(w_ref, w_rec) / denom)


def recovery_report(reference: Sequence[int], recovered: Sequence[int]) -> dict[str, float]:
    """Bundle the recovery metrics into one dict for logging/serialisation."""
    return {
        "ber": bit_error_rate(reference, recovered),
        "bit_accuracy": bit_accuracy(reference, recovered),
        "nc": normalized_correlation(reference, recovered),
    }


def quality_report(reference: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> dict[str, float]:
    """Bundle the image-quality metrics into one dict for logging/serialisation."""
    return {
        "psnr": psnr(reference, test, data_range=data_range),
        "ssim": ssim(reference, test, data_range=data_range),
        "mse": mse(reference, test),
    }
