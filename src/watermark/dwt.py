"""Dimension-preserving two-dimensional discrete wavelet transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt


SUPPORTED_WAVELETS = frozenset({"haar", "db2", "db4"})
SUPPORTED_MODES = frozenset({"symmetric", "periodization"})


@dataclass(frozen=True)
class DWTCoefficients:
    """Single-level 2D DWT components and the source shape required for exact cropping."""

    ll: np.ndarray
    lh: np.ndarray
    hl: np.ndarray
    hh: np.ndarray
    source_shape: tuple[int, int]
    wavelet: str
    mode: str


def _validate_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"DWT expects a 2D array; got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("DWT input must have a numeric dtype")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("DWT input must be at least 2×2")
    if not np.isfinite(array).all():
        raise ValueError("DWT input must contain only finite values")
    return array.astype(np.float64, copy=False)


def _validate_options(wavelet: str, mode: str) -> None:
    if wavelet not in SUPPORTED_WAVELETS:
        raise ValueError(f"Unsupported wavelet {wavelet!r}; allowed: {sorted(SUPPORTED_WAVELETS)}")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported boundary mode {mode!r}; allowed: {sorted(SUPPORTED_MODES)}")


def decompose_2d(image: np.ndarray, wavelet: str = "haar", mode: str = "symmetric") -> DWTCoefficients:
    """Apply a single-level DWT, yielding LL, LH, HL, and HH coefficient arrays.

    PyWavelets uses its established convention: ``cH`` is horizontal detail and
    ``cV`` vertical detail. We expose those as LH and HL respectively; later
    embedding code must use the labels rather than assume an orientation.
    """
    _validate_options(wavelet, mode)
    source = _validate_image(image)
    ll, (lh, hl, hh) = pywt.dwt2(source, wavelet=wavelet, mode=mode)
    return DWTCoefficients(ll, lh, hl, hh, source.shape, wavelet, mode)


def reconstruct_2d(coefficients: DWTCoefficients) -> np.ndarray:
    """Invert a single-level DWT and crop padding introduced by the boundary mode."""
    if not isinstance(coefficients, DWTCoefficients):
        raise TypeError("coefficients must be a DWTCoefficients instance")
    _validate_options(coefficients.wavelet, coefficients.mode)
    parts = (coefficients.ll, coefficients.lh, coefficients.hl, coefficients.hh)
    if any(np.asarray(part).ndim != 2 for part in parts):
        raise ValueError("All DWT coefficient arrays must be two-dimensional")
    reconstructed = pywt.idwt2(
        (coefficients.ll, (coefficients.lh, coefficients.hl, coefficients.hh)),
        wavelet=coefficients.wavelet,
        mode=coefficients.mode,
    )
    height, width = coefficients.source_shape
    return np.asarray(reconstructed, dtype=np.float64)[:height, :width]


def reconstruction_error(image: np.ndarray, wavelet: str = "haar", mode: str = "symmetric") -> float:
    """Return maximum absolute round-trip reconstruction error."""
    source = _validate_image(image)
    recovered = reconstruct_2d(decompose_2d(source, wavelet, mode))
    return float(np.max(np.abs(source - recovered)))
