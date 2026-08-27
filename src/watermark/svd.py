"""Singular Value Decomposition utilities for image watermarking.

Mathematical background
-----------------------
For any real matrix A (shape m x n), SVD decomposes it as:

    A = U @ diag(S) @ Vt

where:
  U  (m x k)  — left singular vectors, columns are orthonormal
  S  (k,)     — singular values, non-negative, sorted descending, k = min(m, n)
  Vt (k x n)  — right singular vectors (V transposed), rows are orthonormal

Key properties used for watermarking:
1. Singular values are numerically robust — they change slowly under small perturbations.
2. The largest singular values carry the most perceptual energy.
3. Modifying S slightly and then reconstructing via U @ diag(S_modified) @ Vt
   produces an image that is visually indistinguishable from the original.
4. The modification persists through many common image transformations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SVDComponents:
    """Container for a full SVD decomposition of a 2D array.

    Attributes
    ----------
    U:
        Left singular vectors, shape (m, k), columns are orthonormal.
    S:
        Singular values, shape (k,), sorted descending, k = min(m, n).
    Vt:
        Right singular vectors (V transposed), shape (k, n), rows are orthonormal.
    source_shape:
        Original matrix shape (m, n), required for exact reconstruction verification.
    """

    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    source_shape: tuple[int, int]


def _validate_matrix(matrix: np.ndarray) -> np.ndarray:
    """Validate and coerce input to a 2D float64 array."""
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"SVD expects a 2D array; got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("SVD input must have a numeric dtype")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("SVD input must be at least 1x1")
    if not np.isfinite(array).all():
        raise ValueError("SVD input must contain only finite values")
    return array.astype(np.float64, copy=False)


def decompose(matrix: np.ndarray) -> SVDComponents:
    """Compute the full SVD of a 2D matrix.

    Returns an SVDComponents dataclass with U, S, Vt and the source shape.
    Singular values are sorted in descending order (NumPy default).

    Parameters
    ----------
    matrix:
        A 2D numeric array, e.g. the LL subband of a DWT decomposition.

    Returns
    -------
    SVDComponents with U, S, Vt, source_shape.
    """
    m = _validate_matrix(matrix)
    U, S, Vt = np.linalg.svd(m, full_matrices=False)
    return SVDComponents(U=U, S=S, Vt=Vt, source_shape=(m.shape[0], m.shape[1]))


def reconstruct(components: SVDComponents) -> np.ndarray:
    """Reconstruct the original matrix from SVD components.

    Computes:  A_reconstructed = U @ diag(S) @ Vt

    Parameters
    ----------
    components:
        An SVDComponents instance (from decompose or with modified S).

    Returns
    -------
    2D float64 array with shape == components.source_shape.
    """
    if not isinstance(components, SVDComponents):
        raise TypeError("components must be an SVDComponents instance")
    reconstructed = components.U @ np.diag(components.S) @ components.Vt
    return np.asarray(reconstructed, dtype=np.float64)


def reconstruction_error(matrix: np.ndarray) -> float:
    """Return the maximum absolute round-trip reconstruction error.

    Performs decompose -> reconstruct and returns max|A - A_reconstructed|.
    For a well-conditioned matrix this should be near floating-point epsilon (~1e-12).

    Parameters
    ----------
    matrix:
        A 2D numeric array.

    Returns
    -------
    float: maximum absolute reconstruction error.
    """
    source = _validate_matrix(matrix)
    recovered = reconstruct(decompose(source))
    return float(np.max(np.abs(source - recovered)))


def get_singular_values(matrix: np.ndarray) -> np.ndarray:
    """Extract and return singular values from a matrix without storing U and Vt.

    More memory-efficient than full decompose() when only the values are needed.

    Parameters
    ----------
    matrix:
        A 2D numeric array.

    Returns
    -------
    1D float64 array of singular values, sorted descending.
    """
    m = _validate_matrix(matrix)
    return np.linalg.svd(m, compute_uv=False)


def embed_in_singular_values(
    components: SVDComponents,
    bits: list[int],
    alpha: float,
    start_index: int = 0,
) -> SVDComponents:
    """Embed watermark bits into singular values by additive modification.

    This is the core DWT-SVD embedding operation:

        S_modified[i] = S[i] + alpha * bit[i]

    where bit is 0 or 1, but we centre it to {-1, +1} to ensure the modification
    is always non-zero:

        S_modified[i] = S[i] + alpha * (2 * bit[i] - 1)

    This means:
    - bit = 1  ->  S_modified[i] = S[i] + alpha
    - bit = 0  ->  S_modified[i] = S[i] - alpha

    The receiver can then classify: if S_extracted[i] > threshold -> bit=1.

    Parameters
    ----------
    components:
        Original SVD decomposition.
    bits:
        List of integers, each 0 or 1. Length determines how many singular
        values are modified.
    alpha:
        Embedding strength. Larger = more robust, more visible. Typically 0.005–0.05.
    start_index:
        Which singular value index to start embedding from. Default 0 (largest).

    Returns
    -------
    New SVDComponents with modified S; U and Vt are unchanged.

    Raises
    ------
    ValueError:
        If bits contains values other than 0/1, or if embedding would exceed
        the number of available singular values.
    """
    if not isinstance(components, SVDComponents):
        raise TypeError("components must be an SVDComponents instance")
    if any(b not in (0, 1) for b in bits):
        raise ValueError("bits must contain only 0 or 1")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive; got {alpha}")
    end_index = start_index + len(bits)
    if end_index > len(components.S):
        raise ValueError(
            f"Cannot embed {len(bits)} bits starting at index {start_index}: "
            f"only {len(components.S)} singular values available"
        )

    S_modified = components.S.copy()
    for offset, bit in enumerate(bits):
        S_modified[start_index + offset] += alpha * (2 * bit - 1)

    return SVDComponents(U=components.U, S=S_modified, Vt=components.Vt, source_shape=components.source_shape)


def extract_from_singular_values(
    original_S: np.ndarray,
    modified_S: np.ndarray,
    n_bits: int,
    start_index: int = 0,
) -> list[int]:
    """Extract watermark bits by comparing original and modified singular values.

    This is the NON-BLIND extraction (requires original matrix S).
    The blind extraction (Phase 7) uses CNN instead.

    Decision rule:
        bit[i] = 1 if modified_S[i] > original_S[i] else 0

    Parameters
    ----------
    original_S:
        Singular values of the original (unwatermarked) LL subband.
    modified_S:
        Singular values of the watermarked LL subband.
    n_bits:
        Number of bits to extract.
    start_index:
        Starting index into the singular value array.

    Returns
    -------
    List of extracted bits (0 or 1).
    """
    extracted = []
    for i in range(n_bits):
        idx = start_index + i
        extracted.append(1 if modified_S[idx] > original_S[idx] else 0)
    return extracted
