"""Tests for src/watermark/svd.py — Phase 5 SVD module."""

import numpy as np
import pytest

from src.watermark.svd import (
    SVDComponents,
    decompose,
    embed_in_singular_values,
    extract_from_singular_values,
    get_singular_values,
    reconstruct,
    reconstruction_error,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_matrix(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape)


# ---------------------------------------------------------------------------
# 1. Decompose / reconstruct round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(4, 4), (8, 8), (16, 32), (64, 64), (128, 65)])
def test_round_trip_preserves_shape_and_values(shape: tuple[int, int]) -> None:
    """A = U @ diag(S) @ Vt must recover the original matrix within float64 precision."""
    matrix = _random_matrix(shape)
    recovered = reconstruct(decompose(matrix))
    assert recovered.shape == shape
    np.testing.assert_allclose(recovered, matrix, atol=1e-10, rtol=0)


def test_reconstruction_error_is_near_machine_eps() -> None:
    """For a well-conditioned matrix reconstruction error should be tiny."""
    matrix = _random_matrix((64, 64))
    error = reconstruction_error(matrix)
    assert error < 1e-8, f"Reconstruction error too large: {error}"


def test_singular_values_are_sorted_descending() -> None:
    """numpy.linalg.svd guarantees descending order."""
    components = decompose(_random_matrix((32, 32)))
    assert all(components.S[i] >= components.S[i + 1] for i in range(len(components.S) - 1))


def test_singular_values_are_non_negative() -> None:
    matrix = _random_matrix((16, 16))
    assert (decompose(matrix).S >= 0).all()


def test_decompose_source_shape_recorded() -> None:
    shape = (13, 27)
    components = decompose(_random_matrix(shape))
    assert components.source_shape == shape


@pytest.mark.parametrize("shape", [(4, 4), (5, 10), (10, 5)])
def test_k_equals_min_dimension(shape: tuple[int, int]) -> None:
    """For full_matrices=False SVD, k = min(m, n)."""
    components = decompose(_random_matrix(shape))
    k = min(shape)
    assert components.S.shape == (k,)
    assert components.U.shape == (shape[0], k)
    assert components.Vt.shape == (k, shape[1])


# ---------------------------------------------------------------------------
# 2. get_singular_values (values-only path)
# ---------------------------------------------------------------------------

def test_get_singular_values_matches_decompose() -> None:
    matrix = _random_matrix((32, 32))
    full = decompose(matrix).S
    fast = get_singular_values(matrix)
    np.testing.assert_allclose(full, fast, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# 3. Numerical stability
# ---------------------------------------------------------------------------

def test_identity_matrix_has_all_singular_values_one() -> None:
    n = 8
    components = decompose(np.eye(n))
    np.testing.assert_allclose(components.S, np.ones(n), atol=1e-12)


def test_zero_matrix_has_all_singular_values_zero() -> None:
    components = decompose(np.zeros((8, 12)))
    np.testing.assert_allclose(components.S, np.zeros(8), atol=1e-12)


def test_rank_one_matrix_has_one_nonzero_singular_value() -> None:
    # Outer product of two vectors -> rank-1 matrix
    v = np.arange(1, 9, dtype=float).reshape(-1, 1)
    w = np.arange(1, 7, dtype=float).reshape(1, -1)
    matrix = v @ w  # shape (8, 6)
    components = decompose(matrix)
    # Only first singular value should be significantly non-zero
    assert components.S[0] > 1.0
    np.testing.assert_allclose(components.S[1:], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. Embedding and non-blind extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_bits", [8, 16, 32])
def test_embed_and_extract_exact_on_clean_image(n_bits: int) -> None:
    """Non-blind extraction must recover the embedded bits exactly when there is no attack."""
    rng = np.random.default_rng(42)
    matrix = rng.random((128, 128)) * 255.0
    bits = [int(b) for b in rng.integers(0, 2, size=n_bits)]

    original_components = decompose(matrix)
    embedded_components = embed_in_singular_values(original_components, bits, alpha=0.01)

    extracted = extract_from_singular_values(
        original_components.S, embedded_components.S, n_bits
    )
    assert extracted == bits


def test_embedding_modifies_singular_values_by_alpha() -> None:
    """Verify the bipolar (+/- alpha) modification is applied correctly."""
    matrix = _random_matrix((32, 32)) * 100.0
    bits = [1, 0, 1, 0, 1, 0, 1, 0]
    alpha = 0.05

    orig = decompose(matrix)
    emb = embed_in_singular_values(orig, bits, alpha=alpha)

    for i, bit in enumerate(bits):
        expected_delta = alpha * (2 * bit - 1)
        actual_delta = emb.S[i] - orig.S[i]
        assert abs(actual_delta - expected_delta) < 1e-12, (
            f"Bit {i}: expected delta {expected_delta:.6f}, got {actual_delta:.6f}"
        )


def test_embedding_does_not_change_u_or_vt() -> None:
    """U and Vt must remain identical after embedding."""
    matrix = _random_matrix((16, 16))
    bits = [1, 0] * 4
    orig = decompose(matrix)
    emb = embed_in_singular_values(orig, bits, alpha=0.01)
    np.testing.assert_array_equal(orig.U, emb.U)
    np.testing.assert_array_equal(orig.Vt, emb.Vt)


@pytest.mark.parametrize("alpha", [0.005, 0.010, 0.015])
def test_embedding_strength_matches_alpha(alpha: float) -> None:
    """All three baseline alpha values work without error."""
    matrix = _random_matrix((64, 64)) * 255.0
    bits = [1] * 64  # embed 64 bits of 1s
    orig = decompose(matrix)
    emb = embed_in_singular_values(orig, bits, alpha=alpha)
    # All modified SVs should be larger (bit=1 -> +alpha)
    assert (emb.S[:64] > orig.S[:64]).all()


def test_start_index_respected() -> None:
    """Bits must be embedded starting at the given index, not at index 0."""
    matrix = _random_matrix((32, 32)) * 100.0
    bits = [1, 0, 1]
    orig = decompose(matrix)
    emb = embed_in_singular_values(orig, bits, alpha=0.01, start_index=5)
    # Indices 0-4 must be unchanged
    np.testing.assert_array_equal(orig.S[:5], emb.S[:5])
    # Indices 5-7 must be modified
    for i, bit in enumerate(bits):
        expected = orig.S[5 + i] + 0.01 * (2 * bit - 1)
        assert abs(emb.S[5 + i] - expected) < 1e-12


# ---------------------------------------------------------------------------
# 5. Input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    np.ones((4, 4, 3)),     # 3D
    np.ones((5,)),           # 1D
    np.array([[np.nan, 1], [2, 3]]),  # NaN
    np.array([[np.inf, 1], [2, 3]]),  # Inf
])
def test_invalid_matrix_rejected_by_decompose(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        decompose(bad)


def test_reconstruct_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="SVDComponents"):
        reconstruct("not components")  # type: ignore[arg-type]


def test_embed_rejects_non_binary_bits() -> None:
    components = decompose(_random_matrix((16, 16)))
    with pytest.raises(ValueError, match="0 or 1"):
        embed_in_singular_values(components, [0, 1, 2], alpha=0.01)


def test_embed_rejects_non_positive_alpha() -> None:
    components = decompose(_random_matrix((16, 16)))
    with pytest.raises(ValueError, match="alpha must be positive"):
        embed_in_singular_values(components, [1, 0], alpha=0.0)


def test_embed_rejects_too_many_bits() -> None:
    matrix = _random_matrix((4, 4))  # 4 singular values
    components = decompose(matrix)
    with pytest.raises(ValueError, match="singular values available"):
        embed_in_singular_values(components, [1] * 5, alpha=0.01)  # 5 > 4


def test_embed_rejects_wrong_components_type() -> None:
    with pytest.raises(TypeError, match="SVDComponents"):
        embed_in_singular_values("not components", [1, 0], alpha=0.01)  # type: ignore[arg-type]
