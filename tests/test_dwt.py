import numpy as np
import pytest

from src.watermark.dwt import (
    SUPPORTED_WAVELETS,
    DWTCoefficients,
    decompose_2d,
    reconstruct_2d,
    reconstruction_error,
)


@pytest.mark.parametrize("wavelet", sorted(SUPPORTED_WAVELETS))
@pytest.mark.parametrize("shape", [(64, 64), (63, 65)])
def test_round_trip_preserves_shape_and_values(wavelet: str, shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(42)
    image = rng.random(shape)
    recovered = reconstruct_2d(decompose_2d(image, wavelet=wavelet))
    assert recovered.shape == shape
    np.testing.assert_allclose(recovered, image, atol=1e-10, rtol=0)
    assert reconstruction_error(image, wavelet) <= 1e-10


def test_coefficients_keep_transform_provenance() -> None:
    coefficients = decompose_2d(np.ones((32, 48)), wavelet="haar")
    assert coefficients.source_shape == (32, 48)
    assert coefficients.wavelet == "haar"
    assert coefficients.ll.shape == coefficients.lh.shape == coefficients.hl.shape == coefficients.hh.shape


@pytest.mark.parametrize("bad_input", [np.ones((8, 8, 3)), np.ones((1, 8)), np.array([[np.nan, 1], [2, 3]])])
def test_invalid_input_is_rejected(bad_input: np.ndarray) -> None:
    with pytest.raises(ValueError):
        decompose_2d(bad_input)


@pytest.mark.parametrize("wavelet", ["db3", "unknown"])
def test_unsupported_wavelet_is_rejected(wavelet: str) -> None:
    with pytest.raises(ValueError, match="Unsupported wavelet"):
        decompose_2d(np.ones((8, 8)), wavelet=wavelet)


def test_reconstruction_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="DWTCoefficients"):
        reconstruct_2d("not coefficients")  # type: ignore[arg-type]


def test_reconstruction_rejects_non_2d_coefficient_component() -> None:
    coefficients = decompose_2d(np.ones((8, 8)))
    malformed = DWTCoefficients(
        coefficients.ll[0], coefficients.lh, coefficients.hl, coefficients.hh, coefficients.source_shape, "haar", "symmetric"
    )
    with pytest.raises(ValueError, match="two-dimensional"):
        reconstruct_2d(malformed)
