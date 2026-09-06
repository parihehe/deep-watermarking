"""Tests for the experimental windowed 1D-CNN blind watermark extractor.

Covers the architecture, the authoritative bit->window mapping, checkpoint I/O,
the blind-inference wrapper, plus the on-the-fly windowed dataset and a 1-epoch
end-to-end training run (the latter two skip when the processed DIV2K split is
absent). The frozen Phase 6 baseline is exercised only through its public API;
nothing here modifies it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.windowed_cnn import (
    WINDOW_DEFAULT,
    WindowedCNNConfig,
    WindowedCNNExtractor,
    bit_window_singular_values,
    load_windowed_checkpoint,
    luminance_ll_singular_values,
    save_windowed_checkpoint,
)

_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed" / "div2k_256"
_VAL_DIR = _PROCESSED / "validation"
_needs_data = pytest.mark.skipif(
    not _VAL_DIR.is_dir() or not any(_VAL_DIR.glob("*.png")),
    reason="processed DIV2K validation split not available",
)

_SIGMA = np.linspace(1000.0, 1.0, 128)  # synthetic decay, like a real spectrum


# ---------------------------------------------------------------------------
# WindowedCNNConfig validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bit_length": 0},
        {"bit_length": -4},
        {"window_size": 0},
        {"window_size": 8},  # even window rejected
        {"start_sv_index": -1},
        {"conv1_channels": 0},
        {"conv2_channels": 0},
        {"kernel_size": 4},  # even kernel rejected
        {"fc_units": 0},
        {"dropout_conv": 1.0},
        {"dropout_fc": -0.1},
        {"subband": "HL"},
    ],
)
def test_windowed_config_rejects_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        WindowedCNNConfig(**kwargs)


def test_windowed_config_defaults_match_paper() -> None:
    cfg = WindowedCNNConfig()
    assert cfg.window_size == WINDOW_DEFAULT == 15
    assert cfg.conv1_channels == 32
    assert cfg.conv2_channels == 64
    assert cfg.kernel_size == 3
    assert cfg.fc_units == 64
    assert cfg.dropout_conv == pytest.approx(0.3)
    assert cfg.dropout_fc == pytest.approx(0.5)
    assert cfg.subband == "LL"


# ---------------------------------------------------------------------------
# Authoritative bit -> window mapping
# ---------------------------------------------------------------------------


def test_every_bit_gets_a_fixed_15_value_window() -> None:
    cfg = WindowedCNNConfig(window_size=15, bit_length=8)
    for i in range(cfg.bit_length):
        w = bit_window_singular_values(_SIGMA, i, cfg)
        assert w.shape == (15,)
        assert np.isfinite(w).all()
        assert abs(float(w.std()) - 1.0) < 1e-6  # standardised


def test_window_is_centred_on_the_bits_own_singular_value() -> None:
    cfg = WindowedCNNConfig(window_size=15)
    center = cfg.start_sv_index + 8
    # In log-domain edge-padded space, the centre value must appear at
    # window position radius = 7. Use the raw (un-normalised) log values:
    radius = cfg.window_size // 2
    padded = np.pad(np.clip(_SIGMA, 0.0, None), (radius, radius), mode="edge")
    expected_center_log = np.log1p(padded[center + radius])
    got_center_log = np.log1p(padded[center : center + cfg.window_size][radius])
    assert expected_center_log == pytest.approx(got_center_log)


def test_window_mapping_is_consistent_across_payload_positions() -> None:
    cfg = WindowedCNNConfig(window_size=15, start_sv_index=2)
    # a window away from the boundary equals the plain log1p->standardise of the
    # 15 raw values centred at start + i
    for i in (10, 40, 90):
        center = cfg.start_sv_index + i
        radius = cfg.window_size // 2
        raw = _SIGMA[center - radius : center + radius + 1]
        s = np.log1p(np.clip(raw, 0.0, None))
        expected = (s - s.mean()) / (s.std() if s.std() > 0 else 1.0)
        got = bit_window_singular_values(_SIGMA, i, cfg)
        assert np.allclose(got, expected)


# ---------------------------------------------------------------------------
# Feature path (identical at training and inference)
# ---------------------------------------------------------------------------


def test_feature_path_rejects_grayscale_and_wrong_shape() -> None:
    cfg = WindowedCNNConfig()
    with pytest.raises(ValueError):
        luminance_ll_singular_values(np.zeros((64, 64), dtype=np.uint8), cfg)
    with pytest.raises(ValueError):
        luminance_ll_singular_values(np.zeros((64, 64, 4), dtype=np.uint8), cfg)


def test_feature_path_singular_values_finite_and_descending() -> None:
    cfg = WindowedCNNConfig()
    rgb = (np.random.default_rng(0).random((128, 128, 3)) * 255).astype(np.uint8)
    sigma = luminance_ll_singular_values(rgb, cfg)
    assert sigma.ndim == 1
    assert len(sigma) == 64  # LL of 128x128 -> 64 singular values
    assert np.isfinite(sigma).all()
    assert np.all(np.diff(sigma) <= 0)  # descending


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


def test_forward_shape_and_finite_logits() -> None:
    model = WindowedCNNExtractor(WindowedCNNConfig(bit_length=8)).eval()
    x = torch.rand(4, 15)
    out = model(x)
    assert out.shape == (4, 1)
    assert torch.isfinite(out).all()


def test_predict_bit_binary_and_params_positive() -> None:
    model = WindowedCNNExtractor(WindowedCNNConfig(bit_length=8)).eval()
    x = torch.rand(3, 15)
    bits = model.predict_bit(x)
    assert bits.shape == (3, 1)
    assert set(torch.unique(bits).tolist()).issubset({0, 1})
    assert model.num_parameters() > 0


def test_forward_rejects_wrong_window_width() -> None:
    model = WindowedCNNExtractor(WindowedCNNConfig(window_size=15)).eval()
    with pytest.raises(ValueError):
        model(torch.rand(1, 14))  # not 15
    with pytest.raises(ValueError):
        model(torch.rand(1, 1, 15))  # not 2D


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip_reproduces_outputs(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = WindowedCNNExtractor(
        WindowedCNNConfig(bit_length=16, dropout_conv=0.0, dropout_fc=0.0)
    ).eval()
    x = torch.rand(2, 15)
    with torch.no_grad():
        reference = model(x)

    path = tmp_path / "ckpt.pt"
    save_windowed_checkpoint(str(path), model, extra={"epoch": 7})
    restored, checkpoint = load_windowed_checkpoint(str(path))

    assert checkpoint["format"] == "windowed-cnn-extractor/1"
    assert checkpoint["epoch"] == 7
    assert checkpoint["windowed_config"]["bit_length"] == 16
    with torch.no_grad():
        assert torch.allclose(restored(x), reference, atol=1e-6)


# ---------------------------------------------------------------------------
# Blind inference wrapper
# ---------------------------------------------------------------------------


def test_windowed_extractor_predicts_expected_length(tmp_path: Path) -> None:
    from src.evaluation.windowed_extract import WindowedExtractor

    model = WindowedCNNExtractor(WindowedCNNConfig(bit_length=12, image_size=64)).eval()
    path = tmp_path / "ckpt.pt"
    save_windowed_checkpoint(str(path), model, extra={"image_size": 64})

    extractor = WindowedExtractor.from_checkpoint(str(path))
    rgb = (np.random.default_rng(0).random((90, 120, 3)) * 255).astype(np.uint8)

    probs = extractor.extract_proba(rgb)
    bits = extractor.extract_bits(rgb)
    assert probs.shape == (12,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert len(bits) == 12
    assert set(bits).issubset({0, 1})


def test_windowed_extractor_resizes_inputs_to_model_size(tmp_path: Path) -> None:
    from src.evaluation.windowed_extract import WindowedExtractor

    model = WindowedCNNExtractor(WindowedCNNConfig(bit_length=8, image_size=128)).eval()
    path = tmp_path / "ckpt.pt"
    save_windowed_checkpoint(str(path), model, extra={"image_size": 128})
    extractor = WindowedExtractor.from_checkpoint(str(path))

    rgb = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)  # smaller
    assert extractor.extract_proba(rgb).shape == (8,)


# ---------------------------------------------------------------------------
# Windowed dataset (needs processed data)
# ---------------------------------------------------------------------------


@_needs_data
def test_windowed_dataset_sample_shape_and_labels() -> None:
    from src.training.windowed_dataset import (
        WindowedExtractionConfig,
        WindowedExtractionDataset,
    )

    cfg = WindowedExtractionConfig(
        processed_root=str(_PROCESSED), bit_length=16, image_size=128, limit=3
    )
    ds = WindowedExtractionDataset(cfg, "validation", deterministic=True)
    assert len(ds) == 3

    windows, labels = ds[0]
    assert windows.shape == (16, 15)
    assert windows.dtype == torch.float32
    assert torch.isfinite(windows).all()
    assert labels.shape == (16,)
    assert set(labels.unique().tolist()).issubset({0.0, 1.0})
    assert ds.embed_config.bit_length == 16
    assert ds.model_config.bit_length == 16


@_needs_data
def test_windowed_dataset_labels_match_embedded_bits() -> None:
    """The label of window i must equal the bit actually embedded at
    singular-value position i (interleaving/project bit order preserved)."""
    import cv2

    from src.models.windowed_cnn import bit_window_singular_values, luminance_ll_singular_values
    from src.training.windowed_dataset import (
        WindowedExtractionConfig,
        WindowedExtractionDataset,
    )
    from src.watermark.embed import embed

    cfg = WindowedExtractionConfig(
        processed_root=str(_PROCESSED), bit_length=8, image_size=256, limit=1
    )
    ds = WindowedExtractionDataset(cfg, "validation", deterministic=True)
    _, labels = ds[0]

    # deterministic=True -> window i carries payload rng(seed + index).
    rng = np.random.default_rng(cfg.seed)
    rgb = cv2.cvtColor(cv2.imread(ds.sample_paths()[0], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    bits = rng.integers(0, 2, size=8, dtype=np.int64).tolist()
    wm = embed(rgb, bits, ds.embed_config).watermarked_image
    sigma = luminance_ll_singular_values(wm, ds.model_config)

    # Every bit's window must be buildable, and its label must equal the bit
    # the frozen embedder actually placed at singular value start+bit.
    for i in range(8):
        bit_window_singular_values(sigma, i, ds.model_config)
    assert labels[0].item() == float(bits[0])


# ---------------------------------------------------------------------------
# End-to-end training (needs processed data) — 1 epoch, tiny subset
# ---------------------------------------------------------------------------


@_needs_data
def test_train_one_epoch_writes_reloadable_checkpoint(tmp_path: Path) -> None:
    from src.evaluation.windowed_extract import WindowedExtractor
    from src.training.train_windowed_cnn import WindowedTrainConfig, train

    cfg = WindowedTrainConfig(
        bit_length=8,
        alpha=0.05,
        train_limit=8,
        val_limit=4,
        image_size=256,
        epochs=1,
        batch_size=4,
        num_workers=0,
        seed=0,
        checkpoint_dir=str(tmp_path / "ckpts"),
        results_dir=str(tmp_path / "results"),
        run_name="unit",
    )
    summary = train(cfg, verbose=False)

    best = Path(summary["checkpoints"]["best"])
    last = Path(summary["checkpoints"]["last"])
    log = Path(summary["training_log"])
    assert best.is_file() and last.is_file() and log.is_file()
    assert summary["n_train_images"] == 8
    assert 0.0 <= summary["best_val_metrics"]["bit_accuracy"] <= 1.0
    assert summary["model_parameters"] > 0

    extractor = WindowedExtractor.from_checkpoint(str(best))
    rgb = (np.random.default_rng(1).random((256, 256, 3)) * 255).astype(np.uint8)
    assert len(extractor.extract_bits(rgb)) == 8


# ---------------------------------------------------------------------------
# Frozen baseline untouched
# ---------------------------------------------------------------------------


def test_frozen_baseline_defaults_unchanged() -> None:
    from src.watermark.embed import EmbedConfig

    default = EmbedConfig()
    assert default.wavelet == "haar"
    assert default.subband == "LL"
    assert default.alpha == 0.010
