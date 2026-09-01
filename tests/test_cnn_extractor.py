"""Tests for Phase 8 — blind CNN watermark extractor.

Covers the architecture, checkpoint I/O and the blind-inference wrapper with no
data dependency, plus the on-the-fly watermark dataset and a 1-epoch end-to-end
training run that are skipped when the processed DIV2K split is absent.

The frozen Phase 6 baseline and the Phase 7 web app are exercised only through
their existing public API — nothing here modifies them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.cnn_extractor import (
    BlindCNNExtractor,
    ExtractorConfig,
    load_checkpoint,
    save_checkpoint,
)

_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed" / "div2k_256"
_VAL_DIR = _PROCESSED / "validation"
_needs_data = pytest.mark.skipif(
    not _VAL_DIR.is_dir() or not any(_VAL_DIR.glob("*.png")),
    reason="processed DIV2K validation split not available",
)


# ---------------------------------------------------------------------------
# ExtractorConfig
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"bit_length": 0},
        {"bit_length": -8},
        {"in_channels": 1},
        {"sv_features": 0},
        {"conv_channels": 0},
        {"conv_layers": 0},
        {"kernel_size": 4},
        {"kernel_size": 0},
        {"dropout": 1.0},
        {"dropout": -0.1},
        {"image_size": 0},
    ],
)
def test_extractor_config_rejects_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ExtractorConfig(**kwargs)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bit_length", [8, 16, 64, 128])
def test_forward_output_shape_matches_bit_length(bit_length: int) -> None:
    model = BlindCNNExtractor(ExtractorConfig(bit_length=bit_length)).eval()
    x = torch.rand(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, bit_length)
    assert torch.isfinite(out).all()


def test_model_is_resolution_agnostic() -> None:
    model = BlindCNNExtractor(ExtractorConfig(bit_length=16)).eval()
    for size in (32, 64, 128):
        out = model(torch.rand(1, 3, size, size))
        assert out.shape == (1, 16)


def test_predict_bits_are_binary_and_num_parameters_positive() -> None:
    model = BlindCNNExtractor(ExtractorConfig(bit_length=32)).eval()
    bits = model.predict_bits(torch.rand(4, 3, 64, 64))
    assert bits.shape == (4, 32)
    assert set(torch.unique(bits).tolist()).issubset({0, 1})
    assert model.num_parameters() > 0


def test_forward_rejects_wrong_rank_and_channels() -> None:
    model = BlindCNNExtractor(ExtractorConfig(bit_length=8)).eval()
    with pytest.raises(ValueError):
        model(torch.rand(3, 64, 64))
    with pytest.raises(ValueError):
        model(torch.rand(1, 1, 64, 64))


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def test_checkpoint_round_trip_reproduces_outputs(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = BlindCNNExtractor(ExtractorConfig(bit_length=24, dropout=0.0)).eval()
    x = torch.rand(2, 3, 48, 48)
    with torch.no_grad():
        reference = model(x)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), model, extra={"epoch": 7})
    restored, checkpoint = load_checkpoint(str(path))

    assert checkpoint["phase"] == 8
    assert checkpoint["epoch"] == 7
    assert checkpoint["extractor_config"]["bit_length"] == 24
    with torch.no_grad():
        assert torch.allclose(restored(x), reference, atol=1e-6)


# ---------------------------------------------------------------------------
# Blind inference wrapper
# ---------------------------------------------------------------------------

def test_blind_extractor_from_checkpoint_predicts_expected_length(tmp_path: Path) -> None:
    from src.evaluation.blind_extract import BlindExtractor

    model = BlindCNNExtractor(ExtractorConfig(bit_length=20, image_size=64)).eval()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), model, extra={"image_size": 64})

    extractor = BlindExtractor.from_checkpoint(str(path))
    rgb = (np.random.default_rng(0).random((90, 120, 3)) * 255).astype(np.uint8)

    probs = extractor.extract_proba(rgb)
    bits = extractor.extract_bits(rgb)
    assert probs.shape == (20,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert len(bits) == 20
    assert set(bits).issubset({0, 1})


# ---------------------------------------------------------------------------
# Watermark dataset (needs processed data)
# ---------------------------------------------------------------------------

@_needs_data
def test_watermark_dataset_sample_shape_and_payload() -> None:
    from src.training.watermark_dataset import (
        WatermarkExtractionConfig,
        WatermarkExtractionDataset,
    )

    cfg = WatermarkExtractionConfig(
        processed_root=str(_PROCESSED), bit_length=32, image_size=256, limit=4
    )
    ds = WatermarkExtractionDataset(cfg, "validation", deterministic=True)
    assert len(ds) == 4

    image, target = ds[0]
    assert image.shape == (3, 256, 256)
    assert image.dtype == torch.float32
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0
    assert target.shape == (32,)
    assert set(target.unique().tolist()).issubset({0.0, 1.0})
    # the embed config payload length is aligned to the requested bit_length
    assert ds.embed_config.bit_length == 32


@_needs_data
def test_watermark_dataset_deterministic_vs_random_payloads() -> None:
    from src.training.watermark_dataset import (
        WatermarkExtractionConfig,
        WatermarkExtractionDataset,
    )

    cfg = WatermarkExtractionConfig(
        processed_root=str(_PROCESSED), bit_length=64, image_size=128, limit=2
    )
    fixed = WatermarkExtractionDataset(cfg, "validation", deterministic=True)
    assert torch.equal(fixed[0][1], fixed[0][1])  # same payload every call

    resampled = WatermarkExtractionDataset(cfg, "validation", deterministic=False)
    draws = {tuple(resampled[0][1].tolist()) for _ in range(5)}
    assert len(draws) > 1  # fresh payload per __getitem__


@_needs_data
def test_watermarked_tensor_differs_from_plain_cover() -> None:
    import cv2

    from src.training.watermark_dataset import (
        WatermarkExtractionConfig,
        WatermarkExtractionDataset,
    )

    cfg = WatermarkExtractionConfig(
        processed_root=str(_PROCESSED), bit_length=32, image_size=256, limit=1
    )
    ds = WatermarkExtractionDataset(cfg, "validation", deterministic=True)
    watermarked, _ = ds[0]

    bgr = cv2.imread(ds.sample_paths()[0], cv2.IMREAD_COLOR)
    plain = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    plain_t = torch.from_numpy(plain).permute(2, 0, 1).float().div(255.0)

    assert not torch.equal(watermarked, plain_t)
    assert torch.mean((watermarked - plain_t).abs()).item() < 0.05  # sub-perceptual


# ---------------------------------------------------------------------------
# End-to-end training (needs processed data) — 1 epoch, tiny subset
# ---------------------------------------------------------------------------

@_needs_data
def test_train_one_epoch_writes_reloadable_checkpoint(tmp_path: Path) -> None:
    from src.evaluation.blind_extract import BlindExtractor
    from src.training.train_extractor import TrainConfig, train

    cfg = TrainConfig(
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

    extractor = BlindExtractor.from_checkpoint(str(best))
    rgb = (np.random.default_rng(1).random((256, 256, 3)) * 255).astype(np.uint8)
    assert len(extractor.extract_bits(rgb)) == 8


# ---------------------------------------------------------------------------
# Frozen baseline / Phase 7 API is untouched
# ---------------------------------------------------------------------------

def test_frozen_baseline_and_app_imports_still_work() -> None:
    from src.app.main import app
    from src.watermark.embed import EmbedConfig

    # Phase 6 defaults are unchanged.
    default = EmbedConfig()
    assert default.wavelet == "haar"
    assert default.subband == "LL"
    assert default.alpha == 0.010
    assert any(route.path == "/api/watermark/embed" for route in app.routes)
