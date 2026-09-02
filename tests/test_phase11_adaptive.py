"""Tests for Phase 11 - Adaptive, Content-Aware Embedding (harness).

Two layers: pure harness logic (no data) and data-dependent end-to-end checks
against the real DIV2K test split, skipped when that split is absent. Nothing
here modifies the frozen Phase 6 baseline or the Phase 8 CNN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.phase11_adaptive import (
    IMAGE_QUALITY_METRICS,
    WATERMARK_RECOVERY_METRICS,
    Phase11Config,
    adaptive_config_for,
    evaluate_alpha_point,
    evaluate_robustness_point,
    load_config,
    load_split_images,
    summarise_alpha_sweep,
    summarise_robustness,
)
from src.watermark.adaptive_embed import AdaptiveEmbedConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "phase11_adaptive.yaml"
_TEST_SPLIT = PROJECT_ROOT / "data" / "processed" / "div2k_256" / "test"

_needs_data = pytest.mark.skipif(
    not _TEST_SPLIT.is_dir() or not any(_TEST_SPLIT.glob("*.png")),
    reason="processed DIV2K test split not available",
)


# ---------------------------------------------------------------------------
# Phase11Config
# ---------------------------------------------------------------------------

def test_config_defaults_match_established_operating_point() -> None:
    cfg = Phase11Config()
    assert cfg.block_size == 16
    assert cfg.bit_length == 64
    assert cfg.alphas == (0.005, 0.010, 0.015, 0.020)
    assert cfg.wavelet == "haar" and cfg.mode == "symmetric"
    assert cfg.eval_split == "test"


def test_config_rejects_mismatched_bit_length_for_block_grid() -> None:
    with pytest.raises(ValueError):
        Phase11Config(block_size=32, bit_length=64)  # 32 -> 4x4=16 blocks, not 64


@pytest.mark.parametrize("kwargs", [
    {"alphas": ()},
    {"alphas": (0.0,)},
    {"alphas": (1.0,)},
    {"robustness_alpha": 0.0},
    {"eval_num_images": 0},
])
def test_config_rejects_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Phase11Config(**kwargs)


def test_load_config_from_repo_yaml() -> None:
    cfg = load_config(CONFIG_PATH)
    assert cfg.block_size == 16
    assert cfg.bit_length == 64
    assert cfg.alphas == (0.005, 0.010, 0.015, 0.020)
    assert cfg.eval_split == "test"
    assert cfg.robustness_alpha == 0.020
    assert "jpeg_compress" in cfg.robustness_attacks
    assert cfg.results_dir == "results/phase11_adaptive"


def test_load_config_overrides() -> None:
    cfg = load_config(CONFIG_PATH, {"eval_num_images": 12, "alphas": (0.010, 0.020)})
    assert cfg.eval_num_images == 12
    assert cfg.alphas == (0.010, 0.020)
    # untouched fields survive
    assert cfg.block_size == 16


def test_resolved_dirs_point_at_phase11_only() -> None:
    cfg = Phase11Config()
    assert cfg.resolved_results_dir().name == "phase11_adaptive"
    assert cfg.resolved_split_dir().name == "test"


# ---------------------------------------------------------------------------
# adaptive_config_for
# ---------------------------------------------------------------------------

def test_adaptive_config_for_builds_expected_embed_config() -> None:
    cfg = Phase11Config()
    econf = adaptive_config_for(cfg, 0.015, "adaptive")
    assert isinstance(econf, AdaptiveEmbedConfig)
    assert econf.alpha == 0.015
    assert econf.alpha_mode == "adaptive"
    assert econf.block_size == cfg.block_size
    assert econf.bit_length == cfg.bit_length


def test_adaptive_config_for_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        adaptive_config_for(Phase11Config(), 0.02, "learned")


# ---------------------------------------------------------------------------
# Metric taxonomy
# ---------------------------------------------------------------------------

def test_metric_taxonomy_matches_project_convention() -> None:
    assert IMAGE_QUALITY_METRICS == ("psnr", "ssim", "mse")
    assert WATERMARK_RECOVERY_METRICS == ("ber", "bit_accuracy", "nc")


# ---------------------------------------------------------------------------
# Data-dependent: alpha-sweep evaluation
# ---------------------------------------------------------------------------

@_needs_data
def test_evaluate_alpha_point_row_shape() -> None:
    images = load_split_images(_TEST_SPLIT, 3)
    cfg = Phase11Config()
    econf = adaptive_config_for(cfg, 0.02, "fixed")
    rows = evaluate_alpha_point(images, econf, payload_seed=cfg.payload_seed)
    assert len(rows) == 3
    for row in rows:
        for metric in IMAGE_QUALITY_METRICS:
            assert f"quality_{metric}" in row
        for metric in WATERMARK_RECOVERY_METRICS:
            assert f"recovery_{metric}" in row
        assert row["alpha_mode"] == "fixed"
        assert row["alpha_realised_std"] == 0.0  # fixed mode: constant alpha
        assert np.isfinite(row["quality_psnr"])
        assert 0.0 <= row["recovery_ber"] <= 1.0


@_needs_data
def test_evaluate_alpha_point_adaptive_has_nonzero_realised_std() -> None:
    images = load_split_images(_TEST_SPLIT, 3)
    cfg = Phase11Config()
    econf = adaptive_config_for(cfg, 0.02, "adaptive")
    rows = evaluate_alpha_point(images, econf, payload_seed=cfg.payload_seed)
    assert any(r["alpha_realised_std"] > 0.0 for r in rows)
    # mean-preserving: realised mean alpha equals the configured alpha
    for r in rows:
        assert r["alpha_realised_mean"] == pytest.approx(0.02, abs=1e-6)


@_needs_data
def test_evaluate_alpha_point_is_deterministic() -> None:
    images = load_split_images(_TEST_SPLIT, 3)
    cfg = Phase11Config()
    econf = adaptive_config_for(cfg, 0.015, "adaptive")
    a = evaluate_alpha_point(images, econf, payload_seed=20260901)
    b = evaluate_alpha_point(images, econf, payload_seed=20260901)
    assert a == b


@_needs_data
def test_summarise_alpha_sweep_covers_both_modes() -> None:
    images = load_split_images(_TEST_SPLIT, 3)
    cfg = Phase11Config()
    rows = []
    for alpha in (0.010, 0.020):
        for mode in ("fixed", "adaptive"):
            econf = adaptive_config_for(cfg, alpha, mode)
            rows.extend(evaluate_alpha_point(images, econf, payload_seed=cfg.payload_seed))
    summary = summarise_alpha_sweep(rows)
    assert len(summary) == 4  # 2 alphas x 2 modes
    for entry in summary:
        assert entry["n_images"] == 3
        assert set(entry["image_quality"]) == {f"{m}_{s}" for m in IMAGE_QUALITY_METRICS for s in ("mean", "std")}
        assert set(entry["watermark_recovery"]) == {f"{m}_{s}" for m in WATERMARK_RECOVERY_METRICS for s in ("mean", "std")}


# ---------------------------------------------------------------------------
# Data-dependent: robustness stress evaluation
# ---------------------------------------------------------------------------

@_needs_data
def test_evaluate_robustness_point_row_shape() -> None:
    images = load_split_images(_TEST_SPLIT, 3)
    cfg = Phase11Config()
    econf = adaptive_config_for(cfg, cfg.robustness_alpha, "fixed")
    rows = evaluate_robustness_point(
        images, econf, "jpeg_compress", {"quality": 75},
        payload_seed=cfg.payload_seed, noise_seed=cfg.noise_seed,
    )
    assert len(rows) == 3
    for row in rows:
        assert row["attack"] == "jpeg_compress"
        assert 0.0 <= row["recovery_ber"] <= 1.0


@_needs_data
def test_evaluate_robustness_point_gaussian_noise_is_seeded() -> None:
    images = load_split_images(_TEST_SPLIT, 2)
    cfg = Phase11Config()
    econf = adaptive_config_for(cfg, cfg.robustness_alpha, "fixed")
    a = evaluate_robustness_point(images, econf, "gaussian_noise", {"sigma": 5},
                                   payload_seed=cfg.payload_seed, noise_seed=1234)
    b = evaluate_robustness_point(images, econf, "gaussian_noise", {"sigma": 5},
                                   payload_seed=cfg.payload_seed, noise_seed=1234)
    assert a == b


@_needs_data
def test_summarise_robustness_covers_both_modes() -> None:
    images = load_split_images(_TEST_SPLIT, 3)
    cfg = Phase11Config()
    rows = []
    for mode in ("fixed", "adaptive"):
        econf = adaptive_config_for(cfg, cfg.robustness_alpha, mode)
        rows.extend(evaluate_robustness_point(
            images, econf, "gaussian_blur", {"ksize": 3},
            payload_seed=cfg.payload_seed, noise_seed=cfg.noise_seed,
        ))
    summary = summarise_robustness(rows)
    assert len(summary) == 2  # 1 attack x 2 modes
    assert {e["alpha_mode"] for e in summary} == {"fixed", "adaptive"}
