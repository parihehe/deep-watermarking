"""Tests for Phase 9 - Baseline Validation & Robustness Preparation.

Three layers:
  * pure harness logic (no data): metric taxonomy, config, deterministic
    payloads, frozen-baseline guard;
  * data-dependent end-to-end checks against the real DIV2K test split, skipped
    when that split is absent;
  * CNN-dependent checks against the trained Phase 8 checkpoint, skipped when it
    is absent.

The frozen Phase 6 baseline and the Phase 8 model are exercised only through
their existing public API - nothing here modifies them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation import phase9_validation as p9
from src.evaluation.phase9_validation import (
    IMAGE_QUALITY_METRICS,
    METRIC_GROUPS,
    WATERMARK_RECOVERY_METRICS,
    Phase9Config,
    assert_baseline_frozen,
    build_baseline_embed_config,
    evaluate_baseline_point,
    evaluate_cnn_point,
    frozen_file_hashes,
    load_config,
    load_split_images,
    master_payload,
    payload_bits,
    summarise_baseline,
    summarise_cnn,
)
from src.watermark.embed import EmbedConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "phase9_validation.yaml"
_TEST_SPLIT = PROJECT_ROOT / "data" / "processed" / "div2k_256" / "test"
_CNN_CKPT = PROJECT_ROOT / "models" / "phase8_cnn" / "phase8_cnn_best.pt"

_needs_data = pytest.mark.skipif(
    not _TEST_SPLIT.is_dir() or not any(_TEST_SPLIT.glob("*.png")),
    reason="processed DIV2K test split not available",
)
_needs_cnn = pytest.mark.skipif(
    not _CNN_CKPT.is_file(), reason="Phase 8 CNN checkpoint not available"
)


# ---------------------------------------------------------------------------
# Metric taxonomy
# ---------------------------------------------------------------------------

def test_metric_groups_are_the_two_required_families() -> None:
    assert METRIC_GROUPS["image_quality"] == ("psnr", "ssim", "mse")
    assert METRIC_GROUPS["watermark_recovery"] == ("ber", "bit_accuracy", "nc")


def test_metric_groups_are_disjoint() -> None:
    assert set(IMAGE_QUALITY_METRICS).isdisjoint(WATERMARK_RECOVERY_METRICS)


# ---------------------------------------------------------------------------
# Phase9Config
# ---------------------------------------------------------------------------

def test_config_defaults_match_the_requirement() -> None:
    cfg = Phase9Config()
    assert cfg.payload_bits == (8, 16, 32, 64, 128)
    assert cfg.alphas == (0.005, 0.010, 0.015)
    assert cfg.wavelet == "haar" and cfg.subband == "LL" and cfg.mode == "symmetric"
    assert cfg.eval_split == "test"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"payload_bits": ()},
        {"payload_bits": (8, -1)},
        {"payload_bits": (8, 256)},        # exceeds frozen LL-only capacity
        {"alphas": ()},
        {"alphas": (0.0,)},
        {"alphas": (1.0,)},
        {"eval_num_images": 0},
    ],
)
def test_config_rejects_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Phase9Config(**kwargs)


def test_load_config_from_repo_yaml() -> None:
    cfg = load_config(CONFIG_PATH)
    assert cfg.payload_bits == (8, 16, 32, 64, 128)
    assert cfg.alphas == (0.005, 0.010, 0.015)
    assert cfg.eval_split == "test"
    assert cfg.cnn.enabled is True
    assert cfg.cnn.bit_length == 64
    assert cfg.cnn.alpha == 0.02
    assert cfg.results_dir == "results/phase9_validation"


def test_load_config_quick_overrides() -> None:
    cfg = load_config(CONFIG_PATH, {
        "eval_num_images": 12,
        "payload_bits": (8, 32, 128),
        "alphas": (0.010,),
    })
    assert cfg.eval_num_images == 12
    assert cfg.payload_bits == (8, 32, 128)
    assert cfg.alphas == (0.010,)
    # untouched fields survive
    assert cfg.cnn.bit_length == 64


def test_resolved_dirs_point_at_phase9_only() -> None:
    cfg = Phase9Config()
    assert cfg.resolved_results_dir().name == "phase9_validation"
    assert cfg.resolved_split_dir().name == "test"


# ---------------------------------------------------------------------------
# Frozen-baseline guard
# ---------------------------------------------------------------------------

def test_assert_baseline_frozen_passes_on_this_repo() -> None:
    assert_baseline_frozen()  # must not raise


def test_assert_baseline_frozen_detects_alpha_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p9, "BASELINE_ALPHAS", (0.01, 0.02, 0.03))
    with pytest.raises(AssertionError):
        assert_baseline_frozen()


def test_frozen_file_hashes_cover_every_frozen_file() -> None:
    hashes = frozen_file_hashes()
    for rel in p9.FROZEN_BASELINE_FILES:
        assert rel in hashes
        assert len(hashes[rel]) == 64
        int(hashes[rel], 16)  # valid hex


# ---------------------------------------------------------------------------
# Deterministic payloads
# ---------------------------------------------------------------------------

def test_master_payload_is_deterministic() -> None:
    a = master_payload(20260901, 7)
    b = master_payload(20260901, 7)
    assert np.array_equal(a, b)
    assert a.shape == (128,)
    assert set(np.unique(a)).issubset({0, 1})


def test_master_payload_varies_by_image_index() -> None:
    assert not np.array_equal(master_payload(1, 0), master_payload(1, 1))


def test_payloads_nest_across_sizes() -> None:
    p16 = payload_bits(123, 4, 16)
    p64 = payload_bits(123, 4, 64)
    assert p64[:16] == p16
    assert len(p16) == 16 and len(p64) == 64


def test_payload_bits_rejects_over_capacity() -> None:
    with pytest.raises(ValueError):
        payload_bits(0, 0, 129)


# ---------------------------------------------------------------------------
# Frozen baseline embed config
# ---------------------------------------------------------------------------

def test_build_baseline_embed_config_is_ll_only() -> None:
    cfg = Phase9Config()
    econf = build_baseline_embed_config(cfg, 0.01, 128)
    assert isinstance(econf, EmbedConfig)
    assert econf.subband == "LL"
    assert econf.extra_subbands == ()
    assert econf.subband_order == ("LL",)
    assert econf.alpha == 0.01
    assert econf.bit_length == 128


def test_build_baseline_embed_config_rejects_over_capacity() -> None:
    with pytest.raises(ValueError):
        build_baseline_embed_config(Phase9Config(), 0.01, 256)


# ---------------------------------------------------------------------------
# Data-dependent: frozen DWT-SVD baseline evaluation
# ---------------------------------------------------------------------------

@_needs_data
def test_evaluate_baseline_point_row_shape_and_groups() -> None:
    images = load_split_images(_TEST_SPLIT, 3)
    econf = build_baseline_embed_config(Phase9Config(), 0.01, 32)
    rows = evaluate_baseline_point(images, econf, payload_seed=20260901)
    assert len(rows) == 3
    for row in rows:
        for metric in IMAGE_QUALITY_METRICS:
            assert f"quality_{metric}" in row
        for metric in WATERMARK_RECOVERY_METRICS:
            assert f"recovery_{metric}" in row
        assert row["decoder"] == "non_blind"
        assert row["payload_bits"] == 32
        assert np.isfinite(row["quality_psnr"])
        assert row["quality_psnr"] > 35.0          # frozen baseline is ~45 dB @ alpha 0.01
        assert 0.0 <= row["recovery_ber"] <= 1.0


@_needs_data
def test_baseline_recovers_eight_bits_exactly() -> None:
    """Documented frozen-baseline property: 8-bit payloads are recovered with
    BER 0.0 at every baseline alpha (non-blind, clean channel)."""
    images = load_split_images(_TEST_SPLIT, 5)
    for alpha in (0.005, 0.010, 0.015):
        econf = build_baseline_embed_config(Phase9Config(), alpha, 8)
        rows = evaluate_baseline_point(images, econf, payload_seed=20260901)
        assert all(r["recovery_ber"] == 0.0 for r in rows)


@_needs_data
def test_evaluate_baseline_point_is_deterministic() -> None:
    images = load_split_images(_TEST_SPLIT, 4)
    econf = build_baseline_embed_config(Phase9Config(), 0.015, 64)
    a = evaluate_baseline_point(images, econf, payload_seed=20260901)
    b = evaluate_baseline_point(images, econf, payload_seed=20260901)
    assert a == b


@_needs_data
def test_summarise_baseline_keeps_groups_separate() -> None:
    images = load_split_images(_TEST_SPLIT, 4)
    rows: list[dict] = []
    for alpha in (0.005, 0.010):
        for n_bits in (8, 32):
            econf = build_baseline_embed_config(Phase9Config(), alpha, n_bits)
            rows.extend(evaluate_baseline_point(images, econf, payload_seed=20260901))
    summary = summarise_baseline(rows)
    assert len(summary) == 4  # 2 alphas x 2 payloads
    for entry in summary:
        assert entry["n_images"] == 4
        assert entry["decoder"] == "non_blind"
        assert set(entry["image_quality"]) == {
            f"{m}_{stat}" for m in IMAGE_QUALITY_METRICS for stat in ("mean", "std")
        }
        assert set(entry["watermark_recovery"]) == {
            f"{m}_{stat}" for m in WATERMARK_RECOVERY_METRICS for stat in ("mean", "std")
        }


# ---------------------------------------------------------------------------
# CNN-dependent: Phase 8 blind extractor evaluation (architecture unchanged)
# ---------------------------------------------------------------------------

@_needs_data
@_needs_cnn
def test_evaluate_cnn_point_produces_blind_and_nonblind_recovery() -> None:
    pytest.importorskip("torch")
    from src.evaluation.blind_extract import BlindExtractor

    images = load_split_images(_TEST_SPLIT, 6)
    cnn_cfg = Phase9Config().cnn
    extractor = BlindExtractor.from_checkpoint(str(_CNN_CKPT))
    rows = evaluate_cnn_point(images, extractor, cnn_config=cnn_cfg)
    assert len(rows) == 6
    for row in rows:
        for prefix in ("blind", "nonblind"):
            for metric in WATERMARK_RECOVERY_METRICS:
                assert f"{prefix}_{metric}" in row
            assert 0.0 <= row[f"{prefix}_ber"] <= 1.0
            assert 0.0 <= row[f"{prefix}_bit_accuracy"] <= 1.0
            assert -1.0 <= row[f"{prefix}_nc"] <= 1.0
        assert row["payload_bits"] == cnn_cfg.bit_length

    summary = summarise_cnn(rows, cnn_cfg)
    blind = summary["watermark_recovery_blind_cnn"]
    ref = summary["watermark_recovery_non_blind_reference"]
    # The non-blind decoder is the strong reference at this operating point; the
    # learned blind decoder is clearly above chance but weaker (consistent with
    # the Phase 8 result of ~0.78 bit-accuracy). Thresholds are deliberately
    # loose - this is a smoke check, the real numbers live in the JSON report.
    assert ref["ber_mean"] < blind["ber_mean"]
    assert blind["ber_mean"] < 0.45
    assert blind["bit_accuracy_mean"] > 0.55


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def test_runner_flatten_baseline_summary() -> None:
    from experiments.run_phase9_validation import _flatten_baseline_summary

    summary = [{
        "alpha": 0.01, "payload_bits": 8, "subband_order": "LL",
        "decoder": "non_blind", "n_images": 3,
        "image_quality": {"psnr_mean": 45.0, "psnr_std": 1.0, "ssim_mean": 0.99,
                          "ssim_std": 0.0, "mse_mean": 2.0, "mse_std": 0.1},
        "watermark_recovery": {"ber_mean": 0.0, "ber_std": 0.0, "bit_accuracy_mean": 1.0,
                               "bit_accuracy_std": 0.0, "nc_mean": 1.0, "nc_std": 0.0},
    }]
    flat = _flatten_baseline_summary(summary)[0]
    assert flat["quality_psnr_mean"] == 45.0
    assert flat["recovery_ber_mean"] == 0.0
    assert flat["decoder"] == "non_blind"
