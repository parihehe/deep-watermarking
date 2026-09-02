"""Tests for Phase 12 - High-Capacity Watermarking (harness).

Two layers: pure harness logic (no data) and data-dependent end-to-end checks
against the real DIV2K test split, skipped when that split is absent. Nothing
here modifies the frozen Phase 6 baseline or the Phase 8 CNN.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.phase12_capacity import (
    IMAGE_QUALITY_METRICS,
    WATERMARK_RECOVERY_METRICS,
    ExperimentSpec,
    Phase12Config,
    capacity_bound_table,
    evaluate_experiment,
    load_config,
    load_split_images,
    payload_for_image,
    summarise_experiment,
)
from src.watermark.capacity_embed import CapacityEmbedConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "phase12_capacity.yaml"
_TEST_SPLIT = PROJECT_ROOT / "data" / "processed" / "div2k_256" / "test"

_needs_data = pytest.mark.skipif(
    not _TEST_SPLIT.is_dir() or not any(_TEST_SPLIT.glob("*.png")),
    reason="processed DIV2K test split not available",
)


# ---------------------------------------------------------------------------
# ExperimentSpec
# ---------------------------------------------------------------------------

def test_experiment_spec_builds_embed_config() -> None:
    spec = ExperimentSpec(name="x", dwt_levels=1, subband_plan=((1, "LL"),), bit_length=128, alpha=0.02)
    econf = spec.embed_config()
    assert isinstance(econf, CapacityEmbedConfig)
    assert econf.dwt_levels == 1
    assert econf.bit_length == 128
    assert econf.alpha == 0.02


# ---------------------------------------------------------------------------
# Phase12Config / load_config
# ---------------------------------------------------------------------------

def test_config_rejects_empty_experiments() -> None:
    with pytest.raises(ValueError):
        Phase12Config(experiments=())


def test_config_rejects_duplicate_experiment_names() -> None:
    spec = ExperimentSpec(name="dup", dwt_levels=1, subband_plan=((1, "LL"),), bit_length=128)
    with pytest.raises(ValueError):
        Phase12Config(experiments=(spec, spec))


def test_config_rejects_non_positive_eval_num_images() -> None:
    spec = ExperimentSpec(name="x", dwt_levels=1, subband_plan=((1, "LL"),), bit_length=128)
    with pytest.raises(ValueError):
        Phase12Config(experiments=(spec,), eval_num_images=0)


def test_load_config_from_repo_yaml_covers_all_four_target_payloads() -> None:
    cfg = load_config(CONFIG_PATH)
    bit_lengths = {e.bit_length for e in cfg.experiments}
    assert bit_lengths == {128, 256, 512, 1024}
    names = [e.name for e in cfg.experiments]
    assert len(names) == len(set(names))
    assert cfg.results_dir == "results/phase12_capacity"


def test_load_config_1024_experiments_use_multi_level_attempts() -> None:
    cfg = load_config(CONFIG_PATH)
    attempts_1024 = [e for e in cfg.experiments if e.bit_length == 1024]
    assert len(attempts_1024) == 3
    assert {e.dwt_levels for e in attempts_1024} == {1, 2, 3}


def test_load_config_overrides() -> None:
    cfg = load_config(CONFIG_PATH, {"eval_num_images": 5})
    assert cfg.eval_num_images == 5
    assert len(cfg.experiments) >= 4  # untouched


def test_resolved_dirs_point_at_phase12_only() -> None:
    cfg = load_config(CONFIG_PATH)
    assert cfg.resolved_results_dir().name == "phase12_capacity"
    assert cfg.resolved_split_dir().name == "test"


# ---------------------------------------------------------------------------
# payload_for_image (reuses the frozen Phase 3 generator)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_bits", [128, 256, 512, 1024])
def test_payload_for_image_supports_all_target_bit_lengths(n_bits: int) -> None:
    bits = payload_for_image(20260901, 0, n_bits)
    assert len(bits) == n_bits
    assert set(bits).issubset({0, 1})


def test_payload_for_image_rejects_unsupported_bit_length() -> None:
    with pytest.raises(ValueError):
        payload_for_image(20260901, 0, 384)  # not in SUPPORTED_BIT_LENGTHS


def test_payload_for_image_is_deterministic() -> None:
    a = payload_for_image(20260901, 3, 256)
    b = payload_for_image(20260901, 3, 256)
    assert a == b


def test_payload_for_image_varies_by_index() -> None:
    a = payload_for_image(20260901, 0, 128)
    b = payload_for_image(20260901, 1, 128)
    assert a != b


# ---------------------------------------------------------------------------
# capacity_bound_table
# ---------------------------------------------------------------------------

def test_capacity_bound_table_shows_1024_is_unreachable() -> None:
    table = capacity_bound_table(256, max_levels=6)
    assert all(row["max_capacity_bits"] < 1024 for row in table)
    assert table[-1]["dwt_levels"] == "infinity"
    assert table[-1]["max_capacity_bits"] == 768


# ---------------------------------------------------------------------------
# Metric taxonomy
# ---------------------------------------------------------------------------

def test_metric_taxonomy_matches_project_convention() -> None:
    assert IMAGE_QUALITY_METRICS == ("psnr", "ssim", "mse")
    assert WATERMARK_RECOVERY_METRICS == ("ber", "bit_accuracy", "nc")


# ---------------------------------------------------------------------------
# Data-dependent: evaluate_experiment / summarise_experiment
# ---------------------------------------------------------------------------

@_needs_data
def test_evaluate_experiment_success_row_shape() -> None:
    images = load_split_images(_TEST_SPLIT, 3)
    spec = ExperimentSpec(name="t128", dwt_levels=1, subband_plan=((1, "LL"),), bit_length=128, alpha=0.02)
    outcome = evaluate_experiment(images, spec, payload_seed=20260901, image_size=256)
    assert outcome["status"] == "success"
    assert outcome["n_images"] == 3
    assert len(outcome["rows"]) == 3
    for row in outcome["rows"]:
        for metric in IMAGE_QUALITY_METRICS:
            assert f"quality_{metric}" in row
        for metric in WATERMARK_RECOVERY_METRICS:
            assert f"recovery_{metric}" in row
        assert 0.0 <= row["recovery_ber"] <= 1.0


@_needs_data
def test_evaluate_experiment_insufficient_capacity_row_shape() -> None:
    images = load_split_images(_TEST_SPLIT, 2)
    spec = ExperimentSpec(
        name="t1024fail", dwt_levels=1,
        subband_plan=((1, "LL"), (1, "LH"), (1, "HL"), (1, "HH")),
        bit_length=1024, alpha=0.02,
    )
    outcome = evaluate_experiment(images, spec, payload_seed=20260901, image_size=256)
    assert outcome["status"] == "insufficient_capacity"
    assert outcome["n_images"] == 0
    assert outcome["rows"] == []
    assert outcome["capacity_available"] == 512
    assert "1024" in outcome["reason"]


@_needs_data
def test_evaluate_experiment_is_deterministic() -> None:
    images = load_split_images(_TEST_SPLIT, 3)
    spec = ExperimentSpec(name="t256", dwt_levels=1, subband_plan=((1, "LL"), (1, "HL")), bit_length=256, alpha=0.02)
    a = evaluate_experiment(images, spec, payload_seed=20260901, image_size=256)
    b = evaluate_experiment(images, spec, payload_seed=20260901, image_size=256)
    assert a["rows"] == b["rows"]


@_needs_data
def test_summarise_experiment_success_has_both_metric_groups() -> None:
    images = load_split_images(_TEST_SPLIT, 3)
    spec = ExperimentSpec(name="t128", dwt_levels=1, subband_plan=((1, "LL"),), bit_length=128, alpha=0.02)
    outcome = evaluate_experiment(images, spec, payload_seed=20260901, image_size=256)
    entry = summarise_experiment(outcome)
    assert entry["status"] == "success"
    assert set(entry["image_quality"]) == {f"{m}_{s}" for m in IMAGE_QUALITY_METRICS for s in ("mean", "std")}
    assert set(entry["watermark_recovery"]) == {f"{m}_{s}" for m in WATERMARK_RECOVERY_METRICS for s in ("mean", "std")}


@_needs_data
def test_summarise_experiment_failure_has_no_metric_groups() -> None:
    images = load_split_images(_TEST_SPLIT, 2)
    spec = ExperimentSpec(
        name="t1024fail", dwt_levels=2,
        subband_plan=((1, "LH"), (1, "HL"), (1, "HH"), (2, "LL"), (2, "LH"), (2, "HL"), (2, "HH")),
        bit_length=1024, alpha=0.02,
    )
    outcome = evaluate_experiment(images, spec, payload_seed=20260901, image_size=256)
    entry = summarise_experiment(outcome)
    assert entry["status"] == "insufficient_capacity"
    assert "image_quality" not in entry
    assert "watermark_recovery" not in entry
    assert "reason" in entry


# ---------------------------------------------------------------------------
# Full experiment matrix (data-dependent, small subset)
# ---------------------------------------------------------------------------

@_needs_data
def test_configured_experiments_128_256_512_succeed_and_1024_fails() -> None:
    """Direct check of the Phase 12 research requirement: 128/256/512 bits
    succeed; every 1024-bit attempt (1, 2, 3 DWT levels) fails explicitly."""
    cfg = load_config(CONFIG_PATH, {"eval_num_images": 2})
    images = load_split_images(_TEST_SPLIT, cfg.eval_num_images)
    outcomes = {
        spec.name: evaluate_experiment(images, spec, payload_seed=cfg.payload_seed, image_size=cfg.image_size)
        for spec in cfg.experiments
    }
    for spec in cfg.experiments:
        outcome = outcomes[spec.name]
        if spec.bit_length == 1024:
            assert outcome["status"] == "insufficient_capacity", spec.name
        else:
            assert outcome["status"] == "success", spec.name
