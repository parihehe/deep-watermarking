"""Tests for Phase 10 - Attack Simulation / Robustness Testing.

Layers:
  * pure harness logic (config, embed-config, resynchronise, aggregation);
  * data-dependent end-to-end checks on the real DIV2K test split (skipped when
    absent);
  * CNN-dependent checks against the trained Phase 8 checkpoint (skipped when
    absent).

The frozen Phase 6 baseline, the Phase 7 UI and the Phase 8 model are used only
through their existing public API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.phase10_robustness import (
    DEFAULT_ATTACKS,
    Phase10Config,
    baseline_block,
    build_embed_config,
    evaluate_attack_point,
    load_config,
    prepare_samples,
    resynchronise,
    summarise,
)
from src.watermark.embed import EmbedConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "phase10_robustness.yaml"
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
# Phase10Config
# ---------------------------------------------------------------------------

def test_config_defaults_are_the_phase8_operating_point() -> None:
    cfg = Phase10Config()
    assert cfg.alpha == 0.02
    assert cfg.payload_bits == 64
    assert cfg.wavelet == "haar" and cfg.subband == "LL" and cfg.mode == "symmetric"
    assert cfg.eval_split == "test"
    assert set(cfg.attacks) == set(DEFAULT_ATTACKS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"payload_bits": 0},
        {"payload_bits": 129},          # beyond frozen LL-only capacity
        {"eval_num_images": 0},
        {"attacks": {}},
        {"attacks": {"teleport": [{"x": 1}]}},
        {"attacks": {"jpeg_compress": []}},
        {"attacks": {"jpeg_compress": "nope"}},
    ],
)
def test_config_rejects_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Phase10Config(**kwargs)


def test_load_config_from_repo_yaml() -> None:
    cfg = load_config(CONFIG_PATH)
    assert cfg.alpha == 0.02 and cfg.payload_bits == 64
    assert cfg.eval_split == "test"
    assert "jpeg_compress" in cfg.attacks and "combined" in cfg.attacks
    assert cfg.attacks["jpeg_compress"][0] == {"quality": 90}
    assert cfg.cnn_enabled is True
    assert cfg.results_dir == "results/phase10_robustness"


def test_load_config_overrides() -> None:
    cfg = load_config(CONFIG_PATH, {"eval_num_images": 8,
                                    "attacks": {"rotate": [{"degrees": 5}]}})
    assert cfg.eval_num_images == 8
    assert set(cfg.attacks) == {"rotate"}


# ---------------------------------------------------------------------------
# embed config - frozen baseline, only the operating point set
# ---------------------------------------------------------------------------

def test_build_embed_config_is_frozen_ll_only() -> None:
    econf = build_embed_config(Phase10Config())
    assert isinstance(econf, EmbedConfig)
    assert econf.wavelet == "haar" and econf.mode == "symmetric"
    assert econf.subband == "LL" and econf.extra_subbands == ()
    assert econf.alpha == 0.02 and econf.bit_length == 64
    assert econf.start_sv_index == 0


# ---------------------------------------------------------------------------
# resynchronise
# ---------------------------------------------------------------------------

def test_resynchronise_noop_when_already_sized() -> None:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    assert resynchronise(img, 256) is img


def test_resynchronise_resizes_when_needed() -> None:
    img = np.zeros((128, 200, 3), dtype=np.uint8)
    out = resynchronise(img, 256)
    assert out.shape == (256, 256, 3)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _synthetic_rows() -> list[dict]:
    rows = []
    for img in ("a", "b", "c"):
        rows.append({
            "image": img, "attack": "identity", "severity": "", "params": "",
            "quality_psnr": 99.0, "quality_ssim": 1.0, "quality_mse": 0.0,
            "nonblind_ber": 0.05, "nonblind_bit_accuracy": 0.95, "nonblind_nc": 0.9,
            "blind_ber": 0.20, "blind_bit_accuracy": 0.80, "blind_nc": 0.6,
        })
    for img in ("a", "b", "c"):
        rows.append({
            "image": img, "attack": "jpeg_compress", "severity": 50.0, "params": "quality=50",
            "quality_psnr": 38.0, "quality_ssim": 0.97, "quality_mse": 10.0,
            "nonblind_ber": 0.12, "nonblind_bit_accuracy": 0.88, "nonblind_nc": 0.76,
            "blind_ber": 0.30, "blind_bit_accuracy": 0.70, "blind_nc": 0.4,
        })
    return rows


def test_summarise_keeps_groups_separate_and_adds_deltas() -> None:
    rows = _synthetic_rows()
    identity_entry = summarise([r for r in rows if r["attack"] == "identity"])[0]
    base = baseline_block(identity_entry)
    summary = summarise(rows, baseline=base)

    assert {e["attack"] for e in summary} == {"identity", "jpeg_compress"}
    jpeg = next(e for e in summary if e["attack"] == "jpeg_compress")
    assert jpeg["n_images"] == 3
    assert set(jpeg["image_quality"]) == {
        f"{m}_{s}" for m in ("psnr", "ssim", "mse") for s in ("mean", "std")
    }
    nb = jpeg["watermark_recovery"]["nonblind"]
    assert nb["ber_mean"] == pytest.approx(0.12)
    assert nb["ber_delta_vs_baseline"] == pytest.approx(0.12 - 0.05)
    bl = jpeg["watermark_recovery"]["blind"]
    assert bl["ber_delta_vs_baseline"] == pytest.approx(0.30 - 0.20)


def test_summarise_without_blind() -> None:
    rows = [r for r in _synthetic_rows()]
    for r in rows:
        for k in ("blind_ber", "blind_bit_accuracy", "blind_nc"):
            r.pop(k)
    summary = summarise(rows, has_blind=False)
    for entry in summary:
        assert list(entry["watermark_recovery"]) == ["nonblind"]


# ---------------------------------------------------------------------------
# data-dependent end-to-end
# ---------------------------------------------------------------------------

@_needs_data
def test_prepare_samples_embeds_every_image() -> None:
    cfg = Phase10Config(eval_num_images=4)
    samples = prepare_samples(cfg)
    assert len(samples) == 4
    for s in samples:
        assert s.original.shape == (256, 256, 3)
        assert s.watermarked.shape == (256, 256, 3)
        assert len(s.bits) == 64
        assert not np.array_equal(s.original, s.watermarked)   # something was embedded


@_needs_data
def test_identity_point_matches_clean_recovery() -> None:
    cfg = Phase10Config(eval_num_images=5)
    samples = prepare_samples(cfg)
    econf = build_embed_config(cfg)
    rows = evaluate_attack_point(samples, "identity", {}, embed_config=econf,
                                 size=256, noise_seed=1234)
    for row in rows:
        assert row["attack"] == "identity"
        assert row["severity"] == ""
        assert np.isinf(row["quality_psnr"]) or row["quality_psnr"] > 60
        for key in ("nonblind_ber", "nonblind_bit_accuracy", "nonblind_nc"):
            assert key in row
        assert 0.0 <= row["nonblind_ber"] <= 1.0


@_needs_data
def test_mild_jpeg_keeps_recovery_close_geometric_breaks_it() -> None:
    cfg = Phase10Config(eval_num_images=6)
    samples = prepare_samples(cfg)
    econf = build_embed_config(cfg)

    def mean_nb_ber(name: str, params: dict) -> float:
        rows = evaluate_attack_point(samples, name, params, embed_config=econf,
                                     size=256, noise_seed=1234)
        return float(np.mean([r["nonblind_ber"] for r in rows]))

    clean = mean_nb_ber("identity", {})
    jpeg_hi = mean_nb_ber("jpeg_compress", {"quality": 90})
    rot45 = mean_nb_ber("rotate", {"degrees": 45})

    assert jpeg_hi <= clean + 0.15            # mild JPEG barely moves the needle
    assert rot45 > clean + 0.15              # 45-deg rotation clearly destroys it
    assert rot45 > 0.3


@_needs_data
def test_evaluate_attack_point_is_deterministic() -> None:
    cfg = Phase10Config(eval_num_images=4)
    samples = prepare_samples(cfg)
    econf = build_embed_config(cfg)
    a = evaluate_attack_point(samples, "gaussian_noise", {"sigma": 15},
                              embed_config=econf, size=256, noise_seed=1234)
    b = evaluate_attack_point(samples, "gaussian_noise", {"sigma": 15},
                              embed_config=econf, size=256, noise_seed=1234)
    assert a == b


@_needs_data
def test_combined_attack_point_runs() -> None:
    cfg = Phase10Config(eval_num_images=3)
    samples = prepare_samples(cfg)
    econf = build_embed_config(cfg)
    params = DEFAULT_ATTACKS["combined"][0]
    rows = evaluate_attack_point(samples, "combined", params, embed_config=econf,
                                 size=256, noise_seed=1234)
    assert len(rows) == 3
    for row in rows:
        assert row["attack"] == "combined"
        assert row["severity"] == ""
        assert "->" in row["params"] or "," in row["params"]
        assert 0.0 <= row["nonblind_ber"] <= 1.0


# ---------------------------------------------------------------------------
# CNN-dependent
# ---------------------------------------------------------------------------

@_needs_data
@_needs_cnn
def test_attack_point_with_blind_decoder() -> None:
    pytest.importorskip("torch")
    from src.evaluation.blind_extract import BlindExtractor

    cfg = Phase10Config(eval_num_images=5)
    samples = prepare_samples(cfg)
    econf = build_embed_config(cfg)
    extractor = BlindExtractor.from_checkpoint(str(_CNN_CKPT))

    rows = evaluate_attack_point(samples, "jpeg_compress", {"quality": 75},
                                 embed_config=econf, size=256, noise_seed=1234,
                                 extractor=extractor)
    for row in rows:
        for key in ("blind_ber", "blind_bit_accuracy", "blind_nc"):
            assert key in row
        assert 0.0 <= row["blind_ber"] <= 1.0

    summary = summarise(rows, has_blind=True)
    assert "blind" in summary[0]["watermark_recovery"]
    assert "nonblind" in summary[0]["watermark_recovery"]


# ---------------------------------------------------------------------------
# runner helpers
# ---------------------------------------------------------------------------

def test_runner_flatten_and_ranking() -> None:
    from experiments.run_phase10_robustness import _flatten_summary, _ranking

    rows = _synthetic_rows()
    identity_entry = summarise([r for r in rows if r["attack"] == "identity"])[0]
    summary = summarise(rows, baseline=baseline_block(identity_entry))

    flat = _flatten_summary(summary, has_blind=True)
    jpeg_flat = next(r for r in flat if r["attack"] == "jpeg_compress")
    assert jpeg_flat["quality_psnr_mean"] == pytest.approx(38.0)
    assert jpeg_flat["nonblind_ber_mean"] == pytest.approx(0.12)
    assert jpeg_flat["blind_ber_delta_vs_baseline"] == pytest.approx(0.10)

    ranking = _ranking(summary, has_blind=True)
    assert ranking[0]["attack"] == "jpeg_compress"      # only non-identity family here
    assert ranking[0]["worst_nonblind_ber"] == pytest.approx(0.12)
