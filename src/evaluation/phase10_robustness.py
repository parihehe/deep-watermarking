"""Phase 10 - Attack Simulation / Robustness Testing (harness).

This module is **new Phase 10 code**. It reuses, without modifying:

* the frozen Phase 6 baseline - ``src.watermark.embed.embed`` /
  ``extract_traditional`` (the embedding formula and its defaults are untouched;
  only ``alpha`` and ``bit_length`` are set on ``EmbedConfig``, exactly as
  Phases 8 and 9 already do);
* the trained Phase 8 blind CNN - ``src.evaluation.blind_extract.BlindExtractor``
  (architecture and checkpoint unchanged);
* the Phase 9 harness helpers - ``assert_baseline_frozen``, ``frozen_file_hashes``,
  ``payload_bits``, ``load_split_images``, ``environment_stamp`` and the metric
  taxonomy - imported from ``src.evaluation.phase9_validation``.

What Phase 10 does
------------------
1. Watermarks each held-out DIV2K test image once, at the Phase 8/9 operating
   point (64 bits, alpha = 0.02).
2. Applies each configured attack (JPEG, Gaussian noise, blur, median filter,
   resize round-trip, rotation, cropping, and combined pipelines) at several
   configurable severities.
3. Recovers the payload from every attacked image two ways:
     * frozen non-blind DWT-SVD (``extract_traditional``, original also in), and
     * Phase 8 blind CNN (``BlindExtractor``, image only in).
4. Records the image-quality group (PSNR / SSIM / MSE, attacked vs clean
   watermarked image) and the watermark-recovery group (BER / bit-accuracy / NC)
   for both decoders, plus the delta against the no-attack baseline.

Scope guard (explicitly NOT in Phase 10)
---------------------------------------
* no change to the frozen Phase 6 formula / defaults;
* no CNN architecture change ................. Phase 15;
* no adaptive embedding ...................... Phase 11;
* no high-capacity changes ................... Phase 12;
* no wavelet / subband study ................. Phases 13-14;
* no error-correcting codes .................. Phase 16;
* no geometric re-synchronisation / registration of attacked images - Phase 10
  measures the raw damage. The only preprocessing before extraction is a resize
  back to the embedding resolution so the array shapes line up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pstdev

import cv2
import numpy as np
import yaml

from src.evaluation.attacks import ATTACKS, apply_attack, severity_of
from src.evaluation.metrics import (
    bit_accuracy,
    bit_error_rate,
    mse,
    normalized_correlation,
    psnr,
    ssim,
)
from src.evaluation.phase9_validation import (
    IMAGE_QUALITY_METRICS,
    PROJECT_ROOT,
    WATERMARK_RECOVERY_METRICS,
    load_split_images,
    payload_bits,
)
from src.watermark.embed import EmbedConfig, embed, extract_traditional

__all__ = [
    "DEFAULT_ATTACKS",
    "Phase10Config",
    "WatermarkedSample",
    "build_embed_config",
    "evaluate_attack_point",
    "load_config",
    "prepare_samples",
    "recover_blind",
    "recover_nonblind",
    "resynchronise",
    "summarise",
]

# Default attack grid - every entry is (attack name -> list of parameter dicts).
# Chosen so each family spans "watermark clearly survives" to "watermark
# clearly destroyed"; all values are overridable from the YAML config.
DEFAULT_ATTACKS: dict[str, list[dict]] = {
    "jpeg_compress": [{"quality": q} for q in (90, 75, 50, 30, 10)],
    "gaussian_noise": [{"sigma": s} for s in (2, 5, 10, 20, 40)],
    "gaussian_blur": [{"ksize": k} for k in (3, 5, 7, 9)],
    "median_filter": [{"ksize": k} for k in (3, 5, 7)],
    "resize_roundtrip": [{"scale": s} for s in (0.9, 0.75, 0.5, 0.25)],
    "rotate": [{"degrees": d} for d in (1, 2, 5, 10, 45)],
    "center_crop": [{"keep": k} for k in (0.95, 0.9, 0.75, 0.5)],
    "combined": [
        {"steps": [
            {"name": "jpeg_compress", "params": {"quality": 75}},
            {"name": "gaussian_noise", "params": {"sigma": 3}},
        ]},
        {"steps": [
            {"name": "gaussian_blur", "params": {"ksize": 3}},
            {"name": "jpeg_compress", "params": {"quality": 50}},
            {"name": "gaussian_noise", "params": {"sigma": 5}},
        ]},
        {"steps": [
            {"name": "resize_roundtrip", "params": {"scale": 0.75}},
            {"name": "jpeg_compress", "params": {"quality": 40}},
        ]},
    ],
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Phase10Config:
    """Everything one Phase 10 robustness run needs."""

    # embedding operating point (frozen Phase 6 formula; only these two are set,
    # matching Phase 8 / Phase 9)
    alpha: float = 0.02
    payload_bits: int = 64
    wavelet: str = "haar"
    mode: str = "symmetric"
    subband: str = "LL"
    start_sv_index: int = 0

    # evaluation data
    processed_root: str = "data/processed/div2k_256"
    eval_split: str = "test"
    eval_num_images: int = 50
    image_size: int = 256
    payload_seed: int = 20260901

    # attacks
    attacks: dict[str, list[dict]] = field(default_factory=lambda: {
        k: [dict(p) for p in v] for k, v in DEFAULT_ATTACKS.items()
    })
    noise_seed: int = 1234

    # Phase 8 blind CNN
    cnn_enabled: bool = True
    cnn_checkpoint: str = "models/phase8_cnn/phase8_cnn_best.pt"

    # output
    results_dir: str = "results/phase10_robustness"
    make_plots: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha}")
        if self.payload_bits <= 0 or self.payload_bits > 128:
            raise ValueError(
                f"payload_bits must be in [1, 128] (frozen LL-only capacity); got {self.payload_bits}"
            )
        if self.eval_num_images <= 0:
            raise ValueError(f"eval_num_images must be positive; got {self.eval_num_images}")
        if not self.attacks:
            raise ValueError("attacks must be non-empty")
        for name, param_list in self.attacks.items():
            if name not in ATTACKS:
                raise ValueError(f"unknown attack {name!r}; known: {sorted(ATTACKS)}")
            if not isinstance(param_list, list) or not param_list:
                raise ValueError(f"attack {name!r} needs a non-empty list of parameter dicts")

    def embed_config(self) -> EmbedConfig:
        return build_embed_config(self)

    def resolved_results_dir(self) -> Path:
        return (PROJECT_ROOT / self.results_dir).resolve()

    def resolved_split_dir(self) -> Path:
        return (PROJECT_ROOT / self.processed_root / self.eval_split).resolve()

    def resolved_checkpoint(self) -> Path:
        return (PROJECT_ROOT / self.cnn_checkpoint).resolve()


def build_embed_config(cfg: Phase10Config) -> EmbedConfig:
    """The frozen LL-only baseline config at the Phase 10 operating point."""
    return EmbedConfig(
        wavelet=cfg.wavelet,
        mode=cfg.mode,
        subband=cfg.subband,
        extra_subbands=(),
        alpha=cfg.alpha,
        bit_length=cfg.payload_bits,
        start_sv_index=cfg.start_sv_index,
    )


def load_config(path: str | Path, overrides: dict | None = None) -> Phase10Config:
    """Load ``configs/phase10_robustness.yaml`` (``phase10:`` block)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["phase10"]
    embed_blk = raw.get("embed", {})
    data = raw.get("data", {})
    cnn = raw.get("cnn", {})
    output = raw.get("output", {})
    attacks = raw.get("attacks") or {k: [dict(p) for p in v] for k, v in DEFAULT_ATTACKS.items()}

    cfg = Phase10Config(
        alpha=float(embed_blk.get("alpha", 0.02)),
        payload_bits=int(embed_blk.get("bit_length", 64)),
        wavelet=embed_blk.get("wavelet", "haar"),
        mode=embed_blk.get("mode", "symmetric"),
        subband=embed_blk.get("subband", "LL"),
        start_sv_index=int(embed_blk.get("start_sv_index", 0)),
        processed_root=data.get("processed_root", "data/processed/div2k_256"),
        eval_split=data.get("eval_split", "test"),
        eval_num_images=int(data.get("eval_num_images", 50)),
        image_size=int(data.get("image_size", 256)),
        payload_seed=int(raw.get("payload_seed", 20260901)),
        attacks={k: [dict(p) for p in v] for k, v in attacks.items()},
        noise_seed=int(raw.get("noise_seed", 1234)),
        cnn_enabled=bool(cnn.get("enabled", True)),
        cnn_checkpoint=cnn.get("checkpoint", "models/phase8_cnn/phase8_cnn_best.pt"),
        results_dir=output.get("results_dir", "results/phase10_robustness"),
        make_plots=bool(output.get("make_plots", True)),
    )
    if overrides:
        merged = {**cfg.__dict__}
        merged.update({k: v for k, v in overrides.items() if v is not None})
        cfg = Phase10Config(**merged)
    return cfg


# ---------------------------------------------------------------------------
# Watermarked samples (embedded once, reused for every attack)
# ---------------------------------------------------------------------------

@dataclass
class WatermarkedSample:
    image_id: str
    original: np.ndarray       # RGB uint8, (S, S, 3)
    watermarked: np.ndarray    # RGB uint8, (S, S, 3)
    bits: list[int]


def prepare_samples(cfg: Phase10Config) -> list[WatermarkedSample]:
    """Embed the deterministic payload into every evaluation image once."""
    images = load_split_images(cfg.resolved_split_dir(), cfg.eval_num_images)
    econf = cfg.embed_config()
    samples: list[WatermarkedSample] = []
    for index, (image_id, rgb) in enumerate(images):
        bits = payload_bits(cfg.payload_seed, index, cfg.payload_bits)
        result = embed(rgb, bits, econf)
        samples.append(WatermarkedSample(image_id, rgb, result.watermarked_image, bits))
    return samples


# ---------------------------------------------------------------------------
# Recovery (with shape re-synchronisation only)
# ---------------------------------------------------------------------------

def resynchronise(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an attacked image back to ``size x size`` so array shapes line up.

    This is the *only* preprocessing Phase 10 applies before extraction - it does
    not undo rotation, translation or cropping.
    """
    if image.shape[:2] != (size, size):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return image


def recover_nonblind(
    attacked: np.ndarray, original: np.ndarray, embed_config: EmbedConfig, size: int
) -> list[int]:
    return extract_traditional(resynchronise(attacked, size), original, embed_config)


def recover_blind(attacked: np.ndarray, extractor) -> list[int]:
    # BlindExtractor resizes internally to its own image_size.
    return extractor.extract_bits(attacked)


# ---------------------------------------------------------------------------
# Per-point evaluation
# ---------------------------------------------------------------------------

def _recovery_metrics(prefix: str, reference: list[int], recovered: list[int]) -> dict:
    return {
        f"{prefix}_ber": round(bit_error_rate(reference, recovered), 6),
        f"{prefix}_bit_accuracy": round(bit_accuracy(reference, recovered), 6),
        f"{prefix}_nc": round(normalized_correlation(reference, recovered), 6),
    }


def evaluate_attack_point(
    samples: list[WatermarkedSample],
    attack_name: str,
    params: dict,
    *,
    embed_config: EmbedConfig,
    size: int,
    noise_seed: int,
    extractor=None,
) -> list[dict]:
    """One row per image for a single (attack, parameter) point.

    Image-quality metrics compare the attacked image against the clean
    watermarked image (both at ``size``). Recovery metrics are computed for the
    non-blind decoder always and for the blind CNN when ``extractor`` is given.
    """
    severity = severity_of(attack_name, params)
    rows: list[dict] = []
    for index, sample in enumerate(samples):
        seed = noise_seed + index
        attacked = apply_attack(attack_name, sample.watermarked, params, seed=seed)
        attacked_sync = resynchronise(attacked, size)
        clean_sync = resynchronise(sample.watermarked, size)

        row = {
            "image": sample.image_id,
            "attack": attack_name,
            "severity": severity if severity is not None else "",
            "params": _params_repr(params),
            "quality_psnr": round(psnr(clean_sync, attacked_sync), 4),
            "quality_ssim": round(ssim(clean_sync, attacked_sync), 6),
            "quality_mse": round(mse(clean_sync, attacked_sync), 6),
        }
        row.update(_recovery_metrics(
            "nonblind", sample.bits,
            recover_nonblind(attacked, sample.original, embed_config, size),
        ))
        if extractor is not None:
            row.update(_recovery_metrics("blind", sample.bits, recover_blind(attacked, extractor)))
        rows.append(row)
    return rows


def _params_repr(params: dict) -> str:
    if "steps" in params:
        return " -> ".join(
            f"{s['name']}({','.join(f'{k}={v}' for k, v in s.get('params', {}).items())})"
            for s in params["steps"]
        )
    return ",".join(f"{k}={v}" for k, v in params.items())


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean_std(values) -> tuple[float | None, float | None]:
    """Mean/std over the finite numeric values. Non-finite entries (e.g. the
    ``identity`` point's infinite PSNR) are dropped; an all-non-finite input
    yields ``(None, None)`` so the JSON/CSV carry a clean blank instead of
    ``Infinity``."""
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return (None, None)
    return (round(fmean(vals), 6), round(pstdev(vals), 6) if len(vals) > 1 else 0.0)


def summarise(
    rows: list[dict],
    *,
    baseline: dict | None = None,
    has_blind: bool = True,
) -> list[dict]:
    """Aggregate per-image rows to mean/std per (attack, params), keeping the two
    metric groups separate and adding the delta against the no-attack baseline.

    ``baseline`` is the summarised ``identity`` entry: ``{"nonblind_ber_mean": ..,
    "blind_ber_mean": ..}``. Deltas are ``attacked_mean - baseline_mean`` (a
    positive BER delta means the attack made recovery worse).
    """
    keys: list[tuple[str, str]] = []
    for row in rows:
        k = (row["attack"], row["params"])
        if k not in keys:
            keys.append(k)

    decoders = ["nonblind"] + (["blind"] if has_blind else [])
    summary: list[dict] = []
    for attack, params in keys:
        subset = [r for r in rows if r["attack"] == attack and r["params"] == params]
        entry: dict = {
            "attack": attack,
            "params": params,
            "severity": subset[0]["severity"],
            "n_images": len(subset),
            "image_quality": {},
            "watermark_recovery": {},
        }
        for metric in IMAGE_QUALITY_METRICS:
            m, s = _mean_std([r[f"quality_{metric}"] for r in subset])
            entry["image_quality"][f"{metric}_mean"] = m
            entry["image_quality"][f"{metric}_std"] = s
        for decoder in decoders:
            block: dict = {}
            for metric in WATERMARK_RECOVERY_METRICS:
                m, s = _mean_std([r[f"{decoder}_{metric}"] for r in subset])
                block[f"{metric}_mean"] = m
                block[f"{metric}_std"] = s
                if baseline is not None and f"{decoder}_{metric}_mean" in baseline:
                    block[f"{metric}_delta_vs_baseline"] = round(
                        m - baseline[f"{decoder}_{metric}_mean"], 6
                    )
            entry["watermark_recovery"][decoder] = block
        summary.append(entry)
    return summary


def baseline_block(identity_summary_entry: dict, *, has_blind: bool = True) -> dict:
    """Flatten the summarised ``identity`` entry into the lookup ``summarise``
    expects for its delta columns."""
    out: dict = {}
    for decoder in ["nonblind"] + (["blind"] if has_blind else []):
        block = identity_summary_entry["watermark_recovery"][decoder]
        for metric in WATERMARK_RECOVERY_METRICS:
            out[f"{decoder}_{metric}_mean"] = block[f"{metric}_mean"]
    return out
