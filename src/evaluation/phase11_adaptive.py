"""Phase 11 - Adaptive, content-aware embedding (evaluation harness).

This module is **new Phase 11 code**. It imports and *runs* the new Phase 11
block-SVD embedder (``src.watermark.adaptive_embed``, itself additive - see that
module's docstring for why it exists as a separate path rather than a change to
the frozen Phase 6 baseline), plus the frozen Phase 6 guard machinery and image
loader already built for Phase 9, and (for one bonus robustness comparison) the
already-implemented Phase 10 attack primitives. Nothing under ``src/watermark/``
other than the new ``adaptive_embed.py`` file, and nothing under
``src/evaluation/metrics.py`` or ``src/evaluation/phase9_validation.py``, is
modified.

What Phase 11 measures
-----------------------
1. **Clean-channel alpha sweep.** For each configured ``alpha`` and both
   ``alpha_mode`` values (``fixed`` / ``adaptive``), embed a 64-bit payload in
   every held-out DIV2K test image (64 blocks = the established 64-bit
   operating point) and record the two metric groups (image-quality:
   PSNR/SSIM/MSE; watermark-recovery: BER/bit-accuracy/NC), non-blind. Fixed
   and adaptive share the same per-image *mean* alpha (see
   ``adaptive_embed.AdaptiveEmbedConfig``), so this isolates exactly the
   "uniform vs content-aware allocation" variable.
2. **Robustness stress check.** At the primary operating alpha, additionally
   attack the watermarked image with a small set of realistic distortions
   (reusing the already-implemented, unmodified ``src.evaluation.attacks``
   registry from Phase 10) and recover with the same non-blind decoder, for
   both alpha modes - this is what "quality vs robustness trade-off" is
   ultimately about: does content-aware allocation help *after* a real channel,
   not just on paper.

What Phase 11 deliberately does NOT do (scope guard)
------------------------------------------------------
* No change to any frozen Phase 6 file (embed.py / dwt.py / svd.py /
  watermark_generator.py) - verified via the Phase 9 frozen-file hash guard.
* No Phase 8 CNN architecture change or retraining - the CNN was trained on the
  frozen whole-subband topology and is not evaluated against the new block-SVD
  images (an apples-to-oranges comparison the CNN was never built for).
* No high-capacity change (Phase 12): bit_length stays at the established
  64-bit point, one bit per block, no subband overflow.
* No error-correcting codes (Phase 16), no blockchain, no wavelet/subband study
  (Phases 13-14).
"""

from __future__ import annotations

import platform
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pstdev

import cv2
import numpy as np
import yaml

from src.evaluation.attacks import apply_attack
from src.evaluation.metrics import bit_accuracy, bit_error_rate, mse, normalized_correlation, psnr, ssim
from src.evaluation.phase9_validation import payload_bits
from src.watermark.adaptive_embed import (
    ALPHA_MODES,
    AdaptiveEmbedConfig,
    embed_adaptive,
    expected_bit_length,
    extract_adaptive,
)

__all__ = [
    "IMAGE_QUALITY_METRICS",
    "PROJECT_ROOT",
    "WATERMARK_RECOVERY_METRICS",
    "Phase11Config",
    "adaptive_config_for",
    "evaluate_alpha_point",
    "evaluate_robustness_point",
    "load_config",
    "load_split_images",
    "summarise_alpha_sweep",
    "summarise_robustness",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_QUALITY_METRICS: tuple[str, ...] = ("psnr", "ssim", "mse")
WATERMARK_RECOVERY_METRICS: tuple[str, ...] = ("ber", "bit_accuracy", "nc")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Phase11Config:
    """Everything one Phase 11 adaptive-embedding evaluation run needs."""

    block_size: int = 16
    bit_length: int = 64
    alphas: tuple[float, ...] = (0.005, 0.010, 0.015, 0.020)
    alpha_low_mult: float = 0.4
    alpha_high_mult: float = 1.6
    texture_percentile_clip: tuple[float, float] = (5.0, 95.0)
    wavelet: str = "haar"
    mode: str = "symmetric"

    # evaluation data
    processed_root: str = "data/processed/div2k_256"
    eval_split: str = "test"
    eval_num_images: int = 50
    image_size: int = 256

    # deterministic payloads
    payload_seed: int = 20260901
    noise_seed: int = 1234

    # robustness stress check (reuses the already-implemented Phase 10 attacks)
    robustness_alpha: float = 0.020
    robustness_attacks: dict[str, dict] = field(
        default_factory=lambda: {
            "jpeg_compress": {"quality": 75},
            "gaussian_noise": {"sigma": 5},
            "gaussian_blur": {"ksize": 3},
        }
    )

    # output
    results_dir: str = "results/phase11_adaptive"
    make_plots: bool = True

    def __post_init__(self) -> None:
        if expected_bit_length(self.image_size, self.block_size) != self.bit_length:
            raise ValueError(
                f"bit_length={self.bit_length} does not match the block grid for "
                f"image_size={self.image_size}, block_size={self.block_size} "
                f"(expected {expected_bit_length(self.image_size, self.block_size)})"
            )
        if not self.alphas:
            raise ValueError("alphas must be non-empty")
        if any(not 0.0 < a < 1.0 for a in self.alphas):
            raise ValueError(f"every alpha must be in (0, 1); got {self.alphas}")
        if not 0.0 < self.robustness_alpha < 1.0:
            raise ValueError(f"robustness_alpha must be in (0, 1); got {self.robustness_alpha}")
        if self.eval_num_images <= 0:
            raise ValueError(f"eval_num_images must be positive; got {self.eval_num_images}")

    def resolved_results_dir(self) -> Path:
        return (PROJECT_ROOT / self.results_dir).resolve()

    def resolved_split_dir(self) -> Path:
        return (PROJECT_ROOT / self.processed_root / self.eval_split).resolve()


def load_config(path: str | Path, overrides: dict | None = None) -> Phase11Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["phase11"]
    embed_ = raw.get("embed", {})
    adaptive = raw.get("adaptive", {})
    data = raw.get("data", {})
    robustness = raw.get("robustness", {})
    output = raw.get("output", {})

    cfg = Phase11Config(
        block_size=int(embed_.get("block_size", 16)),
        bit_length=int(embed_.get("bit_length", 64)),
        wavelet=embed_.get("wavelet", "haar"),
        mode=embed_.get("mode", "symmetric"),
        alphas=tuple(float(a) for a in embed_.get("alphas", (0.005, 0.010, 0.015, 0.020))),
        alpha_low_mult=float(adaptive.get("alpha_low_mult", 0.4)),
        alpha_high_mult=float(adaptive.get("alpha_high_mult", 1.6)),
        texture_percentile_clip=tuple(
            float(x) for x in adaptive.get("texture_percentile_clip", (5.0, 95.0))
        ),
        processed_root=data.get("processed_root", "data/processed/div2k_256"),
        eval_split=data.get("eval_split", "test"),
        eval_num_images=int(data.get("eval_num_images", 50)),
        image_size=int(data.get("image_size", 256)),
        payload_seed=int(raw.get("payload_seed", 20260901)),
        noise_seed=int(raw.get("noise_seed", 1234)),
        robustness_alpha=float(robustness.get("alpha", 0.020)),
        robustness_attacks=dict(robustness.get("attacks", {
            "jpeg_compress": {"quality": 75},
            "gaussian_noise": {"sigma": 5},
            "gaussian_blur": {"ksize": 3},
        })),
        results_dir=output.get("results_dir", "results/phase11_adaptive"),
        make_plots=bool(output.get("make_plots", True)),
    )

    if overrides:
        merged = {**cfg.__dict__}
        merged.update({k: v for k, v in overrides.items() if v is not None})
        cfg = Phase11Config(**merged)
    return cfg


def adaptive_config_for(cfg: Phase11Config, alpha: float, alpha_mode: str) -> AdaptiveEmbedConfig:
    if alpha_mode not in ALPHA_MODES:
        raise ValueError(f"alpha_mode must be one of {ALPHA_MODES}; got {alpha_mode!r}")
    return AdaptiveEmbedConfig(
        block_size=cfg.block_size,
        bit_length=cfg.bit_length,
        alpha_mode=alpha_mode,
        alpha=alpha,
        alpha_low_mult=cfg.alpha_low_mult,
        alpha_high_mult=cfg.alpha_high_mult,
        texture_percentile_clip=cfg.texture_percentile_clip,
        wavelet=cfg.wavelet,
        mode=cfg.mode,
    )


# ---------------------------------------------------------------------------
# Image loading (mirrors Phase 9's loader; kept local so Phase 11 has no
# import-time dependency on Phase 9 beyond the pure ``payload_bits`` helper)
# ---------------------------------------------------------------------------

def load_split_images(split_dir: Path, limit: int) -> list[tuple[str, np.ndarray]]:
    paths = sorted(split_dir.glob("*.png"))[:limit]
    if not paths:
        raise FileNotFoundError(f"no processed images in {split_dir}")
    out = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"unreadable image: {path}")
        out.append((path.stem, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return out


# ---------------------------------------------------------------------------
# Clean-channel alpha-sweep evaluation
# ---------------------------------------------------------------------------

def evaluate_alpha_point(
    images: Sequence[tuple[str, np.ndarray]],
    econf: AdaptiveEmbedConfig,
    *,
    payload_seed: int,
) -> list[dict]:
    """One row per image for a single (alpha, alpha_mode) point, clean channel."""
    rows: list[dict] = []
    for index, (image_id, image) in enumerate(images):
        bits = payload_bits(payload_seed, index, econf.bit_length)
        result = embed_adaptive(image, bits, econf)
        wm = result.watermarked_image
        recovered = extract_adaptive(wm, image, econf)
        rows.append({
            "image": image_id,
            "alpha_mode": econf.alpha_mode,
            "alpha": econf.alpha,
            "block_size": econf.block_size,
            "n_blocks": result.grid[0] * result.grid[1],
            "alpha_realised_mean": round(float(result.alpha_map.mean()), 6),
            "alpha_realised_std": round(float(result.alpha_map.std()), 6),
            "texture_mean": round(float(result.texture_scores.mean()), 4),
            "texture_std": round(float(result.texture_scores.std()), 4),
            "quality_psnr": round(psnr(image, wm), 4),
            "quality_ssim": round(ssim(image, wm), 6),
            "quality_mse": round(mse(image, wm), 6),
            "recovery_ber": round(bit_error_rate(bits, recovered), 6),
            "recovery_bit_accuracy": round(bit_accuracy(bits, recovered), 6),
            "recovery_nc": round(normalized_correlation(bits, recovered), 6),
        })
    return rows


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    values = list(values)
    if not values:
        return (0.0, 0.0)
    return (round(fmean(values), 6), round(pstdev(values), 6) if len(values) > 1 else 0.0)


def summarise_alpha_sweep(rows: Sequence[dict]) -> list[dict]:
    keys = sorted({(r["alpha_mode"], r["alpha"]) for r in rows})
    summary: list[dict] = []
    for alpha_mode, alpha in keys:
        subset = [r for r in rows if r["alpha_mode"] == alpha_mode and r["alpha"] == alpha]
        entry: dict = {
            "alpha_mode": alpha_mode,
            "alpha": alpha,
            "n_images": len(subset),
            "alpha_realised_std_mean": _mean_std([r["alpha_realised_std"] for r in subset])[0],
            "image_quality": {},
            "watermark_recovery": {},
        }
        for metric in IMAGE_QUALITY_METRICS:
            m, s = _mean_std([r[f"quality_{metric}"] for r in subset])
            entry["image_quality"][f"{metric}_mean"] = m
            entry["image_quality"][f"{metric}_std"] = s
        for metric in WATERMARK_RECOVERY_METRICS:
            m, s = _mean_std([r[f"recovery_{metric}"] for r in subset])
            entry["watermark_recovery"][f"{metric}_mean"] = m
            entry["watermark_recovery"][f"{metric}_std"] = s
        summary.append(entry)
    return summary


# ---------------------------------------------------------------------------
# Robustness-stress evaluation (reuses the already-implemented Phase 10 attacks)
# ---------------------------------------------------------------------------

def evaluate_robustness_point(
    images: Sequence[tuple[str, np.ndarray]],
    econf: AdaptiveEmbedConfig,
    attack_name: str,
    attack_params: dict,
    *,
    payload_seed: int,
    noise_seed: int,
) -> list[dict]:
    rows: list[dict] = []
    for index, (image_id, image) in enumerate(images):
        bits = payload_bits(payload_seed, index, econf.bit_length)
        result = embed_adaptive(image, bits, econf)
        wm = result.watermarked_image

        params = dict(attack_params)
        if attack_name == "gaussian_noise":
            params.setdefault("seed", noise_seed + index)
        attacked = apply_attack(attack_name, wm, params)
        if attacked.shape != wm.shape:
            attacked = cv2.resize(attacked, (wm.shape[1], wm.shape[0]), interpolation=cv2.INTER_LINEAR)
        recovered = extract_adaptive(attacked, image, econf)

        rows.append({
            "image": image_id,
            "alpha_mode": econf.alpha_mode,
            "alpha": econf.alpha,
            "attack": attack_name,
            "attack_params": str(params),
            "quality_psnr": round(psnr(image, wm), 4),
            "quality_ssim": round(ssim(image, wm), 6),
            "recovery_ber": round(bit_error_rate(bits, recovered), 6),
            "recovery_bit_accuracy": round(bit_accuracy(bits, recovered), 6),
            "recovery_nc": round(normalized_correlation(bits, recovered), 6),
        })
    return rows


def summarise_robustness(rows: Sequence[dict]) -> list[dict]:
    keys = sorted({(r["attack"], r["alpha_mode"]) for r in rows})
    summary: list[dict] = []
    for attack, alpha_mode in keys:
        subset = [r for r in rows if r["attack"] == attack and r["alpha_mode"] == alpha_mode]
        m_ber, s_ber = _mean_std([r["recovery_ber"] for r in subset])
        m_acc, s_acc = _mean_std([r["recovery_bit_accuracy"] for r in subset])
        m_nc, s_nc = _mean_std([r["recovery_nc"] for r in subset])
        summary.append({
            "attack": attack,
            "alpha_mode": alpha_mode,
            "n_images": len(subset),
            "params": subset[0]["attack_params"],
            "recovery_ber_mean": m_ber,
            "recovery_ber_std": s_ber,
            "recovery_bit_accuracy_mean": m_acc,
            "recovery_nc_mean": m_nc,
        })
    return summary


# ---------------------------------------------------------------------------
# Environment stamp
# ---------------------------------------------------------------------------

def environment_stamp() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
