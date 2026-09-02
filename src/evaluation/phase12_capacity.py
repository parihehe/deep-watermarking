"""Phase 12 - High-Capacity Watermarking (evaluation harness).

This module is **new Phase 12 code**. It imports and *runs* the new Phase 12
multi-level capacity embedder (``src.watermark.capacity_embed``, itself
additive - see that module's docstring for why capacity is fundamentally
limited and why a new path is needed), the frozen Phase 6 guard machinery
(reused from Phase 9), the frozen Phase 3 payload generator
(``watermark_generator.generate_from_uuid``), and ``metrics.py``. Nothing
under ``src/watermark/`` other than the new ``capacity_embed.py`` file, and
nothing under ``src/evaluation/metrics.py`` or ``src/evaluation/phase9_validation.py``,
is modified.

What Phase 12 measures
-----------------------
For each configured experiment (a payload size + DWT level count + subband
allocation + alpha), either:

* the plan has enough capacity - embed a deterministic payload in every
  held-out DIV2K test image, recover non-blind, and record the two metric
  groups (image-quality: PSNR/SSIM/MSE; watermark-recovery: BER/bit-accuracy/NC); or
* the plan does **not** have enough capacity - record the failure explicitly
  (``status="insufficient_capacity"``, with the exact available vs required
  bit counts) and skip embedding for that experiment. No configuration is
  ever silently forced or truncated.

What Phase 12 deliberately does NOT do (scope guard)
------------------------------------------------------
* No change to any frozen Phase 6 file - verified via the Phase 9 frozen-file
  hash guard.
* No Phase 8 CNN work - non-blind extraction only.
* No error-correcting codes, no blockchain, no MLflow/PostgreSQL, no
  production backend.
* No silent change to Phase 7/8 behaviour - both are sanity-checked at the end
  of every run.
"""

from __future__ import annotations

import platform
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev

import cv2
import numpy as np
import yaml

from src.evaluation.metrics import bit_accuracy, bit_error_rate, mse, normalized_correlation, psnr, ssim
from src.watermark.capacity_embed import (
    CapacityEmbedConfig,
    InsufficientCapacityError,
    embed_capacity,
    extract_capacity,
    theoretical_capacity,
    theoretical_capacity_limit,
    total_capacity,
)
from src.watermark.watermark_generator import generate_from_uuid, validate_bit_length

__all__ = [
    "IMAGE_QUALITY_METRICS",
    "PROJECT_ROOT",
    "WATERMARK_RECOVERY_METRICS",
    "ExperimentSpec",
    "Phase12Config",
    "evaluate_experiment",
    "load_config",
    "load_split_images",
    "payload_for_image",
    "summarise_experiment",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_QUALITY_METRICS: tuple[str, ...] = ("psnr", "ssim", "mse")
WATERMARK_RECOVERY_METRICS: tuple[str, ...] = ("ber", "bit_accuracy", "nc")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentSpec:
    """One Phase 12 experiment point: a payload size to attempt at a given
    DWT-level / subband-allocation / alpha configuration."""

    name: str
    dwt_levels: int
    subband_plan: tuple[tuple[int, str], ...]
    bit_length: int
    alpha: float = 0.02
    wavelet: str = "haar"
    mode: str = "symmetric"
    start_sv_index: int = 0

    def embed_config(self) -> CapacityEmbedConfig:
        return CapacityEmbedConfig(
            wavelet=self.wavelet,
            mode=self.mode,
            dwt_levels=self.dwt_levels,
            subband_plan=self.subband_plan,
            alpha=self.alpha,
            bit_length=self.bit_length,
            start_sv_index=self.start_sv_index,
        )


@dataclass(frozen=True)
class Phase12Config:
    experiments: tuple[ExperimentSpec, ...]

    processed_root: str = "data/processed/div2k_256"
    eval_split: str = "test"
    eval_num_images: int = 30
    image_size: int = 256

    payload_seed: int = 20260901

    results_dir: str = "results/phase12_capacity"
    make_plots: bool = True

    def __post_init__(self) -> None:
        if not self.experiments:
            raise ValueError("experiments must be non-empty")
        names = [e.name for e in self.experiments]
        if len(set(names)) != len(names):
            raise ValueError(f"experiment names must be unique: {names}")
        if self.eval_num_images <= 0:
            raise ValueError(f"eval_num_images must be positive; got {self.eval_num_images}")

    def resolved_results_dir(self) -> Path:
        return (PROJECT_ROOT / self.results_dir).resolve()

    def resolved_split_dir(self) -> Path:
        return (PROJECT_ROOT / self.processed_root / self.eval_split).resolve()


def _parse_subband_plan(raw: list) -> tuple[tuple[int, str], ...]:
    return tuple((int(level), str(band)) for level, band in raw)


def load_config(path: str | Path, overrides: dict | None = None) -> Phase12Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["phase12"]
    data = raw.get("data", {})
    output = raw.get("output", {})

    experiments = tuple(
        ExperimentSpec(
            name=e["name"],
            dwt_levels=int(e["dwt_levels"]),
            subband_plan=_parse_subband_plan(e["subband_plan"]),
            bit_length=int(e["bit_length"]),
            alpha=float(e.get("alpha", 0.02)),
            wavelet=e.get("wavelet", "haar"),
            mode=e.get("mode", "symmetric"),
            start_sv_index=int(e.get("start_sv_index", 0)),
        )
        for e in raw["experiments"]
    )

    cfg = Phase12Config(
        experiments=experiments,
        processed_root=data.get("processed_root", "data/processed/div2k_256"),
        eval_split=data.get("eval_split", "test"),
        eval_num_images=int(data.get("eval_num_images", 30)),
        image_size=int(data.get("image_size", 256)),
        payload_seed=int(raw.get("payload_seed", 20260901)),
        results_dir=output.get("results_dir", "results/phase12_capacity"),
        make_plots=bool(output.get("make_plots", True)),
    )

    if overrides:
        merged = {**cfg.__dict__}
        merged.update({k: v for k, v in overrides.items() if v is not None})
        cfg = Phase12Config(**merged)
    return cfg


# ---------------------------------------------------------------------------
# Deterministic payloads (reuses the frozen Phase 3 generator)
# ---------------------------------------------------------------------------

def payload_for_image(payload_seed: int, image_index: int, bit_length: int) -> list[int]:
    """Deterministic payload for one image, built from the frozen
    ``generate_from_uuid`` (Phase 3). ``bit_length`` must be one of
    ``watermark_generator.SUPPORTED_BIT_LENGTHS`` (128/256/512/1024 for this
    phase)."""
    validate_bit_length(bit_length)
    identifier = uuid.uuid5(uuid.NAMESPACE_OID, f"phase12:{payload_seed}:{image_index}")
    return generate_from_uuid(identifier, bit_length)


# ---------------------------------------------------------------------------
# Image loading
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
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_experiment(
    images: Sequence[tuple[str, np.ndarray]],
    spec: ExperimentSpec,
    *,
    payload_seed: int,
    image_size: int,
) -> dict:
    """Run (or fail fast on) one experiment. Always returns a dict describing
    the outcome - never raises for an insufficient-capacity config."""
    econf = spec.embed_config()
    capacity = total_capacity(econf, image_size)
    theoretical_full = theoretical_capacity(image_size, spec.dwt_levels)

    base = {
        "name": spec.name,
        "dwt_levels": spec.dwt_levels,
        "subband_plan": "+".join(f"L{lv}{band}" for lv, band in spec.subband_plan),
        "bit_length": spec.bit_length,
        "alpha": spec.alpha,
        "capacity_available": capacity,
        "theoretical_full_plan_capacity": theoretical_full,
    }

    if capacity < spec.bit_length:
        base.update({
            "status": "insufficient_capacity",
            "n_images": 0,
            "rows": [],
            "reason": (
                f"subband_plan provides {capacity} singular-value slots but "
                f"{spec.bit_length} bits were requested"
            ),
        })
        return base

    rows: list[dict] = []
    for index, (image_id, image) in enumerate(images):
        bits = payload_for_image(payload_seed, index, spec.bit_length)
        result = embed_capacity(image, bits, econf)
        wm = result.watermarked_image
        recovered = extract_capacity(wm, image, econf)
        rows.append({
            "experiment": spec.name,
            "image": image_id,
            "quality_psnr": round(psnr(image, wm), 4),
            "quality_ssim": round(ssim(image, wm), 6),
            "quality_mse": round(mse(image, wm), 6),
            "recovery_ber": round(bit_error_rate(bits, recovered), 6),
            "recovery_bit_accuracy": round(bit_accuracy(bits, recovered), 6),
            "recovery_nc": round(normalized_correlation(bits, recovered), 6),
        })

    base.update({"status": "success", "n_images": len(rows), "rows": rows})
    return base


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    values = list(values)
    if not values:
        return (0.0, 0.0)
    return (round(fmean(values), 6), round(pstdev(values), 6) if len(values) > 1 else 0.0)


def summarise_experiment(outcome: dict) -> dict:
    """Aggregate one experiment's per-image rows into mean/std, keeping the
    two metric groups separated. For a failed (insufficient-capacity)
    experiment, the metric blocks are simply absent."""
    entry: dict = {
        "name": outcome["name"],
        "dwt_levels": outcome["dwt_levels"],
        "subband_plan": outcome["subband_plan"],
        "bit_length": outcome["bit_length"],
        "alpha": outcome["alpha"],
        "capacity_available": outcome["capacity_available"],
        "theoretical_full_plan_capacity": outcome["theoretical_full_plan_capacity"],
        "status": outcome["status"],
        "n_images": outcome["n_images"],
    }
    if outcome["status"] != "success":
        entry["reason"] = outcome["reason"]
        return entry

    rows = outcome["rows"]
    entry["image_quality"] = {}
    entry["watermark_recovery"] = {}
    for metric in IMAGE_QUALITY_METRICS:
        m, s = _mean_std([r[f"quality_{metric}"] for r in rows])
        entry["image_quality"][f"{metric}_mean"] = m
        entry["image_quality"][f"{metric}_std"] = s
    for metric in WATERMARK_RECOVERY_METRICS:
        m, s = _mean_std([r[f"recovery_{metric}"] for r in rows])
        entry["watermark_recovery"][f"{metric}_mean"] = m
        entry["watermark_recovery"][f"{metric}_std"] = s
    return entry


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


def capacity_bound_table(image_size: int = 256, max_levels: int = 6) -> list[dict]:
    """The closed-form ``theoretical_capacity`` for levels 1..max_levels, plus
    the L->infinity limit - used in the report/docs to show 1024 bits is
    unreachable at *any* DWT level for this image size."""
    rows = [
        {"dwt_levels": level, "max_capacity_bits": theoretical_capacity(image_size, level)}
        for level in range(1, max_levels + 1)
    ]
    rows.append({"dwt_levels": "infinity", "max_capacity_bits": theoretical_capacity_limit(image_size)})
    return rows
