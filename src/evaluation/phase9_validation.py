"""Phase 9 - Baseline Validation & Robustness Preparation (harness).

This module is **new Phase 9 code**. It imports the frozen Phase 6 baseline
(``src.watermark.embed``) and the already-trained Phase 8 blind extractor
(``src.evaluation.blind_extract``) and *runs* them; it never modifies either.

What Phase 9 does
-----------------
1. Establishes a reproducible, clean-channel (no-attack) reference for the
   frozen DWT-SVD baseline over the held-out DIV2K test split, for the payload
   sizes 8 / 16 / 32 / 64 / 128 bits at the frozen alpha sweep
   {0.005, 0.010, 0.015}.
2. Records every metric in two clearly separated groups:
     * image-quality      -> PSNR, SSIM, MSE
     * watermark-recovery -> BER, bit-accuracy, NC
3. Evaluates the Phase 8 blind CNN extractor separately, on the same unseen
   images, at its trained operating point, and produces a like-for-like
   non-blind vs blind comparison.

What Phase 9 deliberately does NOT do (documented scope guard)
-------------------------------------------------------------
* No JPEG / noise / blur / crop attacks .......... Phase 10
* No adaptive / content-aware embedding .......... Phase 11
* No error-correcting codes ...................... Phase 16
* No CNN architecture changes .................... Phase 15

Everything here is deterministic: given the config, re-running reproduces the
same numbers.
"""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pstdev

import cv2
import numpy as np
import yaml

from src.evaluation.metrics import (
    bit_accuracy,
    bit_error_rate,
    mse,
    normalized_correlation,
    psnr,
    ssim,
)
from src.watermark.embed import (
    BASELINE_ALPHAS,
    EmbedConfig,
    embed,
    extract_traditional,
)

__all__ = [
    "FROZEN_BASELINE_FILES",
    "IMAGE_QUALITY_METRICS",
    "METRIC_GROUPS",
    "PROJECT_ROOT",
    "WATERMARK_RECOVERY_METRICS",
    "Phase9Config",
    "assert_baseline_frozen",
    "build_baseline_embed_config",
    "evaluate_baseline_point",
    "evaluate_cnn_point",
    "frozen_file_hashes",
    "iter_split_images",
    "load_config",
    "load_split_images",
    "master_payload",
    "payload_bits",
    "summarise_baseline",
    "summarise_cnn",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# Metric taxonomy - the Phase 9 requirement to separate the two families.
# --------------------------------------------------------------------------

IMAGE_QUALITY_METRICS: tuple[str, ...] = ("psnr", "ssim", "mse")
WATERMARK_RECOVERY_METRICS: tuple[str, ...] = ("ber", "bit_accuracy", "nc")
METRIC_GROUPS: dict[str, tuple[str, ...]] = {
    "image_quality": IMAGE_QUALITY_METRICS,
    "watermark_recovery": WATERMARK_RECOVERY_METRICS,
}

# Files that make up the frozen Phase 6 baseline. Phase 9 records their hashes
# in every report and refuses to run if the behavioural contract has drifted.
FROZEN_BASELINE_FILES: tuple[str, ...] = (
    "src/watermark/embed.py",
    "src/watermark/dwt.py",
    "src/watermark/svd.py",
    "src/evaluation/metrics.py",
    "configs/baseline.yaml",
)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Phase9CNNConfig:
    enabled: bool = True
    checkpoint: str = "models/phase8_cnn/phase8_cnn_best.pt"
    bit_length: int = 64
    alpha: float = 0.02
    payload_seed: int = 20260901


@dataclass(frozen=True)
class Phase9Config:
    """Everything one Phase 9 validation run needs."""

    # frozen Phase 6 baseline parameters (validated against the real baseline)
    wavelet: str = "haar"
    mode: str = "symmetric"
    subband: str = "LL"
    start_sv_index: int = 0
    alphas: tuple[float, ...] = BASELINE_ALPHAS
    payload_bits: tuple[int, ...] = (8, 16, 32, 64, 128)

    # evaluation data
    processed_root: str = "data/processed/div2k_256"
    eval_split: str = "test"
    eval_num_images: int = 100
    image_size: int = 256

    # deterministic payloads
    payload_seed: int = 20260901

    # Phase 8 blind CNN extractor
    cnn: Phase9CNNConfig = field(default_factory=Phase9CNNConfig)

    # output
    results_dir: str = "results/phase9_validation"
    make_plots: bool = True

    def __post_init__(self) -> None:
        if not self.payload_bits:
            raise ValueError("payload_bits must be non-empty")
        if any(n <= 0 for n in self.payload_bits):
            raise ValueError(f"payload_bits must be positive; got {self.payload_bits}")
        if max(self.payload_bits) > 128:
            raise ValueError(
                "Phase 9 validates the frozen LL-only baseline (native capacity "
                f"128 bits); payload_bits={self.payload_bits} exceeds it. "
                "Larger payloads are a Phase 12 (capacity) concern."
            )
        if not self.alphas:
            raise ValueError("alphas must be non-empty")
        if any(not 0.0 < a < 1.0 for a in self.alphas):
            raise ValueError(f"every alpha must be in (0, 1); got {self.alphas}")
        if self.eval_num_images <= 0:
            raise ValueError(f"eval_num_images must be positive; got {self.eval_num_images}")

    # -- derived ------------------------------------------------------------

    def baseline_embed_config(self, alpha: float, n_bits: int) -> EmbedConfig:
        return build_baseline_embed_config(self, alpha, n_bits)

    def resolved_results_dir(self) -> Path:
        return (PROJECT_ROOT / self.results_dir).resolve()

    def resolved_split_dir(self) -> Path:
        return (PROJECT_ROOT / self.processed_root / self.eval_split).resolve()


def load_config(path: str | Path, overrides: dict | None = None) -> Phase9Config:
    """Load ``configs/phase9_validation.yaml`` (``phase9:`` block) into a
    :class:`Phase9Config`, then apply optional overrides (used by ``--quick``)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["phase9"]
    baseline = raw.get("baseline", {})
    data = raw.get("data", {})
    cnn = raw.get("cnn", {})
    output = raw.get("output", {})

    cfg = Phase9Config(
        wavelet=baseline.get("wavelet", "haar"),
        mode=baseline.get("mode", "symmetric"),
        subband=baseline.get("subband", "LL"),
        start_sv_index=int(baseline.get("start_sv_index", 0)),
        alphas=tuple(float(a) for a in baseline.get("alphas", BASELINE_ALPHAS)),
        payload_bits=tuple(int(b) for b in baseline.get("payload_bits", (8, 16, 32, 64, 128))),
        processed_root=data.get("processed_root", "data/processed/div2k_256"),
        eval_split=data.get("eval_split", "test"),
        eval_num_images=int(data.get("eval_num_images", 100)),
        image_size=int(data.get("image_size", 256)),
        payload_seed=int(raw.get("payload_seed", 20260901)),
        cnn=Phase9CNNConfig(
            enabled=bool(cnn.get("enabled", True)),
            checkpoint=cnn.get("checkpoint", "models/phase8_cnn/phase8_cnn_best.pt"),
            bit_length=int(cnn.get("bit_length", 64)),
            alpha=float(cnn.get("alpha", 0.02)),
            payload_seed=int(cnn.get("payload_seed", 20260901)),
        ),
        results_dir=output.get("results_dir", "results/phase9_validation"),
        make_plots=bool(output.get("make_plots", True)),
    )

    if overrides:
        merged = {**cfg.__dict__}
        merged.update({k: v for k, v in overrides.items() if v is not None})
        cfg = Phase9Config(**merged)
    return cfg


# --------------------------------------------------------------------------
# Frozen-baseline guard
# --------------------------------------------------------------------------

def frozen_file_hashes(root: Path | None = None) -> dict[str, str]:
    """SHA-256 of every frozen Phase 6 file, for provenance in the report."""
    base = root or PROJECT_ROOT
    out: dict[str, str] = {}
    for rel in FROZEN_BASELINE_FILES:
        path = base / rel
        if path.is_file():
            out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def assert_baseline_frozen() -> None:
    """Fail loudly if the frozen Phase 6 embedding contract has drifted.

    This is a *behavioural* check, not a byte check: it verifies the public
    constants, the :class:`EmbedConfig` defaults and the exact multiplicative
    ``sigma'[i] = sigma[i] * (1 +/- alpha)`` rule that every later phase relies
    on. Reformatting ``embed.py`` is allowed; changing what it computes is not.
    """
    if tuple(BASELINE_ALPHAS) != (0.005, 0.010, 0.015):
        raise AssertionError(f"BASELINE_ALPHAS changed: {BASELINE_ALPHAS!r}")

    default = EmbedConfig()
    expected = {
        "wavelet": "haar",
        "subband": "LL",
        "extra_subbands": (),
        "alpha": 0.010,
        "bit_length": 64,
        "start_sv_index": 0,
        "mode": "symmetric",
    }
    for name, want in expected.items():
        got = getattr(default, name)
        if got != want:
            raise AssertionError(f"EmbedConfig.{name} changed: {got!r} != {want!r}")

    # Exact embedding formula: bit 1 -> *(1 + alpha), bit 0 -> *(1 - alpha),
    # verified on the leading singular value of a real DWT-LL subband.
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    alpha = 0.01
    cfg = EmbedConfig(alpha=alpha, bit_length=4)
    from src.watermark.dwt import decompose_2d
    from src.watermark.embed import _rgb_to_ycrcb
    from src.watermark.svd import decompose

    y, _, _ = _rgb_to_ycrcb(image)
    sigma0 = decompose(decompose_2d(y, wavelet="haar", mode="symmetric").ll).S.copy()
    ones = embed(image, [1, 1, 1, 1], cfg)
    zeros = embed(image, [0, 0, 0, 0], cfg)
    ratio_one = ones.watermarked_sv["LL"][:4] / sigma0[:4]
    ratio_zero = zeros.watermarked_sv["LL"][:4] / sigma0[:4]
    if not np.allclose(ratio_one, 1.0 + alpha, atol=1e-9):
        raise AssertionError(f"bit=1 modulation not (1+alpha): {ratio_one}")
    if not np.allclose(ratio_zero, 1.0 - alpha, atol=1e-9):
        raise AssertionError(f"bit=0 modulation not (1-alpha): {ratio_zero}")


# --------------------------------------------------------------------------
# Deterministic payloads
# --------------------------------------------------------------------------

def master_payload(seed: int, image_index: int, n_bits: int = 128) -> np.ndarray:
    """The 128-bit master payload for one image, fixed by ``(seed, index)``."""
    return np.random.default_rng(seed + image_index).integers(0, 2, size=n_bits, dtype=np.int64)


def payload_bits(seed: int, image_index: int, n_bits: int) -> list[int]:
    """The first ``n_bits`` of the master payload - payloads nest across sizes."""
    if n_bits > 128:
        raise ValueError("payloads > 128 bits are out of Phase 9 scope")
    return master_payload(seed, image_index)[:n_bits].tolist()


# --------------------------------------------------------------------------
# Frozen baseline embed config
# --------------------------------------------------------------------------

def build_baseline_embed_config(cfg: Phase9Config, alpha: float, n_bits: int) -> EmbedConfig:
    """The frozen LL-only baseline config for one (alpha, payload) point.

    No ``extra_subbands`` are ever set: Phase 9 stays inside the 128-bit native
    LL capacity, so the frozen baseline runs exactly as in Phase 6.
    """
    if n_bits > 128:
        raise ValueError(
            f"{n_bits}-bit payload exceeds the frozen LL-only capacity (128); "
            "out of Phase 9 scope"
        )
    return EmbedConfig(
        wavelet=cfg.wavelet,
        mode=cfg.mode,
        subband=cfg.subband,
        extra_subbands=(),
        alpha=alpha,
        bit_length=n_bits,
        start_sv_index=cfg.start_sv_index,
    )


# --------------------------------------------------------------------------
# Image loading
# --------------------------------------------------------------------------

def iter_split_images(split_dir: Path, limit: int) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(image_id, rgb_uint8)`` for the first ``limit`` PNGs in a split."""
    paths = sorted(split_dir.glob("*.png"))[:limit]
    if not paths:
        raise FileNotFoundError(f"no processed images in {split_dir}")
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"unreadable image: {path}")
        yield path.stem, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_split_images(split_dir: Path, limit: int) -> list[tuple[str, np.ndarray]]:
    return list(iter_split_images(split_dir, limit))


# --------------------------------------------------------------------------
# Baseline evaluation (frozen DWT-SVD, non-blind decoder)
# --------------------------------------------------------------------------

def evaluate_baseline_point(
    images: Sequence[tuple[str, np.ndarray]],
    embed_config: EmbedConfig,
    *,
    payload_seed: int,
) -> list[dict]:
    """One row per image for a single (alpha, payload) point.

    Each row carries the image-quality group and the watermark-recovery group
    with an explicit ``group`` tag on every metric via the column names
    ``quality_*`` / ``recovery_*``.
    """
    rows: list[dict] = []
    n_bits = embed_config.bit_length
    for index, (image_id, image) in enumerate(images):
        bits = payload_bits(payload_seed, index, n_bits)
        result = embed(image, bits, embed_config)
        wm = result.watermarked_image
        recovered = extract_traditional(wm, image, embed_config)
        rows.append({
            "image": image_id,
            "alpha": embed_config.alpha,
            "payload_bits": n_bits,
            "subband_order": "+".join(embed_config.subband_order),
            "decoder": "non_blind",
            # image-quality group
            "quality_psnr": round(psnr(image, wm), 4),
            "quality_ssim": round(ssim(image, wm), 6),
            "quality_mse": round(mse(image, wm), 6),
            # watermark-recovery group
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


def summarise_baseline(rows: Sequence[dict]) -> list[dict]:
    """Aggregate per-image rows to mean/std per (alpha, payload), keeping the
    two metric groups separated."""
    keys = sorted({(r["alpha"], r["payload_bits"]) for r in rows})
    summary: list[dict] = []
    for alpha, n_bits in keys:
        subset = [r for r in rows if r["alpha"] == alpha and r["payload_bits"] == n_bits]
        entry: dict = {
            "alpha": alpha,
            "payload_bits": n_bits,
            "subband_order": subset[0]["subband_order"],
            "decoder": "non_blind",
            "n_images": len(subset),
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


# --------------------------------------------------------------------------
# Phase 8 blind CNN evaluation (architecture unchanged)
# --------------------------------------------------------------------------

def evaluate_cnn_point(
    images: Sequence[tuple[str, np.ndarray]],
    extractor,  # src.evaluation.blind_extract.BlindExtractor
    *,
    cnn_config: Phase9CNNConfig,
) -> list[dict]:
    """Blind recovery, plus the non-blind decoder on the *same* watermarked
    images, at the CNN's trained operating point (payload / alpha from the
    checkpoint). One row per image; recovery metrics for both decoders.
    """
    embed_config = EmbedConfig(alpha=cnn_config.alpha, bit_length=cnn_config.bit_length)
    rows: list[dict] = []
    for index, (image_id, image) in enumerate(images):
        bits = payload_bits(cnn_config.payload_seed, index, cnn_config.bit_length)
        result = embed(image, bits, embed_config)
        wm = result.watermarked_image

        blind = extractor.extract_bits(wm)
        nonblind = extract_traditional(wm, image, embed_config)
        rows.append({
            "image": image_id,
            "alpha": cnn_config.alpha,
            "payload_bits": cnn_config.bit_length,
            # image quality is a property of the (shared) watermarked image
            "quality_psnr": round(psnr(image, wm), 4),
            "quality_ssim": round(ssim(image, wm), 6),
            "quality_mse": round(mse(image, wm), 6),
            # blind CNN recovery
            "blind_ber": round(bit_error_rate(bits, blind), 6),
            "blind_bit_accuracy": round(bit_accuracy(bits, blind), 6),
            "blind_nc": round(normalized_correlation(bits, blind), 6),
            # non-blind recovery on the identical image (reference)
            "nonblind_ber": round(bit_error_rate(bits, nonblind), 6),
            "nonblind_bit_accuracy": round(bit_accuracy(bits, nonblind), 6),
            "nonblind_nc": round(normalized_correlation(bits, nonblind), 6),
        })
    return rows


def summarise_cnn(rows: Sequence[dict], cnn_config: Phase9CNNConfig) -> dict:
    q = {
        f"{m}_mean": _mean_std([r[f"quality_{m}"] for r in rows])[0]
        for m in IMAGE_QUALITY_METRICS
    }
    blind = {}
    nonblind = {}
    for metric in WATERMARK_RECOVERY_METRICS:
        bm, bs = _mean_std([r[f"blind_{metric}"] for r in rows])
        nm, ns = _mean_std([r[f"nonblind_{metric}"] for r in rows])
        blind[f"{metric}_mean"] = bm
        blind[f"{metric}_std"] = bs
        nonblind[f"{metric}_mean"] = nm
        nonblind[f"{metric}_std"] = ns
    return {
        "operating_point": {
            "payload_bits": cnn_config.bit_length,
            "alpha": cnn_config.alpha,
            "checkpoint": cnn_config.checkpoint,
        },
        "n_images": len(rows),
        "image_quality": q,
        "watermark_recovery_blind_cnn": blind,
        "watermark_recovery_non_blind_reference": nonblind,
    }


# --------------------------------------------------------------------------
# Environment stamp
# --------------------------------------------------------------------------

def environment_stamp() -> dict:
    stamp = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    try:  # torch only needed for the CNN half
        import torch

        stamp["torch"] = torch.__version__
        stamp["cuda"] = bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover - torch always present in this env
        stamp["torch"] = None
    return stamp
