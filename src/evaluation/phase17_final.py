"""Phase 17 - Final Integrated Model (evaluation harness).

This module is **new Phase 17 code**. It introduces **no new embedding path**:
the "final integrated model" is a *selected configuration* of the frozen
Phase 6 ``embed()`` / ``extract_traditional()`` plus the already-trained
Phase 8 blind CNN, with the ECC layer from Phase 16 available but switched off
unless a Phase 17 measurement justifies it. Nothing in Phases 6 / 7 / 8 / 10 /
15 is modified - this harness only calls those modules and records metrics.

What "integrated model" means here
----------------------------------
Every earlier phase that proposed an addition was measured, and most additions
did not help (see ``docs/phase17_final.md`` for the evidence trail). Phase 17
therefore does **not** stack techniques. It:

1. Enumerates a small, evidence-driven set of candidate configurations, all
   built from the frozen DWT-SVD embedder: Haar wavelet, LL subband, a fixed
   alpha (swept over a few values), a practical 64-bit payload, no ECC, no
   adaptive embedding, no multi-level DWT.
2. Adds two *confirmation* candidates that deliberately re-test a rejected
   idea inside Phase 17's own protocol - one with Phase 16 ECC, one using the
   Phase 15 improved CNN for blind extraction - so the rejection is shown, not
   asserted.
3. Evaluates all candidates and the frozen Phase 6 baseline on identical
   images / payloads / seeds, for clean and three realistic attacks, with the
   two extraction paths (non-blind reference and blind Phase 8 CNN) reported
   separately, plus embed/extract timing.
4. Selects the final configuration by an explicit, documented rule
   (:func:`select_final`) and writes a baseline-vs-final comparison.

Blindness
---------
The non-blind path needs the original image; the blind Phase 8 CNN path does
not. Results for the two are never merged. The final model is only described
as "blind" for the CNN path.

Scope guard (deliberately NOT in Phase 17)
------------------------------------------
* no change to the frozen Phase 6 formula / defaults / EmbedConfig
* no CNN retraining or redesign (Phase 8 / Phase 15 architectures untouched)
* no ablation study (Phase 18), generalization (Phase 19), security (Phase 20),
  performance engineering (Phase 21)
* no MLflow / PostgreSQL, no Web UI change
"""

from __future__ import annotations

import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pstdev

import cv2
import numpy as np
import yaml

from src.evaluation.attacks import apply_attack
from src.evaluation.metrics import (
    bit_accuracy,
    bit_error_rate,
    mse,
    normalized_correlation,
    psnr,
    ssim,
)
from src.evaluation.phase9_validation import payload_bits
from src.watermark.ecc import SUPPORTED_METHODS, ECCCapacityError, build_codec, check_capacity
from src.watermark.embed import EmbedConfig, embed, extract_traditional, subband_capacity

__all__ = [
    "IMAGE_QUALITY_METRICS",
    "PROJECT_ROOT",
    "WATERMARK_RECOVERY_METRICS",
    "CandidateSpec",
    "Phase17Config",
    "environment_stamp",
    "evaluate_candidate",
    "load_config",
    "load_split_images",
    "select_final",
    "summarise_candidate",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_QUALITY_METRICS: tuple[str, ...] = ("psnr", "ssim", "mse")
WATERMARK_RECOVERY_METRICS: tuple[str, ...] = ("ber", "bit_accuracy", "nc")

_DEFAULT_ATTACKS: dict[str, dict] = {
    "jpeg_compress": {"quality": 75},
    "gaussian_noise": {"sigma": 5},
    "gaussian_blur": {"ksize": 3},
}


# ---------------------------------------------------------------------------
# Candidate configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateSpec:
    """One configuration to evaluate.

    ``role`` drives selection: only ``"candidate"`` rows are eligible to be
    chosen as the final model. ``"baseline"`` is the frozen Phase 6 reference;
    ``"confirmation"`` rows deliberately re-test a rejected idea.
    """

    name: str
    wavelet: str = "haar"
    subband: str = "LL"
    alpha: float = 0.02
    bit_length: int = 64
    start_sv_index: int = 0
    mode: str = "symmetric"
    ecc_method: str = "none"
    raw_payload_bits: int | None = None      # ECC only; encoded length must == bit_length
    blind_extractor: str = "phase8"          # "phase8" | "phase15" | "none"
    role: str = "candidate"

    def __post_init__(self) -> None:
        if self.role not in ("baseline", "candidate", "confirmation"):
            raise ValueError(f"unknown role {self.role!r}")
        if self.blind_extractor not in ("phase8", "phase15", "none"):
            raise ValueError(f"unknown blind_extractor {self.blind_extractor!r}")
        if self.ecc_method not in SUPPORTED_METHODS:
            raise ValueError(f"unsupported ecc_method {self.ecc_method!r}")
        if self.subband not in ("LL", "LH", "HL", "HH"):
            raise ValueError(f"unsupported subband {self.subband!r}")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha}")
        if self.bit_length <= 0:
            raise ValueError(f"bit_length must be positive; got {self.bit_length}")
        if self.ecc_method != "none" and self.raw_payload_bits is None:
            raise ValueError(f"{self.name}: raw_payload_bits is required when ecc_method != 'none'")
        if self.ecc_method == "none" and self.raw_payload_bits not in (None, self.bit_length):
            raise ValueError(
                f"{self.name}: raw_payload_bits must be None or {self.bit_length} without ECC"
            )

    # -- derived --------------------------------------------------------

    @property
    def message_bits(self) -> int:
        """Bits of *user message* (before ECC). Equals bit_length without ECC."""
        return self.raw_payload_bits if self.ecc_method != "none" else self.bit_length

    def embed_config(self) -> EmbedConfig:
        return EmbedConfig(
            wavelet=self.wavelet,
            subband=self.subband,
            mode=self.mode,
            alpha=self.alpha,
            bit_length=self.bit_length,
            start_sv_index=self.start_sv_index,
        )

    def assert_capacity(self, image_size: int) -> None:
        """Encoded payload must fit the configured subband, and the ECC-encoded
        length must exactly equal the number of embedded bits."""
        probe = EmbedConfig(
            wavelet=self.wavelet, subband=self.subband, mode=self.mode,
            alpha=self.alpha, bit_length=1, start_sv_index=self.start_sv_index,
        )
        capacity = subband_capacity((image_size, image_size), probe)
        check_capacity(self.bit_length, capacity, context=f"{self.name} {self.subband}")
        if self.ecc_method != "none":
            codec = build_codec(self.ecc_method)
            encoded = codec.encoded_length(self.raw_payload_bits)
            if encoded != self.bit_length:
                raise ECCCapacityError(
                    f"{self.name}: ECC-encoded length {encoded} "
                    f"({self.ecc_method} on {self.raw_payload_bits} raw bits) must equal "
                    f"bit_length {self.bit_length}; adjust raw_payload_bits so it matches "
                    f"(encoded data must not be truncated)."
                )


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

def _default_candidates() -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec("phase6_baseline", alpha=0.010, blind_extractor="phase8", role="baseline"),
        CandidateSpec("final_haar_ll_a015", alpha=0.015, role="candidate"),
        CandidateSpec("final_haar_ll_a020", alpha=0.020, role="candidate"),
        CandidateSpec("final_haar_ll_a030", alpha=0.030, role="candidate"),
        CandidateSpec("confirm_ecc_hamming84", alpha=0.020, ecc_method="hamming_8_4",
                      raw_payload_bits=32, blind_extractor="phase8", role="confirmation"),
        CandidateSpec("confirm_blind_cnn_v2", alpha=0.020, blind_extractor="phase15",
                      role="confirmation"),
    )


@dataclass(frozen=True)
class Phase17Config:
    """Everything one Phase 17 final-model evaluation run needs."""

    candidates: tuple[CandidateSpec, ...] = field(default_factory=_default_candidates)

    processed_root: str = "data/processed/div2k_256"
    eval_split: str = "test"
    eval_num_images: int = 40
    image_size: int = 256

    payload_seed: int = 20260901
    noise_seed: int = 1234

    attacks: dict[str, dict] = field(default_factory=lambda: dict(_DEFAULT_ATTACKS))

    timing_runs: int = 3
    timing_num_images: int = 5

    phase8_checkpoint: str = "models/phase8_cnn/phase8_cnn_best.pt"
    phase15_checkpoint: str = "models/phase15_cnn/phase15_cnn_best.pt"

    # selection rule parameters
    psnr_floor_db: float = 38.0
    require_blind_compatible: bool = True
    cnn_bit_length: int = 64

    results_dir: str = "results/phase17_final"
    make_plots: bool = True

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("candidates must be non-empty")
        names = [c.name for c in self.candidates]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate candidate names: {names}")
        if sum(1 for c in self.candidates if c.role == "baseline") != 1:
            raise ValueError("exactly one candidate must have role='baseline'")
        if not any(c.role == "candidate" for c in self.candidates):
            raise ValueError("at least one candidate with role='candidate' is required")
        if self.eval_num_images <= 0:
            raise ValueError(f"eval_num_images must be positive; got {self.eval_num_images}")
        for name in self.attacks:
            if name not in ("jpeg_compress", "gaussian_noise", "gaussian_blur",
                            "median_filter", "resize_roundtrip", "rotate", "center_crop"):
                raise ValueError(f"unknown attack {name!r}")

    # -- derived --------------------------------------------------------

    def baseline(self) -> CandidateSpec:
        return next(c for c in self.candidates if c.role == "baseline")

    def resolved_results_dir(self) -> Path:
        return (PROJECT_ROOT / self.results_dir).resolve()

    def resolved_split_dir(self) -> Path:
        return (PROJECT_ROOT / self.processed_root / self.eval_split).resolve()

    def channel_names(self) -> list[str]:
        return ["clean", *self.attacks.keys()]


def load_config(path: str | Path, overrides: dict | None = None) -> Phase17Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["phase17"]
    data = raw.get("data", {})
    timing = raw.get("timing", {})
    extraction = raw.get("extraction", {})
    selection = raw.get("selection", {})
    output = raw.get("output", {})

    candidates: list[CandidateSpec] = []
    for entry in raw.get("candidates", []):
        candidates.append(CandidateSpec(
            name=entry["name"],
            wavelet=entry.get("wavelet", "haar"),
            subband=entry.get("subband", "LL"),
            alpha=float(entry.get("alpha", 0.02)),
            bit_length=int(entry.get("bit_length", 64)),
            start_sv_index=int(entry.get("start_sv_index", 0)),
            mode=entry.get("mode", "symmetric"),
            ecc_method=entry.get("ecc_method", "none"),
            raw_payload_bits=(int(entry["raw_payload_bits"])
                              if entry.get("raw_payload_bits") is not None else None),
            blind_extractor=entry.get("blind_extractor", "phase8"),
            role=entry.get("role", "candidate"),
        ))

    cfg = Phase17Config(
        candidates=tuple(candidates) if candidates else _default_candidates(),
        processed_root=data.get("processed_root", "data/processed/div2k_256"),
        eval_split=data.get("eval_split", "test"),
        eval_num_images=int(data.get("eval_num_images", 40)),
        image_size=int(data.get("image_size", 256)),
        payload_seed=int(raw.get("payload_seed", 20260901)),
        noise_seed=int(raw.get("noise_seed", 1234)),
        attacks=dict(raw.get("attacks", dict(_DEFAULT_ATTACKS))),
        timing_runs=int(timing.get("runs", 3)),
        timing_num_images=int(timing.get("num_images", 5)),
        phase8_checkpoint=extraction.get("phase8_checkpoint", "models/phase8_cnn/phase8_cnn_best.pt"),
        phase15_checkpoint=extraction.get("phase15_checkpoint",
                                          "models/phase15_cnn/phase15_cnn_best.pt"),
        psnr_floor_db=float(selection.get("psnr_floor_db", 38.0)),
        require_blind_compatible=bool(selection.get("require_blind_compatible", True)),
        cnn_bit_length=int(selection.get("cnn_bit_length", 64)),
        results_dir=output.get("results_dir", "results/phase17_final"),
        make_plots=bool(output.get("make_plots", True)),
    )

    if overrides:
        merged = {**cfg.__dict__}
        merged.update({k: v for k, v in overrides.items() if v is not None})
        cfg = Phase17Config(**merged)
    return cfg


# ---------------------------------------------------------------------------
# Image loading (shared convention with Phases 9/14/16)
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
# Per-candidate evaluation
# ---------------------------------------------------------------------------

def _apply_channel(
    wm: np.ndarray, channel: str, params: dict, *, noise_seed: int, image_index: int
) -> np.ndarray:
    if channel == "clean":
        return wm
    p = dict(params)
    if channel == "gaussian_noise":
        p.setdefault("seed", noise_seed + image_index)
    attacked = apply_attack(channel, wm, p)
    if attacked.shape != wm.shape:
        attacked = cv2.resize(attacked, (wm.shape[1], wm.shape[0]), interpolation=cv2.INTER_LINEAR)
    return attacked


def evaluate_candidate(
    images: Sequence[tuple[str, np.ndarray]],
    spec: CandidateSpec,
    cfg: Phase17Config,
    *,
    blind_extractor=None,
) -> dict:
    """All per-image rows for one candidate, across clean + every attack.

    ``blind_extractor`` is an already-loaded ``BlindExtractor`` /
    ``BlindExtractorV2`` (or ``None``). The blind path is only run when the
    extractor is present and its width equals ``spec.bit_length``.
    """
    spec.assert_capacity(cfg.image_size)
    econf = spec.embed_config()
    codec = build_codec(spec.ecc_method)
    msg_bits = spec.message_bits

    blind_ok = (
        spec.blind_extractor != "none"
        and blind_extractor is not None
        and spec.bit_length == cfg.cnn_bit_length
    )
    if spec.blind_extractor == "none":
        blind_status = "not requested"
    elif blind_extractor is None:
        blind_status = f"{spec.blind_extractor} checkpoint unavailable"
    elif spec.bit_length != cfg.cnn_bit_length:
        blind_status = f"skipped: bit_length {spec.bit_length} != CNN width {cfg.cnn_bit_length}"
    else:
        blind_status = f"evaluated ({spec.blind_extractor})"

    channels = [("clean", {})] + [(n, p) for n, p in cfg.attacks.items()]
    rows: list[dict] = []

    for index, (image_id, image) in enumerate(images):
        message = payload_bits(cfg.payload_seed, index, msg_bits)
        embedded = codec.encode(message) if spec.ecc_method != "none" else list(message)
        if len(embedded) != spec.bit_length:  # pragma: no cover - guarded by assert_capacity
            raise ECCCapacityError(f"{spec.name}: embedded length {len(embedded)} != {spec.bit_length}")

        t0 = time.perf_counter()
        result = embed(image, embedded, econf)
        embed_ms = (time.perf_counter() - t0) * 1000.0
        wm = result.watermarked_image

        quality = {"psnr": psnr(image, wm), "ssim": ssim(image, wm), "mse": mse(image, wm)}

        for channel, params in channels:
            transmitted = _apply_channel(
                wm, channel, params, noise_seed=cfg.noise_seed, image_index=index
            )

            t0 = time.perf_counter()
            nb_encoded = extract_traditional(transmitted, image, econf)
            nb_ms = (time.perf_counter() - t0) * 1000.0
            nb_msg = (codec.decode(nb_encoded, msg_bits).bits
                      if spec.ecc_method != "none" else nb_encoded)
            rows.append(_row(image_id, spec, channel, "non_blind", quality, message, nb_msg,
                             embed_ms, nb_ms))

            if blind_ok:
                t0 = time.perf_counter()
                bl_encoded = blind_extractor.extract_bits(transmitted)
                bl_ms = (time.perf_counter() - t0) * 1000.0
                bl_msg = (codec.decode(bl_encoded, msg_bits).bits
                          if spec.ecc_method != "none" else bl_encoded)
                rows.append(_row(image_id, spec, channel, "blind_cnn", quality, message, bl_msg,
                                 embed_ms, bl_ms))

    return {
        "name": spec.name,
        "role": spec.role,
        "params": {
            "wavelet": spec.wavelet, "subband": spec.subband, "alpha": spec.alpha,
            "bit_length": spec.bit_length, "message_bits": msg_bits,
            "ecc_method": spec.ecc_method, "blind_extractor": spec.blind_extractor,
        },
        "blind_status": blind_status,
        "status": "success",
        "rows": rows,
    }


def _row(image_id, spec, channel, decoder, quality, reference, recovered,
         embed_ms, extract_ms) -> dict:
    return {
        "image": image_id,
        "candidate": spec.name,
        "role": spec.role,
        "channel": channel,
        "decoder": decoder,
        "wavelet": spec.wavelet,
        "subband": spec.subband,
        "alpha": spec.alpha,
        "bit_length": spec.bit_length,
        "message_bits": len(reference),
        "ecc_method": spec.ecc_method,
        "quality_psnr": round(quality["psnr"], 4),
        "quality_ssim": round(quality["ssim"], 6),
        "quality_mse": round(quality["mse"], 6),
        "recovery_ber": round(bit_error_rate(reference, recovered), 6),
        "recovery_bit_accuracy": round(bit_accuracy(reference, recovered), 6),
        "recovery_nc": round(normalized_correlation(reference, recovered), 6),
        "exact_match": int(list(recovered) == list(reference)),
        "embed_ms": round(embed_ms, 4),
        "extract_ms": round(extract_ms, 4),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    values = list(values)
    if not values:
        return (0.0, 0.0)
    return (round(fmean(values), 6), round(pstdev(values), 6) if len(values) > 1 else 0.0)


def summarise_candidate(outcome: dict) -> list[dict]:
    """Per-(channel, decoder) mean/std for one candidate."""
    rows = outcome["rows"]
    summary: list[dict] = []
    keys = sorted({(r["channel"], r["decoder"]) for r in rows})
    for channel, decoder in keys:
        subset = [r for r in rows if r["channel"] == channel and r["decoder"] == decoder]
        entry: dict = {
            "candidate": outcome["name"],
            "role": outcome["role"],
            "channel": channel,
            "decoder": decoder,
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
        entry["exact_match_rate"] = round(fmean(r["exact_match"] for r in subset), 6)
        entry["embed_ms_mean"] = _mean_std([r["embed_ms"] for r in subset])[0]
        entry["extract_ms_mean"] = _mean_std([r["extract_ms"] for r in subset])[0]
        summary.append(entry)
    return summary


# ---------------------------------------------------------------------------
# Final-model selection
# ---------------------------------------------------------------------------

def select_final(summaries: Sequence[dict], cfg: Phase17Config) -> dict:
    """Pick the final integrated configuration from the candidate summaries.

    Rule (documented in ``docs/phase17_final.md``), applied to ``role ==
    "candidate"`` only:

    1. **Invisibility gate.** Mean clean-channel PSNR must be
       >= ``psnr_floor_db``. A candidate below the floor is not eligible.
    2. **Score = mean BER across BOTH extraction paths and all channels**
       (clean + every attack), lower is better. The final model has to serve
       the non-blind reference decoder *and* the blind Phase 8 CNN, so both
       are weighted equally; a candidate with no blind rows is scored on its
       non-blind BER alone (and flagged).
    3. Tie-break: higher clean-channel non-blind exact-message rate, then
       higher clean PSNR.

    Returns the chosen candidate name and a full rationale. If the PSNR gate
    removes every candidate the pool falls back to all candidates and
    ``psnr_gate_left_no_candidate`` is set.
    """
    by_candidate: dict[str, list[dict]] = {}
    for s in summaries:
        by_candidate.setdefault(s["candidate"], []).append(s)

    considered: list[dict] = []
    for name, entries in by_candidate.items():
        if entries[0]["role"] != "candidate":
            continue
        nb = {e["channel"]: e for e in entries if e["decoder"] == "non_blind"}
        bl = {e["channel"]: e for e in entries if e["decoder"] == "blind_cnn"}
        if "clean" not in nb:
            continue
        clean_psnr = nb["clean"]["image_quality"]["psnr_mean"]
        nb_ber = fmean(e["watermark_recovery"]["ber_mean"] for e in nb.values())
        if bl:
            bl_ber = fmean(e["watermark_recovery"]["ber_mean"] for e in bl.values())
            combined = fmean([nb_ber, bl_ber])
        else:
            bl_ber = None
            combined = nb_ber
        considered.append({
            "candidate": name,
            "clean_psnr": round(clean_psnr, 4),
            "mean_non_blind_ber": round(nb_ber, 6),
            "mean_blind_cnn_ber": (round(bl_ber, 6) if bl_ber is not None else None),
            "combined_score": round(combined, 6),
            "clean_non_blind_exact_match_rate": nb["clean"]["exact_match_rate"],
            "blind_rows_present": bool(bl),
        })

    gated = [c for c in considered if c["clean_psnr"] >= cfg.psnr_floor_db]
    rejected_for_psnr = [c["candidate"] for c in considered if c["clean_psnr"] < cfg.psnr_floor_db]

    pool = gated or considered  # never return nothing; note the gate failure
    pool_sorted = sorted(
        pool,
        key=lambda c: (c["combined_score"], -c["clean_non_blind_exact_match_rate"], -c["clean_psnr"]),
    )
    chosen = pool_sorted[0]["candidate"] if pool_sorted else None

    return {
        "chosen": chosen,
        "rule": {
            "psnr_floor_db": cfg.psnr_floor_db,
            "require_blind_compatible": cfg.require_blind_compatible,
            "score": "mean BER across non-blind + blind CNN, over clean + all attacks (lower is better)",
            "tie_break": ["higher clean non-blind exact-match rate", "higher clean PSNR"],
        },
        "considered": sorted(considered, key=lambda c: c["combined_score"]),
        "rejected_below_psnr_floor": rejected_for_psnr,
        "psnr_gate_left_no_candidate": not gated,
    }


# ---------------------------------------------------------------------------
# Environment stamp
# ---------------------------------------------------------------------------

def environment_stamp() -> dict:
    stamp = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    try:
        import torch

        stamp["torch"] = torch.__version__
        stamp["cuda"] = bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover
        stamp["torch"] = None
    return stamp
