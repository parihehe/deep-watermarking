"""Evaluate the experimental windowed 1D-CNN blind extractor.

Real measurement, not projection:
* same held-out DIV2K test split as the production decoder eval;
* the SAME frozen embedding pipeline (``src.watermark.embed.embed``) at the
  Phase 17 final operating point (haar / LL / alpha=0.02 / 64-bit / no ECC);
* blind extraction from the watermarked image ONLY (the original is never
  given to the decoder);
* reports bit accuracy, BER, exact-payload rate, mean confidence, plus PSNR /
  SSIM of the watermarked image and inference time.

If the production Phase 8 checkpoint is present, the same trials are also run
through it (same images, same embedded bits) so the two decoders can be
compared like-for-like. If it is absent, only the windowed decoder is scored
and the comparison columns are left as None.

Works at any bit length the checkpoint declares (e.g. 8, 16, 32, 64, 128).

Run::
    python experiments/run_windowed_cnn_eval.py --n-images 100 --bit-length 64
    python experiments/run_windowed_cnn_eval.py --checkpoint models/experimental/windowed_cnn/windowed_cnn_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from src.evaluation.metrics import (
    bit_accuracy,
    bit_error_rate,
    normalized_correlation,
    quality_report,
)
from src.watermark.embed import EmbedConfig, embed

DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "div2k_256"
CHECKPOINT = PROJECT_ROOT / "models" / "experimental" / "windowed_cnn" / "windowed_cnn_best.pt"
PRODUCTION_CHECKPOINT = PROJECT_ROOT / "models" / "phase8_cnn" / "phase8_cnn_best.pt"
RESULTS_DIR = PROJECT_ROOT / "results" / "windowed_cnn"

EMBED_ALPHA = 0.02
EMBED_WAVELET = "haar"
EMBED_SUBBAND = "LL"
EMBED_MODE = "symmetric"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    ap.add_argument("--production-checkpoint", default=str(PRODUCTION_CHECKPOINT))
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--bit-length", type=int, default=64)
    ap.add_argument(
        "--seed",
        type=int,
        default=20260906,
        help="seed for signing payloads; test split is deterministic-selectable",
    )
    args = ap.parse_args(argv)

    if not Path(args.checkpoint).is_file():
        raise SystemExit(f"windowed checkpoint not found: {args.checkpoint}")

    from src.evaluation.windowed_extract import WindowedExtractor

    windowed = WindowedExtractor.from_checkpoint(args.checkpoint)
    declared = windowed.config.bit_length
    bit_length = args.bit_length
    if bit_length != declared:
        print(f"WARNING: checkpoint declares bit_length={declared}; using {declared} for trials")
        bit_length = declared

    prod_ext = None
    if Path(args.production_checkpoint).is_file():
        from src.evaluation.blind_extract import BlindExtractor

        prod_ext = BlindExtractor.from_checkpoint(args.production_checkpoint)
        if int(prod_ext.config.bit_length) != bit_length:
            print(
                f"WARNING: production checkpoint is {prod_ext.config.bit_length}-bit; "
                f"comparison for {bit_length}-bit trials will be None"
            )

    paths = sorted((DATA_ROOT / "test").glob("*.png"))[: args.n_images]
    if len(paths) < args.n_images:
        print(f"WARNING: only {len(paths)} DIV2K test images available (requested {args.n_images})")
    if not paths:
        raise SystemExit(
            f"no processed DIV2K test images in {DATA_ROOT / 'test'}. Run "
            f"scripts/prepare_div2k.py (or the kagglehub setup script) first."
        )

    embed_config = EmbedConfig(
        wavelet=EMBED_WAVELET,
        subband=EMBED_SUBBAND,
        alpha=EMBED_ALPHA,
        bit_length=bit_length,
        start_sv_index=0,
        mode=EMBED_MODE,
    )
    rng = np.random.default_rng(args.seed)

    trials: list[dict] = []
    t0 = time.time()
    quality_dims = {"psnr": 0.0, "ssim": 0.0, "mse": 0.0}
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not read test image: {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bits = rng.integers(0, 2, size=bit_length, dtype=np.int64).tolist()
        wm = embed(rgb, bits, embed_config).watermarked_image

        q = quality_report(rgb, wm)
        for k in quality_dims:
            quality_dims[k] += q[k]

        t_w = time.time()
        w_probs = windowed.extract_proba(wm)
        w_infer = time.time() - t_w
        w_bits = (w_probs > 0.5).astype(int).tolist()

        trial = {
            "image": p.name,
            "bit_length": bit_length,
            "quality_psnr": round(q["psnr"], 4),
            "quality_ssim": round(q["ssim"], 6),
            "windowed": {
                "bit_accuracy": round(bit_accuracy(bits, w_bits), 6),
                "ber": round(bit_error_rate(bits, w_bits), 6),
                "nc": round(normalized_correlation(bits, w_bits), 6),
                "exact_match": int(w_bits == bits),
                "confidence": round(float(np.mean(np.abs(w_probs - 0.5)) * 2.0), 6),
                "infer_seconds": round(w_infer, 6),
            },
        }
        if prod_ext is not None and int(prod_ext.config.bit_length) == bit_length:
            t_p = time.time()
            p_probs = prod_ext.extract_proba(wm)
            p_infer = time.time() - t_p
            p_bits = (p_probs > 0.5).astype(int).tolist()
            trial["production"] = {
                "bit_accuracy": round(bit_accuracy(bits, p_bits), 6),
                "ber": round(bit_error_rate(bits, p_bits), 6),
                "nc": round(normalized_correlation(bits, p_bits), 6),
                "exact_match": int(p_bits == bits),
                "confidence": round(float(np.mean(np.abs(p_probs - 0.5)) * 2.0), 6),
                "infer_seconds": round(p_infer, 6),
            }
        trials.append(trial)

    n = len(trials)
    elapsed = time.time() - t0
    for k, value in quality_dims.items():
        quality_dims[k] = round(value / n, 6)

    def _agg(key: str):
        vals = [t[key] for t in trials if key in t]
        if not vals:
            return None
        return {
            "bit_accuracy": round(float(np.mean([v["bit_accuracy"] for v in vals])), 6),
            "ber": round(float(np.mean([v["ber"] for v in vals])), 6),
            "exact_match_rate": round(float(np.mean([v["exact_match"] for v in vals])), 6),
            "mean_confidence": round(float(np.mean([v["confidence"] for v in vals])), 6),
            "mean_infer_seconds": round(float(np.mean([v["infer_seconds"] for v in vals])), 6),
        }

    agg = {
        "n_trials": n,
        "bit_length": bit_length,
        "seed": args.seed,
        "embed": {
            "wavelet": EMBED_WAVELET,
            "subband": EMBED_SUBBAND,
            "alpha": EMBED_ALPHA,
            "mode": EMBED_MODE,
        },
        "quality_mean": quality_dims,
        "windowed": _agg("windowed"),
        "production": _agg("production"),
        "elapsed_seconds": round(elapsed, 1),
        "windowed_checkpoint": str(Path(args.checkpoint).resolve().relative_to(PROJECT_ROOT)),
    }

    def _line(label: str, d: dict | None) -> str:
        if not d:
            return f"  {label:12s}: not evaluated (checkpoint absent/wrong width)"
        return (
            f"  {label:12s}: bit_acc={d['bit_accuracy']:.4f}  ber={d['ber']:.4f}  "
            f"exact={d['exact_match_rate']:.4f}  conf={d['mean_confidence']:.4f}  "
            f"infer={d['mean_infer_seconds']:.4f}s"
        )

    print(
        f"windowed window-size={windowed.config.window_size}  "
        f"bits={bit_length}  images={n}  (elapsed {elapsed:.1f}s)"
    )
    print(
        f"  quality mean: PSNR={quality_dims['psnr']:.2f} dB  SSIM={quality_dims['ssim']:.4f}  "
        f"MSE={quality_dims['mse']:.4f}"
    )
    print(_line("WINDOWED CNN", agg["windowed"]))
    print(_line("PRODUCTION", agg["production"]))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"windowed_cnn_eval_{bit_length}bit.json"
    out.write_text(json.dumps(agg, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
