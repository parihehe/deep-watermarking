"""Phase 6 — frozen DWT-SVD baseline characterisation experiment.

Runs the baseline embedder over a fixed set of held-out DIV2K test images for
every (alpha, payload) combination in ``configs/baseline.yaml`` and records the
real image-quality and (non-blind) recovery metrics. Nothing here is simulated:
every number comes from an actual embed -> IDWT -> re-DWT -> compare round trip.

Outputs (under ``results/phase6_baseline/``):
  * ``baseline_metrics.csv``   — one row per (alpha, payload, image)
  * ``baseline_summary.csv``   — mean/std aggregated per (alpha, payload)
  * ``baseline_summary.json``  — same summary plus run metadata
  * ``samples/``               — original / watermarked / residual PNGs for a
                                 few (alpha, payload) points, for the report/demo

Usage:
    .venv/Scripts/python.exe experiments/run_baseline.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    bit_accuracy,
    bit_error_rate,
    mse,
    normalized_correlation,
    psnr,
    ssim,
)
from src.watermark.embed import EmbedConfig, compute_residual, embed, extract_traditional


def load_baseline_config() -> dict:
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / "baseline.yaml").read_text(encoding="utf-8"))
    return raw["baseline"]


def load_images(split: str, limit: int) -> list[tuple[str, np.ndarray]]:
    split_dir = PROJECT_ROOT / "data" / "processed" / "div2k_256" / split
    paths = sorted(split_dir.glob("*.png"))[:limit]
    if not paths:
        raise FileNotFoundError(f"No processed images in {split_dir}")
    out = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Unreadable image: {path}")
        out.append((path.stem, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return out


def payload_config(cfg: dict, alpha: float, n_bits: int) -> EmbedConfig:
    """Frozen baseline is LL-only; overflow to HL/LH/HH only when the payload
    exceeds one subband so the >128-bit points can still be characterised."""
    order = ["HL", "LH", "HH"]
    extra: list[str] = []
    per_band = 128  # 256x256 image -> 128x128 subband
    need = n_bits
    while need > per_band * (1 + len(extra)):
        extra.append(order[len(extra)])
    return EmbedConfig(
        wavelet=cfg["wavelet"],
        mode=cfg["mode"],
        subband=cfg["subband"],
        extra_subbands=tuple(extra),
        alpha=alpha,
        bit_length=n_bits,
        start_sv_index=cfg["start_sv_index"],
    )


def run() -> None:
    cfg = load_baseline_config()
    rng = np.random.default_rng(cfg["seed"])
    images = load_images(cfg["eval_split"], int(cfg["eval_num_images"]))
    alphas = [float(a) for a in cfg["alphas"]]
    payloads = [int(p) for p in cfg["payload_bits"]]

    results_dir = PROJECT_ROOT / cfg["results_dir"]
    samples_dir = results_dir / "samples"
    results_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    # one fixed payload per (payload_size, image) so runs are reproducible
    payload_bits = {n: rng.integers(0, 2, size=n).tolist() for n in payloads}

    for alpha in alphas:
        for n_bits in payloads:
            econf = payload_config(cfg, alpha, n_bits)
            bits = payload_bits[n_bits]
            for idx, (name, image) in enumerate(images):
                result = embed(image, bits, econf)
                wm = result.watermarked_image
                recovered = extract_traditional(wm, image, econf)
                rows.append({
                    "alpha": alpha,
                    "payload_bits": n_bits,
                    "subband_order": "+".join(econf.subband_order),
                    "image": name,
                    "psnr": round(psnr(image, wm), 4),
                    "ssim": round(ssim(image, wm), 6),
                    "mse": round(mse(image, wm), 6),
                    "ber": round(bit_error_rate(bits, recovered), 6),
                    "bit_accuracy": round(bit_accuracy(bits, recovered), 6),
                    "nc": round(normalized_correlation(bits, recovered), 6),
                })

                if idx == 0 and n_bits in (32, 128) and alpha == 0.010:
                    stem = f"a{alpha:.3f}_b{n_bits}_{name}"
                    cv2.imwrite(str(samples_dir / f"{stem}_original.png"),
                                cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(samples_dir / f"{stem}_watermarked.png"),
                                cv2.cvtColor(wm, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(samples_dir / f"{stem}_residual_x15.png"),
                                cv2.cvtColor(compute_residual(image, wm), cv2.COLOR_RGB2BGR))

    metrics_csv = results_dir / "baseline_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def agg(subset: list[dict], key: str) -> tuple[float, float]:
        vals = [r[key] for r in subset]
        return (round(statistics.fmean(vals), 4),
                round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0)

    summary: list[dict] = []
    for alpha in alphas:
        for n_bits in payloads:
            subset = [r for r in rows if r["alpha"] == alpha and r["payload_bits"] == n_bits]
            psnr_m, psnr_s = agg(subset, "psnr")
            ssim_m, ssim_s = agg(subset, "ssim")
            mse_m, _ = agg(subset, "mse")
            ber_m, ber_s = agg(subset, "ber")
            acc_m, _ = agg(subset, "bit_accuracy")
            nc_m, _ = agg(subset, "nc")
            summary.append({
                "alpha": alpha,
                "payload_bits": n_bits,
                "subband_order": subset[0]["subband_order"],
                "n_images": len(subset),
                "psnr_mean": psnr_m, "psnr_std": psnr_s,
                "ssim_mean": ssim_m, "ssim_std": ssim_s,
                "mse_mean": mse_m,
                "ber_mean": ber_m, "ber_std": ber_s,
                "bit_accuracy_mean": acc_m,
                "nc_mean": nc_m,
            })

    summary_csv = results_dir / "baseline_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    (results_dir / "baseline_summary.json").write_text(
        json.dumps({
            "phase": 6,
            "description": "Frozen DWT-SVD baseline characterisation",
            "config": cfg,
            "embedding_rule": "S'[i] = S[i] * (1 + alpha * (2*bit - 1))",
            "extraction": "non-blind: bit = 1 if S_watermarked[i] > S_original[i]",
            "n_eval_images": len(images),
            "eval_image_ids": [name for name, _ in images],
            "summary": summary,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {metrics_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {results_dir / 'baseline_summary.json'}")
    print()
    hdr = f"{'alpha':>7} {'bits':>5} {'bands':>8} {'PSNR':>7} {'SSIM':>7} {'MSE':>8} {'BER':>7} {'acc':>7} {'NC':>7}"
    print(hdr)
    for s in summary:
        print(f"{s['alpha']:>7.3f} {s['payload_bits']:>5} {s['subband_order']:>8} "
              f"{s['psnr_mean']:>7.2f} {s['ssim_mean']:>7.4f} {s['mse_mean']:>8.3f} "
              f"{s['ber_mean']:>7.4f} {s['bit_accuracy_mean']:>7.4f} {s['nc_mean']:>7.4f}")


if __name__ == "__main__":
    run()
