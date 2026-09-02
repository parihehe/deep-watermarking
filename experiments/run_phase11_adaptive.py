"""Phase 11 - Adaptive, Content-Aware Embedding (experiment runner).

Runs the reproducible Phase 11 experiment defined in
``configs/phase11_adaptive.yaml``:

  1. Guard: assert the frozen Phase 6 baseline contract; record frozen-file hashes.
  2. Clean-channel alpha sweep: for every configured alpha and both alpha modes
     (fixed / adaptive), embed a 64-bit payload (block-SVD, one bit per block)
     in every held-out DIV2K test image and record image-quality (PSNR/SSIM/MSE)
     and watermark-recovery (BER/bit-acc/NC), non-blind.
  3. Robustness stress check: at the primary operating alpha, attack the
     watermarked images (JPEG / Gaussian noise / Gaussian blur, reusing the
     already-implemented Phase 10 attack primitives) and recover, for both
     alpha modes.
  4. Write per-image + summary CSV/JSON and comparison plots (including a
     visual example of the texture map and the resulting alpha allocation).
  5. Sanity-check that Phase 7 (the web app) and Phase 8 (the CNN) still load
     and run, and that the frozen Phase 6 defaults are unchanged.

Nothing here modifies the frozen baseline, the Phase 7 UI or the Phase 8 model.

Usage
-----
    .venv/Scripts/python.exe experiments/run_phase11_adaptive.py
    .venv/Scripts/python.exe experiments/run_phase11_adaptive.py --quick
    .venv/Scripts/python.exe experiments/run_phase11_adaptive.py --eval-num-images 100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.phase9_validation import assert_baseline_frozen, frozen_file_hashes
from src.evaluation.phase11_adaptive import (
    Phase11Config,
    adaptive_config_for,
    environment_stamp,
    evaluate_alpha_point,
    evaluate_robustness_point,
    load_config,
    load_split_images,
    summarise_alpha_sweep,
    summarise_robustness,
)
from src.watermark.adaptive_embed import embed_adaptive

CONFIG_PATH = PROJECT_ROOT / "configs" / "phase11_adaptive.yaml"


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_alpha_summary(summary: list[dict]) -> list[dict]:
    flat: list[dict] = []
    for entry in summary:
        row = {
            "alpha_mode": entry["alpha_mode"],
            "alpha": entry["alpha"],
            "n_images": entry["n_images"],
            "alpha_realised_std_mean": entry["alpha_realised_std_mean"],
        }
        for k, v in entry["image_quality"].items():
            row[f"quality_{k}"] = v
        for k, v in entry["watermark_recovery"].items():
            row[f"recovery_{k}"] = v
        flat.append(row)
    return flat


def _delta_by_alpha(summary: list[dict]) -> list[dict]:
    """adaptive - fixed, per alpha, for every metric (positive delta = adaptive worse for BER/MSE, better for PSNR/SSIM/bit-acc/NC)."""
    by_alpha: dict[float, dict[str, dict]] = {}
    for entry in summary:
        by_alpha.setdefault(entry["alpha"], {})[entry["alpha_mode"]] = entry
    out = []
    for alpha, modes in sorted(by_alpha.items()):
        if "fixed" not in modes or "adaptive" not in modes:
            continue
        f, a = modes["fixed"], modes["adaptive"]
        row = {"alpha": alpha}
        for group in ("image_quality", "watermark_recovery"):
            for key in f[group]:
                if key.endswith("_mean"):
                    row[f"{group}.{key}.delta_adaptive_minus_fixed"] = round(a[group][key] - f[group][key], 6)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _make_plots(
    plots_dir: Path,
    alpha_summary: list[dict],
    robustness_summary: list[dict],
    sample_image,
    cfg: Phase11Config,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        print(f"[phase11] matplotlib unavailable ({exc}); skipping plots")
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # 1. quality metrics vs alpha, fixed vs adaptive
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric, label in zip(axes, ("psnr", "ssim", "mse"), ("PSNR (dB)", "SSIM", "MSE")):
        for mode, marker, color in (("fixed", "o", "#4C72B0"), ("adaptive", "s", "#DD8452")):
            entries = sorted((e for e in alpha_summary if e["alpha_mode"] == mode), key=lambda e: e["alpha"])
            xs = [e["alpha"] for e in entries]
            ys = [e["image_quality"][f"{metric}_mean"] for e in entries]
            ax.plot(xs, ys, marker=marker, color=color, label=mode)
        ax.set_xlabel("alpha (mean strength)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs alpha")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Phase 11: image quality, fixed vs adaptive (equal mean alpha)")
    fig.tight_layout()
    path = plots_dir / "quality_vs_alpha.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    # 2. recovery metrics vs alpha, fixed vs adaptive
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric, label in zip(axes, ("ber", "bit_accuracy", "nc"), ("BER", "Bit accuracy", "NC")):
        for mode, marker, color in (("fixed", "o", "#4C72B0"), ("adaptive", "s", "#DD8452")):
            entries = sorted((e for e in alpha_summary if e["alpha_mode"] == mode), key=lambda e: e["alpha"])
            xs = [e["alpha"] for e in entries]
            ys = [e["watermark_recovery"][f"{metric}_mean"] for e in entries]
            ax.plot(xs, ys, marker=marker, color=color, label=mode)
        ax.set_xlabel("alpha (mean strength)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs alpha")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Phase 11: watermark recovery (clean channel), fixed vs adaptive")
    fig.tight_layout()
    path = plots_dir / "recovery_vs_alpha.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    # 3. robustness stress: BER per attack, fixed vs adaptive
    if robustness_summary:
        attacks = sorted({e["attack"] for e in robustness_summary})
        fixed_ber = []
        adaptive_ber = []
        for a in attacks:
            fixed_ber.append(next(e["recovery_ber_mean"] for e in robustness_summary if e["attack"] == a and e["alpha_mode"] == "fixed"))
            adaptive_ber.append(next(e["recovery_ber_mean"] for e in robustness_summary if e["attack"] == a and e["alpha_mode"] == "adaptive"))
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(attacks))
        width = 0.38
        ax.bar([i - width / 2 for i in x], fixed_ber, width, label="fixed", color="#4C72B0")
        ax.bar([i + width / 2 for i in x], adaptive_ber, width, label="adaptive", color="#DD8452")
        ax.set_xticks(list(x))
        ax.set_xticklabels(attacks, rotation=20, ha="right")
        ax.set_ylabel(f"mean BER after attack (alpha={cfg.robustness_alpha})")
        ax.set_title("Phase 11: robustness stress check, fixed vs adaptive")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = plots_dir / "robustness_comparison.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))

    # 4. texture map / alpha-map example on one real image
    econf = adaptive_config_for(cfg, cfg.robustness_alpha, "adaptive")
    example_bits = np.random.default_rng(0).integers(0, 2, size=econf.bit_length).tolist()
    result = embed_adaptive(sample_image, example_bits, econf)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].imshow(sample_image)
    axes[0].set_title("sample image (Y-channel source)")
    axes[0].axis("off")
    im1 = axes[1].imshow(result.texture_scores, cmap="viridis")
    axes[1].set_title("per-block Sobel edge-energy score")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(result.alpha_map, cmap="magma")
    axes[2].set_title(f"resulting per-block alpha (mean={result.alpha_map.mean():.4f})")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.suptitle("Phase 11: content-aware alpha allocation, one example image")
    fig.tight_layout()
    path = plots_dir / "alpha_map_example.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    return written


# ---------------------------------------------------------------------------
# Phase 7 / Phase 8 sanity
# ---------------------------------------------------------------------------

def _sanity_checks() -> dict:
    result: dict = {}
    from src.app.main import app
    from src.watermark.embed import EmbedConfig

    result["phase7_app_route_present"] = any(
        route.path == "/api/watermark/embed" for route in app.routes
    )
    default = EmbedConfig()
    result["phase6_defaults_intact"] = (
        default.wavelet == "haar" and default.subband == "LL"
        and default.alpha == 0.010 and default.mode == "symmetric"
    )

    ckpt = PROJECT_ROOT / "models" / "phase8_cnn" / "phase8_cnn_best.pt"
    if ckpt.is_file():
        import numpy as np

        from src.evaluation.blind_extract import BlindExtractor

        extractor = BlindExtractor.from_checkpoint(str(ckpt))
        dummy = (np.random.default_rng(0).random((256, 256, 3)) * 255).astype("uint8")
        bits = extractor.extract_bits(dummy)
        result["phase8_cnn_loads_and_runs"] = len(bits) == extractor.config.bit_length
    else:
        result["phase8_cnn_loads_and_runs"] = None
    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(cfg: Phase11Config) -> dict:
    started = time.time()

    assert_baseline_frozen()
    hashes = frozen_file_hashes()
    print("[phase11] frozen Phase 6 baseline contract: OK")

    results_dir = cfg.resolved_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    images = load_split_images(cfg.resolved_split_dir(), cfg.eval_num_images)
    print(f"[phase11] loaded {len(images)} test images; block-SVD {cfg.block_size}x{cfg.block_size} "
          f"blocks -> {cfg.bit_length}-bit payload")

    # -- 1. clean-channel alpha sweep -----------------------------------
    alpha_rows: list[dict] = []
    for alpha in cfg.alphas:
        for mode in ("fixed", "adaptive"):
            econf = adaptive_config_for(cfg, alpha, mode)
            point_rows = evaluate_alpha_point(images, econf, payload_seed=cfg.payload_seed)
            alpha_rows.extend(point_rows)
            mean_ber = sum(r["recovery_ber"] for r in point_rows) / len(point_rows)
            mean_psnr = sum(r["quality_psnr"] for r in point_rows) / len(point_rows)
            print(f"[phase11] alpha={alpha:<6} mode={mode:<8} PSNR={mean_psnr:.2f}  BER={mean_ber:.4f}")

    alpha_summary = summarise_alpha_sweep(alpha_rows)

    # -- 2. robustness stress check --------------------------------------
    robust_rows: list[dict] = []
    for mode in ("fixed", "adaptive"):
        econf = adaptive_config_for(cfg, cfg.robustness_alpha, mode)
        for attack_name, attack_params in cfg.robustness_attacks.items():
            point_rows = evaluate_robustness_point(
                images, econf, attack_name, attack_params,
                payload_seed=cfg.payload_seed, noise_seed=cfg.noise_seed,
            )
            robust_rows.extend(point_rows)
            mean_ber = sum(r["recovery_ber"] for r in point_rows) / len(point_rows)
            print(f"[phase11] robustness attack={attack_name:<16} mode={mode:<8} BER={mean_ber:.4f}")

    robustness_summary = summarise_robustness(robust_rows)

    # -- write outputs -----------------------------------------------------
    per_image_csv = results_dir / "phase11_per_image.csv"
    summary_csv = results_dir / "phase11_summary.csv"
    robustness_per_image_csv = results_dir / "phase11_robustness_per_image.csv"
    robustness_summary_csv = results_dir / "phase11_robustness_summary.csv"
    _write_csv(per_image_csv, alpha_rows)
    _write_csv(summary_csv, _flatten_alpha_summary(alpha_summary))
    _write_csv(robustness_per_image_csv, robust_rows)
    _write_csv(robustness_summary_csv, robustness_summary)

    delta = _delta_by_alpha(alpha_summary)

    report = {
        "phase": 11,
        "title": "Adaptive, Content-Aware Embedding",
        "scope_excludes": [
            "change to the frozen Phase 6 formula / defaults",
            "CNN architecture change or retraining (Phase 15)",
            "high-capacity changes (Phase 12)",
            "wavelet/subband study (Phases 13-14)",
            "error-correcting codes (Phase 16)",
            "blockchain",
        ],
        "method": {
            "embedding_topology": "block-SVD: LL subband tiled into non-overlapping "
                                   f"{cfg.block_size}x{cfg.block_size} blocks, one bit per "
                                   "block's leading singular value, frozen multiplicative rule "
                                   "S'[0] = S[0] * (1 + alpha_block * (2*bit - 1))",
            "texture_score": "Sobel gradient magnitude on the full-resolution Y channel, "
                              "2x2-average-pooled to LL resolution, block-averaged",
            "alpha_map_formula": "alpha_block = alpha * weight / mean(weight); "
                                  "weight = alpha_low_mult + normalised_score * (alpha_high_mult - alpha_low_mult); "
                                  "normalised_score = percentile-clipped min-max of the texture score "
                                  "=> mean(alpha_block) == alpha exactly (equal energy budget vs fixed mode)",
            "alpha_low_mult": cfg.alpha_low_mult,
            "alpha_high_mult": cfg.alpha_high_mult,
            "texture_percentile_clip": cfg.texture_percentile_clip,
            "block_size": cfg.block_size,
            "bit_length": cfg.bit_length,
            "wavelet": cfg.wavelet,
            "mode": cfg.mode,
            "non_blind_decoder": "extract_adaptive: bit = 1 if S_w[0] > S_o[0] else 0, per block "
                                  "(mirrors extract_traditional's decision rule)",
        },
        "config": {
            "alphas": cfg.alphas,
            "eval_split": cfg.eval_split,
            "eval_num_images": len(images),
            "image_size": cfg.image_size,
            "payload_seed": cfg.payload_seed,
            "noise_seed": cfg.noise_seed,
            "robustness_alpha": cfg.robustness_alpha,
            "robustness_attacks": cfg.robustness_attacks,
        },
        "environment": environment_stamp(),
        "frozen_baseline_sha256": hashes,
        "alpha_sweep_summary": alpha_summary,
        "alpha_sweep_delta_adaptive_minus_fixed": delta,
        "robustness_summary": robustness_summary,
        "sanity": _sanity_checks(),
    }

    if cfg.make_plots:
        report["plots"] = _make_plots(
            results_dir / "plots", alpha_summary, robustness_summary, images[0][1], cfg
        )
        for p in report.get("plots", []):
            print(f"[phase11] wrote {p}")

    report["elapsed_seconds"] = round(time.time() - started, 1)
    (results_dir / "phase11_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[phase11] wrote {per_image_csv}")
    print(f"[phase11] wrote {summary_csv}")
    print(f"[phase11] wrote {robustness_per_image_csv}")
    print(f"[phase11] wrote {robustness_summary_csv}")
    print(f"[phase11] wrote {results_dir / 'phase11_report.json'}")

    _print_tables(alpha_summary, delta, robustness_summary)
    print(f"\n[phase11] done in {report['elapsed_seconds']}s -> {results_dir}")
    print(f"[phase11] sanity: {report['sanity']}")
    return report


def _print_tables(alpha_summary: list[dict], delta: list[dict], robustness_summary: list[dict]) -> None:
    print("\n=== Phase 11 - clean-channel alpha sweep (mean over images) ===")
    hdr = f"{'mode':<9} {'alpha':>7} | {'PSNR':>7} {'SSIM':>7} {'MSE':>7} | {'BER':>7} {'bit_acc':>8} {'NC':>7}"
    print(hdr)
    print("-" * len(hdr))
    for e in sorted(alpha_summary, key=lambda e: (e["alpha"], e["alpha_mode"])):
        q, r = e["image_quality"], e["watermark_recovery"]
        print(f"{e['alpha_mode']:<9} {e['alpha']:>7} | {q['psnr_mean']:>7.2f} {q['ssim_mean']:>7.4f} "
              f"{q['mse_mean']:>7.3f} | {r['ber_mean']:>7.4f} {r['bit_accuracy_mean']:>8.4f} {r['nc_mean']:>7.4f}")

    print("\n=== Phase 11 - adaptive minus fixed, per alpha (positive = adaptive higher) ===")
    for d in delta:
        print(f"  alpha={d['alpha']}: "
              f"dPSNR={d['image_quality.psnr_mean.delta_adaptive_minus_fixed']:+.3f}  "
              f"dSSIM={d['image_quality.ssim_mean.delta_adaptive_minus_fixed']:+.5f}  "
              f"dBER={d['watermark_recovery.ber_mean.delta_adaptive_minus_fixed']:+.4f}  "
              f"dNC={d['watermark_recovery.nc_mean.delta_adaptive_minus_fixed']:+.4f}")

    if robustness_summary:
        print("\n=== Phase 11 - robustness stress check (mean BER after attack) ===")
        attacks = sorted({e["attack"] for e in robustness_summary})
        for a in attacks:
            f = next(e for e in robustness_summary if e["attack"] == a and e["alpha_mode"] == "fixed")
            adpt = next(e for e in robustness_summary if e["attack"] == a and e["alpha_mode"] == "adaptive")
            print(f"  {a:<16} fixed BER={f['recovery_ber_mean']:.4f}   adaptive BER={adpt['recovery_ber_mean']:.4f}"
                  f"   (delta {adpt['recovery_ber_mean'] - f['recovery_ber_mean']:+.4f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _quick_overrides() -> dict:
    return {"eval_num_images": 10, "alphas": (0.010, 0.020)}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 11 - adaptive, content-aware embedding")
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--quick", action="store_true", help="fast subset: 10 images, 2 alphas")
    p.add_argument("--eval-num-images", type=int, dest="eval_num_images", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    overrides: dict = {}
    if args.eval_num_images is not None:
        overrides["eval_num_images"] = args.eval_num_images
    if args.quick:
        overrides = {**overrides, **_quick_overrides()}
        if args.eval_num_images is not None:
            overrides["eval_num_images"] = args.eval_num_images
    cfg = load_config(args.config, overrides or None)
    run(cfg)


if __name__ == "__main__":
    main()
