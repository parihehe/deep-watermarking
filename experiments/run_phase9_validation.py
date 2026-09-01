"""Phase 9 - Baseline Validation & Robustness Preparation (experiment runner).

Runs the reproducible Phase 9 validation experiment described in
``configs/phase9_validation.yaml`` and ``docs/phase9_validation.md``:

  1. Guard: assert the frozen Phase 6 baseline contract is intact; record the
     SHA-256 of every frozen file.
  2. Frozen DWT-SVD baseline (non-blind decoder) over the held-out DIV2K test
     split, for payloads 8/16/32/64/128 bits at alpha in {0.005, 0.010, 0.015}.
     Metrics are split into an image-quality group (PSNR/SSIM/MSE) and a
     watermark-recovery group (BER/bit-accuracy/NC).
  3. Phase 8 blind CNN extractor (architecture unchanged), evaluated separately
     on the same unseen images at its trained operating point, with a
     like-for-like non-blind vs blind comparison.
  4. Structured CSV/JSON output + plots.

Nothing here modifies the frozen baseline or the Phase 8 model. It only reads
and runs them.

Usage
-----
    .venv/Scripts/python.exe experiments/run_phase9_validation.py
    .venv/Scripts/python.exe experiments/run_phase9_validation.py --quick
    .venv/Scripts/python.exe experiments/run_phase9_validation.py --no-cnn
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.phase9_validation import (
    METRIC_GROUPS,
    Phase9Config,
    assert_baseline_frozen,
    environment_stamp,
    evaluate_baseline_point,
    evaluate_cnn_point,
    frozen_file_hashes,
    load_config,
    load_split_images,
    summarise_baseline,
    summarise_cnn,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "phase9_validation.yaml"


# --------------------------------------------------------------------------
# CSV helpers
# --------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_baseline_summary(summary: list[dict]) -> list[dict]:
    flat: list[dict] = []
    for entry in summary:
        row = {
            "alpha": entry["alpha"],
            "payload_bits": entry["payload_bits"],
            "subband_order": entry["subband_order"],
            "decoder": entry["decoder"],
            "n_images": entry["n_images"],
        }
        for key, value in entry["image_quality"].items():
            row[f"quality_{key}"] = value
        for key, value in entry["watermark_recovery"].items():
            row[f"recovery_{key}"] = value
        flat.append(row)
    return flat


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def _make_plots(
    plots_dir: Path,
    baseline_summary: list[dict],
    cnn_summary: dict | None,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - matplotlib is in this env
        print(f"[phase9] matplotlib unavailable ({exc}); skipping plots")
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    alphas = sorted({e["alpha"] for e in baseline_summary})
    payloads = sorted({e["payload_bits"] for e in baseline_summary})

    def series(alpha: float, group: str, metric: str) -> list[float]:
        out = []
        for p in payloads:
            match = next(e for e in baseline_summary if e["alpha"] == alpha and e["payload_bits"] == p)
            out.append(match[group][f"{metric}_mean"])
        return out

    # 1. PSNR vs payload (image quality)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for alpha in alphas:
        ax.plot(payloads, series(alpha, "image_quality", "psnr"), marker="o", label=f"alpha={alpha}")
    ax.set_xlabel("payload (bits)")
    ax.set_ylabel("PSNR (dB)  - image quality")
    ax.set_title("Phase 9: frozen DWT-SVD baseline - imperceptibility")
    ax.set_xscale("log", base=2)
    ax.set_xticks(payloads)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "psnr_vs_payload.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    # 2. BER vs payload (watermark recovery, non-blind)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for alpha in alphas:
        ax.plot(payloads, series(alpha, "watermark_recovery", "ber"), marker="o", label=f"alpha={alpha}")
    ax.set_xlabel("payload (bits)")
    ax.set_ylabel("BER  - watermark recovery (non-blind)")
    ax.set_title("Phase 9: frozen DWT-SVD baseline - clean-channel recovery")
    ax.set_xscale("log", base=2)
    ax.set_xticks(payloads)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "ber_vs_payload_nonblind.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    # 3. quality vs recovery tradeoff (scatter)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for alpha in alphas:
        xs = series(alpha, "image_quality", "psnr")
        ys = series(alpha, "watermark_recovery", "bit_accuracy")
        ax.plot(xs, ys, marker="o", label=f"alpha={alpha}")
        for p, x, y in zip(payloads, xs, ys):
            ax.annotate(f"{p}b", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("PSNR (dB)  - image quality")
    ax.set_ylabel("bit accuracy  - watermark recovery")
    ax.set_title("Phase 9: image quality vs watermark recovery")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "quality_vs_recovery.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    # 4. non-blind vs blind BER at the CNN operating point
    if cnn_summary is not None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        labels = ["non-blind\n(reference)", "blind CNN\n(Phase 8)"]
        bers = [
            cnn_summary["watermark_recovery_non_blind_reference"]["ber_mean"],
            cnn_summary["watermark_recovery_blind_cnn"]["ber_mean"],
        ]
        bars = ax.bar(labels, bers, color=["#4C72B0", "#DD8452"])
        op = cnn_summary["operating_point"]
        ax.set_ylabel("mean BER")
        ax.set_title(
            f"Phase 9: decoder comparison @ {op['payload_bits']} bits, alpha={op['alpha']}"
        )
        for rect, value in zip(bars, bers):
            ax.annotate(f"{value:.3f}", (rect.get_x() + rect.get_width() / 2, value),
                        ha="center", va="bottom", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path = plots_dir / "nonblind_vs_blind_ber.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))

    return written


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run(cfg: Phase9Config, *, run_cnn: bool = True) -> dict:
    started = time.time()

    # 1. frozen-baseline guard -------------------------------------------------
    assert_baseline_frozen()
    hashes = frozen_file_hashes()
    print("[phase9] frozen Phase 6 baseline contract: OK")
    for rel, digest in hashes.items():
        print(f"[phase9]   {rel}  sha256={digest[:16]}...")

    results_dir = cfg.resolved_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    split_dir = cfg.resolved_split_dir()
    images = load_split_images(split_dir, cfg.eval_num_images)
    print(f"[phase9] evaluation split: {split_dir.name}  images: {len(images)}")

    # 2. frozen DWT-SVD baseline (non-blind) --------------------------------
    baseline_rows: list[dict] = []
    for alpha in cfg.alphas:
        for n_bits in cfg.payload_bits:
            econf = cfg.baseline_embed_config(alpha, n_bits)
            point_rows = evaluate_baseline_point(images, econf, payload_seed=cfg.payload_seed)
            baseline_rows.extend(point_rows)
            print(
                f"[phase9] baseline alpha={alpha:<6} payload={n_bits:>3}b  "
                f"done ({len(point_rows)} images)"
            )
    baseline_summary = summarise_baseline(baseline_rows)

    per_image_csv = results_dir / "phase9_baseline_per_image.csv"
    summary_csv = results_dir / "phase9_baseline_summary.csv"
    _write_csv(per_image_csv, baseline_rows)
    _write_csv(summary_csv, _flatten_baseline_summary(baseline_summary))
    (results_dir / "phase9_baseline_summary.json").write_text(
        json.dumps({
            "phase": 9,
            "component": "frozen_dwt_svd_baseline",
            "decoder": "non_blind",
            "metric_groups": METRIC_GROUPS,
            "n_images": len(images),
            "eval_split": cfg.eval_split,
            "image_ids": [name for name, _ in images],
            "summary": baseline_summary,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    # 3. Phase 8 blind CNN extractor (separate) ---------------------------
    cnn_summary: dict | None = None
    cnn_rows: list[dict] = []
    if run_cnn and cfg.cnn.enabled:
        checkpoint = (PROJECT_ROOT / cfg.cnn.checkpoint).resolve()
        if not checkpoint.is_file():
            print(f"[phase9] CNN checkpoint not found ({checkpoint}); skipping CNN evaluation")
        else:
            from src.evaluation.blind_extract import BlindExtractor
            from src.models.cnn_extractor import load_checkpoint

            _, meta = load_checkpoint(str(checkpoint))
            ckpt_bits = int(meta.get("bit_length", meta["extractor_config"]["bit_length"]))
            ckpt_alpha = float(meta.get("train_config", {}).get("alpha", cfg.cnn.alpha))
            if ckpt_bits != cfg.cnn.bit_length:
                raise SystemExit(
                    f"checkpoint bit_length {ckpt_bits} != config {cfg.cnn.bit_length}"
                )
            if abs(ckpt_alpha - cfg.cnn.alpha) > 1e-9:
                print(
                    f"[phase9] note: checkpoint trained at alpha={ckpt_alpha}, "
                    f"config says {cfg.cnn.alpha}; using config value"
                )
            extractor = BlindExtractor.from_checkpoint(str(checkpoint))
            cnn_rows = evaluate_cnn_point(images, extractor, cnn_config=cfg.cnn)
            cnn_summary = summarise_cnn(cnn_rows, cfg.cnn)
            print(
                f"[phase9] blind CNN @ {cfg.cnn.bit_length}b alpha={cfg.cnn.alpha}: "
                f"blind BER={cnn_summary['watermark_recovery_blind_cnn']['ber_mean']:.4f}  "
                f"non-blind BER={cnn_summary['watermark_recovery_non_blind_reference']['ber_mean']:.4f}"
            )

            _write_csv(results_dir / "phase9_cnn_per_image.csv", cnn_rows)
            _write_csv(results_dir / "phase9_nonblind_vs_blind.csv", [
                {
                    "decoder": "non_blind_reference",
                    "payload_bits": cfg.cnn.bit_length,
                    "alpha": cfg.cnn.alpha,
                    **{f"recovery_{k}": v
                       for k, v in cnn_summary["watermark_recovery_non_blind_reference"].items()},
                },
                {
                    "decoder": "blind_cnn_phase8",
                    "payload_bits": cfg.cnn.bit_length,
                    "alpha": cfg.cnn.alpha,
                    **{f"recovery_{k}": v
                       for k, v in cnn_summary["watermark_recovery_blind_cnn"].items()},
                },
            ])
            (results_dir / "phase9_cnn_summary.json").write_text(
                json.dumps({
                    "phase": 9,
                    "component": "phase8_blind_cnn_extractor",
                    "architecture_changed": False,
                    "metric_groups": {"watermark_recovery": list(METRIC_GROUPS["watermark_recovery"])},
                    "eval_split": cfg.eval_split,
                    "summary": cnn_summary,
                }, indent=2) + "\n",
                encoding="utf-8",
            )

    # 4. plots -----------------------------------------------------------
    plot_paths: list[str] = []
    if cfg.make_plots:
        plot_paths = _make_plots(results_dir / "plots", baseline_summary, cnn_summary)
        for p in plot_paths:
            print(f"[phase9] wrote {p}")

    # 5. combined report ----------------------------------------------
    elapsed = round(time.time() - started, 1)
    report = {
        "phase": 9,
        "title": "Baseline Validation & Robustness Preparation",
        "scope_excludes": [
            "JPEG/noise/blur/crop attacks (Phase 10)",
            "adaptive/content-aware embedding (Phase 11)",
            "error-correcting codes (Phase 16)",
            "CNN architecture changes (Phase 15)",
        ],
        "config": {
            "wavelet": cfg.wavelet,
            "mode": cfg.mode,
            "subband": cfg.subband,
            "start_sv_index": cfg.start_sv_index,
            "alphas": list(cfg.alphas),
            "payload_bits": list(cfg.payload_bits),
            "eval_split": cfg.eval_split,
            "eval_num_images": len(images),
            "image_size": cfg.image_size,
            "payload_seed": cfg.payload_seed,
            "embedding_rule": "S'[i] = S[i] * (1 + alpha * (2*b - 1))  [frozen Phase 6]",
            "baseline_decoder": "non_blind: bit = 1 if S_watermarked[i] > S_original[i]",
            "cnn": {
                "enabled": bool(run_cnn and cfg.cnn.enabled and cnn_summary is not None),
                "checkpoint": cfg.cnn.checkpoint,
                "bit_length": cfg.cnn.bit_length,
                "alpha": cfg.cnn.alpha,
            },
        },
        "environment": environment_stamp(),
        "frozen_baseline_sha256": hashes,
        "metric_groups": METRIC_GROUPS,
        "baseline_summary": baseline_summary,
        "cnn_summary": cnn_summary,
        "artifacts": {
            "baseline_per_image_csv": str(per_image_csv),
            "baseline_summary_csv": str(summary_csv),
            "baseline_summary_json": str(results_dir / "phase9_baseline_summary.json"),
            "cnn_per_image_csv": str(results_dir / "phase9_cnn_per_image.csv") if cnn_rows else None,
            "cnn_summary_json": str(results_dir / "phase9_cnn_summary.json") if cnn_rows else None,
            "nonblind_vs_blind_csv": str(results_dir / "phase9_nonblind_vs_blind.csv") if cnn_rows else None,
            "plots": plot_paths,
        },
        "elapsed_seconds": elapsed,
    }
    (results_dir / "phase9_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[phase9] wrote {results_dir / 'phase9_report.json'}")

    _print_tables(baseline_summary, cnn_summary)
    print(f"\n[phase9] done in {elapsed}s -> {results_dir}")
    return report


def _print_tables(baseline_summary: list[dict], cnn_summary: dict | None) -> None:
    print("\n=== Phase 9 - frozen DWT-SVD baseline (non-blind), clean channel ===")
    print("  IMAGE QUALITY                     |  WATERMARK RECOVERY")
    hdr = (
        f"{'alpha':>6} {'bits':>5} | {'PSNR':>7} {'SSIM':>7} {'MSE':>8} | "
        f"{'BER':>7} {'bit_acc':>8} {'NC':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for e in baseline_summary:
        q = e["image_quality"]
        r = e["watermark_recovery"]
        print(
            f"{e['alpha']:>6.3f} {e['payload_bits']:>5} | "
            f"{q['psnr_mean']:>7.2f} {q['ssim_mean']:>7.4f} {q['mse_mean']:>8.3f} | "
            f"{r['ber_mean']:>7.4f} {r['bit_accuracy_mean']:>8.4f} {r['nc_mean']:>7.4f}"
        )

    if cnn_summary is not None:
        op = cnn_summary["operating_point"]
        blind = cnn_summary["watermark_recovery_blind_cnn"]
        ref = cnn_summary["watermark_recovery_non_blind_reference"]
        print(
            f"\n=== Phase 9 - Phase 8 blind CNN extractor "
            f"(architecture unchanged) @ {op['payload_bits']} bits, alpha={op['alpha']} ==="
        )
        print(f"{'decoder':>22} | {'BER':>7} {'bit_acc':>8} {'NC':>7}")
        print("-" * 50)
        print(f"{'non-blind (reference)':>22} | {ref['ber_mean']:>7.4f} "
              f"{ref['bit_accuracy_mean']:>8.4f} {ref['nc_mean']:>7.4f}")
        print(f"{'blind CNN (Phase 8)':>22} | {blind['ber_mean']:>7.4f} "
              f"{blind['bit_accuracy_mean']:>8.4f} {blind['nc_mean']:>7.4f}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 9 - baseline validation & robustness preparation")
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--quick", action="store_true",
                   help="fast subset: 12 images, payloads [8, 32, 128], alpha [0.010]")
    p.add_argument("--no-cnn", dest="no_cnn", action="store_true",
                   help="skip the Phase 8 blind CNN evaluation")
    p.add_argument("--eval-num-images", type=int, dest="eval_num_images", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    overrides: dict = {}
    if args.eval_num_images is not None:
        overrides["eval_num_images"] = args.eval_num_images
    if args.quick:
        overrides.update({
            "eval_num_images": min(args.eval_num_images or 12, 12),
            "payload_bits": (8, 32, 128),
            "alphas": (0.010,),
            "make_plots": True,
        })
    cfg = load_config(args.config, overrides)
    run(cfg, run_cnn=not args.no_cnn)


if __name__ == "__main__":
    main()
