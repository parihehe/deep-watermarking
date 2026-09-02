"""Phase 12 - High-Capacity Watermarking (experiment runner).

Runs the reproducible Phase 12 capacity study defined in
``configs/phase12_capacity.yaml``:

  1. Guard: assert the frozen Phase 6 baseline contract; record frozen-file hashes.
  2. For each configured experiment (payload size x DWT levels x subband
     allocation x alpha): check capacity; if sufficient, embed a deterministic
     payload (reusing the frozen Phase 3 generator) in every held-out DIV2K
     test image, recover non-blind, and record image-quality (PSNR/SSIM/MSE)
     and watermark-recovery (BER/bit-acc/NC); if insufficient, record the
     failure explicitly (never force or truncate the payload).
  3. Write per-image + summary CSV/JSON, the closed-form capacity-vs-level
     bound table, and comparison plots.
  4. Sanity-check that Phase 7 (the web app) and Phase 8 (the CNN) still load
     and run, and that the frozen Phase 6 defaults are unchanged.

Nothing here modifies the frozen baseline, the Phase 7 UI or the Phase 8 model.

Usage
-----
    .venv/Scripts/python.exe experiments/run_phase12_capacity.py
    .venv/Scripts/python.exe experiments/run_phase12_capacity.py --quick
    .venv/Scripts/python.exe experiments/run_phase12_capacity.py --eval-num-images 100
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

from src.evaluation.phase9_validation import assert_baseline_frozen, frozen_file_hashes
from src.evaluation.phase12_capacity import (
    Phase12Config,
    capacity_bound_table,
    environment_stamp,
    evaluate_experiment,
    load_config,
    load_split_images,
    summarise_experiment,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "phase12_capacity.yaml"


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


def _flatten_summary(summary: list[dict]) -> list[dict]:
    flat: list[dict] = []
    for entry in summary:
        row = {
            "name": entry["name"],
            "dwt_levels": entry["dwt_levels"],
            "subband_plan": entry["subband_plan"],
            "bit_length": entry["bit_length"],
            "alpha": entry["alpha"],
            "capacity_available": entry["capacity_available"],
            "theoretical_full_plan_capacity": entry["theoretical_full_plan_capacity"],
            "status": entry["status"],
            "n_images": entry["n_images"],
        }
        if entry["status"] == "success":
            for k, v in entry["image_quality"].items():
                row[f"quality_{k}"] = v
            for k, v in entry["watermark_recovery"].items():
                row[f"recovery_{k}"] = v
        else:
            row["reason"] = entry["reason"]
        flat.append(row)
    return flat


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _make_plots(plots_dir: Path, summary: list[dict], bound_table: list[dict]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        print(f"[phase12] matplotlib unavailable ({exc}); skipping plots")
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    successes = [e for e in summary if e["status"] == "success"]
    successes = sorted(successes, key=lambda e: (e["bit_length"], e["name"]))
    failures = [e for e in summary if e["status"] != "success"]

    labels = [f"{e['bit_length']}b\n({e['name']})" for e in successes]

    # 1-4: payload size vs BER / PSNR / SSIM / bit-accuracy (one 2x2 figure)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics_plan = [
        (axes[0, 0], "watermark_recovery", "ber_mean", "BER", "#C44E52"),
        (axes[0, 1], "image_quality", "psnr_mean", "PSNR (dB)", "#4C72B0"),
        (axes[1, 0], "image_quality", "ssim_mean", "SSIM", "#55A868"),
        (axes[1, 1], "watermark_recovery", "bit_accuracy_mean", "Bit accuracy", "#8172B2"),
    ]
    xs = [e["bit_length"] for e in successes]
    for ax, group, key, label, color in metrics_plan:
        ys = [e[group][key] for e in successes]
        ax.bar(range(len(successes)), ys, color=color)
        ax.set_xticks(range(len(successes)))
        ax.set_xticklabels(labels, fontsize=8, rotation=0)
        ax.set_ylabel(label)
        ax.set_title(f"Payload size vs {label}")
        ax.grid(True, axis="y", alpha=0.3)
        if key == "ber_mean":
            ax.axhline(0.5, color="grey", ls=":", lw=1)
    fig.suptitle("Phase 12: capacity vs quality/recovery (successful configurations)")
    fig.tight_layout()
    path = plots_dir / "capacity_vs_quality_recovery.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    # 5. theoretical capacity vs DWT level, with the 1024-bit target line and
    #    the L->infinity ceiling
    fig, ax = plt.subplots(figsize=(8, 5))
    levels = [r["dwt_levels"] for r in bound_table[:-1]]
    caps = [r["max_capacity_bits"] for r in bound_table[:-1]]
    limit = bound_table[-1]["max_capacity_bits"]
    ax.plot(levels, caps, marker="o", color="#4C72B0", label="max capacity at this many DWT levels")
    ax.axhline(limit, color="#C44E52", ls="--", label=f"L->infinity ceiling ({limit} bits)")
    ax.axhline(1024, color="black", ls=":", label="1024-bit target")
    ax.set_xlabel("DWT levels")
    ax.set_ylabel("max capacity (bits)")
    ax.set_title("Phase 12: theoretical capacity ceiling vs DWT levels (256x256 image)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "capacity_ceiling_vs_levels.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    # 6. success/failure map
    fig, ax = plt.subplots(figsize=(9, 4.5))
    all_entries = sorted(summary, key=lambda e: (e["bit_length"], e["dwt_levels"]))
    names = [e["name"] for e in all_entries]
    colors = ["#55A868" if e["status"] == "success" else "#C44E52" for e in all_entries]
    caps = [e["capacity_available"] for e in all_entries]
    reqs = [e["bit_length"] for e in all_entries]
    x = range(len(all_entries))
    ax.bar(x, caps, color=colors, alpha=0.7, label="capacity available")
    ax.plot(x, reqs, "kD", label="bits requested")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("bits")
    ax.set_title("Phase 12: capacity available vs requested (green=success, red=insufficient capacity)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "success_vs_failure.png"
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

def run(cfg: Phase12Config) -> dict:
    started = time.time()

    assert_baseline_frozen()
    hashes = frozen_file_hashes()
    print("[phase12] frozen Phase 6 baseline contract: OK")

    results_dir = cfg.resolved_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    images = load_split_images(cfg.resolved_split_dir(), cfg.eval_num_images)
    print(f"[phase12] loaded {len(images)} test images; {len(cfg.experiments)} experiments configured")

    all_rows: list[dict] = []
    summary: list[dict] = []
    for spec in cfg.experiments:
        outcome = evaluate_experiment(images, spec, payload_seed=cfg.payload_seed, image_size=cfg.image_size)
        entry = summarise_experiment(outcome)
        summary.append(entry)
        all_rows.extend(outcome["rows"])
        if entry["status"] == "success":
            r = entry["watermark_recovery"]
            q = entry["image_quality"]
            print(f"[phase12] {spec.name:<22} bits={spec.bit_length:<5} L{spec.dwt_levels} "
                  f"capacity={entry['capacity_available']:<4} PSNR={q['psnr_mean']:.2f} BER={r['ber_mean']:.4f}")
        else:
            print(f"[phase12] {spec.name:<22} bits={spec.bit_length:<5} L{spec.dwt_levels} "
                  f"capacity={entry['capacity_available']:<4} FAILED: {entry['reason']}")

    bound_table = capacity_bound_table(cfg.image_size)

    per_image_csv = results_dir / "phase12_per_image.csv"
    summary_csv = results_dir / "phase12_summary.csv"
    bound_csv = results_dir / "phase12_capacity_bound_table.csv"
    _write_csv(per_image_csv, all_rows)
    _write_csv(summary_csv, _flatten_summary(summary))
    _write_csv(bound_csv, bound_table)

    report = {
        "phase": 12,
        "title": "High-Capacity Watermarking",
        "scope_excludes": [
            "change to the frozen Phase 6 formula / defaults",
            "Phase 8 CNN architecture change (non-blind extraction only here)",
            "error-correcting codes",
            "blockchain",
            "MLflow/PostgreSQL",
            "production backend",
        ],
        "method": {
            "embedding_topology": "multi-level DWT (frozen dwt.py called recursively on the LL "
                                   "subband) + whole-subband SVD per configured (level, band) "
                                   "window, frozen multiplicative rule "
                                   "S'[i] = S[i] * (1 + alpha * (2*b - 1))",
            "capacity_formula": "capacity(L) = 3*sum_{l=1}^{L-1}(N/2^l) + 4*(N/2^L); "
                                 "L->infinity limit = 3*N",
            "capacity_ceiling_256x256": theoretical_capacity_bound_note(cfg.image_size),
            "non_blind_decoder": "extract_capacity: bit = 1 if S_w[i] > S_o[i] else 0, per "
                                  "(level, band) window (mirrors extract_traditional)",
        },
        "config": {
            "eval_split": cfg.eval_split,
            "eval_num_images": len(images),
            "image_size": cfg.image_size,
            "payload_seed": cfg.payload_seed,
            "experiments": [
                {"name": e.name, "dwt_levels": e.dwt_levels, "subband_plan": list(e.subband_plan),
                 "bit_length": e.bit_length, "alpha": e.alpha}
                for e in cfg.experiments
            ],
        },
        "environment": environment_stamp(),
        "frozen_baseline_sha256": hashes,
        "capacity_bound_table": bound_table,
        "summary": summary,
        "sanity": _sanity_checks(),
    }

    if cfg.make_plots:
        report["plots"] = _make_plots(results_dir / "plots", summary, bound_table)
        for p in report.get("plots", []):
            print(f"[phase12] wrote {p}")

    report["elapsed_seconds"] = round(time.time() - started, 1)
    (results_dir / "phase12_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[phase12] wrote {per_image_csv}")
    print(f"[phase12] wrote {summary_csv}")
    print(f"[phase12] wrote {bound_csv}")
    print(f"[phase12] wrote {results_dir / 'phase12_report.json'}")

    _print_tables(summary, bound_table)
    print(f"\n[phase12] done in {report['elapsed_seconds']}s -> {results_dir}")
    print(f"[phase12] sanity: {report['sanity']}")
    return report


def theoretical_capacity_bound_note(image_size: int) -> str:
    from src.watermark.capacity_embed import theoretical_capacity_limit

    return (
        f"{theoretical_capacity_limit(image_size)} bits as DWT levels -> infinity "
        f"(3 * {image_size}); 1024 bits is unreachable at any level"
    )


def _print_tables(summary: list[dict], bound_table: list[dict]) -> None:
    print("\n=== Phase 12 - experiment summary ===")
    hdr = f"{'name':<22} {'bits':>5} {'L':>2} {'cap':>5} {'status':<20} | {'PSNR':>6} {'SSIM':>7} | {'BER':>7} {'bit_acc':>8} {'NC':>7}"
    print(hdr)
    print("-" * len(hdr))
    for e in summary:
        if e["status"] == "success":
            q, r = e["image_quality"], e["watermark_recovery"]
            print(f"{e['name']:<22} {e['bit_length']:>5} {e['dwt_levels']:>2} {e['capacity_available']:>5} "
                  f"{e['status']:<20} | {q['psnr_mean']:>6.2f} {q['ssim_mean']:>7.4f} | "
                  f"{r['ber_mean']:>7.4f} {r['bit_accuracy_mean']:>8.4f} {r['nc_mean']:>7.4f}")
        else:
            print(f"{e['name']:<22} {e['bit_length']:>5} {e['dwt_levels']:>2} {e['capacity_available']:>5} "
                  f"{e['status']:<20} | {'--':>6} {'--':>7} | {'--':>7} {'--':>8} {'--':>7}")

    print("\n=== Phase 12 - theoretical capacity ceiling by DWT level (256x256 image) ===")
    for r in bound_table:
        print(f"  levels={r['dwt_levels']!s:<10} max capacity = {r['max_capacity_bits']} bits")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 12 - high-capacity watermarking")
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--quick", action="store_true", help="fast subset: 6 images")
    p.add_argument("--eval-num-images", type=int, dest="eval_num_images", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    overrides: dict = {}
    if args.eval_num_images is not None:
        overrides["eval_num_images"] = args.eval_num_images
    if args.quick and args.eval_num_images is None:
        overrides["eval_num_images"] = 6
    cfg = load_config(args.config, overrides or None)
    run(cfg)


if __name__ == "__main__":
    main()
