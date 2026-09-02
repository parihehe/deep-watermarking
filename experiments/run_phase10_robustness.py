"""Phase 10 - Attack Simulation / Robustness Testing (experiment runner).

Runs the reproducible Phase 10 robustness experiment defined in
``configs/phase10_robustness.yaml``:

  1. Guard: assert the frozen Phase 6 baseline contract; record frozen-file hashes.
  2. Watermark every held-out DIV2K test image once at the Phase 8/9 operating
     point (64 bits, alpha = 0.02).
  3. For each attack x severity: attack the watermarked image, recover with the
     frozen non-blind DWT-SVD decoder and with the Phase 8 blind CNN, and record
     image-quality (PSNR/SSIM/MSE) and watermark-recovery (BER/bit-acc/NC) for
     both, plus the delta against the no-attack baseline.
  4. Write per-image + summary CSV/JSON and robustness plots.
  5. Sanity-check that Phase 7 (the web app) and Phase 8 (the CNN) still load and
     run.

Nothing here modifies the frozen baseline, the Phase 7 UI or the Phase 8 model.

Usage
-----
    .venv/Scripts/python.exe experiments/run_phase10_robustness.py
    .venv/Scripts/python.exe experiments/run_phase10_robustness.py --quick
    .venv/Scripts/python.exe experiments/run_phase10_robustness.py --no-cnn
    .venv/Scripts/python.exe experiments/run_phase10_robustness.py --eval-num-images 100
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
    assert_baseline_frozen,
    environment_stamp,
    frozen_file_hashes,
)
from src.evaluation.phase10_robustness import (
    Phase10Config,
    baseline_block,
    evaluate_attack_point,
    load_config,
    prepare_samples,
    summarise,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "phase10_robustness.yaml"


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


def _flatten_summary(summary: list[dict], has_blind: bool) -> list[dict]:
    decoders = ["nonblind"] + (["blind"] if has_blind else [])
    flat: list[dict] = []
    for entry in summary:
        row = {
            "attack": entry["attack"],
            "params": entry["params"],
            "severity": entry["severity"],
            "n_images": entry["n_images"],
        }
        for key, value in entry["image_quality"].items():
            row[f"quality_{key}"] = value
        for decoder in decoders:
            for key, value in entry["watermark_recovery"][decoder].items():
                row[f"{decoder}_{key}"] = value
        flat.append(row)
    return flat


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _make_plots(plots_dir: Path, summary: list[dict], has_blind: bool) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        print(f"[phase10] matplotlib unavailable ({exc}); skipping plots")
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    by_attack: dict[str, list[dict]] = {}
    for entry in summary:
        if entry["attack"] == "identity":
            continue
        by_attack.setdefault(entry["attack"], []).append(entry)

    # 1. per-attack: BER vs severity (non-blind and blind on the same axes)
    scalar_attacks = {
        a: sorted(v, key=lambda e: float(e["severity"]))
        for a, v in by_attack.items()
        if all(e["severity"] != "" for e in v)
    }
    n = len(scalar_attacks)
    if n:
        cols = 3
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 3.6 * rows_n), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for idx, (attack, entries) in enumerate(sorted(scalar_attacks.items())):
            ax = axes.flat[idx]
            ax.set_visible(True)
            sev = [float(e["severity"]) for e in entries]
            nb = [e["watermark_recovery"]["nonblind"]["ber_mean"] for e in entries]
            ax.plot(sev, nb, marker="o", label="non-blind")
            if has_blind:
                bl = [e["watermark_recovery"]["blind"]["ber_mean"] for e in entries]
                ax.plot(sev, bl, marker="s", label="blind CNN")
            ax.axhline(0.5, color="grey", ls=":", lw=1, label="chance")
            ax.set_title(attack)
            ax.set_xlabel(f"severity ({attack})")
            ax.set_ylabel("mean BER")
            ax.set_ylim(-0.02, 0.62)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("Phase 10: BER vs attack strength", y=1.0)
        fig.tight_layout()
        path = plots_dir / "ber_vs_severity.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))

    # 2. per-attack: PSNR vs severity (attack distortion)
    if scalar_attacks:
        fig, ax = plt.subplots(figsize=(8, 5))
        for attack, entries in sorted(scalar_attacks.items()):
            sev = [float(e["severity"]) for e in entries]
            q = [e["image_quality"]["psnr_mean"] for e in entries]
            ax.plot(range(len(sev)), q, marker="o", label=attack)
            for i, (s, y) in enumerate(zip(sev, q)):
                ax.annotate(f"{s:g}", (i, y), textcoords="offset points", xytext=(3, 3), fontsize=7)
        ax.set_xlabel("severity index (labels = parameter value)")
        ax.set_ylabel("PSNR (dB): attacked vs clean watermarked")
        ax.set_title("Phase 10: attack distortion vs strength")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = plots_dir / "psnr_vs_severity.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))

    # 3. worst-case BER per attack family (ranking)
    fig, ax = plt.subplots(figsize=(9, 5))
    families = sorted(by_attack)
    worst_nb = [max(e["watermark_recovery"]["nonblind"]["ber_mean"] for e in by_attack[a]) for a in families]
    x = range(len(families))
    width = 0.38
    ax.bar([i - width / 2 for i in x], worst_nb, width, label="non-blind", color="#4C72B0")
    if has_blind:
        worst_bl = [max(e["watermark_recovery"]["blind"]["ber_mean"] for e in by_attack[a]) for a in families]
        ax.bar([i + width / 2 for i in x], worst_bl, width, label="blind CNN", color="#DD8452")
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_ylabel("worst-case mean BER over the tested severities")
    ax.set_title("Phase 10: which attacks hurt the watermark most")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "worst_case_ber_by_attack.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    return written


# ---------------------------------------------------------------------------
# Phase 7 / Phase 8 sanity
# ---------------------------------------------------------------------------

def _sanity_checks(cfg: Phase10Config) -> dict:
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

    ckpt = cfg.resolved_checkpoint()
    if cfg.cnn_enabled and ckpt.is_file():
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

def run(cfg: Phase10Config, *, run_cnn: bool = True) -> dict:
    started = time.time()

    assert_baseline_frozen()
    hashes = frozen_file_hashes()
    print("[phase10] frozen Phase 6 baseline contract: OK")

    results_dir = cfg.resolved_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    samples = prepare_samples(cfg)
    print(f"[phase10] watermarked {len(samples)} test images @ "
          f"{cfg.payload_bits} bits, alpha={cfg.alpha}")

    extractor = None
    has_blind = False
    if run_cnn and cfg.cnn_enabled:
        ckpt = cfg.resolved_checkpoint()
        if ckpt.is_file():
            from src.evaluation.blind_extract import BlindExtractor

            extractor = BlindExtractor.from_checkpoint(str(ckpt))
            has_blind = True
            print(f"[phase10] blind CNN extractor loaded from {ckpt.name}")
        else:
            print(f"[phase10] CNN checkpoint missing ({ckpt}); non-blind only")

    econf = cfg.embed_config()

    # no-attack baseline first (identity), then every configured attack
    grid: list[tuple[str, dict]] = [("identity", {})]
    for attack_name, param_list in cfg.attacks.items():
        for params in param_list:
            grid.append((attack_name, params))

    all_rows: list[dict] = []
    for attack_name, params in grid:
        point_rows = evaluate_attack_point(
            samples, attack_name, params,
            embed_config=econf, size=cfg.image_size,
            noise_seed=cfg.noise_seed, extractor=extractor,
        )
        all_rows.extend(point_rows)
        nb = sum(r["nonblind_ber"] for r in point_rows) / len(point_rows)
        tail = f"  blind BER={sum(r['blind_ber'] for r in point_rows) / len(point_rows):.4f}" if has_blind else ""
        label = attack_name if not point_rows[0]["severity"] else f"{attack_name}[{point_rows[0]['severity']}]"
        print(f"[phase10] {label:<28} non-blind BER={nb:.4f}{tail}")

    # summarise: identity baseline -> deltas on every other point
    identity_rows = [r for r in all_rows if r["attack"] == "identity"]
    identity_summary = summarise(identity_rows, has_blind=has_blind)[0]
    base = baseline_block(identity_summary, has_blind=has_blind)
    summary = summarise(all_rows, baseline=base, has_blind=has_blind)

    per_image_csv = results_dir / "phase10_per_image.csv"
    summary_csv = results_dir / "phase10_summary.csv"
    _write_csv(per_image_csv, all_rows)
    _write_csv(summary_csv, _flatten_summary(summary, has_blind))

    baseline_recovery = identity_summary["watermark_recovery"]
    (results_dir / "phase10_baseline.json").write_text(
        json.dumps({
            "phase": 10,
            "component": "no_attack_baseline",
            "operating_point": {"payload_bits": cfg.payload_bits, "alpha": cfg.alpha},
            "n_images": len(samples),
            "watermark_recovery": baseline_recovery,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    ranking = _ranking(summary, has_blind)
    report = {
        "phase": 10,
        "title": "Attack Simulation / Robustness Testing",
        "scope_excludes": [
            "change to the frozen Phase 6 formula / defaults",
            "CNN architecture change (Phase 15)",
            "adaptive embedding (Phase 11)",
            "high-capacity changes (Phase 12)",
            "wavelet/subband study (Phases 13-14)",
            "error-correcting codes (Phase 16)",
            "geometric re-synchronisation of attacked images",
        ],
        "config": {
            "operating_point": {"payload_bits": cfg.payload_bits, "alpha": cfg.alpha,
                                "wavelet": cfg.wavelet, "mode": cfg.mode, "subband": cfg.subband,
                                "start_sv_index": cfg.start_sv_index},
            "embedding_rule": "S'[i] = S[i] * (1 + alpha * (2*b - 1))  [frozen Phase 6]",
            "non_blind_decoder": "extract_traditional (original also in), shape-resync only",
            "blind_decoder": "Phase 8 BlindCNNExtractor (image only in)" if has_blind else None,
            "eval_split": cfg.eval_split,
            "eval_num_images": len(samples),
            "image_size": cfg.image_size,
            "payload_seed": cfg.payload_seed,
            "noise_seed": cfg.noise_seed,
            "attacks": cfg.attacks,
        },
        "environment": environment_stamp(),
        "frozen_baseline_sha256": hashes,
        "no_attack_baseline": baseline_recovery,
        "robustness_ranking": ranking,
        "summary": summary,
        "sanity": _sanity_checks(cfg),
    }
    if cfg.make_plots:
        report["plots"] = _make_plots(results_dir / "plots", summary, has_blind)
        for p in report.get("plots", []):
            print(f"[phase10] wrote {p}")

    report["elapsed_seconds"] = round(time.time() - started, 1)
    (results_dir / "phase10_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[phase10] wrote {per_image_csv}")
    print(f"[phase10] wrote {summary_csv}")
    print(f"[phase10] wrote {results_dir / 'phase10_report.json'}")

    _print_tables(summary, baseline_recovery, ranking, has_blind)
    print(f"\n[phase10] done in {report['elapsed_seconds']}s -> {results_dir}")
    print(f"[phase10] sanity: {report['sanity']}")
    return report


def _ranking(summary: list[dict], has_blind: bool) -> list[dict]:
    """Attacks ordered by worst-case non-blind BER increase over baseline."""
    families: dict[str, dict] = {}
    for entry in summary:
        if entry["attack"] == "identity":
            continue
        nb = entry["watermark_recovery"]["nonblind"]
        worst = families.setdefault(entry["attack"], {"attack": entry["attack"],
                                                      "worst_nonblind_ber": 0.0,
                                                      "worst_nonblind_ber_delta": 0.0,
                                                      "at_severity": entry["severity"]})
        if nb["ber_mean"] > worst["worst_nonblind_ber"]:
            worst["worst_nonblind_ber"] = nb["ber_mean"]
            worst["worst_nonblind_ber_delta"] = nb.get("ber_delta_vs_baseline", 0.0)
            worst["at_severity"] = entry["severity"]
        if has_blind:
            bl = entry["watermark_recovery"]["blind"]
            worst["worst_blind_ber"] = max(worst.get("worst_blind_ber", 0.0), bl["ber_mean"])
    return sorted(families.values(), key=lambda d: d["worst_nonblind_ber"], reverse=True)


def _print_tables(summary, baseline_recovery, ranking, has_blind) -> None:
    print("\n=== Phase 10 - no-attack baseline (64 bits, alpha 0.02) ===")
    nb = baseline_recovery["nonblind"]
    line = f"  non-blind: BER={nb['ber_mean']:.4f} bit_acc={nb['bit_accuracy_mean']:.4f} NC={nb['nc_mean']:.4f}"
    print(line)
    if has_blind:
        bl = baseline_recovery["blind"]
        print(f"  blind CNN: BER={bl['ber_mean']:.4f} bit_acc={bl['bit_accuracy_mean']:.4f} NC={bl['nc_mean']:.4f}")

    print("\n=== Phase 10 - per attack point (mean over images) ===")
    hdr = f"{'attack':<16} {'severity':>9} | {'PSNR':>6} {'SSIM':>6} | {'nb BER':>7} {'nb dBER':>8}"
    if has_blind:
        hdr += f" | {'bl BER':>7} {'bl dBER':>8}"
    print(hdr)
    print("-" * len(hdr))
    for e in summary:
        if e["attack"] == "identity":
            continue
        q = e["image_quality"]
        nbk = e["watermark_recovery"]["nonblind"]
        sev = e["severity"] if e["severity"] != "" else "-"
        row = (f"{e['attack']:<16} {sev!s:>9} | {q['psnr_mean']:>6.2f} {q['ssim_mean']:>6.4f} | "
               f"{nbk['ber_mean']:>7.4f} {nbk.get('ber_delta_vs_baseline', 0.0):>+8.4f}")
        if has_blind:
            blk = e["watermark_recovery"]["blind"]
            row += f" | {blk['ber_mean']:>7.4f} {blk.get('ber_delta_vs_baseline', 0.0):>+8.4f}"
        print(row)

    print("\n=== Phase 10 - attacks ranked by worst-case non-blind BER ===")
    for i, r in enumerate(ranking, 1):
        extra = f"  (blind worst {r['worst_blind_ber']:.4f})" if has_blind and "worst_blind_ber" in r else ""
        print(f"  {i:>2}. {r['attack']:<16} worst BER {r['worst_nonblind_ber']:.4f} "
              f"(+{r['worst_nonblind_ber_delta']:.4f} vs baseline) @ severity {r['at_severity']}{extra}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _quick_overrides(cfg_attacks: dict) -> dict:
    trimmed: dict[str, list[dict]] = {}
    for name, params in cfg_attacks.items():
        trimmed[name] = [params[0], params[len(params) // 2]] if len(params) > 1 else params[:1]
    return {"eval_num_images": 10, "attacks": trimmed}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 10 - attack simulation / robustness testing")
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--quick", action="store_true",
                   help="fast subset: 10 images, 2 severities per attack")
    p.add_argument("--no-cnn", dest="no_cnn", action="store_true", help="skip the blind CNN decoder")
    p.add_argument("--eval-num-images", type=int, dest="eval_num_images", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    overrides: dict = {}
    if args.eval_num_images is not None:
        overrides["eval_num_images"] = args.eval_num_images
    cfg = load_config(args.config, overrides or None)
    if args.quick:
        cfg = load_config(args.config, {**overrides, **_quick_overrides(cfg.attacks)})
    run(cfg, run_cnn=not args.no_cnn)


if __name__ == "__main__":
    main()
