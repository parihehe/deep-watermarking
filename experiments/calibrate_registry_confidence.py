"""Calibrate RELIABLE_REGISTRY_ID_CONFIDENCE against the actual confidence
distribution of the NEW registry-ID decode path (interleaved + masked
encoding, ``src/watermark/id_registry.py``), on real images through the real
blind CNN checkpoint - no synthetic data, no assumed distribution.

Why this exists: ``RELIABLE_TEXT_CONFIDENCE`` (0.85, ``src/app/final_model.py``)
was set before the registry-ID decode path existed, calibrated (per
``docs/blind_extraction_diagnosis.md`` S6) against a *different* confidence
metric - the raw per-bit CNN sigmoid margin on a single-copy payload, back
when blind decode accuracy was ~0%, so *any* gate value trivially suppressed
all output. The registry-ID path's confidence is a different metric entirely
(mean majority-vote agreement over 4 slots, quantized to multiples of 1/15),
and was never separately calibrated. This script measures its real precision/
recall trade-off across the full 16-ID space so a threshold can be chosen
from evidence instead of inherited from an unrelated metric.

Run::
    .venv/Scripts/python.exe experiments/calibrate_registry_confidence.py --n-images 30
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

from src.evaluation.blind_extract import BlindExtractor
from src.watermark import id_registry
from src.watermark.embed import EmbedConfig, embed

DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "div2k_256"
CHECKPOINT = PROJECT_ROOT / "models" / "phase8_cnn" / "phase8_cnn_best.pt"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase18_id_registry"

EMBED_CONFIG = EmbedConfig(wavelet="haar", subband="LL", alpha=0.02,
                            bit_length=id_registry.PAYLOAD_BITS, start_sv_index=0, mode="symmetric")

CALIBRATION_SEED = 42  # independent of the verification sweeps' seeds (none / 20260905)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-images", type=int, default=30)
    args = ap.parse_args(argv)

    extractor = BlindExtractor.from_checkpoint(str(CHECKPOINT))
    all_images = sorted((DATA_ROOT / "test").glob("*.png"))
    rng = np.random.default_rng(CALIBRATION_SEED)
    idx = rng.choice(len(all_images), size=min(args.n_images, len(all_images)), replace=False)
    paths = [all_images[i] for i in idx]
    print(f"calibration seed={CALIBRATION_SEED}  n_images={len(paths)}  ids=all 16\n")

    trials: list[dict] = []
    t0 = time.time()
    for id_ in range(16):
        bits = id_registry.encode_id_bits(id_)
        for p in paths:
            rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            wm = embed(rgb, bits, EMBED_CONFIG).watermarked_image
            probs = extractor.extract_proba(wm)
            recovered = (probs > 0.5).astype(int).tolist()
            decoded_id, mean_conf, min_conf = id_registry.decode_id_bits(
                recovered[: id_registry.ENCODED_BITS]
            )
            trials.append({
                "id": id_, "image": p.name, "decoded_id": decoded_id,
                "correct": decoded_id == id_,
                "confidence_mean": round(mean_conf, 4), "confidence_min": round(min_conf, 4),
            })

    n = len(trials)
    n_correct = sum(t["correct"] for t in trials)
    print(f"overall: {n_correct}/{n} ({n_correct/n:.2%})  elapsed={time.time()-t0:.1f}s\n")

    # Precision/recall sweep over candidate thresholds on confidence_mean.
    thresholds = sorted({round(k / 60, 4) for k in range(32, 61)} | {0.85})
    table = []
    for th in thresholds:
        passing = [t for t in trials if t["confidence_mean"] >= th]
        if not passing:
            continue
        precision = sum(t["correct"] for t in passing) / len(passing)
        recall = len(passing) / n
        table.append({"threshold": th, "n_pass": len(passing), "coverage": round(recall, 4),
                      "precision": round(precision, 4)})

    print(f"{'thresh':>8} {'n_pass':>8} {'coverage':>10} {'precision':>10}")
    for row in table:
        print(f"{row['threshold']:>8.4f} {row['n_pass']:>8} {row['coverage']:>10.2%} {row['precision']:>10.2%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "calibration_seed": CALIBRATION_SEED, "n_images_per_id": len(paths), "n_ids": 16,
        "n_trials": n, "overall_accuracy": round(n_correct / n, 4),
        "threshold_table": table, "trials": trials,
    }
    (RESULTS_DIR / "confidence_calibration.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {RESULTS_DIR / 'confidence_calibration.json'}")


if __name__ == "__main__":
    main()
