"""Real measurement: does the registry-ID + repetition(15) blind payload
(``src/watermark/id_registry.py``) actually recover exact text through the
blind CNN, at the alpha=0.02 operating point?

This is the empirical follow-up to the feasibility math (K=4 ID bits, r=15,
60/64 bits used, projected 96.7% exact-match under an i.i.d.-bit-error model).
That projection has a known blind spot: the CNN's 1-D conv feature extractor
(kernel=7) mixes neighboring output positions, so real errors may be
correlated in a way the i.i.d. model can't see. This script measures the
actual rate, not the projection - no CNN retraining, existing checkpoint only.

For each of the 5 canonical test messages (hello, hi, owner-2026, Vestigia,
日本語 - the same vocabulary used throughout this project, e.g.
experiments/train_blind_size_models.py), on N held-out DIV2K test images:
  1. encode the message's registry ID with id_registry.encode_id_bits (64 bits)
  2. embed with the frozen embed() at the Phase 17 final-model operating point
     (haar, LL, alpha=0.02, no ECC)
  3. run the shipped blind CNN checkpoint (models/phase8_cnn/phase8_cnn_best.pt)
  4. majority-vote decode the recovered bits back to an ID (id_registry.decode_id_bits)
  5. look the ID up in the same registry and compare to the expected message

Run::
    .venv/Scripts/python.exe experiments/run_id_registry_eval.py --n-images 40
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

# The exact frozen final-model operating point (src/app/final_model.py FINAL_MODEL_SPEC):
# haar wavelet, LL subband only, alpha=0.02, 64-bit payload, no ECC, no adaptive embedding.
EMBED_CONFIG = EmbedConfig(
    wavelet="haar", subband="LL", alpha=0.02, bit_length=id_registry.PAYLOAD_BITS,
    start_sv_index=0, mode="symmetric",
)

# Same vocabulary as src/app/final_model.py's MESSAGE_REGISTRY and
# experiments/train_blind_size_models.py's TEXTS.
MESSAGES = ["hello", "hi", "owner-2026", "Vestigia", "日本語"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Real exact-match evaluation of the registry-ID blind payload")
    ap.add_argument("--n-images", type=int, default=40)
    args = ap.parse_args(argv)

    if not CHECKPOINT.is_file():
        raise SystemExit(f"checkpoint not found: {CHECKPOINT}")
    extractor = BlindExtractor.from_checkpoint(str(CHECKPOINT))
    print(f"checkpoint: {CHECKPOINT.relative_to(PROJECT_ROOT)}  "
          f"declared bit_length={extractor.config.bit_length}")

    registry = id_registry.MessageRegistry(MESSAGES)

    paths = sorted((DATA_ROOT / "test").glob("*.png"))[: args.n_images]
    if len(paths) < args.n_images:
        print(f"WARNING: only {len(paths)} test images available (requested {args.n_images})")
    print(f"n_images per message: {len(paths)}  |  messages: {MESSAGES}\n")

    per_message: list[dict] = []
    all_trials: list[dict] = []
    t0 = time.time()

    for message in MESSAGES:
        msg_id = registry.id_for(message)
        bits = id_registry.encode_id_bits(msg_id)

        exact = 0
        id_correct = 0
        confidences = []
        for p in paths:
            rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            wm = embed(rgb, bits, EMBED_CONFIG).watermarked_image
            probs = extractor.extract_proba(wm)
            recovered_bits = (probs > 0.5).astype(int).tolist()
            decoded_id, mean_conf, min_conf = id_registry.decode_id_bits(
                recovered_bits[: id_registry.ENCODED_BITS]
            )
            decoded_text = registry.message_for(decoded_id)
            ok_id = decoded_id == msg_id
            ok_text = decoded_text == message
            id_correct += int(ok_id)
            exact += int(ok_text)
            confidences.append(mean_conf)
            all_trials.append({
                "message": message, "image": p.name, "expected_id": msg_id,
                "decoded_id": decoded_id, "id_correct": ok_id, "text_correct": ok_text,
                "vote_confidence_mean": round(mean_conf, 4), "vote_confidence_min": round(min_conf, 4),
            })

        n = len(paths)
        rate = exact / n if n else 0.0
        print(f'[{message!r:14s}] id={msg_id:2d}  exact-match {exact:3d}/{n} ({rate:.1%})  '
              f'mean vote confidence {np.mean(confidences):.4f}')
        per_message.append({
            "message": message, "id": msg_id, "n": n, "exact_match": exact,
            "exact_match_rate": round(rate, 4), "mean_vote_confidence": round(float(np.mean(confidences)), 4),
        })

    total_n = sum(m["n"] for m in per_message)
    total_exact = sum(m["exact_match"] for m in per_message)
    overall_rate = total_exact / total_n if total_n else 0.0

    print(f"\n{'='*72}")
    print(f"OVERALL: {total_exact}/{total_n} exact-match ({overall_rate:.2%})")
    print(f"theoretical projection (i.i.d.-error model, measured BER=0.222): 96.73%")
    gap = overall_rate - 0.9673
    print(f"gap vs. projection: {gap:+.2%}")
    print(f"{'='*72}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)),
        "embed_config": {"wavelet": "haar", "subband": "LL", "alpha": 0.02,
                          "bit_length": id_registry.PAYLOAD_BITS},
        "id_bits": id_registry.ID_BITS, "repetition": id_registry.REPETITION,
        "encoded_bits": id_registry.ENCODED_BITS, "messages": MESSAGES,
        "per_message": per_message,
        "overall": {"n": total_n, "exact_match": total_exact, "exact_match_rate": round(overall_rate, 4)},
        "theoretical_projection": 0.9673,
        "gap_vs_projection": round(gap, 4),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (RESULTS_DIR / "per_trial.json").write_text(json.dumps(all_trials, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {RESULTS_DIR / 'summary.json'}")
    print(f"wrote {RESULTS_DIR / 'per_trial.json'}")


if __name__ == "__main__":
    main()
