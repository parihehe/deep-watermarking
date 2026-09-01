"""Phase 8 — CPU smoke experiment for the blind CNN watermark extractor.

Purpose: prove end to end, on real DIV2K images, that

  * the frozen Phase 6 embedder synthesises training data,
  * the blind CNN actually learns — validation bit-accuracy rises well above
    the 0.5 chance line with the watermark embedded at the frozen baseline
    strength (alpha = 0.02), original image never seen, and
  * a self-describing checkpoint is written and reloads for blind extraction.

It is deliberately small so it finishes in a few minutes on a CPU: 32-bit
payload, 300 train / 60 validation images, 12 epochs (~12 s/epoch). The full
training run uses ``configs/cnn_extractor.yaml`` (all 700 / 100 images).

Usage:
    .venv/Scripts/python.exe experiments/run_phase8_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.blind_extract import BlindExtractor
from src.training.train_extractor import TrainConfig, train
from src.training.watermark_dataset import WatermarkExtractionConfig, WatermarkExtractionDataset

MIN_BIT_ACCURACY = 0.60  # comfortably above the 0.5 chance line


def run() -> None:
    cfg = TrainConfig(
        bit_length=32,
        alpha=0.02,                # frozen Phase 6 baseline strength
        train_limit=300,
        val_limit=60,
        image_size=256,
        epochs=12,
        batch_size=32,
        lr=1e-3,
        num_workers=0,
        seed=20260901,
        run_name="phase8_smoke",
    )
    summary = train(cfg)

    init = summary["init_val_metrics"]
    best = summary["best_val_metrics"]
    print()
    print("=" * 68)
    print("PHASE 8 SMOKE RESULT")
    print("=" * 68)
    print(f"payload / alpha        : {cfg.bit_length} bits / alpha={cfg.alpha}")
    print(f"train / val images     : {summary['n_train_images']} / {summary['n_val_images']}")
    print(f"learnable parameters   : {summary['model_parameters']:,}")
    print(f"init  val bit-accuracy : {init['bit_accuracy']:.4f}  (chance = 0.5)")
    print(f"best  val bit-accuracy : {best['bit_accuracy']:.4f}  (epoch {summary['best_epoch']})")
    print(f"best  val BER          : {best['ber']:.4f}")
    print(f"best  val NC           : {best['nc']:.4f}")
    print(f"best  val exact-match  : {best['exact_match']:.4f}")
    print(f"checkpoint             : {summary['checkpoints']['best']}")

    # Reload the checkpoint and run a real blind extraction on held-out
    # validation images to show the inference path works.
    extractor = BlindExtractor.from_checkpoint(summary["checkpoints"]["best"])
    val_ds = WatermarkExtractionDataset(
        WatermarkExtractionConfig(
            processed_root=str((PROJECT_ROOT / cfg.processed_root).resolve()),
            bit_length=cfg.bit_length,
            embed_config=cfg.embed_config(),
            image_size=cfg.image_size,
            limit=5,
            seed=cfg.seed,
        ),
        "validation",
        deterministic=True,
    )
    print()
    print("blind extraction on 5 held-out validation images (image only in):")
    for i in range(len(val_ds)):
        image_tensor, target = val_ds[i]
        rgb_u8 = (image_tensor.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")
        predicted = extractor.extract_bits(rgb_u8)
        reference = [int(b) for b in target.tolist()]
        wrong = sum(int(a != b) for a, b in zip(reference, predicted))
        print(f"  image {i}: {wrong:2d}/{len(reference)} bit errors  (BER {wrong / len(reference):.3f})")

    if best["bit_accuracy"] < MIN_BIT_ACCURACY:
        raise SystemExit(
            f"SMOKE FAILED: best validation bit-accuracy {best['bit_accuracy']:.4f} "
            f"< {MIN_BIT_ACCURACY:.2f} — the blind extractor did not learn."
        )
    print()
    print(
        f"SMOKE PASSED: blind bit-accuracy {best['bit_accuracy']:.3f} "
        f">> 0.5 chance, checkpoint saved and reloaded."
    )


if __name__ == "__main__":
    run()
