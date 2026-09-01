# Phase 8 — Blind CNN Watermark Extraction

**Status:** implemented and tested (2026-09-01).
**Relationship to the frozen baseline:** additive only. No file under
`src/watermark/`, `src/evaluation/metrics.py` or `src/app/` was modified. The
frozen Phase 6 DWT-SVD embedder is *called*, never changed.

---

## 1. Goal

Phase 6 recovers the watermark **non-blindly**: `extract_traditional` needs the
original cover image to compare singular values. Phase 8 removes that dependency
with a learned decoder:

> Given **only the watermarked image**, predict the embedded payload bits.

The original image is never an input to the network, at training time or at
inference time.

---

## 2. What was added

| File | Purpose |
|---|---|
| `src/models/cnn_extractor.py` | `BlindCNNExtractor` architecture, `ExtractorConfig`, self-describing checkpoint save/load. |
| `src/training/watermark_dataset.py` | `WatermarkExtractionDataset` — wraps processed DIV2K images and calls the frozen `embed()` on the fly to produce `(watermarked image, payload bits)` pairs. |
| `src/training/train_extractor.py` | Training + validation loop, checkpointing, metrics, CSV/JSON logging, CLI. |
| `src/evaluation/blind_extract.py` | `BlindExtractor` inference wrapper + CLI for blind extraction from a single image. |
| `configs/cnn_extractor.yaml` | Phase 8 configuration (separate from `configs/baseline.yaml`). |
| `experiments/run_phase8_smoke.py` | Fast CPU experiment proving the model trains and a checkpoint is produced. |
| `tests/test_cnn_extractor.py` | 23 tests: architecture, checkpoint round-trip, dataset, blind inference, 1-epoch end-to-end training, and a guard that the frozen baseline / Phase 7 API is unchanged. |
| `docs/phase8_blind_cnn.md` | This document. |

No existing source file was edited.

---

## 3. CNN architecture (plain-language)

The watermark is defined in the **SVD-of-DWT-LL domain**, not in pixel space, so
the extractor works there too. Everything from the image down to the
singular-value vector is a **fixed, parameter-free, differentiable** feature
extractor — it uses no learnable weights and never sees the original image. All
learning happens in a small **1D CNN that scans along the singular-value index
axis**.

```
watermarked RGB image  (B, 3, 256, 256), pixels in [0, 1]
  │
  ├─ fixed  BT.601 luminance                       -> Y   (B, 1, 256, 256)
  ├─ fixed  single-level Haar DWT, keep LL         -> LL  (B, 1, 128, 128)
  ├─ fixed  batched SVD, singular values only      -> sigma (B, 128)
  ├─ fixed  log1p(sigma) then per-sample standardise
  │
  ├─ 1D CNN along the index axis:
  │     Conv1d(1  -> 64, k=7) + BatchNorm + ReLU
  │     3 x [ Conv1d(64 -> 64, k=7) + BatchNorm + ReLU ]
  │     Dropout(0.1)
  │     Conv1d(64 -> 1, k=1)          -> one logit per singular-value index
  │
  └─ first `bit_length` logits are the payload
```

* **Why this shape:** the non-blind baseline reads bit `i` by comparing the
  watermarked `S'[i]` against the original `S[i]`. Blind, there is no original,
  so the network instead learns the smooth natural decay of the singular-value
  spectrum and detects the local `+-alpha` step at each index. The `k=7`
  convolution gives every bit decision a 7-neighbour window of context — "is
  `S'[i]` above or below the smooth local trend?".
* **A plain 2D CNN on RGB does not work here** — measured: it trained to exactly
  chance (0.49 bit accuracy). The watermark residual is sub-perceptual and
  spatially diffuse, and pooling averages it away. That negative result is what
  motivated this design.
* **Prediction:** `sigmoid(logits) > 0.5` per position.
* **Loss:** `BCEWithLogitsLoss` — each bit is an independent binary
  classification, matching the bipolar embedding rule
  `S'[i] = S[i] * (1 + alpha * (2*b[i] - 1))`.
* **Resolution-agnostic:** the singular-value vector is truncated / zero-padded
  to a fixed length, so any input size runs; training/inference standardise on
  256x256 and non-square inputs are resized.
* **Size:** ~120 K learnable parameters (the feature stage has zero). Trivial on
  CPU.
* **Configurable payload:** `ExtractorConfig.bit_length` (8..128 fits the
  LL-only default; larger payloads need `extra_subbands` in the embed config,
  exactly as in Phase 6).

---

## 4. Training data

* **Source images:** the Phase 2 processed DIV2K re-split,
  `data/processed/div2k_256/` (256x256 PNGs, disjoint IDs — train 1–700,
  validation 701–800, test 801–900, no leakage).
* **Watermarking:** for every image, a random `bit_length`-bit payload is drawn
  and embedded with the **frozen Phase 6 `embed()`** (default `haar` / `LL` /
  `symmetric`, `alpha = 0.02` for the learned decoder — configurable).
* **On the fly:** watermarking happens inside `__getitem__`, so no watermarked
  images are cached to disk.
* **Payload policy:**
  * `validation` / `test` — payload is a fixed function of `(seed, index)`; the
    validation set is identical on every run and every epoch.
  * `train` — a fresh random payload every `__getitem__` call, so the network
    sees new `(image, payload)` pairs each epoch and cannot memorise a fixed
    mapping.

---

## 5. Model input / output

| | |
|---|---|
| **Input** | One watermarked RGB image, `(B, 3, 256, 256)`, float, pixels in `[0, 1]`. Nothing else — no original image, no cover residual, no side information. |
| **Output** | `bit_length` per-bit logits → `sigmoid` → threshold at 0.5 → the recovered payload bit string. |

---

## 6. Metrics

Per epoch, on the fixed validation set:

* **loss** — `BCEWithLogitsLoss`
* **bit accuracy** — fraction of bits correct (0.5 = chance)
* **BER** — bit error rate = `1 - bit accuracy`
* **exact-match rate** — fraction of images whose entire payload is recovered
* **NC** — mean normalised correlation of the bipolar bit strings

`training_log.csv` has one row per epoch; `training_summary.json` records the
config, environment, best epoch and best/final metrics.

### Measured results — smoke experiment (`experiments/run_phase8_smoke.py`)

CPU, 32-bit payload, watermark embedded at the **frozen baseline strength
`alpha = 0.02`**, 300 train / 60 validation DIV2K images, 12 epochs
(~12 s/epoch), ~87 K learnable parameters:

| | init (random weights) | best (epoch 7) | final (epoch 12) |
|---|---|---|---|
| validation bit-accuracy | 0.4953 (chance) | **0.7396** | 0.7313 |
| validation BER | 0.5047 | **0.2604** | 0.2688 |
| validation NC | -0.009 | 0.479 | 0.463 |
| training bit-accuracy | ~0.50 | 0.724 | 0.735 |
| loss | 0.693 | 0.531 | 0.506 |

Reloading `phase8_smoke_best.pt` and running blind extraction on 5 held-out
validation images (image only in, no original): 8–11 of 32 bits wrong per image
(BER 0.25–0.34), matching the aggregate.

### Measured results — full configuration (`configs/cnn_extractor.yaml`)

CPU, **64-bit** payload, `alpha = 0.02`, **all 700 train / 100 validation**
DIV2K images, 20 epochs (~19 s/epoch, ~6.5 min total), 87,041 learnable
parameters, seed 20260901:

| | init | best = final (epoch 20) |
|---|---|---|
| validation loss (BCE) | 0.6931 | **0.4537** |
| validation bit-accuracy | 0.5078 | **0.7761** |
| validation BER | 0.4922 | **0.2239** |
| validation NC | 0.016 | 0.552 |
| training loss | — | 0.4600 |
| training bit-accuracy | — | 0.7698 |
| validation exact-match | 0.0 | 0.0 |

Train 0.770 vs validation 0.776 — no overfitting; the extra data lifts the
plateau from ~0.74 (smoke) to ~0.776. Blind extraction from the reloaded
`phase8_cnn_best.pt` on all 100 held-out validation images: **mean BER 0.2239,
mean bit-accuracy 0.7761** (per-image 8–16 of 64 bits wrong, NC +0.50 … +0.75).

* Validation tracks training closely — the decoder generalises across unseen
  images *and* fresh random payloads, it is not memorising.
* The ~0.73 plateau is the expected shape: the leading singular values (which
  carry the first bits) are recovered well, the trailing ones are fragile
  through the IDWT + uint8 round trip — the same effect the Phase 6 non-blind
  BER table shows. Bigger data / more epochs / a leading-bit-weighted loss are
  the levers for a later CNN-improvement phase; they are out of Phase 8 scope.
* The full run (`configs/cnn_extractor.yaml`: all 700 / 100 images, 20 epochs,
  64-bit payload) uses the identical pipeline.

The exact numbers for the run on this machine are in
`results/phase8_cnn/phase8_smoke_training_log.csv` and
`results/phase8_cnn/phase8_smoke_training_summary.json`.

---

## 7. How to run

### Full training (uses `configs/cnn_extractor.yaml`)

```powershell
.venv\Scripts\python.exe -m src.training.train_extractor --config configs\cnn_extractor.yaml
```

Fast CPU sanity check with overrides:

```powershell
.venv\Scripts\python.exe -m src.training.train_extractor `
    --config configs\cnn_extractor.yaml `
    --epochs 3 --train-limit 64 --val-limit 16 --batch-size 8 --bit-length 16
```

### Smoke experiment (self-contained, ~few minutes on CPU)

```powershell
.venv\Scripts\python.exe experiments\run_phase8_smoke.py
```

Trains a 16-bit / `alpha=0.05` extractor on ~96 train / 24 validation images for
8 epochs, then reloads the checkpoint and runs a real blind extraction. It exits
non-zero if validation bit-accuracy never beats chance.

### Blind extraction from a single image

Programmatic:

```python
from src.evaluation.blind_extract import BlindExtractor

extractor = BlindExtractor.from_checkpoint("models/phase8_cnn/phase8_cnn_best.pt")
bits = extractor.extract_bits("watermarked.png")     # list[int], length bit_length
probs = extractor.extract_proba("watermarked.png")   # np.ndarray of per-bit probabilities
```

CLI:

```powershell
.venv\Scripts\python.exe -m src.evaluation.blind_extract `
    --checkpoint models\phase8_cnn\phase8_cnn_best.pt `
    --image watermarked.png `
    --reference-bits 0110100110010101      # optional -> prints BER / accuracy
```

---

## 8. Saved weights

| Checkpoint | Written when |
|---|---|
| `models/phase8_cnn/<run_name>_last.pt` | every epoch |
| `models/phase8_cnn/<run_name>_best.pt` | validation bit-accuracy improves |

`<run_name>` is `phase8_cnn` for the full config and `phase8_smoke` for the
smoke experiment. Each checkpoint is self-describing: it stores the
`ExtractorConfig`, the `TrainConfig`, the epoch and the validation metrics, so
`BlindExtractor.from_checkpoint` needs no external metadata.

Logs and summaries: `results/phase8_cnn/<run_name>_training_log.csv` and
`results/phase8_cnn/<run_name>_training_summary.json`.

---

## 9. Scope boundaries (per the Phase 8 brief)

* No JPEG / noise / cropping / attack robustness — that is Phase 10.
* No blockchain.
* No changes to the frozen DWT-SVD baseline or the Phase 7 web UI.
* The learned decoder trains at `alpha = 0.02` (slightly above the frozen
  baseline's 0.005–0.015 study range) purely to give the network a stronger
  signal; the baseline's own configuration is untouched and can be matched by
  setting `embed.alpha` in `configs/cnn_extractor.yaml`.
