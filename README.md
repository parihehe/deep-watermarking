# Deep Watermarking

Research framework for **invisible, blind image watermarking** based on a reproducible **DWT–SVD–CNN** pipeline. It embeds a secret payload into an image so that the payload can later be read back **without** needing the original image (blind extraction) or a neural-network lookup — all using the same frozen embedder.

This repository is self-contained; it is intentionally independent from any other application in the parent workspace.

---

## What this project does

```text
Input image + secret payload
        │
        ▼
┌──────────────────────────────┐
│ 1. Embed (frozen DWT-SVD     │
│    multiplicative rule)      │
└──────────────────────────────┘
        │  watermarked image (visually ~ identical)
        ▼
┌──────────────────────────────┐
│ 2. Decode — one of three    │
│    extractors:               │
│    • non-blind (needs        │
│      original image)         │
│    • Phase 8 blind CNN       │
│    • windowed 1D-CNN         │
│      (experimental decoder)  │
└──────────────────────────────┘
        │
        ▼
   recovered payload / registry ID
```

In one sentence: **stamp an image with an invisible ID, then read the ID back — with or without the original — and measure how much the image changed.**

---

## Why blind watermarking?

Watermarking lets a content owner prove ownership or trace distribution of an image.

- **Non-blind extraction** compares the watermarked image against the *original* image. It is accurate but useless if the original is lost — in practice the original is often what was stolen.
- **Blind extraction** reads the payload from the watermarked image alone. That is the practically interesting case (marketing-use analytics, provenance tracking, leak tracing), but it is harder because the decoder must separate the hidden signal from the image content itself.

This project ships a frozen, classical **DWT–SVD** embedder and evaluates **neural blind decoders** against it. No "production-grade" or deployment guarantees are claimed anywhere in this document; everything here is research-grade.

---

## Architecture

```mermaid
flowchart LR
    subgraph Embed["Embedding (frozen, src/watermark/embed.py)"]
        I[RGB image] --> Y[YCbCr - use Y luminance]
        Y --> DW[DWT 1-level - wavelets like haar]
        DW --> LL["LL sub-band"]
        LL --> SV[SVD - singular values S]
        B[payload bits b_i] --> M["S' = S * (1 + alpha * (2b_i - 1))"]
        SV --> M
        M --> IDWT[IDWT + convert back]
        IDWT --> W["watermarked image"]
    end
    W --> Ext
    subgraph Ext["Extraction"]
        Ext["DWT + SVD on watermarked image"] --> NB["non-blind: residual vs original"]
        Ext --> CNNW["Phase 8 CNN (blind)"]
        Ext --> WC["windowed 1D-CNN (blind, experimental)"]
    end
    NB --> RES["payload / registry ID"]
    CNNW --> RES
    WC --> RES
```

### Embedding pipeline (the fixed rule)

Embedding never changes across decoders — it is the ground-truth signal the decoders learn:

1. Convert RGB to **YCbCr** and keep the luminance channel `Y`.
2. Apply a **1-level DWT** (default wavelet `haar`, symmetric mode); embed in the **LL** sub-band by default.
3. Compute the **SVD** of LL. For each payload bit `b_i`, scale the `i`-th singular value multiplicatively:

   ```
   S'[i] = S[i] * (1 + alpha * (2*b[i] - 1))
   ```

4. Rebuild LL via ISVD, invert the DWT, and convert back to RGB.

The final model uses `alpha = 0.020`, `bit_length = 64`, single-level LL, no error-correcting code (an ECC module exists at `src/watermark/ecc.py` but is **not** part of the final model).

### Payload generation

Supported payload sources (`src/app/service.py`):

| Source | Meaning |
|---|---|
| `message` | Free text, UTF-8, length-prefixed, repeated (odd, up to 9x) and majority-voted at decode |
| `text` | SHA-256 digest of a text string |
| `uuid` | A UUID serialized to bits |
| `random` | A random UUID |
| `bits` | An explicit 0/1 string (zero-padded to `bit_length`) |

### Registry IDs

The registry (in-memory, `src/watermark/id_registry.py`) supports a **4-bit ID** code that is **interleaved 15 times** and **XOR-masked** so a handful of bit errors can be corrected by the repetition. Decoding a registry ID returns `(id, mean_conf, min_conf)` where `mean_conf` is the average per-slot confidence across the repeats. A decode is treated as **reliable only when `mean_conf >= 0.78`** (`RELIABLE_REGISTRY_ID_CONFIDENCE`). There is **no database** — the registry is an in-memory codec with a fixed ID table.

```
payload bits ──► 4-bit ID ──► interleave x15 ──► XOR mask ──► embed
extract ──► un-mask ──► de-interleave ──► confidence vote ──► (id, mean_conf, min_conf)
```

### Extraction pipeline

- **Non-blind decoder** (`src/app/final_model.py`): needs both the watermarked and the original image; yields near-perfect bit recovery and is the reference for how well embedding worked at all.
- **Phase 8 blind CNN**: the original neural blind decoder. **Note: its checkpoint is not present in this repo** (`models/phase8_cnn/` is missing), so it cannot currently be served; the loader reports `checkpoint_available=false`.
- **Windowed 1D-CNN** (experimental, opt-in): predicts one bit at a time from a **fixed local window of 15 DWT-LL singular values** centered on the singular value carrying that bit. Enabled with the environment variable `DECODER_MODE=windowed_cnn`.

---

## The CNN models

Both neural decoders are **PyTorch**.

### Phase 8 CNN (default blind decoder)

Default decoder in the app, **but no checkpoint ships with this repo today** — see **Known limitations**.

### Windowed 1D-CNN (experimental)

A small per-bit network (~68k params) shared by all bits, whose input is the 15-value singular-value window around the target bit:

```
input: window of 15 log1p-singular values
  └─► Conv1d(1 → 32, kernel 3, same) + ReLU
  └─► Conv1d(32 → 64, kernel 3, same) + ReLU
  └─► Dropout(0.3)
  └─► Flatten ──► Dense(64, ReLU)
  └─► Dropout(0.5)
  └─► Dense(1) + sigmoid  →  P(bit = 1)
```

Config (`configs/windowed_cnn.yaml`): `window_size 15`, `image_size 256`, `alpha 0.02`, `bit_length 64`, `epochs 50`, `batch_size 64`, `lr 1e-3`, `weight_decay 1e-4`, `grad_clip 1.0`, `early_stop_patience 10`, `seed 20260906`.

Feature path (shared with the Phase 8 decoder): **luminance → DWT-LL → log1p SVD**, so the only difference between the decoders is the windowing hypothesis, not the features.

---

## Dataset: DIV2K

- Official **DIV2K** (900 released images).
- Ingested by `scripts/ingest_div2k_stream.py` directly from the official HR archives to `data/processed/div2k_256` at **256×256** (`INTER_AREA`), without ever writing full-res PNGs. Result: **900 files, ~117 MB**.
- Split: **700 train** (`0001–0700`), **100 validation** (`0701–0800`), **100 test** (`0801–0900`, the official VALID set).

Each training sample is synthesised **on the fly** by the frozen embedder: pick a DIV2K image, generate a random registry-style payload, run the real `embed()` at `alpha = 0.02`, compute the per-bit 15-value windows, and label each window with its bit. The model therefore learns the exact signal the production system embeds — no simulated data.

---

## Training & results

### How to train (local, macOS — MPS)

```bash
# 0) env (Python 3.12)
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[training,tracking,dev]"

# 1) stream-ingest the official DIV2K archives -> data/processed/div2k_256
python scripts/ingest_div2k_stream.py

# 2) real 50-epoch training + honest evaluation + artifact packaging
python training/colab_train_decoder.py --project-root . --processed

# 3) (optional) cloud-GPU alternative via Google Colab
python scripts/build_colab_payload.py -o /tmp/deep_watermarking_colab.zip
#    then run training/colab_windowed_cnn_training.ipynb in Colab
```

The training script auto-selects `cuda → mps → cpu` and writes real metrics to `models/experimental/windowed_cnn/metrics.json`.

### Measured results (windowed 1D-CNN, test split, real runs)

| Metric | Value | Source |
|---|---|---|
| Test bit accuracy | **0.7778** | `metrics.json` `test_bit_level.bit_accuracy` |
| Test BER | **0.2222** | `metrics.json` `test_bit_level.ber` |
| Test precision / recall / F1 | 0.7741 / 0.7437 / 0.7586 | `metrics.json` |
| Best validation bit accuracy | **0.7842** (epoch 49) | `metrics.json` |
| Exact registry-ID recovery | **96 / 100** | `metrics.json` `full_payload` |
| Registry decodes passing the 0.78 threshold | **7 / 100** | `metrics.json` (i.e. most decodes fall just below the calibrated threshold) |
| False-positive registrations on clean images | **0 / 100** | `metrics.json` `false_positives` |
| Robustness — mean BER, PNG re-encode (n=8) | 0.205 | `metrics.json` `robustness` |
| Robustness — mean BER, JPEG q90 (n=8) | 0.221 | `metrics.json` `robustness` |
| Robustness — mean BER, gentle blur k3 (n=8) | **0.383** | `metrics.json` `robustness` |

Full suite: **437 tests pass, 2 skipped** (the 2 skips are Phase 8 CNN tests awaiting its checkpoint).

Anything not listed above (e.g. known-class accuracy, other robustness transforms, throughput) is **not reported in the current implementation**.

### What the metrics mean

- **PSNR / SSIM / MSE** — image-quality: how much the watermarked image differs from the original. The web app's non-blind reference decoder reports these per image (values depend on `alpha`, image content and payload; e.g. alpha 0.02 on typical content gives ≈ 40 dB / ≈ 0.9997 / MSE ≈ 7 — impressions only, not a benchmark).
- **BER** — bit-error rate: fraction of the 64 recovered bits that are wrong (0.0 = perfect).
- **Bit accuracy** — `1 − BER`.
- **Normalized correlation (NC)** — similarity between inserted and recovered payloads (reported by the non-blind embed route).
- **Registry confidence** — `mean/min_conf` of the repeated ID vote; decodes below 0.78 are shown as unreliable.

### The alpha trade-off

| `alpha` | Effect |
|---|---|
| Higher (e.g. 0.05–0.1) | More robust extraction, but more visible distortion (lower PSNR/SSIM) |
| Lower (e.g. 0.005) | Nearly invisible but bits flip more easily |

- Frozen embedder default in `src/watermark/embed.py` and the trained CNN: **`alpha = 0.020`**.
- The general web-app API default is **`alpha = 0.010`** (`src/app/main.py`) — this is an API default, not the trained model's operating point. Always pass `alpha` explicitly when you depend on it.

---

## App: API + web UI

FastAPI app served from `src/app/main.py` (UI is static **HTML/CSS/JS**, jQuery-backed; no build step). Run:

```bash
uvicorn src.app.main:app --reload --port 8000     # or
python scripts/run_app.py --port 8000
```

Open <http://127.0.0.1:8000/>.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/api/health` | Status + torch/cuda info |
| GET | `/api/options` | Supported wavelets/subbands/bit-lengths/alphas/payload sources |
| POST | `/api/watermark/embed` | Classic embed (alpha, wavelet, subband, extra_subbands, payload source) → quality + recovery metrics, PNG data URIs |
| GET | `/api/final-model/info` | Decoder status (Phase 8 vs windowed CNN, checkpoint availability) |
| POST | `/api/final-model/embed` | Final-model embed (registry-ID / message payloads) |
| POST | `/api/final-model/extract/blind` | Blind extraction — watermarked image only |
| POST | `/api/final-model/extract/nonblind` | Non-blind extraction — watermarked + original |

### UI panels

- **Embed panel**: payload source (message / text / uuid / bits / random), alpha, wavelet, subband, extra overflow subbands, live PSNR/SSIM/MSE, and a **"Use in Verify →"** button that carries the payload into the verify panel.
- **Verify panel**: dropzone for the watermarked image, registry-id + mean/min confidence, BER, active decoder + fallback reason, ownership/registry-match verdict, and hash-based routing of the image.
- Extraction routes report exactly how well they did (`status`: recovered / partial / failed), never hiding errors.

---

## Configuration

- `configs/windowed_cnn.yaml` — windowed-CNN training/config (embed, payload, data, model, train, output).
- Model files live under `models/experimental/windowed_cnn/`:

  | File | What it is |
  |---|---|
  | `windowed_cnn_best.pt` | Best-epoch checkpoint (epoch 49) |
  | `windowed_cnn_last.pt` | Last-epoch checkpoint |
  | `normalization.npz` | Feature normalization stats |
  | `decoder_config.json` | Bit length, window size, threshold (0.78) |
  | `training_config.json` | Resolved training hyperparameters |
  | `dataset_split.json` | Train/validation split provenance |
  | `metrics.json` | Honest test metrics (cited above) |
  | `training_history.csv` | Per-epoch training/validation history |

---

## Repository layout

```text
configs/       reproducible experiment configurations (windowed_cnn.yaml …)
data/          datasets (data/processed/div2k_256 - not committed)
docs/          research & operational docs (baseline, phase8, windowed_cnn, dataset, …)
experiments/   experiment definitions + calibration scripts
models/        checkpoints (experimental/windowed_cnn/…; phase8 checkpoint absent)
results/       run outputs (windowed_cnn/)
scripts/       ingest_div2k_stream.py, run_app.py, build_colab_payload.py …
src/
  app/         FastAPI + static web UI (service.py, main.py, final_model.py)
  evaluation/  blind_extract, metrics, attacks, decoder_loader, windowed_extract, phase*
  models/      windowed_cnn.py, cnn_extractor.py
  training/    datasets + trainer scripts
  utils/       dataset_pipeline.py
  watermark/   FROZEN embed.py, dwt.py, svd.py, id_registry.py, ecc.py …
tests/         437 passing tests (2 skipped)
training/      colab_train_decoder.py, colab_windowed_cnn_training.ipynb
```

---

## Honest limitations (read this first)

- **Phase 8 CNN checkpoint is not shipped** — the default blind decoder reports `checkpoint_available=false`. Use `DECODER_MODE=windowed_cnn` for a blind extractor.
- The **windowed CNN does not yet beat** the Phase 8 results it was meant to replace; it is experimental.
- Registry decodes mostly sit **below** the calibrated 0.78 reliability threshold (7/100 pass it), and **registry-ID reliability is not a mathematical guarantee** — the threshold is calibrated, not proven. Before registering more than the current few messages, re-run `experiments/calibrate_registry_confidence.py` (see `docs/phase18_id_registry.md`).
- **Robustness is weak** under blur (mean BER 0.383 for a single 3×3 blur) and only mild transforms were tested.
- No ECC is applied in the final model (`src/watermark/ecc.py` is unused in it).
- No production claims: single dataset (DIV2K), single resolution (256×256), no throughput/scale testing.

## Future work (not implemented)

- Phase 8 CNN checkpoint recovery / retraining so the default blind decoder is actually servable.
- ECC integration for the final model (the codec exists at `src/watermark/ecc.py`).
- Stronger augmentation during synthesis (crop/JPEG/blur) to harden the decoder.
- Comparing windowed-CNN vs Phase 8 bit-for-bit under identical conditions.
- Everything listed here is **planned, not implemented**.

---

## Reproducibility & troubleshooting

- Python **3.12 only** (`>=3.12,<3.13`); PyTorch wheel must match your machine (CPU fine, CUDA for cloud GPU, MPS on Apple Silicon).
- No host datasets, checkpoints, result images, or secrets are committed to Git.
- Every experiment should run from a committed YAML config and fixed seed.
- If a checkpoint/registry errors with HTTP 503, that is the app correctly reporting a missing artifact — check `GET /api/final-model/info`.
- Train with MPS locally may be slow; the Colab notebook (`training/colab_windowed_cnn_training.ipynb`) is the cloud-GPU alternative and is resume-safe mid-run.

## Commands cheat-sheet

```bash
uvicorn src.app.main:app --reload --port 8000        # run web app
python scripts/ingest_div2k_stream.py                 # build processed DIV2K split
python training/colab_train_decoder.py --project-root . --processed   # train + evaluate
DECODER_MODE=windowed_cnn uvicorn src.app.main:app --port 8000        # serve windowed decoder
pytest                                                 # 437 passed, 2 skipped
```

See `docs/windowed_cnn.md`, `docs/phase18_id_registry.md`, and `docs/webapp.md` for deeper write-ups. All metrics in this README trace to `models/experimental/windowed_cnn/metrics.json` or the test suite.