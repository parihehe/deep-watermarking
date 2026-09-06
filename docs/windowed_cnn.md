# Windowed 1D-CNN blind extractor (experimental)

**Status: EXPERIMENTAL with an opt-in app switch.** The decoder is NOT the
default anywhere: the app uses it only when run with
`DECODER_MODE=windowed_cnn` (see [App integration](#app-integration)), and it
must independently pass the project's own evaluation on the project's own
embedding pipeline before it may *replace* anything. The production Phase 8
decoder and the Phase 18 registry-ID app behaviour are untouched by default.

## Motivation

The shipped 64-bit blind CNN plateaus around 0.76–0.78 bit accuracy with **0
exact-payload matches** on real DIV2K (see `docs/blind_extraction_diagnosis.md`,
~22.2 % bit error rate). The paper under study predicts **one bit at a time**
from a **fixed local window of 15 singular values** centred on the singular
value that carries that bit. The hypothesis is that judging each bit against its
*smooth local spectrum context* — rather than squeezing the whole spectrum
through one shared CNN head — yields tighter per-bit decisions.

## How this decoder differs from the Phase 8 CNN

| | Phase 8 production CNN | Windowed experiment |
|---|---|---|
| Input | whole standardised log1p spectrum (one tensor/image) | one 15-value window per bit |
| Decision | one logit per bit, single forward pass | one logit per bit, per-bit forward pass |
| Network | 1D CNN (k=7) → per-bit heads | Conv1d(1→32,k=3) → Conv1d(32→64,k=3) → dropout(0.3) → Dense(64) → dropout(0.5) → Dense(1) |
| Bit alignment | single spectrum position | window centred on `start_sv_index + bit_index` |
| Params | ~? (Phase 8) | ≈ 68,001 (paper-scale, per the windowed config) |

Non-negotiable shared physics: **both decoders consume the exact same feature
path** — BT.601 luma of the watermarked RGB image → DWT-LL → SVD →
`log1p` singular values — so any accuracy gap is attributable to the windowing +
per-bit network, not to different features.

## The authoritative bit → window mapping

The frozen Phase 6 embedder (`src/watermark/embed.py`) writes bit `i` into
singular value `start_sv_index + i` of the luminance DWT-LL subband
(`S'[i] = S[i] * (1 + alpha * (2b[i] - 1))`). The windowed decoder centres its
15-value window on that same index. The single source of truth for
`bit_index -> window` is `bit_window_singular_values` in
`src/models/windowed_cnn.py`, used identically by the dataset generator and the
live extractor, so they cannot drift apart. Boundary truncation is handled by
**edge-padding** so every window is exactly 15 values (fixed output kept the
batches stackable).

## Files

- `src/models/windowed_cnn.py` — `WindowedCNNConfig`, `WindowedCNNExtractor`,
  the shared feature path (`luminance_ll_singular_values`,
  `bit_window_singular_values`) and checkpoint I/O.
- `src/training/windowed_dataset.py` — cover → (watermarked cover → per-bit
  windows, ground-truth bits) on the fly through the frozen embedder.
- `src/training/train_windowed_cnn.py` — training loop + CLI + resume.
- `src/evaluation/windowed_extract.py` — blind per-bit inference wrapper/CLI.
- `configs/windowed_cnn.yaml` — experimental config (alpha 0.02, 64-bit,
  window 15).
- `experiments/run_windowed_cnn_eval.py` — same-image, same-payload comparison
  of windowed vs production decoders (bit-acc / BER / exact-match / confidence /
  PSNR / SSIM / inference time).
- `training/colab_train_decoder.py` — the single pipeline (data → cache → train
  → eval → package); used by both the local runner (via `--processed`) and the
  optional Colab notebook.
- `scripts/ingest_div2k_stream.py` — stream-extracts the official archives into
  `data/processed/div2k_256` without ever writing the full-resolution files.
- `training/colab_windowed_cnn_training.ipynb` — optional Google-Colab GPU
  notebook over the same pipeline (KaggleHub / official-mirror data sources).
- `src/evaluation/decoder_loader.py` — `DECODER_MODE` selection between the
  production (`"current"`) and windowed (`"windowed_cnn"`) decoders.
- `tests/test_windowed_cnn.py`, `tests/test_decoder_loader.py` — unit + e2e
  tests including the blind end-to-end recovery path.
- `docs/windowed_cnn.md` — this document.

## Quick start

```bash
# train (50 epochs, paper references)
python -m src.training.train_windowed_cnn --config configs/windowed_cnn.yaml

# resume
python -m src.training.train_windowed_cnn --config configs/windowed_cnn.yaml \
    --resume models/experimental/windowed_cnn/windowed_cnn_last.pt

# evaluate vs production on the project's own test split
python experiments/run_windowed_cnn_eval.py --bit-length 64 --n-images 100
```

`best.pt` is persisted on every epoch whose validation bit-accuracy beats the
running best (and always on the first epoch, so it exists even for a
chance-level run). `last.pt` is written after the best-update each epoch so
`--resume` inherits correct best metadata; on resume the true best is read from
the surviving `best.pt`, never trusted to `last.pt`.

## Training pipeline (runs locally, no GPU needed)

The real-data training runs on this machine (Apple Silicon MPS; CPU also
works) from the DL-ready DIV2K already processed by the project:

```bash
# 1) (once) stream-ingest the official archives -> data/processed/div2k_256
#    reads straight from data/raw/archives/*.zip, resizes to 256px, writes the
#    700/100/100 split - no full-resolution files ever touch disk.
.venv/bin/python scripts/ingest_div2k_stream.py

# 2) (once) run the real 50-epoch train + honest eval + packaging
.venv/bin/python training/colab_train_decoder.py --project-root . --processed
```

`--processed` reads `data/processed/div2k_256/{train,validation,test}` and feeds
`make_split` exactly like the downloader does; `--smoke` + `--fake` still rebuild
a tiny synthetic tree for fast pipeline checks without the real data. Produced
artifacts (`windowed_cnn_best.pt`, `windowed_cnn_last.pt`, `normalization.npz`,
`decoder_config.json`, `training_config.json`, `dataset_split.json`,
`metrics.json`, `training_history.csv`, packaged as
`windowed_cnn_model_package.zip`) are copied into
`models/experimental/windowed_cnn/`.

The pipeline is **self-honest by construction**:

1. **Real encoder.** Every training window comes from the frozen Phase 6
   `embed()` + the same `bit_window_singular_values` mapping used at inference.
2. **Real codec payloads.** Payloads are `id_registry.encode_id_bits(id)` — the
   exact physical bits the ownership flow embeds.
3. **Split = 700 / 100 / 100.** Released DIV2K has 900 images (800 train + 100
   valid); the master-brief "800/100/100" needs 1000, so the project's split is
   used: train 0001-0700, validation 0701-0800, test 0801-0900 (the official
   valid set — the same test the local project uses).
4. **Cached windows.** Embed-then-window is a separate cached stage (`npz` +
   config hash), so re-runs reuse the arrays. Images are 256×256 (the
   extractor's operating resolution; the embed then window extraction matches
   live inference exactly).
5. **Honest metrics only.** Test BER, exact registry-ID recovery (via the real
   `decode_id_bits`), false-positive rate on never-watermarked test pixels, and
   robustness (PNG / JPEG / blur). No numbers are fabricated.

### Current trained result (64-bit, DIV2K 256px, 50 epochs)

| metric | value |
|---|---|
| best validation bit-acc (epoch 49) | 0.784 |
| test bit-accuracy / BER / F1 | 0.777 / 0.223 / 0.758 |
| exact registry-ID recovery on test | 96 / 100 |
| certified ownership matches @ 0.78 | 10 / 100 |
| false positives on clean images | 0 / 100 |
| robustness mean BER (PNG / JPEG q90 / blur×3) | 0.205 / 0.221 / 0.383 |

Every number above comes from `metrics.json` produced by the project's own
evaluation on the held-out official-valid test set (0801-0900). The architecture
is deliberately small and the alpha operating point is shared with the
production pipeline; it does **not** beat the production Phase 8 CNN — this
decoder is an experimental candidate, not a replacement (see
[Acceptance bar](#acceptance-bar-unchanged-from-project-rules)).

## Blind-extraction fix (2026-09-06): native-resolution decoding + soft decode

Blind verification in the app was returning wrong IDs at ~chance bit accuracy.
Two decoder-side defects, both fixed; **the embedder was not changed**.

**1. The decoder rescaled its input (root cause).** `WindowedExtractor` resized
every image to `image_size` (256) before reading the singular values. But
`embed()` writes bit *i* into singular value *i* of the LL sub-band **at the
image's own resolution**, and DWT-SVD singular values are not resize-invariant —
rescaling remaps the whole spectrum, so the decoder read the wrong values.
256×256 covers were unaffected (no resize happened), which is why the offline
evaluation — run entirely on the 256×256 processed split — never saw it, while
every real user upload at any other size was corrupted. Measured on DIV2K
covers embedded at their native size:

| cover | rescaled to 256 (old) | native resolution (fixed) |
|---|---|---|
| 256×256 | BER 0.225, ID 15/15 | BER 0.225, ID 15/15 |
| 512×512 | BER 0.365, ID 8/15 | BER 0.242, ID 14/15 |
| 800×600 | BER 0.365, ID 9/15 | BER 0.235, ID 14/15 |
| 1024×768 | BER 0.359, ID 8/15 | BER 0.239, ID 15/15 |

**2. Hard majority voting discarded the decoder's confidence.**
`decode_id_bits` thresholded the probabilities to 0/1 first, so a 0.51 copy and
a 0.99 copy counted the same. `decode_id_bits_soft` sums log-likelihood ratios
over the 15 copies instead (additive; the hard codec is unchanged and still used
by the non-blind path). Held-out DIV2K, 1680 trials over three resolutions:

| | exact ID | gate coverage | wrong decodes shown |
|---|---|---|---|
| hard vote, gate 0.78 | 0.96–0.975 | 0.58 | 11 |
| soft LLR, gate 0.9999 | **0.980–0.984** | **0.79** | **0** |

Because the soft confidence is a posterior (saturates near 1) and the hard one is
a vote fraction, they need different thresholds — hence
`RELIABLE_REGISTRY_ID_POSTERIOR` alongside the untouched
`RELIABLE_REGISTRY_ID_CONFIDENCE`. This is a strict improvement on both axes,
not a relaxed bar.

### Known ceiling (not a bug)

Raw 64-bit BER stays ~0.19–0.24 and **cannot approach 0** at the frozen
α = 0.02 operating point: the embedded sign survives the IDWT + uint8 round trip
only **89.6 %** of the time (measured), so ~0.90 bit accuracy is the hard ceiling
for *any* blind decoder here, and the non-blind decoder measures ~0.90 too.
Exact payload recovery comes from the 15× repetition code on top, which is why
registry-ID recovery is ~98 % while raw BER is ~0.22. A tiny-set overfit probe
also showed the standardised window feature is the accuracy bottleneck (train
bit-acc 0.965 standardised vs 0.994 with a locally-detrended window, centre-value
AUC 0.740 → 0.796), so a detrended feature is the most promising lever if the
decoder is ever retrained.

## Google-Colab alternative (optional)

A cloud-GPU (T4/A100) path exists for machines without MPS/CPU appetite for the
full run: `training/colab_windowed_cnn_training.ipynb` runs the same
`colab_train_decoder.py` (DIV2K via KaggleHub `soumikrakshit/...`, or the
official archives), and its cells are self-contained / resume-safe across
kernel restarts. The pipeline and artifacts are identical.

## App integration (opt-in)

`src/evaluation/decoder_loader.py` selects the blind decoder:

```bash
.venv/bin/python scripts/run_app.py            # production Phase 8 CNN (default)
DECODER_MODE=windowed_cnn .venv/bin/python scripts/run_app.py   # windowed CNN
```

The windowed decoder is used only when its declared bit-length matches the
payload width; an unavailable or width-mismatched checkpoint falls back to the
production CNN with a visible `decoder_fallback` reason in the API response, and
`/final-model/info` reports `decoder_mode` + `windowed_cnn.checkpoint_available`.
The embedding path is never affected by this switch.

## Acceptance bar (unchanged from project rules)

The experiment runs the project's own evaluation protocol on the project's own
test split (`data/processed/div2k_256/test`), at alpha 0.02 haar/LL with 64-bit
payloads. It is only a candidate replacement if it **outperforms** the existing
Phase 8 CNN on the same images and payloads (bit-acc / BER / exact-match), and
still not without re-running the registry-confidence calibration
(`docs/phase18_id_registry.md` S5) before any app integration.