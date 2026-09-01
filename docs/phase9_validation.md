# Phase 9 — Baseline Validation & Robustness Preparation

**Status:** implemented, tested and run (2026-09-01).
**Relationship to earlier phases:** additive only. No file under `src/watermark/`,
`src/evaluation/metrics.py`, `src/app/`, `src/models/` or `src/training/` was
modified. The frozen Phase 6 DWT-SVD baseline and the trained Phase 8 blind CNN
extractor are *run*, never changed. `experiments/run_phase9_validation.py` asserts
the frozen Phase 6 contract before it does anything and records the SHA-256 of
every frozen file in its report.

---

## 1. Goal

Phase 9 is a **validation / experimental** phase, not a modelling phase. It:

1. Establishes a **reproducible, clean-channel (no-attack) reference** for the
   frozen DWT-SVD baseline over the held-out DIV2K **test** split, across the
   payload sizes the requirement calls out — **8, 16, 32, 64, 128 bits** — at the
   frozen alpha sweep **{0.005, 0.010, 0.015}**.
2. Reports every metric in two clearly separated families:
   * **image-quality:** PSNR, SSIM, MSE
   * **watermark-recovery:** BER, bit-accuracy, NC
3. Evaluates the **Phase 8 blind CNN extractor separately**, on the same unseen
   test images, at its trained operating point (64 bits, alpha = 0.02), with a
   like-for-like non-blind vs blind comparison. Its architecture is **not**
   touched.

The clean-channel numbers here are the reference that Phase 10 (attack
simulation) will measure degradation against — that is the "robustness
preparation" part of the phase name.

### Explicitly out of scope (deferred, by design)

| Not in Phase 9 | Where it belongs |
|---|---|
| JPEG / Gaussian noise / blur / crop attacks | Phase 10 |
| adaptive / content-aware embedding strength | Phase 11 |
| error-correcting codes (BCH/…) | Phase 16 |
| any change to the CNN architecture | Phase 15 |

`configs/phase9_validation.yaml` and `Phase9Config.__post_init__` enforce the
scope: payloads are capped at the frozen **LL-only** native capacity of 128 bits,
so the baseline runs exactly as in Phase 6 (no multi-subband overflow).

---

## 2. What was added

| File | Purpose |
|---|---|
| `src/evaluation/phase9_validation.py` | Pure, tested harness: metric taxonomy (`METRIC_GROUPS`), `Phase9Config` + `load_config`, deterministic payload generation, `assert_baseline_frozen`, per-image evaluation of the frozen baseline, separate CNN evaluation, aggregation. Imports the frozen baseline; never mutates it. |
| `configs/phase9_validation.yaml` | Reproducible Phase 9 config, separate from `baseline.yaml` and `cnn_extractor.yaml`. |
| `experiments/run_phase9_validation.py` | Single entry point. Writes all CSV/JSON + plots. `--quick` runs a 12-image / 3-payload / 1-alpha subset; `--no-cnn` skips the CNN half. |
| `tests/test_phase9_validation.py` | 28 tests: metric-taxonomy, config validation, deterministic payloads, the frozen-baseline guard (incl. a drift-detection test), data-dependent end-to-end baseline checks, and a CNN-dependent blind-vs-non-blind check. Data / checkpoint dependent tests skip when the inputs are absent. |
| `docs/phase9_validation.md` | This document. |

No existing source file was edited.

---

## 3. Method

### 3.1 Deterministic payloads

For image index `i`, a 128-bit master payload is drawn from
`numpy.random.default_rng(payload_seed + i)` (`payload_seed = 20260901`). An
`N`-bit payload is its first `N` bits, so payloads **nest** across sizes and are
identical on every run. This is a deliberate change from Phase 6's
`run_baseline.py` (which used one fixed payload per size, shared by all images):
per-image distinct payloads are a stronger validation and the nesting keeps the
size sweep comparable within an image.

### 3.2 Frozen DWT-SVD baseline (non-blind decoder)

For every `(alpha, payload)` pair the harness calls the **frozen**
`src.watermark.embed.embed` with
`EmbedConfig(wavelet="haar", mode="symmetric", subband="LL", extra_subbands=(),
start_sv_index=0, alpha=alpha, bit_length=N)` on each test image, then decodes
with the frozen non-blind `extract_traditional` (which compares the watermarked
and original singular values). One row per image records the image-quality group
(`quality_psnr/ssim/mse`) and the watermark-recovery group
(`recovery_ber/bit_accuracy/nc`); rows are aggregated to mean/std per
`(alpha, payload)` with the two groups kept structurally separate in the JSON.

### 3.3 Phase 8 blind CNN extractor (architecture unchanged)

The trained checkpoint `models/phase8_cnn/phase8_cnn_best.pt` (64-bit payload,
alpha = 0.02, ~87 K params) is loaded through the existing
`src.evaluation.blind_extract.BlindExtractor`. Each test image is watermarked
with the frozen `embed()` at the checkpoint's operating point, then decoded two
ways from the **same** watermarked image:

* **blind** — `BlindExtractor.extract_bits` (image only in), and
* **non-blind reference** — `extract_traditional` (original also in).

The test split (DIV2K IDs 0801–0900) was **not** used to train the Phase 8 model
(it trained on the train + validation splits), so this is a genuine
generalisation check.

---

## 4. Measured results (real experiment)

Command: `.venv\Scripts\python.exe experiments\run_phase9_validation.py`
Data: full DIV2K **test** split, **100 images**, 256×256, held out.
Environment: Python 3.12.10, NumPy 2.5.2, OpenCV 4.14.0, PyTorch 2.13.0+cpu
(CUDA false). Runtime ≈ 2 min 45 s on CPU.

Full data: `results/phase9_validation/`.

### 4.1 Frozen DWT-SVD baseline — image quality (mean over 100 images)

| alpha | PSNR (dB) | SSIM | MSE |
|---|---|---|---|
| 0.005 | 49.01 – 49.04 | 0.9986 | ~0.86 |
| 0.010 | 45.60 – 45.72 | 0.9982 | ~1.95 |
| 0.015 | 42.74 – 42.87 | 0.9975 | ~3.9 |

Payload size has negligible effect on image quality (the modulation energy sits
in a few singular values). Every operating point is visually lossless
(PSNR > 42 dB, SSIM > 0.997).

### 4.2 Frozen DWT-SVD baseline — watermark recovery, non-blind, clean channel (mean BER over 100 images)

| payload | alpha 0.005 | alpha 0.010 | alpha 0.015 |
|---|---|---|---|
| 8 bits | 0.0000 | 0.0000 | 0.0013 |
| 16 bits | 0.0081 | 0.0088 | 0.0156 |
| 32 bits | 0.0075 | 0.0166 | 0.0338 |
| 64 bits | 0.0266 | 0.0372 | 0.0616 |
| 128 bits | 0.1313 | 0.1216 | 0.1361 |

Corresponding bit-accuracy = `1 − BER`; NC = `1 − 2·BER` (both in the CSV/JSON).

Reading:

* **8-bit payloads are recovered essentially perfectly** at every baseline alpha;
  BER grows smoothly as the payload reaches deeper into the singular spectrum.
* Around 128 bits the LL subband is saturated — its deepest ~30–40 singular
  values are `O(1–10)` and their ordering is not stable through the
  `IDWT → uint8 → DWT → SVD` round trip — so ~12–14 % of those bits flip
  regardless of alpha.
* Larger alpha slightly **worsens** BER at high payloads: a bigger perturbation
  of the leading singular values adds more reconstruction noise onto the fragile
  trailing ones.
* These numbers reproduce the Phase 6 characterisation (which used the first 40
  test images) on the **full 100-image** held-out split — same shape, same story,
  slightly higher absolute BER from the larger, more varied sample.

### 4.3 Phase 8 blind CNN extractor vs non-blind reference (100 images, 64 bits, alpha = 0.02)

| decoder | BER | bit-accuracy | NC |
|---|---|---|---|
| non-blind (reference) | 0.1013 | 0.8988 | 0.7975 |
| **blind CNN (Phase 8)** | **0.2220** | **0.7780** | **0.5559** |

* The blind CNN on the **unseen test split** (BER 0.222 / bit-acc 0.778) matches
  its Phase 8 **validation** result (BER 0.224 / bit-acc 0.776) almost exactly —
  it generalises to images it never saw, no overfitting.
* The non-blind reference is stronger at this operating point but not perfect:
  64-bit recovery at alpha = 0.02 already sits in the fragile part of the
  spectrum (cf. the 64-bit row of 4.2 trending upward with alpha).
* The ~0.22 blind-BER plateau is the documented Phase 8 finding: trailing
  singular values are fragile through the round trip, so BER rises with payload
  depth. Closing that gap (LR schedule, leading-bit-weighted loss, deeper model)
  is Phase 15, not Phase 9.

### 4.4 Plots (`results/phase9_validation/plots/`)

| File | Shows |
|---|---|
| `psnr_vs_payload.png` | PSNR vs payload, one line per alpha (imperceptibility is flat in payload). |
| `ber_vs_payload_nonblind.png` | non-blind BER vs payload, one line per alpha (the capacity knee at 128 bits). |
| `quality_vs_recovery.png` | image-quality (PSNR) vs watermark-recovery (bit-accuracy) trade-off, points labelled by payload. |
| `nonblind_vs_blind_ber.png` | non-blind vs blind CNN BER at the 64-bit / alpha 0.02 operating point. |

---

## 5. Output files (`results/phase9_validation/`)

| File | Content |
|---|---|
| `phase9_baseline_per_image.csv` | 1500 rows = 3 alphas × 5 payloads × 100 images; image-quality and watermark-recovery columns. |
| `phase9_baseline_summary.csv` | 15 rows = mean/std per `(alpha, payload)`, flat columns. |
| `phase9_baseline_summary.json` | Same summary with the two metric groups as nested objects, plus the evaluated image IDs. |
| `phase9_cnn_per_image.csv` | 100 rows; blind and non-blind recovery per image at the CNN operating point. |
| `phase9_cnn_summary.json` | CNN operating point + blind vs non-blind aggregates. |
| `phase9_nonblind_vs_blind.csv` | 2-row head-to-head table. |
| `phase9_report.json` | Combined report: full config, environment stamp, **SHA-256 of every frozen Phase 6 file**, both summaries, artifact paths, runtime. |
| `plots/*.png` | The four figures above. |

`results/**` and `models/**` are git-ignored (as for Phases 6 and 8); the code,
config and this doc are what is committed.

---

## 6. Reproduce

```powershell
.venv\Scripts\python.exe experiments\run_phase9_validation.py           # full: 100 images, ~3 min CPU
.venv\Scripts\python.exe experiments\run_phase9_validation.py --quick   # 12 images, payloads [8,32,128], alpha [0.010]
.venv\Scripts\python.exe experiments\run_phase9_validation.py --no-cnn  # baseline only
```

Deterministic: payloads come from `numpy.random.default_rng(payload_seed + i)`
with `payload_seed` from `configs/phase9_validation.yaml`. Re-running overwrites
`results/phase9_validation/` with identical numbers.

---

## 7. Tests

`tests/test_phase9_validation.py` (28 tests):

* metric taxonomy is exactly the two required families and they are disjoint;
* `Phase9Config` rejects empty/negative/over-capacity payloads and bad alphas;
* `load_config` reads the repo YAML and honours `--quick` overrides;
* **frozen-baseline guard**: `assert_baseline_frozen()` passes on this repo and
  *fails* when `BASELINE_ALPHAS` or the `1 ± alpha` modulation rule drifts;
  `frozen_file_hashes()` covers every frozen file;
* deterministic-payload nesting and reproducibility;
* data-dependent: per-image row shape carries both metric groups; 8-bit payloads
  recover at BER 0.0 at every baseline alpha; `evaluate_baseline_point` is
  bit-for-bit reproducible; `summarise_baseline` keeps the groups separate;
* CNN-dependent: `evaluate_cnn_point` produces both blind and non-blind recovery,
  the non-blind reference beats the blind CNN, and the blind CNN is clearly above
  chance.

Full suite after Phase 9: **176 passed** (was 148).
