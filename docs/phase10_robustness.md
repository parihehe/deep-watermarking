# Phase 10 — Attack Simulation / Robustness Testing

**Status:** implemented, tested and run (2026-09-01).
**Relationship to earlier phases:** additive only. `git diff` on `src/watermark/`,
`src/evaluation/metrics.py`, `src/evaluation/blind_extract.py`,
`src/evaluation/phase9_validation.py`, `src/models/`, `src/training/`, `src/app/`
and every earlier config is empty. The frozen Phase 6 baseline, the Phase 7 web
UI and the Phase 8 CNN are *run*, never modified.
`experiments/run_phase10_robustness.py` asserts the frozen Phase 6 contract,
records the SHA-256 of every frozen file, and finishes with a sanity block
confirming the Phase 7 route and the Phase 8 checkpoint still load and run.

---

## 1. Goal

Measure how well the watermark survives real-world channel and adversarial
distortions, for **both** decoders, against the Phase 9 clean-channel reference.

1. Watermark each held-out DIV2K **test** image once, at the Phase 8/9 operating
   point — **64 bits, alpha = 0.02** (the frozen Phase 6 formula and all its
   structural defaults are unchanged; only `alpha` and `bit_length` are set on
   `EmbedConfig`, exactly as Phases 8 and 9 already do, so the non-blind and
   blind decoders see identical images).
2. Apply each attack at several configurable severities.
3. Recover the payload two ways from every attacked image:
   * **non-blind** — frozen `extract_traditional` (original cover also in), and
   * **blind** — Phase 8 `BlindCNNExtractor` (image only in).
4. Record the **image-quality group** (PSNR / SSIM / MSE — attacked vs clean
   watermarked image) and the **watermark-recovery group** (BER / bit-accuracy /
   NC) for both decoders, plus the delta against the no-attack baseline.

### Deliberately out of scope (kept separate)

| Not in Phase 10 | Where it belongs |
|---|---|
| any change to the frozen Phase 6 formula / defaults | — (frozen) |
| CNN architecture improvement | Phase 15 |
| adaptive / content-aware embedding | Phase 11 |
| high-capacity changes | Phase 12 |
| wavelet / subband studies | Phases 13–14 |
| error-correcting codes | Phase 16 |
| geometric re-synchronisation / registration of attacked images | later (sync) |

Phase 10 measures the **raw** damage. The only preprocessing before extraction
is a resize back to 256×256 so array shapes line up (`resynchronise`); no
attack is inverted, no rotation angle or crop offset is estimated or undone.

---

## 2. What was added

| File | Purpose |
|---|---|
| `src/evaluation/attacks.py` | Nine pure attack primitives + a registry. Imports nothing from the watermarking pipeline. Every attack: RGB uint8 → RGB uint8, same source shape. |
| `src/evaluation/phase10_robustness.py` | Harness: `Phase10Config` + `load_config`, `build_embed_config` (frozen LL-only), `prepare_samples` (embed once per image), `resynchronise`, `recover_nonblind` / `recover_blind`, `evaluate_attack_point`, `summarise` (two metric groups + baseline deltas). Reuses the Phase 9 helpers (`payload_bits`, `load_split_images`, metric taxonomy, frozen-file hashes). |
| `configs/phase10_robustness.yaml` | Reproducible attack grid + operating point + output config. Separate from every earlier config. |
| `experiments/run_phase10_robustness.py` | Single entry point. `--quick` (10 images, 2 severities/attack), `--no-cnn`, `--eval-num-images N`. Writes all CSV/JSON + plots + a Phase 7/8 sanity block. |
| `tests/test_attacks.py` | 42 tests: output contract, per-attack behaviour (monotone distortion, seed determinism, kernel coercion, validation), dispatch, severity. |
| `tests/test_phase10_robustness.py` | 24 tests: config validation, frozen embed-config, resynchronise, aggregation + deltas, and data/CNN-dependent end-to-end checks (mild JPEG ≈ clean, 45° rotation breaks it, determinism). |
| `docs/phase10_robustness.md` | This document. |

No existing source file was edited.

---

## 3. Attacks and exact parameters used

All values live in `configs/phase10_robustness.yaml` and are knobs. Gaussian-noise
attacks are seeded from `noise_seed (1234) + image_index`, so every run
reproduces.

| Attack | Parameter (severity axis) | Values run | Direction |
|---|---|---|---|
| `jpeg_compress` | JPEG quality | 90, 75, 50, 30, 10 | lower = stronger |
| `gaussian_noise` | sigma (0–255 units) | 2, 5, 10, 20, 40 | higher = stronger |
| `gaussian_blur` | kernel size | 3, 5, 7, 9 | higher = stronger |
| `median_filter` | kernel size | 3, 5, 7 | higher = stronger |
| `resize_roundtrip` | down-scale factor (then up again) | 0.9, 0.75, 0.5, 0.25 | lower = stronger |
| `rotate` | degrees about centre (BORDER_REFLECT, not undone) | 1, 2, 5, 10, 45 | higher = stronger |
| `center_crop` | central fraction kept, rest zero-padded | 0.95, 0.9, 0.75, 0.5 | lower = stronger |
| `combined` | ordered pipeline | ① JPEG75→noise σ3 ② blur k3→JPEG50→noise σ5 ③ resize 0.75→JPEG40 | — |
| `identity` | — | no-op (the severity-zero reference) | — |

Run configuration: DIV2K **test** split, **50 images**, 256×256, `payload_seed
20260901`. 34 (attack, severity) points × 50 images = **1700 per-image rows** per
decoder. Environment: Python 3.12.10, NumPy 2.5.2, OpenCV 4.14.0, PyTorch
2.13.0+cpu (CUDA false). Runtime ≈ 3 min on CPU.

---

## 4. Results

Full data: `results/phase10_robustness/`.

### 4.1 No-attack baseline (identity, 64 bits, alpha 0.02, 50 images)

| decoder | BER | bit-accuracy | NC |
|---|---|---|---|
| non-blind (reference) | **0.0956** | 0.9044 | 0.8087 |
| blind CNN (Phase 8) | **0.2216** | 0.7784 | 0.5569 |

These reproduce the Phase 9 numbers at the same operating point (non-blind
0.1013 / blind 0.2220 on 100 images) — the harness is consistent.

### 4.2 Watermark recovery under attack — mean BER (50 images)

`dBER` = increase over the no-attack baseline (positive = worse). Image-quality
(PSNR of the attacked vs clean watermarked image) is shown for context.

| attack | severity | PSNR dB | non-blind BER | non-blind dBER | blind BER | blind dBER |
|---|---|---|---|---|---|---|
| jpeg_compress | 90 | 33.8 | 0.111 | +0.015 | 0.252 | +0.031 |
| jpeg_compress | 75 | 30.7 | 0.161 | +0.066 | 0.287 | +0.065 |
| jpeg_compress | 50 | 28.6 | 0.229 | +0.133 | 0.344 | +0.122 |
| jpeg_compress | 30 | 27.2 | 0.281 | +0.186 | 0.382 | +0.160 |
| jpeg_compress | 10 | 24.2 | 0.366 | +0.270 | 0.437 | +0.215 |
| gaussian_noise | 2 | 42.1 | 0.104 | +0.009 | 0.240 | +0.018 |
| gaussian_noise | 5 | 34.3 | 0.173 | +0.077 | 0.274 | +0.053 |
| gaussian_noise | 10 | 28.3 | 0.282 | +0.186 | 0.340 | +0.119 |
| gaussian_noise | 20 | 22.5 | 0.378 | +0.283 | 0.402 | +0.181 |
| gaussian_noise | 40 | 16.9 | 0.469 | +0.373 | 0.457 | +0.236 |
| gaussian_blur | 3 | 28.2 | 0.472 | +0.376 | 0.412 | +0.190 |
| gaussian_blur | 5 | 26.0 | 0.482 | +0.387 | 0.451 | +0.230 |
| gaussian_blur | 7 | 24.2 | 0.488 | +0.392 | 0.468 | +0.247 |
| gaussian_blur | 9 | 23.3 | 0.489 | +0.394 | 0.471 | +0.249 |
| median_filter | 3 | 27.4 | 0.442 | +0.346 | 0.443 | +0.222 |
| median_filter | 5 | 24.1 | 0.472 | +0.376 | 0.470 | +0.248 |
| median_filter | 7 | 22.4 | 0.481 | +0.385 | 0.478 | +0.257 |
| resize_roundtrip | 0.9 | 29.7 | 0.466 | +0.370 | 0.389 | +0.168 |
| resize_roundtrip | 0.75 | 28.3 | 0.471 | +0.375 | 0.406 | +0.185 |
| resize_roundtrip | 0.5 | 25.8 | 0.481 | +0.385 | 0.425 | +0.203 |
| resize_roundtrip | 0.25 | 22.2 | 0.490 | +0.394 | 0.504 | +0.283 |
| rotate | 1 | 20.3 | 0.464 | +0.369 | 0.497 | +0.275 |
| rotate | 2 | 17.5 | 0.478 | +0.382 | 0.513 | +0.292 |
| rotate | 5 | 14.9 | 0.486 | +0.390 | 0.509 | +0.288 |
| rotate | 10 | 13.4 | 0.494 | +0.398 | 0.503 | +0.281 |
| rotate | 45 | 10.8 | 0.495 | +0.399 | 0.494 | +0.272 |
| center_crop | 0.95 | 17.1 | 0.466 | +0.370 | 0.428 | +0.206 |
| center_crop | 0.9 | 14.1 | 0.489 | +0.393 | 0.465 | +0.243 |
| center_crop | 0.75 | 10.5 | 0.494 | +0.398 | 0.502 | +0.280 |
| center_crop | 0.5 | 8.0 | 0.499 | +0.404 | 0.488 | +0.266 |
| combined ① (JPEG75→noise σ3) | — | 30.0 | 0.181 | +0.085 | 0.299 | +0.078 |
| combined ② (blur k3→JPEG50→noise σ5) | — | 25.4 | 0.473 | +0.377 | 0.434 | +0.213 |
| combined ③ (resize 0.75→JPEG40) | — | 26.1 | 0.470 | +0.374 | 0.417 | +0.195 |

### 4.3 Which attacks hurt the watermark most

Attacks ranked by worst-case non-blind BER over the tested severities
(`results/phase10_robustness/phase10_report.json → robustness_ranking`):

| # | attack | worst non-blind BER | Δ vs baseline | worst blind BER |
|---|---|---|---|---|
| 1 | center_crop | 0.499 | +0.404 | 0.502 |
| 2 | rotate | 0.495 | +0.399 | 0.513 |
| 3 | resize_roundtrip | 0.490 | +0.394 | 0.504 |
| 4 | gaussian_blur | 0.489 | +0.394 | 0.471 |
| 5 | median_filter | 0.481 | +0.385 | 0.478 |
| 6 | combined (heavy) | 0.473 | +0.377 | 0.434 |
| 7 | gaussian_noise | 0.469 | +0.373 | 0.457 |
| 8 | jpeg_compress | 0.366 | +0.270 | 0.437 |

**Reading of the results**

* **The watermark only meaningfully survives JPEG and light Gaussian noise.**
  Non-blind BER stays under ~0.23 down to JPEG quality 50 and under ~0.17 up to
  noise σ = 5, and degrades smoothly from there. `combined ①` (JPEG75 + σ3)
  is still at BER 0.18. This is the useful operating envelope of the frozen
  baseline.
* **Every low-pass or geometric attack collapses the non-blind decoder to
  chance (~0.47–0.50) even at its mildest setting** — blur k = 3, resize 0.9,
  1° rotation, 5 % crop. The frozen non-blind decoder is a bare magnitude
  comparison `σ_w[i] > σ_o[i]` with no amplitude normalisation: any filter that
  scales the LL singular values down (blur, resize, median) makes almost every
  comparison read "smaller", and any geometric change destroys the singular
  vectors' identity. This fragility is exactly what `docs/baseline.md` flagged
  in Phase 6; Phase 10 now quantifies it.
* **The blind CNN is measurably more robust to blur and resize** than the
  non-blind decoder — e.g. resize 0.9: 0.389 vs 0.466, blur k3: 0.412 vs 0.472,
  combined ②/③: ~0.42–0.43 vs ~0.47 — because it learned a spectrum-shape prior
  rather than a raw magnitude test. It is **not** more robust to rotation or
  cropping (both decoders ≈ chance there).
* **Rotation and crop also crush the image-quality metrics** (PSNR 8–20 dB, SSIM
  down to 0.16) because the whole frame is displaced — a reminder that these
  numbers are un-synchronised worst cases, not what a registration-aware
  receiver would see.
* **Nothing here recovers a full 64-bit payload error-free under any attack**;
  the baseline's clean exact-match is already 0. Payload-level robustness needs
  error-correcting coding (Phase 16) and/or attack-aware training and
  synchronisation (later phases). Phase 10's job is to measure, not to fix.

### 4.4 Plots (`results/phase10_robustness/plots/`)

| File | Shows |
|---|---|
| `ber_vs_severity.png` | one panel per scalar attack: non-blind and blind BER vs severity, with the 0.5 chance line. |
| `psnr_vs_severity.png` | attack distortion (PSNR of attacked vs clean watermarked) vs severity, all attacks. |
| `worst_case_ber_by_attack.png` | worst-case BER per attack family, non-blind vs blind — the robustness ranking at a glance. |

---

## 5. Output files (`results/phase10_robustness/`)

| File | Content |
|---|---|
| `phase10_per_image.csv` | 1700 rows = 34 (attack, severity) points × 50 images; `quality_*`, `nonblind_*`, `blind_*` columns. |
| `phase10_summary.csv` | 34 rows = mean/std per point, flat columns, incl. `*_delta_vs_baseline`. |
| `phase10_baseline.json` | the no-attack (identity) recovery block, both decoders. |
| `phase10_report.json` | combined: config (full attack grid), environment, **SHA-256 of every frozen Phase 6 file**, no-attack baseline, `robustness_ranking`, the two-group `summary`, `sanity` (Phase 7 route + Phase 6 defaults + Phase 8 checkpoint all OK), plot paths, runtime. |
| `plots/*.png` | the three figures above. |

`results/**` and `models/**` are git-ignored (as for Phases 6/8/9); the code,
config and this doc are what is committed.

---

## 6. Reproduce

```powershell
.venv\Scripts\python.exe experiments\run_phase10_robustness.py                       # 50 images, ~3 min CPU
.venv\Scripts\python.exe experiments\run_phase10_robustness.py --quick               # 10 images, 2 severities/attack
.venv\Scripts\python.exe experiments\run_phase10_robustness.py --no-cnn              # non-blind decoder only
.venv\Scripts\python.exe experiments\run_phase10_robustness.py --eval-num-images 100 # full test split
```

Deterministic: payloads from `numpy.random.default_rng(payload_seed + i)`,
noise from `numpy.random.default_rng(noise_seed + i)`. Re-running overwrites
`results/phase10_robustness/` with identical numbers.

---

## 7. Tests

`tests/test_attacks.py` (42) + `tests/test_phase10_robustness.py` (24):

* every attack obeys the RGB-uint8 → RGB-uint8 same-shape contract;
* JPEG distortion grows as quality drops; Gaussian noise is seed-deterministic
  and scales with sigma; blur/median reduce Laplacian variance; even kernel
  sizes are coerced odd; resize/rotate/crop degrade monotonically and validate
  their ranges; `combined` runs steps in order and is seed-deterministic;
* `apply_attack` dispatch, unknown-name errors, seed injection only for
  randomised attacks; `severity_of`;
* `Phase10Config` rejects bad alpha / payload > 128 / empty or unknown attacks;
  `load_config` reads the repo YAML and honours overrides;
* `build_embed_config` is the frozen LL-only config with only the operating
  point set;
* `summarise` keeps the two metric groups separate and adds `*_delta_vs_baseline`;
* data-dependent: `prepare_samples` embeds every image; the `identity` point
  reproduces clean recovery; mild JPEG barely moves BER while a 45° rotation
  pushes it well past baseline; `evaluate_attack_point` is bit-for-bit
  reproducible; the `combined` pipeline runs;
* CNN-dependent: the blind decoder path produces `blind_*` metrics and
  `summarise(has_blind=True)` carries both decoder blocks.

Full suite after Phase 10: **242 passed** (was 176).
