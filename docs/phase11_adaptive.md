# Phase 11 — Adaptive, Content-Aware Embedding

**Status:** implemented, tested and run (2026-09-02).
**Relationship to earlier phases:** additive only. `git diff` on `src/watermark/embed.py`,
`dwt.py`, `svd.py`, `watermark_generator.py`, `src/evaluation/metrics.py`,
`src/evaluation/blind_extract.py`, `src/evaluation/phase9_validation.py`,
`src/evaluation/phase10_robustness.py`, `src/models/`, `src/training/`, `src/app/`
and every earlier config is **empty**. The frozen Phase 6 baseline, the Phase 7
web UI and the Phase 8 CNN are untouched. Full test suite: **297 passed** (was 242).

---

## 1. Goal

Replace a single fixed embedding strength (`alpha`) with a **content-aware**
one: lower `alpha` in smooth image regions, higher `alpha` in textured/edge
regions, and measure whether that improves the quality/robustness trade-off
compared with the classic uniform-alpha baseline — at an **equal average
embedding strength**, so the comparison isolates allocation, not budget.

## 2. Why a new embedding path (not a smarter `alpha` argument to `embed()`)

The frozen Phase 6 baseline applies **one SVD to the whole LL subband** and
modulates its leading singular values. Each singular value's rank-1 update
(`u_i @ v_i.T`) spans the **entire image** — there is no spatial region a
single global "singular value index 7" corresponds to, so a scalar `alpha`
cannot be made region-adaptive without changing what `embed()` computes, which
would break the Phase 6 freeze.

Phase 11 instead adds `src/watermark/adaptive_embed.py`: it partitions the LL
subband into a grid of non-overlapping blocks and runs one small SVD per
block, embedding exactly one bit per block via the frozen baseline's own
multiplicative rule applied locally:

```
S'[0] = S[0] * (1 + alpha_block * (2*bit - 1))
```

Now a per-block `alpha_block` is meaningful — it can depend on that block's
local image content. With `block_size = 16` and the project's 256×256
processed images, the LL subband (128×128) yields an 8×8 = 64-block grid, so
the established 64-bit operating point (Phase 8/9/10) needs no change.

Two `alpha_mode` values share this exact machinery, so the comparison isolates
only the one variable under test:

* **`fixed`** — every block gets the same `alpha` (uniform allocation).
* **`adaptive`** — each block's alpha is derived from a texture/edge score of
  that block, then rescaled so the **per-image mean alpha equals the
  configured alpha exactly** — the same total "energy budget" as `fixed`, just
  reallocated.

## 3. How the alpha map is calculated

1. **Texture/edge score** (`texture_map`): Sobel gradient magnitude
   (`cv2.Sobel`, `ksize=3`, both axes, combined as `sqrt(gx² + gy²)`) computed
   on the **full-resolution** Y channel, then 2×2-average-pooled to exactly the
   LL subband's resolution (matching a single-level Haar DWT's implicit
   downsampling), then averaged within each `block_size × block_size` region.
   Higher score ⇒ more edges/texture in that block's source region.
2. **Normalisation**: the raw per-image block scores are clipped to their
   [5th, 95th] percentile (configurable) — so one very sharp edge does not
   compress every other block's weight toward one end — then linearly rescaled
   to `[0, 1]`.
3. **Weight**: `weight = alpha_low_mult + normalised_score * (alpha_high_mult − alpha_low_mult)`,
   default `alpha_low_mult = 0.4`, `alpha_high_mult = 1.6` — the least-textured
   block in the image gets weight 0.4, the most-textured gets weight 1.6.
4. **Mean-preserving rescale**: `alpha_block = alpha * weight / mean(weight)`,
   so `mean(alpha_block)` over the image's 64 blocks equals the configured
   `alpha` **exactly** — verified by `test_adaptive_mode_preserves_mean_alpha_energy_budget`
   and reproduced in every run (`alpha_realised_mean == alpha` to 1e-6 in the
   per-image CSV).

Non-blind extraction (`extract_adaptive`) mirrors the frozen baseline's own
decision rule, applied per block: `bit = 1 if S_w[0] > S_o[0] else 0`.

## 4. Scope guard (deliberately NOT in Phase 11)

| Not in Phase 11 | Where it belongs |
|---|---|
| any change to the frozen Phase 6 formula / defaults | — (frozen) |
| Phase 8 CNN architecture change or retraining | Phase 15 / not planned here |
| high-capacity changes (payload > 64 bits, subband overflow) | Phase 12 |
| wavelet / subband studies (only `haar` supported here) | Phases 13–14 |
| error-correcting codes | Phase 16 |
| blockchain | — |

Extraction is **non-blind only**. The Phase 8 CNN was trained on the frozen
whole-subband topology (LL, global SVD, 64 singular-value indices); it was
never built for, and is not evaluated against, the new block-SVD images — that
would be an apples-to-oranges comparison, not a CNN architecture change.

## 5. What was added

| File | Purpose |
|---|---|
| `src/watermark/adaptive_embed.py` | `AdaptiveEmbedConfig`, `embed_adaptive`, `extract_adaptive`, `texture_map`, `block_grid`, `expected_bit_length`. Imports only the frozen `dwt.py`/`svd.py` primitives (`decompose_2d`/`reconstruct_2d`, `decompose`/`reconstruct`/`SVDComponents`/`get_singular_values`); does not import or alter `embed.py`. |
| `src/evaluation/phase11_adaptive.py` | Harness: `Phase11Config`/`load_config`, `adaptive_config_for`, `evaluate_alpha_point`/`summarise_alpha_sweep` (clean-channel sweep), `evaluate_robustness_point`/`summarise_robustness` (attack stress check, reuses the already-implemented Phase 10 `apply_attack`). |
| `configs/phase11_adaptive.yaml` | Block size, bit length, alpha sweep, adaptive-map multipliers, robustness-check attacks, output config. |
| `experiments/run_phase11_adaptive.py` | Entry point (`--quick`, `--eval-num-images`). Writes CSV/JSON + plots + a Phase 6/7/8 sanity block. |
| `tests/test_adaptive_embed.py` (35) + `tests/test_phase11_adaptive.py` (20) | 55 tests total. |
| `docs/phase11_adaptive.md` | This document. |

No existing source file was edited.

## 6. Experiment configuration

Deterministic: payloads from `numpy.random.default_rng(payload_seed + i)` per
Phase 9's convention; Gaussian-noise attacks from `numpy.random.default_rng(noise_seed + i)`.

* Data: DIV2K **test** split, **50 images**, 256×256.
* Embedding: `block_size = 16` → 8×8 = **64 blocks = 64-bit payload**, `wavelet = haar`, `mode = symmetric`.
* Clean-channel alpha sweep: **{0.005, 0.010, 0.015, 0.020}**, each run in both `fixed` and `adaptive` mode (8 points × 50 images = 400 rows).
* Adaptive map: `alpha_low_mult = 0.4`, `alpha_high_mult = 1.6`, texture percentile clip `[5, 95]`.
* Robustness stress check at the primary operating point (`alpha = 0.020`): `jpeg_compress(quality=75)`, `gaussian_noise(sigma=5)`, `gaussian_blur(ksize=3)` — reusing the already-implemented, unmodified Phase 10 attack primitives — each run in both modes (6 points × 50 images = 300 rows).
* Environment: Python 3.12.10, NumPy, OpenCV, PyTorch 2.13.0+cpu. Runtime ≈ 54 s CPU for the full run.

## 7. Results (50 images, full run)

### 7.1 Clean-channel alpha sweep (mean over images)

| mode | alpha | PSNR | SSIM | MSE | BER | bit-acc | NC |
|---|---|---|---|---|---|---|---|
| fixed | 0.005 | 49.19 | 0.9986 | 0.828 | 0.0678 | 0.9322 | 0.8644 |
| adaptive | 0.005 | 49.00 | 0.9987 | 0.845 | 0.1338 | 0.8662 | 0.7325 |
| fixed | 0.010 | 45.67 | 0.9977 | 1.917 | 0.0197 | 0.9803 | 0.9606 |
| adaptive | 0.010 | 45.06 | 0.9981 | 2.140 | 0.0456 | 0.9544 | 0.9087 |
| fixed | 0.015 | 42.75 | 0.9963 | 3.909 | 0.0128 | 0.9872 | 0.9744 |
| adaptive | 0.015 | 42.10 | 0.9970 | 4.295 | 0.0238 | 0.9762 | 0.9525 |
| fixed | 0.020 | 40.55 | 0.9946 | 6.601 | 0.0112 | 0.9888 | 0.9775 |
| adaptive | 0.020 | 39.82 | 0.9957 | 7.317 | 0.0169 | 0.9831 | 0.9663 |

**Adaptive − fixed** (positive = adaptive higher):

| alpha | dPSNR | dSSIM | dMSE | dBER | dNC |
|---|---|---|---|---|---|
| 0.005 | −0.193 | +0.00006 | +0.017 | **+0.0659** | −0.1319 |
| 0.010 | −0.609 | +0.00031 | +0.223 | **+0.0259** | −0.0519 |
| 0.015 | −0.649 | +0.00068 | +0.386 | **+0.0109** | −0.0219 |
| 0.020 | −0.722 | +0.00111 | +0.716 | **+0.0056** | −0.0112 |

### 7.2 Robustness stress check (alpha = 0.020, mean over images)

| attack | fixed BER | adaptive BER | delta |
|---|---|---|---|
| jpeg_compress (q75) | 0.0156 | 0.0266 | +0.0109 |
| gaussian_noise (σ5) | 0.0184 | 0.0284 | +0.0100 |
| gaussian_blur (k3) | 0.0175 | 0.0288 | +0.0112 |

## 8. Does adaptive embedding improve the quality/robustness trade-off?

**No — not on these aggregate metrics, and the reason is understood, not a bug.**
At an equal mean-alpha energy budget, `adaptive` is consistently:

* **Worse on BER/bit-accuracy/NC**, clean channel and under every stress
  attack tested (+0.006 to +0.066 BER depending on alpha — worst at the
  smallest alpha, where a handful of blocks pushed toward `alpha_low_mult`
  cross into unreliable-decode territory).
* **Worse on PSNR/MSE** at every alpha (−0.19 to −0.72 dB).
* **Marginally better on global SSIM** (+0.00006 to +0.00111 — small but
  monotonically increasing with alpha).

**Why**: BER and MSE are both approximately *convex, decreasing-then-flat*
functions of a single block's alpha in this low-alpha regime (a small change
in alpha near the fragile end of the curve moves BER much more than the same
change near the strong end). By Jensen's inequality, spreading a fixed mean
alpha unevenly across blocks (adaptive) therefore *increases* the mean of a
convex loss (BER, MSE) compared with applying the same mean uniformly (fixed)
— exactly what was measured. This is a real, mathematically expected property
of variance in the embedding strength, not an implementation defect; it is
confirmed by `test_higher_alpha_lowers_ber_at_fixed_mode` and by the monotone
alpha sweep in the frozen baseline (`docs/baseline.md`).

The one metric that does move in adaptive's favour — global SSIM — is
consistent with the actual design intent: SSIM is more structure/edge-aware
than raw PSNR/MSE, so concentrating distortion energy into already-textured
blocks (where the HVS is less sensitive and local structural variance is
already high) reads as marginally *more* similar to SSIM even though the
allocation is worse for decoding. The effect is small (<0.0011) at this
`block_size`/multiplier configuration, and neither global PSNR nor the
robustness stress check found a compensating benefit.

**Honest conclusion**: with the mean-preserving, Sobel-edge-based allocation
implemented here, uniform (`fixed`) alpha dominates adaptive alpha on every
robustness metric and on PSNR, at every tested strength and under every
tested attack; adaptive only wins a small, and by itself inconclusive, SSIM
margin. Adaptive embedding is not free — realising its classical benefit (HVS
masking hides visible artefacts in flat regions without paying for it in
decode reliability) would need either a smaller `alpha_low_mult`/`alpha_high_mult`
spread, a decoder that also knows the local alpha (this non-blind decoder
does, in principle, but the naive S[0]-vs-S[0] comparison does not use it —
a threshold-aware decoder is a natural follow-up), or an image-quality metric
that is itself HVS-weighted rather than global PSNR/SSIM. None of that is in
Phase 11's scope; the measurement, not a fix, was the deliverable.

## 9. Output files (`results/phase11_adaptive/`)

| File | Content |
|---|---|
| `phase11_per_image.csv` | 400 rows = 4 alphas × 2 modes × 50 images; quality + recovery + realised alpha/texture stats. |
| `phase11_summary.csv` | 8 rows = mean/std per (alpha, mode). |
| `phase11_robustness_per_image.csv` | 300 rows = 3 attacks × 2 modes × 50 images. |
| `phase11_robustness_summary.csv` | 6 rows = mean/std per (attack, mode). |
| `phase11_report.json` | method description, config, environment, frozen-file SHA-256, both summaries, the adaptive-minus-fixed delta table, sanity block, plot paths, runtime. |
| `plots/quality_vs_alpha.png` | PSNR / SSIM / MSE vs alpha, fixed vs adaptive. |
| `plots/recovery_vs_alpha.png` | BER / bit-accuracy / NC vs alpha, fixed vs adaptive. |
| `plots/robustness_comparison.png` | Bar chart: mean BER per attack, fixed vs adaptive. |
| `plots/alpha_map_example.png` | One real image alongside its per-block Sobel texture score and the resulting per-block alpha allocation — visual confirmation that smooth blocks get lower alpha and textured/edge blocks get higher alpha. |

`results/**` is git-ignored, same as Phases 6/8/9/10.

## 10. Reproduce

```powershell
.venv\Scripts\python.exe experiments\run_phase11_adaptive.py                        # 50 images, ~55s CPU
.venv\Scripts\python.exe experiments\run_phase11_adaptive.py --quick                # 10 images, 2 alphas
.venv\Scripts\python.exe experiments\run_phase11_adaptive.py --eval-num-images 100  # full test split
```

## 11. Tests

`tests/test_adaptive_embed.py` (35) + `tests/test_phase11_adaptive.py` (20) = 55 tests:

* `expected_bit_length`/`block_grid` arithmetic and its rejection of non-dividing block sizes;
* `AdaptiveEmbedConfig` validation (wavelet restricted to haar, alpha/mult/percentile bounds);
* `texture_map`: zero on a flat image, elevated at a synthetic hard edge, requires even dimensions;
* fixed mode uses a literal constant alpha map; adaptive mode preserves the mean-alpha energy budget to 1e-9/1e-6, varies across blocks on real images, stays within the configured multiplier bounds, and gives a smooth (flattened) region strictly lower alpha than the image mean;
* `embed_adaptive`/`extract_adaptive` contract (uint8, same shape, near-lossless, deterministic, bit_length/block_size mismatches rejected, non-RGB rejected);
* extraction decision rule matches the recorded per-block `S[0]` comparison exactly; higher alpha does not increase BER;
* harness: config load/override/validation, `evaluate_alpha_point`/`evaluate_robustness_point` row shape and determinism, `summarise_alpha_sweep`/`summarise_robustness` grouping.

Full suite after Phase 11: **297 passed** (was 242).
