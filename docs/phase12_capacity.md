# Phase 12 — High-Capacity Watermarking

**Status:** implemented, tested and run (2026-09-02).
**Relationship to earlier phases:** additive only. `git diff` on `src/watermark/embed.py`,
`dwt.py`, `svd.py`, `watermark_generator.py`, `src/evaluation/metrics.py`,
`src/evaluation/blind_extract.py`, `src/evaluation/phase9_validation.py`,
`src/evaluation/phase10_robustness.py`, `src/evaluation/phase11_adaptive.py`,
`src/watermark/adaptive_embed.py`, `src/models/`, `src/training/`, `src/app/`
and every earlier config is **empty**. The frozen Phase 6 baseline, the
Phase 7 web UI and the Phase 8 CNN are untouched. Full test suite: **369
passed** (was 297).

---

## 1. Goal

Investigate how far watermark payload capacity can be pushed beyond the
frozen baseline's 128-bit LL-only limit — specifically **128 / 256 / 512 /
1024 bits** — and measure the resulting trade-off between payload size,
image quality (PSNR/SSIM/MSE) and watermark recovery (BER/bit-accuracy/NC).
This is an **experimental capacity study**, not a claim that more capacity is
automatically better.

## 2. Why capacity is limited

The frozen baseline embeds one bit per leading singular value of a DWT
subband. A single-level DWT of an `N x N` image gives four `N/2 x N/2`
subbands (LL, LH, HL, HH), each with `N/2` singular values — **128 bits per
subband** for the project's 256×256 processed images. The frozen baseline
already supports spending more than one subband via `EmbedConfig.extra_subbands`
(up to all four = **512 bits**, documented in `docs/baseline.md`); that
ceiling is real and single-level DWT cannot exceed it, however the four
subbands are combined.

To go further, **multi-level DWT** recursively decomposes the LL subband
again. A 2-level decomposition of a 256×256 image gives three untouched
128×128 detail subbands (LH1/HL1/HH1) plus four 64×64 subbands from
decomposing LL1 a second time (LL2/LH2/HL2/HH2). Each new subband still only
contributes its own side length in singular values, and that contribution
shrinks geometrically with depth. In closed form, for an `N x N` image
decomposed to `L` levels, using every detail subband at levels `1..L-1` and
all four subbands at the final level `L`:

```
capacity(L) = 3 * sum_{l=1}^{L-1} (N / 2^l)  +  4 * (N / 2^L)
```

| DWT levels | max capacity (256×256 image) |
|---|---|
| 1 | 512 bits |
| 2 | 640 bits |
| 3 | 704 bits |
| 4 | 736 bits |
| 5 | 752 bits |
| 6 | 760 bits |
| **L → ∞** | **768 bits (= 3×256, a hard ceiling)** |

As `L → ∞`, `capacity(L)` converges to **3N = 768 bits**, because
`sum_{l=1}^{∞} N/2^l = N`. This is a **hard theoretical ceiling** for this
project's watermarking scheme (one bit per leading singular value,
whole-subband SVD, 256×256 images): **no number of DWT levels can reach 1024
bits this way.** Going deep enough to approach 768 also makes the smallest
subbands (single-digit pixels wide well before the limit is approached)
useless as an embedding domain in practice.

**This bound was verified experimentally, not just assumed**: 1024-bit
embedding was attempted at 1, 2 and 3 DWT levels and failed explicitly every
time (§5), and the closed-form formula (`theoretical_capacity` /
`theoretical_capacity_limit` in `src/watermark/capacity_embed.py`) is
unit-tested (`test_theoretical_capacity_never_reaches_1024`).

## 3. How the new allocation works

`src/watermark/capacity_embed.py` (new, Phase 12) is a separate embedding
path built on the frozen `dwt.py`/`svd.py` primitives, never on `embed.py`
directly:

* **Multi-level DWT**: `dwt.decompose_2d`/`reconstruct_2d` (frozen, unchanged)
  are called recursively on the LL subband, `dwt_levels` times. Reconstruction
  propagates each level's output back into its parent's `ll` field, from the
  deepest level up to level 1.
* **Configurable subband allocation**: `CapacityEmbedConfig.subband_plan` is
  an ordered list of `(level, band)` pairs. An intermediate level's LL
  (`level < dwt_levels`) is invalid — it is always decomposed further — only
  the final level's LL and any level's LH/HL/HH are valid targets. This is
  validated at config construction time (`ValueError` on an intermediate LL).
* **Embedding rule**: identical to the frozen baseline's own multiplicative
  rule, `S'[i] = S[i] * (1 + alpha * (2*b[i] - 1))`, applied per configured
  window (same formula, same convention as `embed.py`'s `_modulate`).
* **Bit allocation**: `plan_windows` walks `subband_plan` in order, taking as
  many bits as each subband can hold (its full side length, minus
  `start_sv_index` for the first window only) until the payload is placed.
  If the plan cannot hold the full payload, `InsufficientCapacityError` (a
  `ValueError` subclass) is raised with the exact available-vs-required
  count — **no configuration is ever silently forced or truncated**.
* **Non-blind extraction**: `bit = 1 if S_w[i] > S_o[i] else 0` per window —
  the same decision rule as `extract_traditional`, generalised across levels.

### Reuse (per the instruction to avoid unnecessary duplication)

| Reused unchanged | From |
|---|---|
| `decompose_2d` / `reconstruct_2d` (called recursively) | `src/watermark/dwt.py` (frozen) |
| `decompose` / `reconstruct` / `SVDComponents` / `get_singular_values` | `src/watermark/svd.py` (frozen) |
| `generate_from_uuid` / `validate_bit_length` / `SUPPORTED_BIT_LENGTHS` | `src/watermark/watermark_generator.py` (frozen) — its allowed set is exactly `{8,16,32,64,128,256,512,1024}`, which is why this study's four target payloads use that generator directly |
| `psnr` / `ssim` / `mse` / `bit_error_rate` / `bit_accuracy` / `normalized_correlation` | `src/evaluation/metrics.py` (frozen) |
| Colour-space (RGB↔YCrCb) round trip, the `S'[i]=S[i]*(1+α(2b−1))` modulation formula | Re-implemented as tiny, self-contained ~10-line helpers (same pattern already used in Phase 11's `adaptive_embed.py`), so this phase does not import private (`_`-prefixed) symbols from `embed.py` or depend on another phase's file being unmodified |

No existing source file was edited.

## 4. Scope guard (deliberately NOT in Phase 12)

Not implemented here: error-correcting codes, blockchain, MLflow/PostgreSQL,
the production backend, any change to the frozen Phase 6 formula, any Phase 8
CNN redesign (extraction here is non-blind only), any silent change to Phase
7/8 behaviour (both are sanity-checked at the end of every run).

## 5. Experiment configuration (`configs/phase12_capacity.yaml`)

Deterministic payloads: `generate_from_uuid(uuid.uuid5(NAMESPACE_OID, f"phase12:{payload_seed}:{i}"), bit_length)`
— reuses the frozen Phase 3 generator directly; `bit_length` must be one of
`SUPPORTED_BIT_LENGTHS` (128/256/512/1024 here). Data: DIV2K **test** split,
**30 images**, 256×256. `alpha = 0.02` throughout (the project's established
operating point since Phase 8).

| experiment | bits | DWT levels | subband allocation | capacity available | expected |
|---|---|---|---|---|---|
| `cap128_L1_LL` | 128 | 1 | LL | 128 | success |
| `cap256_L1_LL_HL` | 256 | 1 | LL + HL | 256 | success |
| `cap256_L1_LL_LH` | 256 | 1 | LL + LH | 256 | success |
| `cap512_L1_all4` | 512 | 1 | LL + LH + HL + HH | 512 | success (single-level max) |
| `cap512_L2_detail` | 512 | 2 | L1: LH+HL+HH, L2: LL+LH+HL+HH | 640 | success (multi-level, same payload as above for comparison) |
| `cap1024_L1_attempt` | 1024 | 1 | LL + LH + HL + HH | 512 | **fail** (insufficient capacity) |
| `cap1024_L2_attempt` | 1024 | 2 | L1: LH+HL+HH, L2: all 4 | 640 | **fail** (insufficient capacity) |
| `cap1024_L3_attempt` | 1024 | 3 | L1/L2 details, L3: all 4 | 704 | **fail** (insufficient capacity) |

## 6. Results (30 images, full run)

### 6.1 Successful configurations (30 test images, real run)

| name | bits | DWT levels | PSNR (dB) | SSIM | MSE | BER | bit-acc | NC |
|---|---|---|---|---|---|---|---|---|
| cap128_L1_LL | 128 | 1 | 40.46 | 0.9964 | 6.494 | 0.1591 | 0.8409 | 0.6818 |
| cap256_L1_LL_HL | 256 | 1 | 40.43 | 0.9963 | 6.532 | 0.1635 | 0.8365 | 0.6729 |
| cap256_L1_LL_LH | 256 | 1 | 40.44 | 0.9963 | 6.526 | 0.1656 | 0.8344 | 0.6688 |
| cap512_L1_all4 | 512 | 1 | 40.39 | 0.9962 | 6.575 | 0.1743 | 0.8257 | 0.6513 |
| cap512_L2_detail | 512 | 2 | 40.41 | 0.9963 | 6.532 | 0.1662 | 0.8338 | 0.6676 |

### 6.2 Failed configurations (1024 bits)

All three 1024-bit attempts failed explicitly, exactly as the closed-form
bound in §2 predicts:

| name | DWT levels | capacity available | bits requested | status |
|---|---|---|---|---|
| cap1024_L1_attempt | 1 | 512 | 1024 | `insufficient_capacity` |
| cap1024_L2_attempt | 2 | 640 | 1024 | `insufficient_capacity` |
| cap1024_L3_attempt | 3 | 704 | 1024 | `insufficient_capacity` |

No payload was embedded for these three rows — `evaluate_experiment` returns
`n_images: 0`, `rows: []` and a `reason` string, and `summarise_experiment`
omits the `image_quality`/`watermark_recovery` blocks entirely rather than
reporting fabricated metrics.

### 6.3 Reading the results

* **PSNR/SSIM/MSE are essentially flat across 128 → 512 bits** (PSNR 40.39–40.46
  dB, SSIM 0.9962–0.9964): at a fixed alpha, image quality is governed by how
  many singular values are touched *within already-used subbands*, not by how
  many additional subbands are opened up — each new subband's own leading
  values are just as robust to a ±2% multiplicative nudge as the first
  subband's, so quality does not visibly degrade as capacity grows this way.
* **BER rises steadily with payload size at single-level allocations**: 0.159
  (128b) → 0.164 (256b) → 0.174 (512b). Every additional subband used is
  filled to its *full* 128-slot depth, so the extra bits always land partly in
  the same fragile trailing-singular-value region that already hurts recovery
  at 128 bits (§7) — more subbands means more of that fragile region is used,
  not a fresh supply of robust slots.
* **`cap512_L2_detail` (512 bits spread across 7 smaller subbands with 640
  bits of headroom) recovers measurably better than `cap512_L1_all4` (512
  bits exactly saturating 4 subbands' 512-bit capacity)**: BER 0.1662 vs
  0.1743, bit-accuracy 0.8338 vs 0.8257. Using more, smaller subbands with
  spare capacity — rather than maxing out fewer, larger ones — modestly
  improves recovery for the *same* payload size, at effectively the same PSNR
  (40.41 vs 40.39 dB). This is the practical benefit multi-level DWT offers
  even though it can't reach 1024 bits: a *better-conditioned* allocation for
  capacities single-level DWT can already technically reach.
* **256-bit allocation choice (LL+HL vs LL+LH) makes almost no difference**
  (BER 0.1635 vs 0.1656) — Haar's horizontal and vertical detail subbands are
  statistically similar for natural images, so which one is chosen second is
  not an important design decision at this operating point.

All numbers above are reproduced exactly by `results/phase12_capacity/phase12_summary.csv`
and `phase12_report.json` (which also carries per-metric standard deviations
and the full per-image breakdown in `phase12_per_image.csv`).

## 7. An additional finding: alpha is not monotonically helpful at full payload depth

While validating the modulation direction, a genuine and reproducible
property of the frozen multiplicative rule surfaced: **at full 128-bit LL
depth** (every singular value of the subband is modified simultaneously,
including the fragile, near-degenerate trailing ones), **increasing alpha
does not reliably reduce BER, and can make it worse.** This reproduces
identically on the *frozen* `embed()`/`extract_traditional()` at the same
operating point (bit_length=128, alpha swept 0.005→0.05) — it is not a defect
introduced by the new Phase 12 code, and is asserted directly in
`test_alpha_vs_ber_is_not_monotonic_at_full_ll_depth`. The likely mechanism:
modifying all 128 singular values of a subband simultaneously with a larger
relative step increases coupling between the already closely-spaced trailing
values through the lossy reconstruct → uint8-requantize → re-decompose round
trip, so larger perturbations create more cross-talk among near-degenerate
components rather than a cleaner separation. This is the opposite of the
well-established *shallow*-payload behaviour (8–64 bits, `docs/baseline.md`),
where higher alpha reliably lowers BER — confirmed here too
(`test_higher_alpha_lowers_mean_ber_for_a_shallow_payload`, 16-bit payload).

**Practical implication**: pushing payload depth close to a subband's full
capacity does not benefit from simply raising alpha to compensate for the
extra fragility — the frozen operating point (alpha≈0.02) is close to a
reasonable choice already, and further increases can be counter-productive
exactly where the extra bits live.

## 8. Conclusions and limitations

* **128, 256 and 512 bits are all reachable with a single-level DWT** by
  spending one, two, or all four subbands respectively — the frozen
  baseline's own `extra_subbands` mechanism already anticipated this; Phase
  12's contribution is a) a general, level-aware embedding path that is not
  restricted to a single DWT level, and b) the actual measurement of what
  quality/recovery cost each capacity point pays.
* **1024 bits is unreachable with this architecture on a 256×256 image, at
  any DWT level.** The theoretical ceiling as levels → ∞ is 3×256 = 768 bits,
  strictly below 1024, and was independently confirmed by three real failed
  embedding attempts (1, 2, 3 levels). Reaching 1024 bits would require a
  fundamentally different strategy — e.g. a larger image resolution, more
  than one bit per singular value (finer quantisation / QIM), a different
  transform domain, or error-correcting codes to recover a shorter raw
  payload reliably at higher redundancy. All of these are explicitly out of
  Phase 12's scope (ECC is reserved for Phase 16; no other strategy was
  implemented here) — this is reported as a limitation, not solved.
* **Multi-level DWT does increase usable capacity** (512 → 640 → 704 bits for
  1/2/3 levels) and the mechanism was verified to work correctly end-to-end
  (embed → reconstruct → re-decompose → extract), but the additional capacity
  it unlocks (128 bits at 2 levels, 192 at 3 levels beyond the single-level
  512) is nowhere near enough to bridge the gap to 1024.
* **Adding more subbands/levels for a given payload size is not free** —
  compare `cap512_L1_all4` (single-level, all four subbands fully used) with
  `cap512_L2_detail` (2-level, using detail bands only lightly and going deep
  into the smaller level-2 subbands) for the *same* 512-bit payload: see the
  measured numbers in §6.3 for which allocation is actually preferable at
  equal capacity usage.
* **Alpha does not compensate for capacity depth** (§7) — a further,
  independently useful finding for anyone tuning this scheme.

## 9. Output files (`results/phase12_capacity/`)

| File | Content |
|---|---|
| `phase12_per_image.csv` | one row per (successful experiment × image); quality + recovery metrics. |
| `phase12_summary.csv` | one row per experiment (8 total): mean/std for successes, status+reason for failures. |
| `phase12_capacity_bound_table.csv` | the closed-form `capacity(L)` for levels 1–6 plus the L→∞ limit. |
| `phase12_report.json` | method description, full config, environment, frozen-file SHA-256, capacity bound table, summary, sanity block, plot paths, runtime. |
| `plots/capacity_vs_quality_recovery.png` | payload size vs BER / PSNR / SSIM / bit-accuracy, successful configurations. |
| `plots/capacity_ceiling_vs_levels.png` | theoretical capacity vs DWT levels, with the 1024-bit target and the 768-bit L→∞ ceiling marked. |
| `plots/success_vs_failure.png` | capacity available vs bits requested per experiment, coloured by success/failure. |

`results/**` is git-ignored, same as Phases 6/8/9/10/11.

## 10. Reproduce

```powershell
.venv\Scripts\python.exe experiments\run_phase12_capacity.py                        # 30 images, full 8-experiment matrix
.venv\Scripts\python.exe experiments\run_phase12_capacity.py --quick                # 6 images
.venv\Scripts\python.exe experiments\run_phase12_capacity.py --eval-num-images 100  # full test split
```

## 11. Tests

`tests/test_capacity_embed.py` (39) + `tests/test_phase12_capacity.py` (33) =
72 tests:

* capacity arithmetic (`subband_side_length`, `full_subband_plan`,
  `theoretical_capacity`, `theoretical_capacity_limit`) against the closed-form
  formula, including the explicit "1024 bits unreachable at any level" check
  and the "256 halves exactly 8 times, a 9th level is impossible" boundary;
* `CapacityEmbedConfig` validation (wavelet restricted to haar, level/band
  range, intermediate-LL rejection, duplicate-window rejection, alpha/bit_length
  bounds);
* `plan_windows`/`total_capacity` bit-allocation logic, including the
  `InsufficientCapacityError` path;
* embed/extract contract (uint8, same shape, near-lossless, square-image
  requirement, bit_length mismatches rejected);
* **128-bit single-level output matches the frozen baseline byte-for-byte**
  (`test_128_bit_single_level_matches_frozen_baseline_exactly`) — the
  strongest possible regression guard that Phase 12 did not silently change
  Phase 6 behaviour;
* 512-bit single-level, 2-level and 3-level round trips;
  determinism of both embed and extract;
* the modulation-direction sanity check at a shallow payload, and the
  documented non-monotonic-at-full-depth finding (§7), each asserting the
  measured direction so a silent behavioural change is caught;
* insufficient-capacity handling for all three 1024-bit attempts (1/2/3
  levels), at both the primitive (`embed_capacity` raising) and harness
  (`evaluate_experiment` returning a structured failure, never raising) layers;
* harness: config load/override/validation (including that the shipped YAML
  covers exactly `{128, 256, 512, 1024}`), deterministic payload generation
  reusing the frozen Phase 3 generator, `evaluate_experiment`/`summarise_experiment`
  row shape for both outcomes, and the full 8-experiment matrix's
  success/failure pattern end-to-end.

Full suite after Phase 12: **369 passed** (was 297).
