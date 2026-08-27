# DWT-SVD Baseline — Phase 6

This is the project's **frozen reference system**. Every learned, adaptive or
error-corrected variant introduced in later phases is measured against the numbers
recorded here. Once accepted, `src/watermark/embed.py`, `configs/baseline.yaml`
and the results in `results/phase6_baseline/` are not modified.

## Pipeline

```
RGB uint8
  -> YCrCb (operate on Y only; Cr, Cb pass through untouched)
  -> single-level 2D DWT of Y            (haar, symmetric boundary)
  -> subband S  (baseline: LL)
  -> SVD:  S = U diag(sigma) V^T
  -> modulate the leading singular values with the payload bits
  -> reconstruct S,  inverse DWT
  -> clip Y to [0,255] uint8,  merge Cr, Cb,  -> RGB uint8
```

## Embedding rule (selected and frozen in Phase 6)

Phase 0 deliberately left the bit-to-singular-value rule open. Phase 6 selects
**multiplicative (relative) modulation**:

```
sigma'[i] = sigma[i] * (1 + alpha * (2*b[i] - 1))
    b[i] = 1  ->  sigma[i] * (1 + alpha)
    b[i] = 0  ->  sigma[i] * (1 - alpha)
```

Why multiplicative rather than the generic additive candidate
`sigma'[i] = sigma[i] + alpha*f(b[i])`:

* The singular values of an LL subband span ~1e4 down to ~1e0. A single additive
  `alpha` is simultaneously invisible for the small values and destructive for the
  large ones; a relative step keeps the perturbation proportional and keeps every
  modified value above the uint8 quantization floor for realistic `alpha`.
* The roadmap's own strength sweep — `alpha in {0.005, 0.010, 0.015}`, i.e.
  0.5%–1.5% — is only meaningful as a relative quantity.
* The blind decoder's decision statistic (Phase 7) becomes scale-free:
  `sign(sigma_w[i] / sigma_o[i] - 1)`.

## Non-blind extraction (reference decoder)

```
bit[i] = 1  if  sigma_watermarked[i] > sigma_original[i]  else  0
```

This requires the original image and exists only to establish a perfect-condition
reference. The **blind** decoder that does not need the original is the Phase 7
CNN; replacing this fragile magnitude comparison with a learned decision is the
main reason Phase 7 exists.

## Payload capacity and the documented limitation

A single-level DWT of an `N x N` image produces subbands of size `N/2 x N/2`, each
with `N/2` singular values. For the processed dataset (`N = 256`) that is
**128 bits per subband**.

| Payload | How it is carried | Status |
|---|---|---|
| 8–128 bits | LL subband only | frozen baseline |
| 256 bits | LL then HL (documented overflow order) | characterised, not part of the frozen baseline |
| 512 bits | LL, HL, LH, HH | reachable, deferred to Phase 12 |
| 1024 bits | needs a multi-level DWT | Phase 12 (high-capacity study) |

`EmbedConfig(subband="LL", extra_subbands=(...))` sets the overflow order.
`embed()` raises `ValueError` with the exact available slot count when a payload
does not fit the configured subbands. The frozen baseline itself is **LL-only**,
so its native capacity is **128 bits**; anything larger is explicitly an
extension studied later.

## Measured results (real experiment)

`experiments/run_baseline.py` embeds a fixed per-size random payload into the
**first 40 held-out DIV2K test images** (IDs 0801–0840) for every
`(alpha, payload)` pair and records real metrics. No value below is simulated.

Full data: `results/phase6_baseline/baseline_metrics.csv` (720 rows).
Aggregates: `results/phase6_baseline/baseline_summary.csv` / `.json`.
Example original / watermarked / residual (x15) images:
`results/phase6_baseline/samples/`.

### Imperceptibility (mean over 40 images)

| alpha | PSNR (dB) | SSIM | MSE |
|---|---|---|---|
| 0.005 | 48.9 – 49.0 | 0.9984 – 0.9988 | ~0.85 |
| 0.010 | 45.5 – 45.7 | 0.9976 – 0.9984 | ~1.9 |
| 0.015 | 42.6 – 42.8 | 0.9963 – 0.9979 | ~3.9 |

Payload size has almost no effect on PSNR/SSIM (the modification energy sits in a
few singular values). All operating points are visually lossless
(PSNR > 42 dB, SSIM > 0.996).

### Non-blind recovery (mean BER over 40 images)

| payload | alpha 0.005 | alpha 0.010 | alpha 0.015 |
|---|---|---|---|
| 8 bits | 0.000 | 0.000 | 0.000 |
| 16 bits | 0.005 | 0.003 | 0.003 |
| 32 bits | 0.006 | 0.011 | 0.018 |
| 64 bits | 0.019 | 0.027 | 0.050 |
| 128 bits | 0.118 | 0.117 | 0.128 |
| 256 bits (LL+HL) | 0.117 | 0.119 | 0.137 |

Reading of the results:

* **Recovery is exact for an 8-bit payload** at every baseline alpha and degrades
  gracefully as the payload reaches deeper into the singular spectrum.
* Around 128 bits the LL subband is saturated: its deepest ~30–40 singular values
  are `O(1–10)` and their ordering is not stable through the
  `IDWT -> uint8 -> DWT -> SVD` round trip, so ~12% of those bits flip regardless
  of alpha.
* Larger alpha slightly *worsens* BER at high payloads, because a bigger
  perturbation of the leading singular values adds more reconstruction noise onto
  the fragile trailing ones.
* This fragility of magnitude-comparison decoding — not of the embedding — is the
  motivation for the learned blind decoder in Phase 7 and the capacity study in
  Phase 12.

## Reproduce

```powershell
.venv\Scripts\python.exe experiments\run_baseline.py
```

Deterministic: payloads are drawn from `numpy.random.default_rng(seed)` with
`seed` from `configs/baseline.yaml`. Re-running overwrites
`results/phase6_baseline/` with identical numbers.

## Tests

* `tests/test_metrics.py` — PSNR / SSIM / MSE / BER / bit-accuracy / NC (11 tests).
* `tests/test_embed.py` — YCrCb round-trip fidelity, config validation, embedding
  invisibility, capacity guard, multi-subband overflow, and non-blind recovery
  bounds that bracket the measured results (24 tests; data-dependent tests skip
  if the processed test split is absent).
