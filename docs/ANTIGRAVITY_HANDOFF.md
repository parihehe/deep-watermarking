# SESSION HANDOFF — Running Project Status

**Purpose:** authoritative, verified state of the `deep-watermarking` project so any
future session can continue without conversation memory.

**Last updated:** 2026-08-27 (Claude Code session — re-audit + **Phase 6 completed**)

---

## 0. HOW TO READ THIS FILE

- Every status below is backed by **actual file inspection and real command execution**
  in the session that wrote it. No status is inferred from "a file exists".
- If a later session changes reality, it MUST update this file in the same session.
- Classification vocabulary: **COMPLETE / PARTIALLY COMPLETE / BROKEN / MISSING / NOT VERIFIED**.

---

## 1. TIMELINE — WHO DID WHAT

### Codex (original author)
- Created Phase 0 research docs (`docs/phase0/*`).
- Built repository scaffolding: `pyproject.toml`, `requirements.txt`, `environment.yml`,
  `.gitignore`, `configs/`, `src/` package layout, `tests/`.
- Implemented and tested Phases 1–5 (environment contract, dataset pipeline,
  watermark generation, DWT module, SVD module).
- Left `src/models/`, `src/evaluation/` as empty stubs.

### Antigravity session (Claude, ended on usage limit)
- Re-audited the repository, corrected the earlier claim that Phase 5 was "not started"
  (it was fully implemented and passing).
- Ran `pytest tests/ -v` → 77 passed.
- Confirmed dataset: 900 raw DIV2K PNGs, 700/100/100 processed 256×256 split.
- Wrote the first version of `docs/CODEX_HANDOFF.md`.
- Was about to re-run the pytest suite when the session ended.
- **Started Phase 6**: created `src/watermark/embed.py` (DWT-SVD baseline embed +
  non-blind extraction + residual). This file was **never validated** — see §4.

### This session (Claude Code, 2026-08-27)
- Full re-audit (this file).
- Smoke-tested `src/watermark/embed.py` on a real DIV2K image → **found it BROKEN**
  (details in §4): colour round-trip alone lost ~26 grey levels (PSNR 17 dB) before
  any watermark was embedded; non-blind BER ≈ 0.5.
- **Completed Phase 6.** All five known problems resolved and validated on real
  held-out DIV2K test images (see §4). Added `src/evaluation/metrics.py`,
  rewrote `src/watermark/embed.py`, added `configs/baseline.yaml`,
  `experiments/run_baseline.py`, `tests/test_metrics.py`, `tests/test_embed.py`,
  `docs/baseline.md`. Real results in `results/phase6_baseline/`.
- Test suite: **111 passed** (was 77).
- **Next: Phase 7 — blind CNN extractor + real training.**

---

## 2. ENVIRONMENT (verified this session)

| Component | Version | Notes |
|---|---|---|
| Python | 3.12.10 | in `.venv/`, matches `requires-python = ">=3.12,<3.13"` |
| PyTorch | 2.13.0+cpu | **CUDA: False** — CPU-only machine |
| NumPy | 2.5.2 | |
| SciPy | 1.18.1 | |
| PyWavelets | 1.8.0 | |
| OpenCV | 4.14.0 | `opencv-python` |
| scikit-image | 0.26.0 | used for PSNR/SSIM |
| MLflow | 3.15.1 | installed, **not yet used in code** |
| DVC | 3.67.1 | `.dvc/config` has `no_scm = True`; local cache only, no remote |
| pytest | 8.x | 77 tests, all pass |

**Git:** repository initialized, branch `main`, **zero commits**. Nothing is committed yet.

---

## 3. PHASE STATUS TABLE

| Phase | Description | Status | Evidence |
|---|---|---|---|
| 0 | Research documentation | COMPLETE | `docs/phase0/*` present, test enforces their existence |
| 1 | Environment + repository | COMPLETE | versions above; `test_environment_contract.py` passes |
| 2 | Dataset pipeline | COMPLETE | 700/100/100 processed 256² PNGs; `images.csv` + `dataset_summary.json`; `Div2KHostDataset` loader; leakage prevented by disjoint ID ranges 1–700 / 701–800 / 801–900 |
| 3 | Watermark generation | COMPLETE | `watermark_generator.py`: UUID→SHA-256→bits, text→bits, 8…1024-bit lengths, round-trip tested |
| 4 | DWT module | COMPLETE | `dwt.py`: haar/db2/db4, symmetric/periodization, round-trip max err ≤ 1e-10, 11 tests |
| 5 | SVD module | COMPLETE | `svd.py`: full SVD, reconstruct, singular-value extract/select, bipolar embed, non-blind extract, 20 tests |
| 6 | DWT-SVD baseline embed + evaluation | **COMPLETE (frozen)** | multiplicative relative SV modulation; correct YCrCb path; `metrics.py`; `run_baseline.py`; real results over 40 held-out test images in `results/phase6_baseline/`; `docs/baseline.md`; 111 tests pass. See §4. |
| 7 | Blind CNN extractor + training | MISSING — **NEXT** | `src/models/` empty stub |
| 8 | Baseline validation | MISSING | |
| 9 | Attack simulation | MISSING | |
| 10 | Adaptive/content-aware embedding | MISSING | |
| 11 | Attack-aware training | MISSING | |
| 12 | High-capacity study | MISSING | note: 256² image → LL is 128×128 → only 128 singular values; payloads > 128 bits need multi-subband or multi-channel allocation |
| 13 | Wavelet study | MISSING | |
| 14 | Subband study | MISSING | |
| 15 | CNN improvement | MISSING | |
| 16 | Error correction (BCH) | MISSING | |
| 17 | Final integrated model | MISSING | |
| 18 | Ablation | MISSING | |
| 19 | Generalization | MISSING | |
| 20 | Security / false positives | MISSING | |
| 21 | Performance | MISSING | |
| 22 | MLflow + PostgreSQL | MISSING | MLflow installed only |
| 23 | FastAPI backend | MISSING | |
| 24 | React frontend | MISSING | |
| 25 | Docker | MISSING | |
| 26 | Final experiments | MISSING | |
| 27 | Documentation | PARTIAL | README is setup-only; per-module docs exist for dataset/dwt/watermark |
| 28 | PPT + viva | MISSING | |

---

## 4. Phase 6 — the five known problems and how each was resolved

The previous session's `embed.py` draft did not work. Status of each issue after
this session, all **validated on real held-out DIV2K test images** (not fixtures):

| # | Problem (as found) | Resolution | Validation |
|---|---|---|---|
| 1 | RGB↔YCrCb round-trip wrong (`COLOR_BGR2YCrCb` output Y,Cr,Cb read as Y,Cb,Cr and re-stacked inconsistently) | New `_rgb_to_ycrcb` / `_ycrcb_to_rgb`: OpenCV `COLOR_RGB2YCrCb`, keep Cr/Cb as original uint8, only Y is modified | Round-trip with **no watermark**: PSNR **52.5 dB** (was 17.3). `test_ycrcb_round_trip_is_high_fidelity` |
| 2 | No-watermark reconstruction ~17 dB PSNR | Same root cause as #1 | Watermarked-image PSNR now **42–49 dB** across the whole alpha×payload grid |
| 3 | Embedding strength / quantization unreliable (additive ±alpha on raw singular values ~1e4) | Switched to **multiplicative relative** rule `σ'[i] = σ[i]·(1 + α(2b−1))` | 8-bit payload: **BER = 0.000** at every baseline alpha; 64-bit ≤ 0.05 |
| 4 | BER ≈ 0.5 (random recovery) | Consequence of #1+#3; fixed with them | Mean BER over 40 images: **0.000** (8 b) → 0.006–0.018 (32 b) → 0.12 (128 b). Full table in `docs/baseline.md` |
| 5 | 256-bit payload exceeds the 128 singular values of LL | `_plan_windows` distributes the payload across an ordered subband list; `EmbedConfig.extra_subbands` sets the overflow order (LL→HL→LH→HH). `embed()` raises a precise `ValueError` when a payload genuinely does not fit | 256-bit payload embeds via `("LL","HL")`, mean BER ≈ 0.12. `test_multi_subband_payload_256_bits`. >512 bits needs a multi-level DWT → deferred to Phase 12 (documented) |

### Baseline decisions frozen in Phase 6
- Rule: multiplicative relative SV modulation (rationale in `docs/baseline.md`).
- Domain: luminance (Y) channel, single-level **haar** DWT, **symmetric** boundary,
  **LL** subband, `start_sv_index = 0`.
- Alpha sweep: `{0.005, 0.010, 0.015}`. Native LL capacity: **128 bits**.
- Non-blind reference decoder: `bit[i] = 1 if σ_w[i] > σ_o[i] else 0`.
- The fragility of magnitude-comparison decoding at large payloads is real,
  measured, and documented — it is the motivation for Phase 7 (learned blind
  decoder) and Phase 12 (capacity study), not a bug to hide.

### Phase 6 artefacts (this session)
- `src/evaluation/metrics.py` — PSNR, SSIM, MSE, BER, bit accuracy, NC (pure, tested).
- `src/watermark/embed.py` — rewritten (colour path + multiplicative rule + multi-subband payload planner + residual).
- `configs/baseline.yaml` — frozen baseline config.
- `experiments/run_baseline.py` — deterministic characterisation over 40 test images.
- `tests/test_metrics.py` (11), `tests/test_embed.py` (24 incl. skips).
- `docs/baseline.md` — full method + measured results.
- `results/phase6_baseline/` — `baseline_metrics.csv` (720 rows), `baseline_summary.csv/json`, `samples/` (original/watermarked/residual PNGs).

---

## 5. FILE INVENTORY (source)

| File | Lines | Status |
|---|---|---|
| `src/watermark/watermark_generator.py` | 77 | COMPLETE |
| `src/watermark/dwt.py` | 82 | COMPLETE |
| `src/watermark/svd.py` | 241 | COMPLETE |
| `src/watermark/embed.py` | ~330 | COMPLETE / frozen (rewritten this session) |
| `src/evaluation/metrics.py` | ~130 | COMPLETE (new) |
| `src/utils/dataset_pipeline.py` | 153 | COMPLETE |
| `src/training/dataset.py` | 33 | COMPLETE (host-image loader; no watermark logic yet) |
| `src/models/__init__.py` | 1 | MISSING content — Phase 7 |
| `scripts/prepare_div2k.py` | 21 | COMPLETE (thin CLI over `dataset_pipeline`) |
| `experiments/run_baseline.py` | ~200 | COMPLETE (new) |

**Test suite: 111 tests, all passing** (`test_dwt`, `test_svd`,
`test_watermark_generator`, `test_dataset_pipeline`, `test_environment_contract`,
`test_metrics`, `test_embed`).

---

## 6. ROADMAP (reorganised 2026-08-27 by user request)

The Web UI was pulled forward ahead of the CNN. Completed phases are unchanged.

| Phase | Description | Status |
|---|---|---|
| 6 | DWT-SVD baseline embedding + evaluation | ✅ COMPLETE (frozen) |
| 7 | **Web Application / UI** around the existing `embed()` pipeline | **NEXT** |
| 8 | Blind CNN extractor + training | pending |
| 9 | Baseline validation and robustness testing | pending |
| 10 | Attack simulation | pending |
| 11 | Adaptive / content-aware embedding | pending |
| 12+ | (unchanged: capacity, wavelet/subband, CNN improvement, ECC, final model, ablation, generalization, security, performance, MLflow+PG, FastAPI, React, Docker, final experiments, docs, PPT) | pending |

### Phase 7 scope (this reorg)
Simple but polished web UI on top of the **existing, frozen** DWT-SVD backend —
no changes to `src/watermark/` or `src/evaluation/` maths. The UI must:
upload an image; generate/select a watermark payload; choose embedding parameters
(alpha, payload size, wavelet, subband); call the existing `embed()`; show the
original and watermarked images; show PSNR / SSIM / MSE and the non-blind
recovery metrics (BER / bit accuracy / NC via `extract_traditional`).
**No CNN work in Phase 7.**

Entry points the UI calls (all already implemented and tested):
- `src.watermark.embed.EmbedConfig`, `embed`, `extract_traditional`, `compute_residual`
- `src.watermark.watermark_generator.generate_random`, `generate_from_text`, `generate_from_uuid`
- `src.evaluation.metrics.quality_report`, `recovery_report`
