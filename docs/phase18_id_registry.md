# Phase 18 - Registry-ID Blind Payload & Reliability Calibration

**Status:** implemented, evaluated, and tested (2026-09-05).
**Relationship to earlier phases:** additive to Phase 17B (`src/app/final_model.py`).
Does not retrain the Phase 8 blind CNN checkpoint (`models/phase8_cnn/phase8_cnn_best.pt`)
and does not change `alpha`, `r` outside what's documented below. Builds on
the diagnosis in `docs/blind_extraction_diagnosis.md` (why raw text cannot
round-trip through the blind CNN) and the registry-ID codec design already
present in `src/watermark/id_registry.py`.

---

## 1. Problem

`docs/blind_extraction_diagnosis.md` established that the blind CNN cannot
recover arbitrary text (P(all 64 bits correct) ~ 8e-9 at its measured 22.2%
BER). The fix already in place before this phase: don't embed text, embed a
small registered ID (`src/watermark/id_registry.py`, K=4 bits, r=15
repetition, 60/64-bit budget) and look the text up locally. The first real
measurement of that design (`results/phase18_id_registry`, contiguous
repetition layout, no degenerate-ID exclusion) came in far below the 95%
target:

| metric | value |
|---|---|
| exact-match (40 images x 5 messages) | **30.5%** (61/200) |
| theoretical i.i.d. projection | 96.7% |

## 2. The fix: interleaved + masked codec

Two independent problems, found and fixed in sequence, both without touching
`r`, `alpha`, or the CNN checkpoint:

**2.1 Contiguous-block correlated errors.** The CNN's kernel=7 receptive
field correlates errors between neighbouring output positions. A contiguous
layout (`[bit0]*15 + [bit1]*15 + ...`) put all 15 copies of one ID bit in one
neighbourhood, so a single correlated burst could flip that bit's whole vote.
**Fix:** round-robin interleaving (`encoded[r*4+i]` = copy `r` of ID-bit `i`).
Result: **77.0%** (154/200).

**2.2 Global bit-balance sensitivity.** A 16-ID sweep (not just the 5
registered messages) found exact-match rate is a sharp function of the ID's
Hamming weight, independent of `r`: weight 2 (`0011`, `0101`, ...) scored
90-100%, weight 1/3 scored 35-70%, weight 0/4 (`0000`/`1111`) collapsed to
0-5%. Because every ID-bit's 15 copies share one value, Hamming weight fixes
the payload's *global* 1-fraction at `weight/4`, and only weight 2 (47%) sits
near the ~50% balance the CNN likely calibrated on during training. **Fix:**
a fixed, seeded, globally-balanced XOR mask (`MASK_SEED = 1729`,
`src/watermark/id_registry.py`) applied to all 60 encoded positions at both
encode and decode time - invertible per bit, decouples the physically
embedded stream's balance from the ID's own weight. Result: **95.5%**
(191/200), confirmed on a second, independently reseeded 16-ID sweep at
**96.2%** (308/320, correlation(weight, accuracy) 0.000).

Degenerate IDs (0 = `0000`, 15 = `1111`) are additionally excluded from
`MessageRegistry` allocation (`DEGENERATE_IDS`, `USABLE_ENTRIES = 14`) as a
policy safeguard, independent of the mask fix.

## 3. Confidence gate calibration

`RELIABLE_TEXT_CONFIDENCE = 0.85` (`src/app/final_model.py`) predates the
registry-ID path and was calibrated against a *different* metric (raw
per-bit CNN sigmoid margin, back when blind accuracy was ~0%). Applied
wholesale to the registry-ID path's metric (mean majority-vote agreement
across 4 slots), it passed only 6% of real decodes - correctly precise, but
far too conservative for this metric, silently suppressing nearly all
now-correct output.

**Fix:** a separate constant, `RELIABLE_REGISTRY_ID_CONFIDENCE = 0.78`,
calibrated from `experiments/calibrate_registry_confidence.py` (480 real
trials, all 16 IDs x 30 images, seed 42,
`results/phase18_id_registry/confidence_calibration.json`) - the lowest
achievable confidence value (47/60) with zero observed errors in that
calibration set. `RELIABLE_TEXT_CONFIDENCE` is untouched for the legacy
non-registry path.

## 4. Live-app end-to-end verification

Real FastAPI routes (`TestClient`, not the eval script), watermarked image
only, `/api/final-model/embed` -> `/api/final-model/extract/blind` (the
pairing that actually exercises `id_registry.py` -
`/api/watermark/embed` does not).

n=20 images per message, fresh images not reused from any calibration run:

| message | underlying accuracy | shown reliable | of those, correct |
|---|---|---|---|
| hello | 90.0% | 15% | 100% |
| hi | 100.0% | 20% | 100% |
| owner-2026 | 95.0% | 40% | 100% |
| 日本語 | 95.0% | 35% | 100% |
| **combined** | **95.0%** | **27.5%** | **100%** |

## 5. Residual risk: reliability is currently real, but not a mathematical guarantee

**"0 wrong answers ever shown to a user" is true so far, but it reflects the
current registry's sparsity (5 of 14 usable slots populated), not a proof
that `confidence >= 0.78` implies correctness.**

Precise count, directly from the data (not inferred): across the 560 trials
gathered to validate this phase (480-trial calibration set +
80-trial live-app message set), **exactly 1** wrong decode occurred at
confidence >= 0.78 (`"owner-2026"` on `0818.png`, decoded id=11 instead of 3,
confidence 0.8333). It was never shown to a user only because id=11 has no
registered message (`MESSAGE_REGISTRY.message_for(11) is None`) - the
registry's sparsity caught it, not the confidence gate. A follow-up 400-trial
sweep run to check id=15 specifically (`results/phase18_id_registry/id15_large_sample_check.json`)
found 4 more such cases (ids 15 and 3, all mis-decoding to unregistered ids 7
or 11). Combined: **5 wrong decodes at confidence >= 0.78 out of 960 total
trials (~0.52%)**, and 0 of them were ever displayed, purely because none
happened to land on one of the 5 currently-registered IDs.

If the registry is later populated closer to its 14-slot capacity, this same
~0.5% underlying rate has a correspondingly higher chance of landing on a
now-registered ID and being confidently displayed as **wrong** text.

### Required check before expanding the registry beyond its current 5 slots

**Before adding any message beyond the current 5 (`hello`, `hi`,
`owner-2026`, `Vestigia`, `日本語`), re-run
`experiments/calibrate_registry_confidence.py` against the intended larger
registry and re-verify `RELIABLE_REGISTRY_ID_CONFIDENCE` still holds at
acceptable precision at the new size.** Do not assume the current clean
record (0 wrong answers shown) carries over automatically - it is a function
of which 5 of 16 possible IDs happen to be populated, not a property of the
codec alone. If precision at 0.78 degrades as more slots fill, the threshold
needs to be raised (accepting lower coverage) before shipping the larger
registry.

## 6. Files

| file | role |
|---|---|
| `src/watermark/id_registry.py` | codec: interleaving, masking, degenerate-ID exclusion |
| `src/app/final_model.py` | `RELIABLE_REGISTRY_ID_CONFIDENCE`, `MESSAGE_REGISTRY`, blind extract wiring |
| `experiments/run_id_registry_eval.py` | 5-message x 40-image exact-match eval |
| `experiments/calibrate_registry_confidence.py` | confidence threshold calibration (16 IDs x 30 images) |
| `results/phase18_id_registry/summary.json`, `per_trial.json` | latest 5-message eval (95.5%) |
| `results/phase18_id_registry_archive/*_interleaved_77pct.json` | interleaving-only run, preserved for comparison |
| `results/phase18_id_registry/confidence_calibration.json` | 480-trial threshold calibration data |
| `results/phase18_id_registry/id15_large_sample_check.json` | 400-trial degenerate-ID large-sample check |
| `tests/test_id_registry.py` | interleaving, masking, and degenerate-exclusion regression tests |
