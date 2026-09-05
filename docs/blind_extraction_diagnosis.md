# Blind Watermark Extraction — Diagnosis

**Date:** 2026-09-04
**Scope:** Why the web app's blind extraction does not recover the embedded
watermark text (e.g. `"hello"`), and why it fails for every payload size.
**Status:** diagnosis only — no code changed, nothing committed. The frozen
Phase 6 baseline (`src/watermark/`) was inspected but not modified.

---

## 0. How this was reproduced

* Full test suite: `.venv/Scripts/python.exe -m pytest -q` → **644 passed**.
  The suite does **not** contain a single test that embeds text and asserts it
  is recovered, so it is green while the feature is broken.
* Real app driven through `fastapi.testclient` and the frozen embedder on the
  held-out DIV2K test/validation images (`data/processed/div2k_256`).
* Blind decoder under test: `models/phase8_cnn/phase8_cnn_best.pt` (the only
  checkpoint wired into the app), loaded exactly as the app loads it.

Measured blind bit-accuracy of the shipped 64-bit CNN on freshly embedded,
clean, 256×256 images (no attack, α = 0.02, the training operating point):

| Payload | Blind CNN bit-acc | Non-blind reference bit-acc | Exact-match rate (blind) |
|--------:|------------------:|----------------------------:|-------------------------:|
| 8       | 0.58             | 1.00                        | 0 |
| 16      | 0.61             | 0.98                        | 0 |
| 32      | 0.73             | 0.96                        | 0 |
| 64      | 0.76–0.78        | 0.90                        | 0 |
| 128     | 0.77             | 0.90                        | n/a (CNN only emits 64) |
| 256     | 0.77             | 0.89                        | n/a |
| 512     | 0.77             | 0.92                        | n/a |
| 1024    | —                | —                           | does not fit 256×256 (capacity 512) |

Per-bit accuracy of the 64-bit CNN across the 64 payload positions:
**min 0.40, max 0.93, mean ≈ 0.75**. Estimated probability that all 64 bits of
one payload are correct: **≈ 8 × 10⁻⁹**. This matches the project's own record
in `docs/phase15_closure.md` ("Exact 64-bit match rate: 0.0" for both Phase 8
and Phase 15) and `results/phase15_cnn/phase15_cnn_training_summary.json`
(`val_bit_accuracy` 0.776, `val_exact_match` 0.0).

`"hello"` through the actual app today:

* 256×256 cover → message payload is **48 bits**, repetition = 1,
  `blind_extractable = false`. Blind endpoint still runs the 64-bit CNN and
  returns 64 unrelated bits → `"bits recovered; not decodable as text"`.
* Even the **non-blind** reference decode of that same 48-bit payload returns
  `"he��/"` (garbled) at α = 0.02.
* 512×512 cover → message payload is **240 bits** (rep = 5); non-blind decode
  returns `"`e"`; blind CNN still emits only 64 bits.

---

## 1. What currently works

* **Embedding** — `src/watermark/embed.py` (frozen) and `src/app/service.py`
  `run_embed`. All payload sources (`message`, `text`, `uuid`, `random`,
  `bits`) embed correctly; PSNR/SSIM are good; the watermarked PNG is
  byte-identical to calling the frozen `embed()` directly
  (`tests/test_app_final_model.py::test_final_embed_matches_direct_frozen_pipeline_at_alpha_020`).
* **Non-blind extraction** — `extract_traditional()` with the original image.
  Exact or near-exact for ≤ 32-bit payloads at α ≤ 0.01; degrades gracefully
  after that. This path is correct; it just needs the original image.
* **The message text codec** — `src/app/service.py`
  `message_base_bits` / `choose_repetition` / `decode_message`. Header +
  UTF-8 + odd-count repetition + per-position majority vote is internally
  consistent and round-trips perfectly on clean bits.
* **CNN checkpoint loading** — `src/evaluation/blind_extract.py`
  `BlindExtractor.from_checkpoint` loads `phase8_cnn_best.pt`, rebuilds
  `ExtractorConfig(bit_length=64, image_size=256)`, runs in eval mode. The
  right file is loaded; the format tag and config round-trip is fine.
* **Blind endpoint plumbing** — `/api/final-model/extract/blind`
  (`src/app/main.py`) takes the watermarked image *only*; there is no
  `original` parameter. The "no original required" contract is honoured.
* **Honest reliability gating** — when the CNN output is not trustworthy the
  UI shows *"Not reliably recovered"* rather than a wrong string. This is
  correct behaviour and must be preserved.

---

## 2. What currently fails

**F1 — Blind extraction never recovers the payload for any size.**
The 64-bit CNN decoder averages ~75 % per-bit accuracy on clean images and
0 % exact-match. A 48-bit `"hello"` payload comes back with ~10 wrong bits, so
UTF-8 decoding produces garbage and the reliability gate (correctly) suppresses
it. There is no confidence level at which the current CNN output is a correct
payload.

**F2 — `"hello"` (and any text) cannot round-trip through the blind path even
in principle**, because embedding and blind decoding use **different payload
formats**:

| | Embed (`service.run_embed`, `message` source) | Blind decode (`final_model._decode_message_bits`) |
|---|---|---|
| Payload length | `base_len × reps` (48, 96, 240, … — depends on image size) | assumes 64 |
| Repetition | odd count 1–9, chosen from LL capacity | assumes **1 copy** (comment: *"blind extraction has only the single embedded copy"*) |
| Majority vote | yes (`decode_message`) | **no** |
| Header | 1-byte length + UTF-8 body | same, but read from the wrong bit range |

So the blind decoder is parsing a 64-bit slice of a 48/96/240-bit repeated
structure that it never aligns to. `blind_extractable` is
`config.bit_length == 64` (`src/app/final_model.py:337`), which is **false**
for essentially every real text message, yet the UI's Extract button is still
enabled and still calls the blind endpoint.

**F3 — Payload sizes 8/16/32/128/256/512/1024 have no working blind path.**
The UI payload-size dropdown is filled from `/api/options`
(`service.options()["bit_lengths"]` = `{8,16,32,64,128,256,512,1024}`,
`src/app/static/index.html:382`), and `/api/watermark/embed` will embed any of
them. But:
* the CNN checkpoint is **fixed at 64 bits** (`ExtractorConfig.bit_length`,
  baked into the weights) — it always emits exactly 64 logits
  (`src/models/cnn_extractor.py` `forward`: `return logits[:, :64]`);
* for N < 64 the CNN is fed an out-of-distribution image (singular values
  64…N were never modulated during training) and does *worse* than at 64
  (0.58 at N = 8);
* for N > 64 the CNN physically cannot return the missing bits;
* 1024 bits does not fit a 256×256 image at all (LL+HL+LH+HH = 512 slots).

`final_model.SUPPORTED_PAYLOAD_BITS` is `(16, 32, 64)`
(`src/app/final_model.py:69`) — inconsistent with both the generator's
`SUPPORTED_BIT_LENGTHS` and the UI dropdown.

**F4 — The message-mode Embed and the blind Extract are wired to two
different backends that were never integrated.**
The UI Embed button posts to **`/api/watermark/embed`** (Phase 7
`service.run_embed`), the Extract button posts to
**`/api/final-model/extract/blind`** (Phase 17B `final_model`). Embed chooses a
repetition factor and a variable `bit_length`; Extract assumes the Phase 17
"final model" fixed 64-bit contract. Nothing carries the repetition count,
`base_len`, or true `bit_length` from embed to extract, and the UI only
forwards `expected_bits` when `lastBitLen === 64`
(`src/app/static/index.html:508-509,544`).

**F5 — Non-blind text recovery is itself lossy at α = 0.02.**
`final_model.run_final_embed` and `service.run_embed` (message mode) embed at
α = 0.02 with repetition chosen only from raw LL capacity. On a 256×256 image a
5–6 character message gets **repetition = 1** (48 bits ≤ 128, 96 bits > 128),
so a single flipped trailing-singular-value bit corrupts a character with no
majority-vote protection. `"hello"` non-blind → `"he��/"`.

**F6 — The existing test suite does not cover recovery.**
`tests/test_app_final_model.py` asserts only structural shape and
`text_reliable in (True, False)`; `tests/test_app.py` asserts message-mode
round-trip only through the **non-blind** `service.run_embed` path with
`alpha = 0.01`. No test embeds text and asserts the blind path returns it.

---

## 3. Why it fails

### 3.1 Root cause of F1 (the core failure): the blind decoder is a fixed, undertrained 64-bit model, and the task is close to its information limit

* `models/phase8_cnn/phase8_cnn_best.pt` was trained (`configs/cnn_extractor.yaml`,
  `src/training/train_extractor.py`) for **20 epochs on 700 images at exactly
  `bit_length = 64`, `alpha = 0.02`**, and plateaued at **0.776 val
  bit-accuracy, 0.0 exact-match**. `models/phase15_cnn/phase15_cnn_best.pt`
  (bigger architecture, `src/models/cnn_extractor_v2.py`) reached **0.778** —
  the same wall. `docs/phase15_closure.md` §3 attributes this to a physical
  limit: the deep LL singular values are fragile through the IDWT + uint8
  round trip, and a blind decoder with no reference cannot recover what the
  round trip did not preserve.
* The feature stage (`_SpectrumFeatureExtractor` in
  `src/models/cnn_extractor.py`) standardises `log1p` of the singular-value
  vector, then a 1-D CNN judges each value against its local neighbourhood.
  A ±2 % multiplicative nudge on the **leading** singular values (which are
  large and widely separated) is tiny next to the gap to their neighbours, so
  the CNN is *worst* on exactly the low-order bits where the signal is
  cleanest (measured per-bit accuracy 0.40–0.63 on positions 0–7). The
  non-blind decoder, which compares against the true original value, gets
  those same bits 100 % right.
* Net effect: mean 0.75 per-bit, `P(all 64 correct) ≈ 8e-9`. **No decode-path
  change, header fix, or threshold change can make this checkpoint return a
  correct payload.** The bits coming out are wrong before any decoding starts.

### 3.2 Root cause of F2/F4: two un-integrated subsystems

Phase 7 (`service.py`, the flexible embed pipeline, variable `bit_length`,
repetition + majority vote) and Phase 17B (`final_model.py`, a fixed 64-bit
"final model" + Phase 8 CNN) were both wired into the same page without a
shared payload contract. `final_model._decode_message_bits` explicitly assumes
"a single embedded copy" and a 64-bit field; `service`'s message codec
produces a repeated, variable-length field. They cannot interoperate as
written.

### 3.3 Root cause of F3: `bit_length` is a compile-time constant of the checkpoint

`ExtractorConfig.bit_length` is serialised into the weights and the final
`forward` slice. Supporting 8/16/32/128/256/512 blind means **one trained
decoder per width** (or a width-agnostic decoder), per
`docs/phase15_closure.md` ("`bit_length` is fixed per checkpoint at training
time … neither model can decode a different payload size without a separate
model trained from scratch"). None of those checkpoints exist. 1024 bits is
additionally impossible at 256×256 regardless of decoder (Phase 12 finding).

### 3.4 Root cause of F5: α and repetition tuned for signal strength, not text integrity

α = 0.02 was chosen (`configs/cnn_extractor.yaml`, `docs/phase17_final.md`) to
give the CNN a *stronger* signal to read, accepting a higher raw
non-blind BER (0.019 → 0.05 as α goes 0.005 → 0.015; ~0.10 at 0.02 for 64
bits — `results/phase6_baseline/baseline_summary.json`). Message mode inherits
that α and only repeats the payload if raw LL capacity allows, so short
messages on normal-sized images get no redundancy.

---

## 4. Which files are responsible

| File | Role in the failure |
|---|---|
| `models/phase8_cnn/phase8_cnn_best.pt` | The blind decoder actually used. Fixed 64-bit, 0.776 bit-acc, 0 exact-match. **Primary cause of F1.** |
| `src/app/final_model.py` | `SUPPORTED_PAYLOAD_BITS=(16,32,64)` (L69) vs generator's 8 sizes; `BLIND_BIT_LENGTH=64` (L70); `_decode_message_bits` assumes single copy + 64-bit layout (L354-369); `run_blind_extract` always runs the 64-bit CNN and has no notion of repetition / true bit_length (L385-456); `blind_extractable` gate (L337). **F2, F3, F4.** |
| `src/app/service.py` | `message_base_bits` / `choose_repetition` / `decode_message` (L151-176) define the *embed-side* payload format that the blind path never mirrors; `options()` advertises all 8 bit lengths (L349). **F2, F3.** |
| `src/app/static/index.html` | Embed → `/api/watermark/embed`, Extract → `/api/final-model/extract/blind` (L483, L545) — two backends, no shared payload metadata; payload-size dropdown filled from `options()` (L382-383); `lastBits`/`expected_bits` only sent when `lastBitLen === 64` (L508-509, L544); Extract button enabled regardless of `blind_extractable` (L436, L520). **F3, F4.** |
| `src/app/main.py` | `/api/final-model/extract/blind` (L119-134) accepts only the image + optional `expected_*`; no channel for `bit_length` / repetition / base_len. **F4.** |
| `src/models/cnn_extractor.py` / `src/models/cnn_extractor_v2.py` | `ExtractorConfig.bit_length` baked into weights; `forward` returns exactly `bit_length` logits. **Mechanism of F3.** |
| `src/training/train_extractor.py` / `train_extractor_v2.py` + `configs/cnn_extractor.yaml` / `configs/phase15_cnn.yaml` | Trained one width (64) at one α (0.02); no per-size or width-agnostic training run. **F1, F3.** |
| `src/training/watermark_dataset.py` | Generates payloads at a single fixed `bit_length`, every one of the first 64 singular values always carrying a bit → model is out-of-distribution for N < 64. **F3.** |
| `src/evaluation/blind_extract.py` | `_to_batch_tensor` (L84-89) resizes any input to 256×256; for non-256 covers this misaligns the watermarked LL sub-band the CNN reads. Secondary contributor. |
| `tests/test_app_final_model.py`, `tests/test_app.py` | No blind text round-trip assertion, so CI stays green. **F6.** |
| `src/watermark/embed.py` (frozen) | Not a bug. Its docstring already states clean-image recovery is exact only for well-separated leading singular values and a fraction of trailing bits flip on the uint8 round trip — the ceiling the blind model runs into. **Do not modify.** |

---

## 5. Exact implementation changes required

Two layers. Layer A is a decoder-capability change and is unavoidable for
requirement 1 ("recover `"hello"` as `"hello"`") and requirement 3 (all
sizes). Layer B is the pipeline-correctness work that is needed regardless and
that is worthless without Layer A.

### Layer A — make the blind decoder actually decode (required for req. 1 & 3)

The shipped checkpoint cannot return a correct payload. Pick one:

* **A1 (recommended): retrain the existing `BlindCNNExtractor` architecture as
  a width-agnostic / per-size decoder, and lower α for text.**
  * Train `src/training/train_extractor.py` with `bit_length` **sampled per
    sample** from `{8,16,32,64,128}` (needs a small change in
    `src/training/watermark_dataset.py` to draw a random width and zero-pad the
    target to a fixed 128-wide head; the frozen `embed()` is called unchanged).
    Keep `forward` emitting 128 logits and slice to the requested width at
    inference.
  * Add a second training α (e.g. 0.006) or train at 0.01 so the *embed* side
    can drop to a text-safe α without starving the decoder. Embedding math is
    untouched — only `EmbedConfig.alpha` passed in.
  * New checkpoints under `models/phase8_cnn/` (or a new folder); no
    architecture edit, no new "phase", frozen Phase 6 untouched.
  * Expected outcome from the measurements above: ≤ 32-bit payloads become
    reliably exact; 64-bit becomes usable with repetition; text via a
    ≤ 32-bit-per-copy repeated format becomes recoverable.
  * 128/256/512 blind: only feasible if a 128-wide head plus repetition gets
    there; **1024 stays unsupported at 256×256** and the UI must say so.

* **A2: keep the checkpoint, restrict the blind feature to what it can do.**
  Only offer blind extraction for a short text embedded as an **N ≤ 24
  bit-per-copy, high-repetition** payload at low α, and drive recovery by
  majority vote over many copies. This narrows "all payload sizes" to "short
  text only" and still needs Layer B. Lower confidence — likely still not
  reliable given per-bit 0.75.

* **A3: add a separate, opt-in blind-decodable embedding mode** (e.g. QIM on
  the LL singular values) wired only into the `final_model` routes, leaving
  `src/watermark/embed.py` frozen and the default path unchanged. Largest
  change; the only option that makes 128–512-bit blind genuinely exact.

### Layer B — pipeline correctness (required regardless; do after A)

1. **Unify the payload contract.** Route the message-mode **embed** and the
   **blind extract** through the same module (`final_model`), or have
   `/api/final-model/embed` return, and the UI echo back to
   `/api/final-model/extract/blind`, the values needed to invert the format:
   `bit_length`, `base_len`, `repetition`. Add those as optional form fields on
   `main.py:120` (`final_model_extract_blind`).

2. **Replace `final_model._decode_message_bits`** (L354-369) with a call to the
   existing `service.decode_message(bits, base_len, reps)` so the blind path
   uses the *same* header + repetition + majority-vote logic as embed. Delete
   the "single embedded copy" assumption.

3. **Fix `SUPPORTED_PAYLOAD_BITS`** (`final_model.py:69`) to match
   `watermark_generator.SUPPORTED_BIT_LENGTHS`, and make `run_blind_extract`
   slice/pad the decoder output to the requested width, returning an explicit
   `blind_supported: false` + reason for widths with no trained decoder and for
   1024-bit / capacity-exceeded cases instead of silently running the 64-bit
   model.

4. **Gate the UI Extract button on `blind_extractable`/`blind_supported`**
   (`index.html` ~L436/L520) and show the returned reason when false. Keep the
   *"Not reliably recovered"* branch (L549-552) exactly as is.

5. **Message-mode α + repetition for text integrity** (F5): in the message
   branch of the embed service, use a text-safe α (≤ 0.01) and force
   `repetition ≥ 3` (raise the friendly "text too long / image too small"
   error otherwise). This is an `EmbedConfig` argument change only — the frozen
   rule is not touched. Confirm `tests/test_app.py` message round-trip still
   passes.

6. **Do NOT** change `RELIABLE_TEXT_CONFIDENCE` (`final_model.py:75`) and do
   NOT hardcode expected text. The 0.85 gate is doing its job; measured
   confidence on real extracts is ~0.57, and it *should* suppress today's
   wrong output. It will pass naturally once Layer A produces correct bits.

### Regression tests to add (`tests/test_app_final_model.py`)

* embed `"hello"` (message mode, 256×256) → blind extract with the watermarked
  image only → `recovered.text == "hello"` **and** `text_reliable is True`.
* parametrised over payload sizes `{8,16,32,64,128,256,512}`: embed a known
  random payload, blind extract, assert returned `bit_length` is correct and
  (for widths with a trained decoder) `bit_accuracy == 1.0`; for unsupported
  widths assert a clean `blind_supported is False` + reason, not a 500 and not
  a silent 64-bit result.
* `1024` bits at 256×256 → embed rejected / `blind_supported False` with the
  capacity reason.
* negative: a deliberately mismatched / noisy image → `text_reliable is False`
  and `recovered.text is None` (no false string shown).
* blind endpoint request body contains no `original` field and extraction
  still succeeds (already covered — keep).
