# Web Application — Phase 7

A small FastAPI service with a single polished page that drives the **frozen
Phase 6 DWT-SVD baseline**. It contains no watermarking mathematics of its own:
every number comes from `src.watermark` / `src.evaluation`.

## Architecture

```
browser (src/app/static/index.html — vanilla HTML/CSS/JS, no build step)
        |  multipart POST
FastAPI  src/app/main.py        -- HTTP marshalling only
        |
adapter  src/app/service.py     -- bytes<->ndarray, payload construction, base64
        |
frozen   src/watermark/embed.py         embed(), extract_traditional(), compute_residual()
research src/watermark/watermark_generator.py    generate_random/from_text/from_uuid
         src/evaluation/metrics.py      quality_report(), recovery_report()
```

The research code is imported, never modified. This is the split the README
mandates ("the API and frontend ... call this code rather than duplicating its
mathematics") and it is the same backend the later FastAPI/React phases build on.

## Run

```powershell
.venv\Scripts\python.exe -m pip install -e ".[api]"     # first time
.venv\Scripts\python.exe scripts\run_app.py             # http://127.0.0.1:8000
# or:  .venv\Scripts\python.exe -m uvicorn src.app.main:app --reload
```

## What the UI does

The page is a three-step guided flow aimed at a non-specialist:

1. **Upload image** — drag/drop or click; JPEG/PNG/BMP/TIFF/WEBP, up to
   2048×2048. Shows a preview and the file name.
2. **Enter watermark** — one plain-text field ("Enter your watermark text").
   The service turns the text into payload bits and, on recovery, turns the
   recovered bits back into text, so the result section can show the watermark
   the user actually typed.
3. **Watermark settings** — just the embedding strength α as a labelled slider
   (default 0.005). Everything else — payload source, payload size, wavelet,
   sub-band, overflow sub-bands, explicit UUID / bit-string / SHA-256 fingerprint
   modes — lives in a collapsed **Advanced settings** panel and is ignored while
   the simple watermark-text field is in use.

Then **Embed watermark** shows a loading state, calls the API, and renders:

* **Watermark verification** — the original text, the recovered text, a
  three-tier verdict (recovered / partially recovered / failed), and BER + NC
  with one-line explanations.
* **Images** — Original, Watermarked, and Difference (×15 residual).
* **Image quality** — PSNR (dB), SSIM, MSE, each with a beginner-friendly note.
* **Technical details** (collapsed) — effective config, capacity, bit accuracy,
  and the raw embedded / recovered bit strings, for the project demonstration.
* A **How it works** strip (Original → DWT → SVD → Embedding → Watermarked →
  Recovery) and a **Next phase** card describing the not-yet-built Phase 8 CNN
  blind decoder.

API errors are mapped to plain-language messages (e.g. "Your watermark is too
large for this image…") with the raw `detail` kept behind a "Technical details"
disclosure.

### The `message` payload source

`payload_source=message` (used by the simple flow) is a thin, fully reversible
codec in `src/app/service.py`: `[1-byte length][UTF-8 body]`, repeated an odd
number of times (≤ 9) to fill the LL sub-band, majority-voted on recovery. It
does **not** change the embedding rule — repetition is just more payload bits.

Exact character recovery is **not guaranteed**: the frozen non-blind decoder is
only reliable on the well-separated leading singular values of LL, and a fraction
of the trailing ones flip on the uint8 round trip (see `src/watermark/embed.py`).
Short watermarks on a clean image recover exactly; longer text comes back with a
few character errors. The response reports this honestly via
`summary.status`, `summary.char_accuracy` and `recovery_metrics.text_match` —
nothing is faked or hardcoded. Removing this limitation is the point of Phase 8.

The other payload sources (`random`, `text` SHA-256 fingerprint, `uuid`,
`bits`) and all `EmbedConfig` knobs are unchanged and still reachable from the
Advanced panel. The recovery metrics still come from `extract_traditional`,
which compares against the original image.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | the single-page UI |
| GET | `/api/health` | `{status, torch, cuda}` |
| GET | `/api/options` | wavelets, subbands, bit lengths, baseline alphas, payload sources |
| POST | `/api/watermark/embed` | multipart: `image` + optional `alpha` (default 0.010), `bit_length` (default 64), `wavelet`, `subband`, `extra_subbands`, `payload_source` (default `message`), `payload_text`/`payload_uuid`/`payload_bits`. Returns `config`, `payload` (now incl. `text`), `image_info`, `quality_metrics`, `recovery_metrics` (now incl. `recovered_text`, `text_match`, `char_accuracy`), a `summary` block (`status`, `headline`, `original_text`, `recovered_text`, `char_accuracy`, `repetition`, `watermark_recovered`), and base64 PNG data URIs for original / watermarked / residual. |

User-correctable problems (payload too large for the image, bad α, undecodable
upload) return HTTP 400 with a plain-text explanation in `detail`.

## Tests

`tests/test_app.py` (13 tests) uses `fastapi.testclient.TestClient`. The key test
embeds through the HTTP endpoint and asserts the returned watermarked image is
**bit-for-bit identical** to a direct `embed()` call with the same inputs, and
that the reported PSNR matches `metrics.psnr` to 1e-3 — i.e. the web layer is a
faithful pass-through to the frozen pipeline. Added tests cover the reversible
`message` codec (real LL-only config, payload sized from the text, honest
`status` / `char_accuracy`), exact round-trip of a 1-character watermark, the
friendly "too long for this image" 400, and the simplified page markup.

## Not in Phase 7

No CNN, no training, no attack simulation, no database, no experiment history.
Those are Phases 8+. The backend module (`src/app`) is structured so those
endpoints can be added later without touching the research code.
