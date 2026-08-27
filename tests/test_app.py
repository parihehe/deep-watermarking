"""Tests for the Phase 7 web application (src/app).

These verify the HTTP layer actually drives the frozen Phase 6 pipeline: an
end-to-end embed request must return real PSNR/SSIM/MSE and recovery metrics that
match a direct call to ``src.watermark.embed``.
"""

import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.app.main import app
from src.evaluation.metrics import psnr as psnr_metric
from src.watermark.embed import EmbedConfig, embed
from src.watermark.watermark_generator import generate_from_uuid

client = TestClient(app)


def _png_bytes(size: int = 128, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    radius = np.sqrt(fy**2 + fx**2)
    radius[0, 0] = 1.0
    falloff = 1.0 / radius**1.1
    img = np.empty((size, size, 3), dtype=np.float64)
    for c in range(3):
        field = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size))) * falloff))
        field = (field - field.min()) / (field.max() - field.min())
        img[..., c] = 20 + 210 * field
    rgb = np.clip(np.round(img), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


def _decode_data_uri(uri: str) -> np.ndarray:
    assert uri.startswith("data:image/png;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def test_health() -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_options_lists_real_choices() -> None:
    opts = client.get("/api/options").json()
    assert "haar" in opts["wavelets"]
    assert opts["subbands"][0] == "LL"
    assert 64 in opts["bit_lengths"]
    assert 0.010 in opts["baseline_alphas"]


def test_index_served() -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "DWT-SVD" in r.text


def test_embed_endpoint_matches_direct_pipeline() -> None:
    """The API result must equal a direct embed() call with the same inputs."""
    img_bytes = _png_bytes(size=128, seed=1)
    resp = client.post(
        "/api/watermark/embed",
        files={"image": ("test.png", img_bytes, "image/png")},
        data={
            "alpha": "0.01", "bit_length": "32", "wavelet": "haar", "subband": "LL",
            "payload_source": "uuid",
            "payload_uuid": "00000000-0000-0000-0000-000000000001",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # reproduce the same computation directly
    bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bits = generate_from_uuid("00000000-0000-0000-0000-000000000001", 32)
    assert body["payload"]["bits"] == bits

    direct = embed(rgb, bits, EmbedConfig(alpha=0.01, bit_length=32, wavelet="haar", subband="LL"))
    api_wm = _decode_data_uri(body["images"]["watermarked"])
    assert np.array_equal(api_wm, direct.watermarked_image)

    # metrics are real and internally consistent
    q = body["quality_metrics"]
    assert q["psnr_db"] > 35.0
    assert 0.9 < q["ssim"] <= 1.0
    assert q["mse"] > 0.0
    assert psnr_metric(rgb, api_wm) == pytest.approx(q["psnr_db"], abs=1e-3)
    assert 0.0 <= body["recovery_metrics"]["ber"] <= 1.0
    assert body["image_info"] == {"width": 128, "height": 128, "subband_capacity_bits": 64}


def test_embed_random_payload_source() -> None:
    resp = client.post(
        "/api/watermark/embed",
        files={"image": ("t.png", _png_bytes(96), "image/png")},
        data={"alpha": "0.015", "bit_length": "16", "payload_source": "random"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["payload"]["bits"]) == 16
    assert "UUID" in body["payload"]["description"]


def test_embed_message_source_is_a_real_reversible_pipeline() -> None:
    """The simple 'message' source embeds the text in LL and decodes it back.

    Exact character recovery is *not* guaranteed by the frozen non-blind
    baseline (its trailing singular values flip on the uint8 round trip), so the
    contract asserted here is that the wiring is real: LL-only config, a payload
    sized from the text, genuine metrics, and an honest status/accuracy report.
    """
    resp = client.post(
        "/api/watermark/embed",
        files={"image": ("t.png", _png_bytes(256, seed=3), "image/png")},
        data={"alpha": "0.005", "payload_source": "message",
              "payload_text": "ACME"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["payload"]["source"] == "message"
    assert body["payload"]["text"] == "ACME"
    assert body["summary"]["original_text"] == "ACME"
    assert body["config"]["subband_order"] == ["LL"]
    # payload = (1 length byte + 4 body bytes) * 8 bits * repetition factor
    reps = body["summary"]["repetition"]
    assert reps in (1, 3, 5, 7, 9)
    assert body["config"]["bit_length"] == (1 + len("ACME")) * 8 * reps
    # genuine pass-through metrics, nothing hardcoded
    assert body["quality_metrics"]["psnr_db"] > 30.0
    assert 0.0 <= body["recovery_metrics"]["ber"] <= 1.0
    assert isinstance(body["summary"]["recovered_text"], str)
    assert 0.0 <= body["summary"]["char_accuracy"] <= 1.0
    assert body["summary"]["status"] in ("recovered", "partial", "failed")
    assert isinstance(body["recovery_metrics"]["text_match"], bool)


def test_short_message_recovers_exactly_on_a_clean_image() -> None:
    """A 1-character watermark round-trips exactly on a clean image (heavy repetition)."""
    resp = client.post(
        "/api/watermark/embed",
        files={"image": ("t.png", _png_bytes(256, seed=1), "image/png")},
        data={"alpha": "0.005", "payload_source": "message", "payload_text": "A"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["summary"]["recovered_text"] == "A"
    assert body["summary"]["watermark_recovered"] is True
    assert body["summary"]["status"] == "recovered"
    assert body["recovery_metrics"]["text_match"] is True


def test_message_too_long_returns_friendly_400() -> None:
    resp = client.post(
        "/api/watermark/embed",
        files={"image": ("t.png", _png_bytes(64), "image/png")},
        data={"alpha": "0.01", "payload_source": "message",
              "payload_text": "this watermark is far too long for a tiny image"},
    )
    assert resp.status_code == 400
    assert "too long for this image" in resp.json()["detail"]


def test_embed_text_payload_source() -> None:
    resp = client.post(
        "/api/watermark/embed",
        files={"image": ("t.png", _png_bytes(160), "image/png")},  # LL 80x80 -> 80 slots
        data={"alpha": "0.01", "bit_length": "64", "payload_source": "text",
              "payload_text": "owner:alice"},
    )
    assert resp.status_code == 200
    assert len(resp.json()["payload"]["bits"]) == 64


def test_payload_too_large_returns_400_with_explanation() -> None:
    resp = client.post(
        "/api/watermark/embed",
        files={"image": ("t.png", _png_bytes(64), "image/png")},  # LL is 32x32 -> 32 slots
        data={"alpha": "0.01", "bit_length": "128", "payload_source": "random"},
    )
    assert resp.status_code == 400
    assert "does not fit" in resp.json()["detail"]


def test_bad_alpha_returns_400() -> None:
    resp = client.post(
        "/api/watermark/embed",
        files={"image": ("t.png", _png_bytes(96), "image/png")},
        data={"alpha": "0", "bit_length": "16", "payload_source": "random"},
    )
    assert resp.status_code == 400


def test_undecodable_image_returns_400() -> None:
    resp = client.post(
        "/api/watermark/embed",
        files={"image": ("t.png", b"not an image", "image/png")},
        data={"alpha": "0.01", "bit_length": "16", "payload_source": "random"},
    )
    assert resp.status_code == 400
    assert "decode" in resp.json()["detail"]
