"""Decoder selection (production vs opt-in windowed CNN) and the blind
end-to-end recovery path.

The windowed-CNN decoder is the experimental DIV2K-trained decoder from
``training/colab_windowed_cnn_training.ipynb``. These tests verify the
*selection machinery* and that the **production decoder stays the default** -
they do not assert any accuracy. (A deterministic tiny windowed model is used to
exercise the real filesystem loader; a stub is used only to exercise wiring.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

import src.evaluation.decoder_loader as dl
from src.app import final_model as fm
from src.app.final_model import CheckpointError, ExtractRequest
from src.models.windowed_cnn import (
    WindowedCNNConfig,
    WindowedCNNExtractor,
    save_windowed_checkpoint,
)
from src.watermark.embed import EmbedConfig, embed


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(dl.DECODER_MODE_ENV, raising=False)
    fm.reset_extractor_cache()


def _rgb(seed: int = 1, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((size, size, 3)) * 255).astype(np.uint8)


def _png_bytes(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


def _save_tiny_windowed_model(
    tmp_path: Path, *, bit_length: int = 16, image_size: int = 64
) -> Path:
    cfg = WindowedCNNConfig(bit_length=bit_length, image_size=image_size)
    model = WindowedCNNExtractor(cfg)
    model.eval()
    path = tmp_path / f"windowed_cnn_{bit_length}bit_test.pt"
    save_windowed_checkpoint(
        str(path),
        model,
        extra={
            "bit_length": bit_length,
            "window_size": cfg.window_size,
            "image_size": image_size,
            "epoch": 1,
            "best_val_bit_acc": 0.5,
            "best_epoch": 0,
        },
    )
    return path


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------


def test_default_decoder_mode_is_current() -> None:
    assert dl.default_decoder_mode() == "current"


def test_env_selects_windowed_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(dl.DECODER_MODE_ENV, "windowed_cnn")
    assert dl.default_decoder_mode() == "windowed_cnn"


def test_unknown_env_falls_back_to_current(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(dl.DECODER_MODE_ENV, "bogus")
    assert dl.default_decoder_mode() == "current"


def test_windowed_decoder_missing_checkpoint_raises(tmp_path: Path) -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(dl, "WINDOWED_CHECKPOINT", tmp_path / "missing.pt")
    with pytest.raises(dl.DecoderUnavailableError):
        dl.windowed_decoder()
    monkeypatch.undo()


def test_windowed_decoder_loads_real_model(tmp_path: Path) -> None:
    ckpt = _save_tiny_windowed_model(tmp_path, bit_length=16, image_size=64)
    ext = dl.windowed_decoder(checkpoint=str(ckpt))
    probs = ext.extract_proba(_rgb(seed=2))
    assert ext.config.bit_length == 16
    assert probs.shape == (16,)
    assert np.isfinite(probs).all()
    assert 0.0 <= probs.min() <= probs.max() <= 1.0
    # deterministic given the model + image
    assert np.array_equal(probs, ext.extract_proba(_rgb(seed=2)))


# ---------------------------------------------------------------------------
# extractor resolution
# ---------------------------------------------------------------------------


@dataclass
class StubWindowed:
    config: object
    image_size: int = 256
    # extract_proba returns PROBABILITIES that each bit is 1, so 0.5 is the
    # uninformative decoder (0.0 would mean "certain every bit is 0").
    proba: float = 0.5

    def extract_proba(self, image) -> np.ndarray:
        return np.full(self.config.bit_length, self.proba)


def _windowed_stub(bit_length: int, proba: float = 0.5):
    cfg = WindowedCNNConfig(bit_length=bit_length, image_size=256)
    return StubWindowed(config=cfg, proba=proba)


def test_resolve_default_uses_current() -> None:
    fm._EXTRACTOR = _windowed_stub(64)  # test hook
    _ext, mode, reason = fm._resolve_blind_extractor(64)
    assert mode == "current" and reason is None


def test_resolve_windowed_unavailable_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(dl.DECODER_MODE_ENV, "windowed_cnn")

    def boom() -> dl.WindowedExtractor:
        raise dl.DecoderUnavailableError("no checkpoint")

    monkeypatch.setattr(dl, "windowed_decoder", boom)
    fm._EXTRACTOR = _windowed_stub(64)
    _ext, mode, reason = fm._resolve_blind_extractor(64)
    assert mode == "current" and reason is not None and "unavailable" in reason


def test_resolve_windowed_width_mismatch_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(dl.DECODER_MODE_ENV, "windowed_cnn")
    fm._EXTRACTOR = _windowed_stub(64)
    monkeypatch.setattr(dl, "windowed_decoder", lambda **_: _windowed_stub(32))
    _ext, mode, reason = fm._resolve_blind_extractor(64)
    assert mode == "current" and reason is not None and "declares" in reason


def test_resolve_windowed_used_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(dl.DECODER_MODE_ENV, "windowed_cnn")
    monkeypatch.setattr(dl, "windowed_decoder", lambda **_: _windowed_stub(64))
    _ext, mode, reason = fm._resolve_blind_extractor(64)
    assert mode == "windowed_cnn" and reason is None


# ---------------------------------------------------------------------------
# blind end-to-end through the final-model path
# ---------------------------------------------------------------------------


def test_blind_e2e_real_tiny_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuine embed -> PNG -> re-load -> blind-decode cycle using a real
    (untrained) 16-bit windowed checkpoint, wiring only - not accuracy."""
    ckpt = _save_tiny_windowed_model(tmp_path, bit_length=16, image_size=64)
    monkeypatch.setattr(dl, "WINDOWED_CHECKPOINT", ckpt)
    monkeypatch.setenv(dl.DECODER_MODE_ENV, "windowed_cnn")

    bits = [0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
    wm = embed(_rgb(seed=3), bits, EmbedConfig(alpha=0.02, bit_length=16))
    result = fm.run_blind_extract(
        _png_bytes(wm.watermarked_image), ExtractRequest(payload_bit_length=16, repetition=1)
    )
    assert result["extraction"] == "blind"
    assert result["decoder"] == "windowed_cnn"
    assert result["decoder_fallback"] is None
    assert result["model"] == "windowed_cnn_16bit"
    assert result["checkpoint_bit_length"] == 16
    assert result["recovered"]["bit_length"] == 16
    assert len(result["recovered"]["bit_string"]) == 16


def test_blind_e2e_registry_id_through_windowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """registry-ID payload kind routed through the windowed decoder: the 60
    encoded bits are decoded by the real id_registry codec, proving the exact
    contract the ownership flow uses."""
    monkeypatch.setenv(dl.DECODER_MODE_ENV, "windowed_cnn")
    monkeypatch.setattr(dl, "windowed_decoder", lambda **_: _windowed_stub(64))
    result = fm.run_blind_extract(
        _png_bytes(_rgb(seed=5)),
        ExtractRequest(payload_bit_length=64, repetition=1, payload_kind="registry_id"),
    )
    assert result["decoder"] == "windowed_cnn"
    assert result["recovered"]["bit_length"] == 64
    assert result["recovered"]["registry_id"] is not None
    assert result["recovered"]["confidence_kind"] == "soft_posterior"
    # An uninformative decoder (p = 0.5 on every bit) carries no evidence, so
    # the soft decode must bottom out at chance and never surface a message.
    assert result["recovered"]["confidence_mean"] == pytest.approx(0.5)
    assert result["recovered"]["text"] is None
    assert result["recovered"]["text_reliable"] is False


def test_blind_default_production_path_unchanged() -> None:
    """Regression: with no DECODER_MODE the response identifies the production
    decoder and the exact same fields the pre-integration path returned."""
    fm._EXTRACTOR = _windowed_stub(64)  # test hook stands in for the phase-8 CNN
    result = fm.run_blind_extract(
        _png_bytes(_rgb(seed=4)),
        ExtractRequest(payload_bit_length=64),
    )
    assert result["decoder"] == "current"
    assert result["decoder_fallback"] is None
    assert result["model"] == "blind_cnn_64bit"
    assert result["checkpoint_bit_length"] == 64
    assert result["extraction"] == "blind"
    assert "requires_original_image" in result and result["requires_original_image"] is False


def test_blind_no_checkpoint_raises_503(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When neither the requested (windowed) nor the production decoder is
    available, the caller gets a 503-equivalent CheckpointError - the windowed
    fallback never masks a missing production decoder."""
    monkeypatch.setenv(dl.DECODER_MODE_ENV, "windowed_cnn")

    def boom() -> dl.WindowedExtractor:
        raise dl.DecoderUnavailableError("no checkpoint here")

    monkeypatch.setattr(dl, "windowed_decoder", boom)
    # production checkpoints live in the repo; point the production dir to tmp
    monkeypatch.setattr(fm, "BLIND_MODEL_DIR", tmp_path)
    with pytest.raises(CheckpointError):
        fm.run_blind_extract(
            _png_bytes(_rgb()),
            ExtractRequest(payload_bit_length=64),
        )
