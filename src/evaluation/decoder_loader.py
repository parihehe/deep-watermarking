"""Decoder selection for the app's blind-extraction path.

The app has two CNN blind decoders:

* ``"current"`` - the production Phase 8 ``BlindExtractor`` (one CNN pass over
  the whole leading spectrum). **Default** - the shipped behaviour is never
  changed by this module.
* ``"windowed_cnn"`` - the experimental per-bit windowed 1D-CNN (trained on
  DIV2K with the actual project embedder; see ``docs/windowed_cnn.md`` and
  ``training/colab_windowed_cnn_training.ipynb``). Opt-in.

Selection is done by the ``DECODER_MODE`` environment variable::

    DECODER_MODE=windowed_cnn python scripts/run_app.py

A requested decoder that is not usable (no checkpoint / width mismatch) falls
back to ``"current"`` - the production behaviour stays a safe default. The
resulting ``(decoder, mode, reason)`` is surfaced in the extraction response so
results are never silently produced by an unexpected network.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.evaluation.windowed_extract import WindowedExtractor

__all__ = [
    "DECODER_MODES",
    "DECODER_MODE_ENV",
    "WINDOWED_CHECKPOINT",
    "WINDOWED_MODEL_DIR",
    "DecoderUnavailableError",
    "DefaultDecoder",
    "default_decoder_mode",
    "windowed_decoder",
]

DECODER_MODE_ENV = "DECODER_MODE"
DECODER_MODES = ("current", "windowed_cnn")

# Trained artifacts live alongside the shipped (production) blind checkpoints.
WINDOWED_MODEL_DIR = (
    Path(__file__).resolve().parents[2] / "models" / "experimental" / "windowed_cnn"
)
WINDOWED_CHECKPOINT = WINDOWED_MODEL_DIR / "windowed_cnn_best.pt"


class DefaultDecoder:
    """Sentinel for the production phase-8 CNN (selected through ``blind_extractor``)."""


class DecoderUnavailableError(RuntimeError):
    """The requested decoder exists as a concept but cannot be loaded here."""


def default_decoder_mode() -> str:
    """The decoder the blind-extraction path should use.

    ``DECODER_MODE`` overrides the default; an unknown value is rejected with a
    warning and falls back to ``"current"`` (never silently changes what runs).
    """
    raw = os.environ.get(DECODER_MODE_ENV, "").strip().lower()
    if not raw or raw == "current":
        return "current"
    if raw in DECODER_MODES:
        return raw
    print(
        f"[decoder_loader] warning: {DECODER_MODE_ENV}={raw!r} is not a known decoder "
        f"mode (known: {', '.join(DECODER_MODES)}); falling back to 'current'."
    )
    return "current"


def windowed_decoder(
    *, device: str = "cpu", checkpoint: str | Path | None = None
) -> WindowedExtractor:
    """Load the experimental windowed 1D-CNN decoder (opt-in mode).

    Raises :class:`DecoderUnavailableError` when the trained artifacts are not
    present, with actionable instructions to produce them (the Colab pipeline).
    """
    path = Path(checkpoint) if checkpoint else WINDOWED_CHECKPOINT
    if not path.is_file():
        raise DecoderUnavailableError(
            f"windowed-cnn decoder checkpoint not found at {path}. Train it via "
            f"training/colab_windowed_cnn_training.ipynb (DIV2K + the actual project "
            f"embedder) and place the package in models/experimental/windowed_cnn/ "
            f"as documented in docs/windowed_cnn.md."
        )
    try:
        return WindowedExtractor.from_checkpoint(str(path), device=device)
    except Exception as exc:
        raise DecoderUnavailableError(
            f"could not load windowed-cnn decoder {path.name}: {exc!r}"
        ) from exc
