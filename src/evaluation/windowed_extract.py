"""Blind inference wrapper for the experimental windowed 1D-CNN extractor.

Load a trained :class:`~src.models.windowed_cnn.WindowedCNNExtractor` checkpoint
and recover the payload bits from a **watermarked image alone**. The original /
cover image is never used - this is the blind counterpart of the production
``BlindExtractor`` (Phase 8), evaluated under the same rules.

Unlike the production decoder (a single forward pass over the whole spectrum),
the windowed decoder predicts each bit independently from its own local 15-value
singular-value window, in the same order the frozen embedder placed the bits
(bit ``i`` -> singular value ``start_sv_index + i``).

Programmatic use::

    from src.evaluation.windowed_extract import WindowedExtractor

    extractor = WindowedExtractor.from_checkpoint("models/experimental/windowed_cnn/windowed_cnn_best.pt")
    probs = extractor.extract_proba("watermarked.png")   # (bit_length,) in [0,1]
    bits  = extractor.extract_bits("watermarked.png")    # list[int]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.models.windowed_cnn import (
    WindowedCNNConfig,
    WindowedCNNExtractor,
    bit_window_singular_values,
    load_windowed_checkpoint,
    luminance_ll_singular_values,
)

__all__ = ["WindowedExtractor"]


class WindowedExtractor:
    """Thin, stateful wrapper around a trained windowed blind extractor."""

    # This decoder reads the payload at the image's native resolution (see
    # _window_batch); ``image_size`` records only the resolution it was TRAINED
    # at. The production Phase 8 BlindExtractor does rescale, hence the flag.
    resizes_input = False

    def __init__(
        self,
        model: WindowedCNNExtractor,
        *,
        device: str | torch.device = "cpu",
        image_size: int | None = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.config: WindowedCNNConfig = model.config
        self.image_size = image_size or self.config.image_size

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> WindowedExtractor:
        model, checkpoint = load_windowed_checkpoint(str(path), map_location=device)
        image_size = int(checkpoint.get("image_size", model.config.image_size))
        return cls(model, device=device, image_size=image_size)

    # -- input handling -------------------------------------------------

    def _to_rgb_uint8(self, image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"could not read image: {image}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected an RGB image (H, W, 3); got shape {array.shape}")
        if array.dtype != np.uint8:
            array = np.clip(np.round(array), 0, 255).astype(np.uint8)
        return array

    def _window_batch(self, rgb_u8: np.ndarray) -> np.ndarray:
        # Decode at the image's NATIVE resolution. The embedder writes bit i into
        # singular value i of the LL sub-band at whatever size the image is, and
        # DWT-SVD singular values are not resize-invariant: rescaling first
        # remaps the whole spectrum and destroys the bit<->index correspondence.
        # Measured on DIV2K covers embedded at their native size, resizing to
        # 256 before decoding cost ~0.12 BER and dropped exact registry-ID
        # recovery from ~95% to ~55%.
        sigma = luminance_ll_singular_values(rgb_u8, self.config)
        needed = self.config.start_sv_index + self.config.bit_length
        if len(sigma) < needed:
            raise ValueError(
                f"image is too small to carry a {self.config.bit_length}-bit payload: its "
                f"{self.config.subband} sub-band yields {len(sigma)} singular values, "
                f"but bit {self.config.bit_length - 1} lives at index {needed - 1}. "
                f"Supply the watermarked image at its original size (at least "
                f"{2 * needed}x{2 * needed} pixels)."
            )
        windows = np.stack(
            [
                bit_window_singular_values(sigma, i, self.config)
                for i in range(self.config.bit_length)
            ]
        )  # (bit_length, window_size)
        return windows

    # -- extraction ---------------------------------------------------

    @torch.no_grad()
    def extract_proba(self, image) -> np.ndarray:
        """Per-bit probabilities in ``[0, 1]``, shape ``(bit_length,)``."""
        windows = self._window_batch(self._to_rgb_uint8(image))
        t = torch.from_numpy(windows).float().to(self.device)
        logits = self.model(t).squeeze(1)
        probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy()

    def extract_bits(self, image) -> list[int]:
        """Hard 0/1 payload prediction, length ``bit_length``."""
        return (self.extract_proba(image) > 0.5).astype(int).tolist()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_reference_bits(text: str, bit_length: int) -> list[int]:
    cleaned = text.strip().replace(" ", "").replace(",", "")
    if not cleaned or any(c not in "01" for c in cleaned):
        raise SystemExit("--reference-bits must be a string of 0/1")
    if len(cleaned) != bit_length:
        raise SystemExit(
            f"--reference-bits has {len(cleaned)} bits; checkpoint expects {bit_length}"
        )
    return [int(c) for c in cleaned]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Experimental windowed-CNN blind watermark extraction"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference-bits", dest="reference_bits", default=None)
    args = parser.parse_args(argv)

    extractor = WindowedExtractor.from_checkpoint(args.checkpoint, device=args.device)
    probs = extractor.extract_proba(args.image)
    bits = (probs > 0.5).astype(int).tolist()

    print(f"checkpoint      : {args.checkpoint}")
    print(f"image           : {args.image}")
    print(f"bit_length      : {extractor.config.bit_length}")
    print(f"window_size     : {extractor.config.window_size}")
    print(f"predicted bits  : {''.join(map(str, bits))}")
    print(f"mean confidence : {float(np.mean(np.abs(probs - 0.5)) * 2):.4f}")

    if args.reference_bits is not None:
        reference = _parse_reference_bits(args.reference_bits, extractor.config.bit_length)
        wrong = sum(int(a != b) for a, b in zip(reference, bits))
        ber = wrong / len(reference)
        print(f"reference bits  : {''.join(map(str, reference))}")
        print(f"bit errors      : {wrong}/{len(reference)}")
        print(f"BER             : {ber:.4f}")
        print(f"bit accuracy    : {1.0 - ber:.4f}")


if __name__ == "__main__":
    main()
