"""Phase 8 — blind watermark extraction inference.

Load a trained :class:`~src.models.cnn_extractor.BlindCNNExtractor` checkpoint and
recover the payload bits from a **watermarked image alone**. The original/cover
image is never used — this is the blind counterpart of the frozen baseline's
``extract_traditional`` (which needs the original).

Programmatic use::

    from src.evaluation.blind_extract import BlindExtractor

    extractor = BlindExtractor.from_checkpoint("models/phase8_cnn/phase8_cnn_best.pt")
    bits = extractor.extract_bits("some_watermarked.png")          # list[int]
    probs = extractor.extract_proba(rgb_uint8_array)               # np.ndarray

CLI::

    .venv/Scripts/python.exe -m src.evaluation.blind_extract \
        --checkpoint models/phase8_cnn/phase8_cnn_best.pt \
        --image path/to/watermarked.png \
        [--reference-bits 010110...]        # optional, prints BER / accuracy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.models.cnn_extractor import BlindCNNExtractor, ExtractorConfig, load_checkpoint

__all__ = ["BlindExtractor"]

ImageLike = str | Path | np.ndarray


class BlindExtractor:
    """Thin, stateful wrapper around a trained blind CNN extractor."""

    def __init__(
        self,
        model: BlindCNNExtractor,
        *,
        device: str | torch.device = "cpu",
        image_size: int | None = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.config: ExtractorConfig = model.config
        self.image_size = image_size or self.config.image_size

    # -- construction ------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> BlindExtractor:
        model, checkpoint = load_checkpoint(str(path), map_location=device)
        image_size = int(checkpoint.get("image_size", model.config.image_size))
        return cls(model, device=device, image_size=image_size)

    # -- input handling -------------------------------------------------

    def _to_rgb_uint8(self, image: ImageLike) -> np.ndarray:
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

    def _to_batch_tensor(self, rgb_u8: np.ndarray) -> torch.Tensor:
        size = self.image_size
        if size is not None and rgb_u8.shape[:2] != (size, size):
            rgb_u8 = cv2.resize(rgb_u8, (size, size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb_u8)).permute(2, 0, 1).float().div(255.0)
        return tensor.unsqueeze(0).to(self.device)

    # -- extraction ---------------------------------------------------

    @torch.no_grad()
    def extract_proba(self, image: ImageLike) -> np.ndarray:
        """Per-bit probabilities in ``[0, 1]``, shape ``(bit_length,)``."""
        batch = self._to_batch_tensor(self._to_rgb_uint8(image))
        probs = torch.sigmoid(self.model(batch))[0]
        return probs.detach().cpu().numpy()

    def extract_bits(self, image: ImageLike) -> list[int]:
        """Hard 0/1 payload prediction, length ``bit_length``."""
        return (self.extract_proba(image) > 0.5).astype(int).tolist()

    def extract_batch_bits(self, images: list[ImageLike]) -> list[list[int]]:
        return [self.extract_bits(image) for image in images]


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
    parser = argparse.ArgumentParser(description="Phase 8 — blind watermark extraction")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference-bits", dest="reference_bits", default=None)
    args = parser.parse_args(argv)

    extractor = BlindExtractor.from_checkpoint(args.checkpoint, device=args.device)
    probs = extractor.extract_proba(args.image)
    bits = (probs > 0.5).astype(int).tolist()

    print(f"checkpoint      : {args.checkpoint}")
    print(f"image           : {args.image}")
    print(f"bit_length      : {extractor.config.bit_length}")
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
