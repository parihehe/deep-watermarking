"""Phase 8 — training data for the blind CNN watermark extractor.

Each sample is produced by running the **frozen Phase 6 DWT-SVD embedder**
(``src.watermark.embed.embed``) on a real processed DIV2K image with a random
binary payload:

    (watermarked RGB image tensor in [0, 1],  payload bit tensor in {0, 1})

The cover image is deliberately **not** returned as a model input — the CNN is
blind. The cover is only used internally, and only by the frozen baseline, to
create the watermarked image.

Payload policy
--------------
* ``validation`` / ``test`` splits (``deterministic=True``): the payload for
  image ``i`` is a fixed function of ``(seed, i)``, so the validation set is
  identical on every run and every epoch.
* ``train`` split (``deterministic=False``): a fresh random payload is drawn on
  every ``__getitem__`` call, so the network sees new (image, payload) pairs
  each epoch and cannot memorise a fixed mapping.

This module imports the frozen baseline but never modifies it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.watermark.embed import EmbedConfig, embed

__all__ = ["WatermarkExtractionConfig", "WatermarkExtractionDataset"]

_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class WatermarkExtractionConfig:
    """Everything needed to turn a folder of cover images into extractor training
    data.

    Attributes
    ----------
    processed_root:
        Directory that contains ``train/``, ``validation/`` and ``test/``
        subfolders of processed PNGs (the Phase 2 output,
        ``data/processed/div2k_256``).
    bit_length:
        Payload size in bits. Must equal ``embed_config.bit_length``.
    embed_config:
        The frozen Phase 6 :class:`EmbedConfig`. Its ``bit_length`` is forced to
        ``bit_length`` in :meth:`__post_init__` of the dataset if left at the
        default, so callers only have to set it once.
    image_size:
        Square size every watermarked image tensor is resized to before being
        returned. ``None`` keeps the native size.
    limit:
        Optional cap on the number of cover images used (handy for CPU smoke
        runs). ``None`` uses the whole split.
    seed:
        Base seed for deterministic (validation/test) payloads.
    """

    processed_root: str
    bit_length: int = 64
    embed_config: EmbedConfig = field(default_factory=EmbedConfig)
    image_size: int | None = 256
    limit: int | None = None
    seed: int = 20260901


class WatermarkExtractionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Cover images -> (watermarked image tensor, payload bit tensor)."""

    def __init__(
        self,
        config: WatermarkExtractionConfig,
        split: str,
        *,
        deterministic: bool | None = None,
    ) -> None:
        if split not in _SPLITS:
            raise ValueError(f"unknown split {split!r}; allowed: {sorted(_SPLITS)}")
        self.config = config
        self.split = split
        # validation/test are fixed by default; train resamples payloads.
        self.deterministic = (split != "train") if deterministic is None else deterministic

        self.embed_config = self._aligned_embed_config(config)

        self.root = Path(config.processed_root) / split
        self.images = sorted(self.root.glob("*.png"))
        if not self.images:
            raise FileNotFoundError(f"no processed images in {self.root}")
        if config.limit is not None:
            self.images = self.images[: config.limit]

    @staticmethod
    def _aligned_embed_config(config: WatermarkExtractionConfig) -> EmbedConfig:
        base = config.embed_config
        if base.bit_length == config.bit_length:
            return base
        # Rebuild with the requested payload length, keeping every other frozen
        # baseline parameter intact.
        return EmbedConfig(
            wavelet=base.wavelet,
            subband=base.subband,
            extra_subbands=base.extra_subbands,
            alpha=base.alpha,
            bit_length=config.bit_length,
            start_sv_index=base.start_sv_index,
            mode=base.mode,
        )

    def __len__(self) -> int:
        return len(self.images)

    # -- payload -------------------------------------------------------------

    def _payload_for(self, index: int) -> np.ndarray:
        if self.deterministic:
            rng = np.random.default_rng(self.config.seed + index)
        else:
            rng = np.random.default_rng()  # fresh OS entropy -> new bits per call
        return rng.integers(0, 2, size=self.config.bit_length, dtype=np.int64)

    # -- image ------------------------------------------------------------

    def _load_rgb(self, path: Path) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"unreadable processed image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _to_tensor(self, rgb_u8: np.ndarray) -> torch.Tensor:
        size = self.config.image_size
        if size is not None and rgb_u8.shape[:2] != (size, size):
            rgb_u8 = cv2.resize(rgb_u8, (size, size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(rgb_u8)).permute(2, 0, 1).float().div(255.0)

    # -- item -----------------------------------------------------------

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cover = self._load_rgb(self.images[index])
        bits = self._payload_for(index)
        result = embed(cover, bits.tolist(), self.embed_config)
        image = self._to_tensor(result.watermarked_image)
        target = torch.from_numpy(bits).float()
        return image, target

    # -- diagnostics ---------------------------------------------------

    def sample_paths(self) -> list[str]:
        return [str(p) for p in self.images]
