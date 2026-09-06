"""Training data for the experimental windowed 1D-CNN blind extractor.

Each sample is synthesized by running the **frozen Phase 6 DWT-SVD embedder**
(``src.watermark.embed.embed``) on a real processed DIV2K image with a random
binary payload, then going back through the SAME extraction-side feature path
(:func:`src.models.windowed_cnn.luminance_ll_singular_values` +
``bit_window_singular_values``) to build a per-bit 15-value window paired with
that bit's ground-truth label.

The label for window ``i`` **necessarily** comes from the actual bit that was
embedded at singular-value position ``start_sv_index + i`` - it cannot drift
from the embedder because both use the single authoritative mapping in
``bit_window_singular_values``.

Split policy matches the project:
* ``train`` (deterministic=False): fresh random payload every ``__getitem__``,
  so the network sees new (image, payload) pairs each epoch.
* ``validation``/``test`` (deterministic=True): payload is a fixed function of
  ``(seed, index)``, identical on every run/epoch.

This module imports the frozen baseline but never modifies it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.windowed_cnn import (
    WindowedCNNConfig,
    bit_window_singular_values,
    luminance_ll_singular_values,
)
from src.watermark.embed import EmbedConfig, embed

__all__ = ["WindowedExtractionConfig", "WindowedExtractionDataset"]

_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class WindowedExtractionConfig:
    """Everything needed to turn a folder of cover images into windowed-CNN
    training data.

    Attributes
    ----------
    processed_root:
        Directory containing ``train/``, ``validation/``, ``test/`` subfolders
        of processed PNGs (the Phase 2 output, ``data/processed/div2k_256``).
    bit_length:
        Payload size in bits. Must equal ``embed_config.bit_length``.
    model_config:
        The :class:`WindowedCNNConfig` (its ``start_sv_index`` / ``wavelet`` /
        ``mode`` / ``subband`` / ``window_size`` drive both the embed config
        alignment and the window construction).
    embed_config:
        The frozen Phase 6 :class:`EmbedConfig`. Its embedding parameters must
        agree with ``model_config`` (same wavelet / mode / subband /
        alpha / bit_length). If ``bit_length`` differs, the embed config's is
        overridden to match and kept consistent with ``model_config``.
    image_size:
        Square size every watermarked image is resized to before being
        processed. ``None`` keeps the native size.
    limit:
        Optional cap on the number of cover images used.
    seed:
        Base seed for deterministic (validation/test) payloads.
    """

    processed_root: str
    bit_length: int = 64
    model_config: WindowedCNNConfig = field(default_factory=WindowedCNNConfig)
    embed_config: EmbedConfig = field(default_factory=EmbedConfig)
    image_size: int | None = 256
    limit: int | None = None
    seed: int = 20260901


class WindowedExtractionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Cover images -> (per-bit 15-value window tensor, ground-truth bit tensor)."""

    def __init__(
        self,
        config: WindowedExtractionConfig,
        split: str,
        *,
        deterministic: bool | None = None,
    ) -> None:
        if split not in _SPLITS:
            raise ValueError(f"unknown split {split!r}; allowed: {sorted(_SPLITS)}")
        self.config = config
        self.split = split
        self.deterministic = (split != "train") if deterministic is None else deterministic

        self.model_config = self._aligned_model_config(config)
        self.embed_config = self._aligned_embed_config(config, self.model_config)

        self.root = Path(config.processed_root) / split
        self.images = sorted(self.root.glob("*.png"))
        if not self.images:
            raise FileNotFoundError(f"no processed images in {self.root}")
        if config.limit is not None:
            self.images = self.images[: config.limit]

    @staticmethod
    def _aligned_model_config(config: WindowedExtractionConfig) -> WindowedCNNConfig:
        base = config.model_config
        if base.bit_length == config.bit_length:
            return base
        return WindowedCNNConfig(
            bit_length=config.bit_length,
            window_size=base.window_size,
            start_sv_index=base.start_sv_index,
            conv1_channels=base.conv1_channels,
            conv2_channels=base.conv2_channels,
            kernel_size=base.kernel_size,
            fc_units=base.fc_units,
            dropout_conv=base.dropout_conv,
            dropout_fc=base.dropout_fc,
            wavelet=base.wavelet,
            mode=base.mode,
            subband=base.subband,
            image_size=base.image_size,
        )

    @staticmethod
    def _aligned_embed_config(
        config: WindowedExtractionConfig,
        model: WindowedCNNConfig,
    ) -> EmbedConfig:
        m = model
        base = config.embed_config
        return EmbedConfig(
            wavelet=m.wavelet,
            subband=m.subband,
            extra_subbands=base.extra_subbands,
            alpha=base.alpha,
            bit_length=m.bit_length,
            start_sv_index=m.start_sv_index,
            mode=m.mode,
        )

    def __len__(self) -> int:
        return len(self.images)

    # -- payload -------------------------------------------------------------

    def _payload_for(self, index: int) -> np.ndarray:
        if self.deterministic:
            rng = np.random.default_rng(self.config.seed + index)
        else:
            rng = np.random.default_rng()
        return rng.integers(0, 2, size=self.config.bit_length, dtype=np.int64)

    # -- image ---------------------------------------------------------------

    def _load_rgb(self, path: Path) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"unreadable processed image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # -- item -----------------------------------------------------------

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cover = self._load_rgb(self.images[index])
        bits = self._payload_for(index)
        result = embed(cover, bits.tolist(), self.embed_config)
        watermarked = result.watermarked_image
        if self.config.image_size is not None and watermarked.shape[:2] != (
            self.config.image_size,
            self.config.image_size,
        ):
            watermarked = cv2.resize(
                watermarked,
                (self.config.image_size, self.config.image_size),
                interpolation=cv2.INTER_AREA,
            )
        # The SAME feature path used at live inference (never the original image).
        sigma = luminance_ll_singular_values(watermarked, self.model_config)
        windows = []
        labels = []
        for i in range(self.config.bit_length):
            w = bit_window_singular_values(sigma, i, self.model_config)
            windows.append(torch.from_numpy(w).float())
            labels.append(int(bits[i]))
        window_t = torch.stack(windows)  # (bit_length, window_size)
        label_t = torch.tensor(labels, dtype=torch.float32)  # (bit_length,)
        return window_t, label_t

    def sample_paths(self) -> list[str]:
        return [str(p) for p in self.images]
