"""Phase 8 — Blind CNN watermark extractor.

This module is **new Phase 8 code** and is completely separate from the frozen
Phase 6 DWT-SVD baseline in ``src/watermark/``. Nothing here modifies the
baseline; the baseline is only *used* (via ``src.watermark.embed.embed``) to
generate training examples elsewhere in Phase 8.

Why the model has this shape
----------------------------
The frozen baseline embeds bit ``i`` by scaling the ``i``-th singular value of
the luminance DWT-LL subband::

    S'[i] = S[i] * (1 + alpha * (2 * b[i] - 1))

The *non-blind* decoder recovers the bit by comparing ``S'[i]`` against the
original ``S[i]``. A blind decoder has no original, so it must instead learn a
prior for the natural (unwatermarked) singular-value spectrum — which decays
smoothly — and detect the local ``+-alpha`` deviation from that smooth trend.

A generic 2D CNN on raw RGB pixels does not do this: the watermark is a
sub-perceptual, spatially diffuse residual, and pooling averages it away (this
was measured — such a model trained to exactly chance). So the extractor is
built to operate where the signal actually lives:

    watermarked RGB image  (B, 3, H, W) in [0, 1]
      -> fixed BT.601 luminance                       Y   (B, 1, H, W)
      -> fixed single-level Haar DWT, LL subband      LL  (B, 1, H/2, W/2)
      -> batched SVD, singular values only            sigma (B, K)
      -> log1p + per-sample standardisation
      -> 1D CNN along the singular-value index axis
      -> 1x1 conv head -> one logit per index
      -> first `bit_length` logits are the payload

Everything from the image to ``sigma`` is a fixed, differentiable feature
extractor (no learnable parameters, no original image). All learning happens in
the 1D CNN, whose receptive field lets each output bit see its singular value
*in the context of its neighbours* — exactly the "is this value above or below
the smooth local trend?" question the blind task reduces to.

The public surface (``ExtractorConfig``, ``BlindCNNExtractor`` with a
``(B, bit_length)`` logit output, ``predict_bits``, and the checkpoint helpers)
is unchanged from the first Phase 8 draft, so the training pipeline, the
inference wrapper and the tests are unaffected by the internal redesign.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "BlindCNNExtractor",
    "ExtractorConfig",
    "build_extractor",
    "load_checkpoint",
    "save_checkpoint",
]

# BT.601 luma weights, matching OpenCV's COLOR_RGB2YCrCb used by the embedder.
_LUMA_RGB = (0.299, 0.587, 0.114)


@dataclass(frozen=True)
class ExtractorConfig:
    """Architecture + input contract for :class:`BlindCNNExtractor`.

    Attributes
    ----------
    bit_length:
        Number of payload bits the head predicts. Must match the payload used
        by the Phase 6 embedder that generated the training data.
    in_channels:
        Input image channels. 3 for RGB (the only mode used in Phase 8).
    sv_features:
        Number of leading singular values fed to the 1D CNN. Gives the network
        trailing-spectrum context beyond the payload region. Clamped at runtime
        to what the input resolution actually provides, and never below
        ``bit_length``.
    conv_channels:
        Width of every hidden 1D convolution.
    conv_layers:
        Number of hidden 1D convolution blocks (Conv1d + BatchNorm + ReLU).
    kernel_size:
        Convolution kernel size along the singular-value axis (odd). Sets how
        many neighbouring singular values each bit decision can see.
    dropout:
        Dropout probability applied before the 1x1 output convolution.
    image_size:
        Square size the training / inference code resizes inputs to before the
        forward pass. The model itself is resolution-agnostic.
    """

    bit_length: int = 64
    in_channels: int = 3
    sv_features: int = 128
    conv_channels: int = 64
    conv_layers: int = 4
    kernel_size: int = 7
    dropout: float = 0.1
    image_size: int = 256

    def __post_init__(self) -> None:
        if self.bit_length <= 0:
            raise ValueError(f"bit_length must be positive; got {self.bit_length}")
        if self.in_channels != 3:
            raise ValueError(f"in_channels must be 3 (RGB); got {self.in_channels}")
        if self.sv_features <= 0:
            raise ValueError(f"sv_features must be positive; got {self.sv_features}")
        if self.conv_channels <= 0:
            raise ValueError(f"conv_channels must be positive; got {self.conv_channels}")
        if self.conv_layers <= 0:
            raise ValueError(f"conv_layers must be positive; got {self.conv_layers}")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd int; got {self.kernel_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {self.dropout}")
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive; got {self.image_size}")

    def to_dict(self) -> dict:
        return asdict(self)


class _SpectrumFeatureExtractor(nn.Module):
    """Fixed (parameter-free) map: watermarked RGB image -> LL singular values.

    Uses buffers, not parameters, so it moves with ``.to(device)`` and is saved
    in the state dict but contributes nothing to ``num_parameters``. The
    original image is never involved.
    """

    def __init__(self) -> None:
        super().__init__()
        luma = torch.tensor(_LUMA_RGB, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("luma", luma)
        # Haar LL analysis filter: average of each 2x2 block, stride 2.
        ll_kernel = torch.full((1, 1, 2, 2), 0.5, dtype=torch.float32)
        self.register_buffer("ll_kernel", ll_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) in [0, 1]  ->  Y in [0, 255] to match the embedder scale
        y = (x * self.luma.to(x.dtype)).sum(dim=1, keepdim=True) * 255.0
        if y.shape[-1] % 2 or y.shape[-2] % 2:
            y = y[..., : y.shape[-2] // 2 * 2, : y.shape[-1] // 2 * 2]
        ll = F.conv2d(y, self.ll_kernel.to(y.dtype), stride=2)  # (B, 1, H/2, W/2)
        sigma = torch.linalg.svdvals(ll.squeeze(1))  # (B, min(H/2, W/2))
        return sigma


class _ConvBlock1d(nn.Module):
    """Conv1d -> BatchNorm1d -> ReLU, length-preserving."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BlindCNNExtractor(nn.Module):
    """Blind CNN watermark extractor (Phase 8).

    Input:  ``(B, 3, H, W)`` float tensor with pixels in ``[0, 1]``.
    Output: ``(B, bit_length)`` float tensor of **logits** (pre-sigmoid).
    """

    def __init__(self, config: ExtractorConfig | None = None) -> None:
        super().__init__()
        self.config = config or ExtractorConfig()

        self.features = _SpectrumFeatureExtractor()

        channels = self.config.conv_channels
        blocks: list[nn.Module] = [_ConvBlock1d(1, channels, self.config.kernel_size)]
        for _ in range(self.config.conv_layers - 1):
            blocks.append(_ConvBlock1d(channels, channels, self.config.kernel_size))
        self.conv = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(self.config.dropout)
        self.head = nn.Conv1d(channels, 1, kernel_size=1)

    # -- fixed feature stage --------------------------------------------

    def _spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, H, W)`` image -> ``(B, L)`` standardised log singular values,
        length ``L = max(bit_length, sv_features)`` (zero-padded if the input
        resolution is too small to supply that many)."""
        sigma = self.features(x)  # (B, K)
        target = max(self.config.bit_length, self.config.sv_features)
        k = sigma.shape[-1]
        if k >= target:
            sigma = sigma[:, :target]
        else:
            sigma = F.pad(sigma, (0, target - k))
        s = torch.log1p(sigma.clamp_min(0.0))
        mean = s.mean(dim=1, keepdim=True)
        std = s.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (s - mean) / std

    # -- forward -------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected a 4D (B, C, H, W) tensor; got shape {tuple(x.shape)}")
        if x.shape[1] != self.config.in_channels:
            raise ValueError(
                f"expected {self.config.in_channels} input channels; got {x.shape[1]}"
            )
        s = self._spectrum(x).unsqueeze(1)  # (B, 1, L)
        h = self.conv(s)
        h = self.dropout(h)
        logits = self.head(h).squeeze(1)  # (B, L)
        return logits[:, : self.config.bit_length]

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Per-bit probabilities in ``[0, 1]`` (caller sets eval mode)."""
        return torch.sigmoid(self.forward(x))

    @torch.no_grad()
    def predict_bits(self, x: torch.Tensor) -> torch.Tensor:
        """Hard 0/1 payload prediction, shape ``(B, bit_length)`` (long dtype)."""
        return (self.predict_proba(x) > 0.5).long()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_extractor(config: ExtractorConfig | None = None) -> BlindCNNExtractor:
    """Small factory kept so callers do not import the class directly."""
    return BlindCNNExtractor(config)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: BlindCNNExtractor,
    *,
    extra: dict | None = None,
) -> None:
    """Write a self-describing checkpoint.

    The checkpoint stores the ``ExtractorConfig`` alongside the weights so that
    :func:`load_checkpoint` can rebuild the exact architecture with no external
    metadata.
    """
    payload: dict = {
        "phase": 8,
        "format": "blind-cnn-extractor/2",
        "extractor_config": model.config.to_dict(),
        "model_state": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[BlindCNNExtractor, dict]:
    """Rebuild a :class:`BlindCNNExtractor` from a checkpoint written by
    :func:`save_checkpoint`.

    Returns ``(model_in_eval_mode, full_checkpoint_dict)``.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    config = ExtractorConfig(**checkpoint["extractor_config"])
    model = BlindCNNExtractor(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
