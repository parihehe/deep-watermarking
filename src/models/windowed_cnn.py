"""Experimental windowed 1D-CNN blind watermark extractor (research-paper style).

**Status: EXPERIMENTAL.** This module is a *separate, isolated* hypothesis
decoder inspired by the attached research paper. It is intentionally not wired
into the production ``src.app.final_model`` path and never replaces the existing
Phase 8 decoder unless it independently passes the project's own evaluation on
the project's own embedding pipeline (see ``docs/windowed_cnn.md``).

How it differs from the production Phase 8 decoder
--------------------------------------------------
* The production decoder feeds the *whole* standardised log1p singular-value
  spectrum into a 1D CNN and emits one logit per bit in a single forward pass.
* This experimental decoder follows the paper: it predicts **one bit at a
  time** from a **fixed local window of 15 singular values** centred on the
  singular value that carries that bit. The network is:

      window of 15 singular values            (1, window_size, 1)?  -> see below
        -> Conv1d(1 -> 32, k=3, same) + ReLU
        -> Conv1d(32 -> 64, k=3, same) + ReLU
        -> dropout(0.3)
        -> Flatten
        -> Dense(64, ReLU)
        -> dropout(0.5)
        -> Dense(1)  -> sigmoid       (the probability bit == 1)

Why the window is centred on the bit's own singular value
---------------------------------------------------------
The frozen Phase 6 embedder (``src.watermark.embed``) puts bit ``i`` into
singular value ``start_sv_index + i`` of the luminance DWT-LL subband using the
relative rule ``S'[i] = S[i] * (1 + alpha * (2*b[i] - 1))``. A blind decoder has
no original, so it must judge whether ``S'[i]`` sits above or below the *smooth
local trend* of the singular-value spectrum. The 15-value window gives each bit
decision 7 neighbours of context on either side - the same idea as the
production decoder's k=7 kernel, but applied per-bit on a local window exactly
as the paper describes.

Feature extraction (identical at training and inference)
-------------------------------------------------------
1. RGB uint8 -> BT.601 luminance Y (matches the embedder's YCrCb Y channel).
2. Single-level DWT (project default: haar, symmetric), keep the LL subband.
3. SVD of the LL subband -> singular values ``sigma`` (descending).
4. ``log1p`` of the singular values.
5. For bit ``i``, take the window ``[idx - r : idx + r + 1]`` (``r = window//2``)
   centred on ``idx = start_sv_index + i``, truncated to the valid range.
6. Per-window standardisation (subtract mean, divide by std) over the 15 values.

Steps 1-4 are the **exact** same feature path the production Phase 8 decoder
uses, so windows are built on the same physics; only the windowing + a
per-bit MLP-on-conv differ. The mapping ``bit_index -> singular-value window``
lives in the single authoritative function ``bit_window_singular_values`` so
dataset creation and live inference cannot drift apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.watermark.dwt import decompose_2d
from src.watermark.svd import decompose

__all__ = [
    "WINDOW_DEFAULT",
    "WindowedCNNConfig",
    "WindowedCNNExtractor",
    "bit_window_singular_values",
    "build_windowed_extractor",
    "load_windowed_checkpoint",
    "luminance_ll_singular_values",
    "save_windowed_checkpoint",
]

# Default window width (number of singular values per bit), matching the paper.
WINDOW_DEFAULT = 15

# BT.601 luma weights, matching OpenCV's COLOR_RGB2YCrCb used by the embedder.
_LUMA_RGB = (0.299, 0.587, 0.114)


@dataclass(frozen=True)
class WindowedCNNConfig:
    """Architecture + feature contract for :class:`WindowedCNNExtractor`.

    Attributes
    ----------
    bit_length:
        Number of payload bits the decoder recovers. Must match the payload
        used by the frozen embedder that generated the training data.
    window_size:
        Number of leading singular values fed as context around each bit's own
        singular value (the paper's 15-value window). Odd; the bit sits at the
        window centre.
    start_sv_index:
        Index of the first singular value modified by the embedder (project
        default 0). The decoder uses the same index to align bit positions.
    conv1_channels / conv2_channels:
        Filter counts of the two 1D conv layers.
    kernel_size:
        1D convolution kernel along the (small) window axis.
    fc_units:
        Width of the fully connected hidden layer.
    dropout_conv / dropout_fc:
        Dropout after the conv stack (paper ~0.3) and before the output
        (paper ~0.5).
    wavelet / mode / subband:
        The DWT parameters of the embedding pipeline that generated the
        training data; the feature extractor must match them exactly.
    image_size:
        Square size inputs are (re)scaled to before the forward pass.
    """

    bit_length: int = 64
    window_size: int = WINDOW_DEFAULT
    start_sv_index: int = 0
    conv1_channels: int = 32
    conv2_channels: int = 64
    kernel_size: int = 3
    fc_units: int = 64
    dropout_conv: float = 0.3
    dropout_fc: float = 0.5
    wavelet: str = "haar"
    mode: str = "symmetric"
    subband: str = "LL"
    image_size: int = 256

    def __post_init__(self) -> None:
        if self.bit_length <= 0:
            raise ValueError(f"bit_length must be positive; got {self.bit_length}")
        if self.window_size <= 0 or self.window_size % 2 == 0:
            raise ValueError(f"window_size must be a positive odd int; got {self.window_size}")
        if self.start_sv_index < 0:
            raise ValueError(f"start_sv_index must be >= 0; got {self.start_sv_index}")
        if self.conv1_channels <= 0 or self.conv2_channels <= 0:
            raise ValueError("conv channel counts must be positive")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd int; got {self.kernel_size}")
        if self.fc_units <= 0:
            raise ValueError(f"fc_units must be positive; got {self.fc_units}")
        if not 0.0 <= self.dropout_conv < 1.0 or not 0.0 <= self.dropout_fc < 1.0:
            raise ValueError("dropouts must be in [0, 1)")
        if self.subband != "LL":
            raise ValueError("the windowed decoder currently supports the LL subband only")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Fixed (parameter-free) feature extraction - shared by training & inference
# ---------------------------------------------------------------------------


def luminance_ll_singular_values(rgb_u8: np.ndarray, config: WindowedCNNConfig) -> np.ndarray:
    """RGB uint8 -> log1p singular values of the DWT-LL luminance subband.

    This is the **single authoritative feature path** used both when generating
    training samples and when running live inference, so the two cannot
    disagree about DWT/SVD/luminance/ordering. Returns ``sigma`` (float64,
    descending), the raw (non-log) singular values; callers apply log1p.
    """
    rgb = np.asarray(rgb_u8, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an RGB image (H, W, 3); got shape {rgb.shape}")
    # BT.601 luma to match the embedder's Y channel exactly.
    y = (rgb * np.asarray(_LUMA_RGB, dtype=np.float64)).sum(axis=2)
    coeffs = decompose_2d(y, wavelet=config.wavelet, mode=config.mode)
    band = coeffs.ll if config.subband == "LL" else None
    if band is None:
        raise ValueError(f"unsupported subband {config.subband!r}")
    return decompose(band).S


def bit_window_singular_values(
    sigma: np.ndarray, bit_index: int, config: WindowedCNNConfig
) -> np.ndarray:
    """Return the standardised 15-value window the decoder uses for ``bit_index``.

    The window is centred on ``start_sv_index + bit_index`` (the singular value
    that carries this bit in the frozen embedder), always exactly
    ``window_size`` wide. The singular-value spectrum is **edge-padded** by
    ``radius = window_size // 2`` on both ends first, so leading bits near index
    0 (whose windows would otherwise truncate) still get a full window: the
    padding repeats the boundary singular value, which is stable for the smooth
    spectrum decay. Values are ``log1p`` then standardised across the window
    (mean-0, std-1), matching the feature scale the network was trained on.

    This is the authoritative ``bit_index -> window`` mapping; both dataset
    creation and live inference must call it (never ``window[i:i+15]``
    directly). The fixed ``window_size`` output is what keeps windows
    stackable in a batch and consistent at every payload position.
    """
    center = config.start_sv_index + bit_index
    radius = config.window_size // 2  # 15 -> 7
    if len(sigma) == 0:
        raise ValueError("cannot build a window from an empty singular-value array")
    padded = np.pad(
        np.clip(sigma, 0.0, None),
        pad_width=(radius, radius),
        mode="edge",
    )  # (len(sigma) + 2*radius,)
    raw = padded[center : center + config.window_size]
    if raw.shape[0] != config.window_size:
        # Only reachable if the padded window runs past both bounds - clamp.
        raise ValueError(
            f"bit_index {bit_index} yields a {raw.shape[0]}-value window; expected "
            f"{config.window_size}"
        )
    s = np.log1p(raw)
    mean = s.mean()
    std = s.std() if s.std() > 0 else 1.0
    return (s - mean) / std


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class WindowedCNNExtractor(nn.Module):
    """Per-bit windowed 1D-CNN blind watermark extractor (experimental).

    Input:  ``(B, window_size)`` float tensor, one standardised window per bit
            decision (the project's DWT-LL singular-value window from
            :func:`bit_window_singular_values`).
    Output: ``(B, 1)`` float tensor of **logits** (pre-sigmoid), one per window
            = the probability bit == 1.
    """

    def __init__(self, config: WindowedCNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or WindowedCNNConfig()

        self.conv_stack = nn.Sequential(
            nn.Conv1d(
                1,
                self.config.conv1_channels,
                self.config.kernel_size,
                padding=self.config.kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(self.config.conv1_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                self.config.conv1_channels,
                self.config.conv2_channels,
                self.config.kernel_size,
                padding=self.config.kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(self.config.conv2_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout_conv = nn.Dropout(self.config.dropout_conv)
        conv_out = self.config.window_size * self.config.conv2_channels
        self.fc1 = nn.Linear(conv_out, self.config.fc_units)
        self.dropout_fc = nn.Dropout(self.config.dropout_fc)
        self.fc2 = nn.Linear(self.config.fc_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.config.window_size:
            raise ValueError(
                f"expected 2D (B, window_size={self.config.window_size}); got {tuple(x.shape)}"
            )
        h = x.unsqueeze(1)  # (B, 1, window_size)
        h = self.conv_stack(h)
        h = self.dropout_conv(h)
        h = h.reshape(h.shape[0], -1)  # flatten
        h = self.dropout_fc(F.relu(self.fc1(h)))
        return self.fc2(h)  # (B, 1) logits

    @torch.no_grad()
    def predict_proba(self, window: torch.Tensor) -> torch.Tensor:
        """Per-bit probabilities in ``[0, 1]`` (caller sets eval mode)."""
        return torch.sigmoid(self.forward(window))

    @torch.no_grad()
    def predict_bit(self, window: torch.Tensor) -> torch.Tensor:
        """Hard 0/1 prediction, shape ``(B, 1)`` (long dtype)."""
        return (self.predict_proba(window) > 0.5).long()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_windowed_extractor(config: WindowedCNNConfig | None = None) -> WindowedCNNExtractor:
    """Factory so callers need not import the class directly."""
    return WindowedCNNExtractor(config)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_windowed_checkpoint(
    path: str,
    model: WindowedCNNExtractor,
    *,
    extra: dict | None = None,
) -> None:
    """Write a self-describing checkpoint (stores the config alongside weights)."""
    payload: dict = {
        "phase": "experimental-windowed-cnn",
        "format": "windowed-cnn-extractor/1",
        "windowed_config": model.config.to_dict(),
        "model_state": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_windowed_checkpoint(
    path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[WindowedCNNExtractor, dict]:
    """Rebuild a :class:`WindowedCNNExtractor` from a checkpoint written by
    :func:`save_windowed_checkpoint`. Returns ``(model_in_eval_mode, checkpoint_dict)``."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    config = WindowedCNNConfig(**checkpoint["windowed_config"])
    model = WindowedCNNExtractor(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
