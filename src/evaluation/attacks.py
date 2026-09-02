"""Phase 10 - image attacks for watermark robustness testing.

This module is **new Phase 10 code**. It contains only forward image
distortions ("attacks") an adversary or a lossy channel could apply to a
*watermarked* image. It imports nothing from the watermarking pipeline: the
frozen Phase 6 baseline, the Phase 7 UI and the Phase 8 CNN are untouched and
unreferenced here.

Contract
--------
Every attack takes an RGB ``uint8`` array ``(H, W, 3)`` and returns an RGB
``uint8`` array. Photometric attacks (JPEG, noise, blur, median) keep the shape;
geometric attacks (resize, rotate, crop) may change the pixels' geometry but
always return a full-frame image of the *same* shape - a receiver still has to
re-synchronise it to the embedding resolution before extraction, which the
Phase 10 harness does explicitly. Phase 10 does **not** attempt to invert a
geometric attack (no de-rotation, no registration); measuring the raw,
un-synchronised damage is the point. Synchronisation is a later-phase concern.

Determinism
-----------
The only randomised attacks are ``gaussian_noise`` and any ``combined`` pipeline
that includes it. Both take an explicit integer ``seed`` so a run reproduces
exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "ATTACKS",
    "AttackSpec",
    "apply_attack",
    "attack_names",
    "center_crop",
    "combined",
    "gaussian_blur",
    "gaussian_noise",
    "identity",
    "jpeg_compress",
    "median_filter",
    "resize_roundtrip",
    "rotate",
    "severity_of",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _validate(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an RGB image (H, W, 3); got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image; got dtype {arr.dtype}")
    return arr


def _to_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def _odd_ksize(ksize: int) -> int:
    k = int(ksize)
    if k < 1:
        raise ValueError(f"ksize must be >= 1; got {ksize}")
    return k if k % 2 == 1 else k + 1


# ---------------------------------------------------------------------------
# individual attacks
# ---------------------------------------------------------------------------

def identity(image: np.ndarray) -> np.ndarray:
    """No-op attack - the severity-zero reference point."""
    return _validate(image).copy()


def jpeg_compress(image: np.ndarray, *, quality: int) -> np.ndarray:
    """Lossy JPEG re-encode/decode at the given quality (1..100; lower = worse)."""
    arr = _validate(image)
    q = int(quality)
    if not 1 <= q <= 100:
        raise ValueError(f"quality must be in [1, 100]; got {quality}")
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("cv2.imencode failed for JPEG")
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def gaussian_noise(image: np.ndarray, *, sigma: float, seed: int = 0) -> np.ndarray:
    """Add zero-mean Gaussian noise of standard deviation ``sigma`` (0..255 units)."""
    arr = _validate(image).astype(np.float64)
    s = float(sigma)
    if s < 0:
        raise ValueError(f"sigma must be >= 0; got {sigma}")
    rng = np.random.default_rng(int(seed))
    noisy = arr + rng.normal(0.0, s, size=arr.shape)
    return _to_u8(noisy)


def gaussian_blur(image: np.ndarray, *, ksize: int) -> np.ndarray:
    """Gaussian low-pass filter with an ``ksize x ksize`` kernel (sigma from ksize)."""
    arr = _validate(image)
    k = _odd_ksize(ksize)
    return cv2.GaussianBlur(arr, (k, k), 0)


def median_filter(image: np.ndarray, *, ksize: int) -> np.ndarray:
    """Median filter with an ``ksize x ksize`` window (non-linear denoise/attack)."""
    arr = _validate(image)
    k = _odd_ksize(ksize)
    return cv2.medianBlur(arr, k)


def resize_roundtrip(image: np.ndarray, *, scale: float) -> np.ndarray:
    """Downscale by ``scale`` (0<scale<=1) then upscale back to the original size.

    Loses the high-frequency detail that cannot survive the smaller grid.
    """
    arr = _validate(image)
    f = float(scale)
    if not 0.0 < f <= 1.0:
        raise ValueError(f"scale must be in (0, 1]; got {scale}")
    h, w = arr.shape[:2]
    sh, sw = max(1, round(h * f)), max(1, round(w * f))
    small = cv2.resize(arr, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def rotate(image: np.ndarray, *, degrees: float, border: str = "reflect") -> np.ndarray:
    """Rotate about the image centre, keeping the original canvas size.

    The rotation is **not** undone anywhere in Phase 10 - this measures the raw
    geometric-desynchronisation damage.
    """
    arr = _validate(image)
    deg = float(degrees)
    border_flag = {
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE,
        "constant": cv2.BORDER_CONSTANT,
        "wrap": cv2.BORDER_WRAP,
    }.get(border)
    if border_flag is None:
        raise ValueError(f"unknown border mode {border!r}")
    h, w = arr.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    return cv2.warpAffine(arr, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=border_flag)


def center_crop(image: np.ndarray, *, keep: float) -> np.ndarray:
    """Keep the central ``keep`` fraction of each axis; zero-pad back to full size.

    ``keep = 1.0`` is a no-op; ``keep = 0.5`` blacks out the outer border.
    """
    arr = _validate(image)
    f = float(keep)
    if not 0.0 < f <= 1.0:
        raise ValueError(f"keep must be in (0, 1]; got {keep}")
    h, w = arr.shape[:2]
    kh, kw = max(1, round(h * f)), max(1, round(w * f))
    top, left = (h - kh) // 2, (w - kw) // 2
    out = np.zeros_like(arr)
    out[top:top + kh, left:left + kw] = arr[top:top + kh, left:left + kw]
    return out


def combined(image: np.ndarray, *, steps: Sequence[dict], seed: int = 0) -> np.ndarray:
    """Apply a sequence of attacks in order.

    Each step is ``{"name": <attack>, "params": {...}}``. A ``gaussian_noise``
    step inside the pipeline is seeded from ``seed`` plus its position so the
    pipeline is deterministic and two noise steps differ.
    """
    out = _validate(image)
    for position, step in enumerate(steps):
        name = step["name"]
        params = dict(step.get("params", {}))
        if name == "combined":
            raise ValueError("combined pipelines cannot nest a 'combined' step")
        if name == "gaussian_noise":
            params.setdefault("seed", int(seed) + 1 + position)
        out = apply_attack(name, out, params)
    return out


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttackSpec:
    """Metadata for one attack family.

    Attributes
    ----------
    name:
        Registry key.
    func:
        The attack callable.
    severity_param:
        The kwarg that is the "strength" axis for plots (empty for ``identity``
        and ``combined``, which the harness indexes ordinally).
    higher_param_is_stronger:
        Whether a larger ``severity_param`` value means a *more damaging* attack.
        ``False`` for JPEG quality, resize scale and crop keep-fraction.
    randomised:
        Whether the attack consumes a ``seed``.
    """

    name: str
    func: Callable[..., np.ndarray]
    severity_param: str
    higher_param_is_stronger: bool
    randomised: bool = False


ATTACKS: dict[str, AttackSpec] = {
    "identity": AttackSpec("identity", identity, "", True),
    "jpeg_compress": AttackSpec("jpeg_compress", jpeg_compress, "quality", False),
    "gaussian_noise": AttackSpec("gaussian_noise", gaussian_noise, "sigma", True, randomised=True),
    "gaussian_blur": AttackSpec("gaussian_blur", gaussian_blur, "ksize", True),
    "median_filter": AttackSpec("median_filter", median_filter, "ksize", True),
    "resize_roundtrip": AttackSpec("resize_roundtrip", resize_roundtrip, "scale", False),
    "rotate": AttackSpec("rotate", rotate, "degrees", True),
    "center_crop": AttackSpec("center_crop", center_crop, "keep", False),
    "combined": AttackSpec("combined", combined, "", True, randomised=True),
}


def attack_names() -> list[str]:
    return list(ATTACKS)


def apply_attack(name: str, image: np.ndarray, params: dict | None = None, *, seed: int | None = None) -> np.ndarray:
    """Dispatch to a registered attack.

    ``seed`` is injected only for randomised attacks and only when the params do
    not already carry one.
    """
    if name not in ATTACKS:
        raise ValueError(f"unknown attack {name!r}; known: {sorted(ATTACKS)}")
    spec = ATTACKS[name]
    kwargs = dict(params or {})
    if spec.randomised and seed is not None:
        kwargs.setdefault("seed", int(seed))
    return spec.func(image, **kwargs)


def severity_of(name: str, params: dict) -> float | None:
    """The numeric strength value for one attack instance, or ``None`` when the
    attack has no single scalar strength (``identity`` / ``combined``)."""
    if name not in ATTACKS:
        raise ValueError(f"unknown attack {name!r}")
    key = ATTACKS[name].severity_param
    if not key or key not in params:
        return None
    return float(params[key])
