"""PyTorch dataset for deterministic processed DIV2K host images."""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class Div2KHostDataset(Dataset[tuple[torch.Tensor, str]]):
    """Load a single processed split as normalized RGB tensors in [0, 1]."""

    def __init__(self, processed_root: str | Path, split: str) -> None:
        self.root = Path(processed_root) / split
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.images = sorted(self.root.glob("*.png"))
        if not self.images:
            raise FileNotFoundError(f"No processed images in {self.root}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.images[index]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Unreadable processed image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        return tensor, path.name
