"""Stream-ingest DIV2K official archives into data/processed/div2k_256.

Disk-friendly alternative to scripts/setup_div2k.py: reads the HR PNGs straight
from the .zip archives one at a time, resizes to 256x256 (INTER_AREA, matching
src/utils/dataset_pipeline.process_dataset), and writes only the small processed
squares - no HR PNG files ever touch disk. Uses the documented 700/100/100 split:

    train      = DIV2K IDs 0001-0700
    validation = DIV2K IDs 0701-0800
    test       = DIV2K IDs 0801-0900

Run: python scripts/ingest_div2k_stream.py
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT / "data" / "raw" / "archives"
IMAGE_SIZE = 256
ID_PNG = re.compile(r"^(?:DIV2K_(?:train|valid)_HR/)?(\d{4})\.png$")


def _split_for(image_id: int) -> str:
    if 1 <= image_id <= 700:
        return "train"
    if 701 <= image_id <= 800:
        return "validation"
    if 801 <= image_id <= 900:
        return "test"
    return ""


def main() -> int:
    train_zip = ARCHIVES / "DIV2K_train_HR.zip"
    valid_zip = ARCHIVES / "DIV2K_valid_HR.zip"
    for z in (train_zip, valid_zip):
        if not z.is_file():
            print(f"missing archive: {z} (curl the official DIV2K HR zips into data/raw/archives/)")
            return 1

    out_root = ROOT / "data" / "processed" / "div2k_256"
    written: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    for z in (train_zip, valid_zip):
        print(f"[ingest] streaming {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            for member in zf.infolist():
                match = ID_PNG.match(member.filename)
                if not match:
                    continue
                image_id = int(match.group(1))
                split = _split_for(image_id)
                if not split:
                    continue
                target = out_root / split / f"{image_id:04d}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_file():
                    continue
                data = zf.read(member)
                arr = np.frombuffer(data, dtype="uint8")
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    print(f"  unreadable {member.filename}")
                    continue
                resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(str(target), resized):
                    print(f"  write failure {target}")
                    return 1
                written[split] += 1
                if written[split] % 100 == 0:
                    print(f"  ... {member.filename} -> {split} (total {written[split]})")

    print(f"[ingest] done: {written}")
    expected = {"train": 700, "validation": 100, "test": 100}
    if written != expected:
        print(f"[ingest] WARNING: expected {expected}, got {written}")
        return 1
    print(f"[ingest] data/processed/div2k_256 ready (image_size={IMAGE_SIZE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())