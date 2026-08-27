"""Deterministic acquisition, preprocessing, and validation for public DIV2K HR data."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class DatasetConfig:
    archive_dir: Path
    extracted_dir: Path
    processed_dir: Path
    image_size: int
    splits: dict[str, tuple[int, int]]
    expected_counts: dict[str, int]


def load_config(path: str | Path) -> DatasetConfig:
    """Load and validate the project dataset configuration."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["dataset"]
    root = Path(path).resolve().parents[1]
    splits = {name: tuple(bounds) for name, bounds in raw["splits"].items()}
    config = DatasetConfig(
        archive_dir=root / raw["raw_archive_dir"],
        extracted_dir=root / raw["extracted_dir"],
        processed_dir=root / raw["processed_dir"],
        image_size=int(raw["image_size"]),
        splits=splits,
        expected_counts={name: int(count) for name, count in raw["expected_counts"].items()},
    )
    if config.image_size <= 0:
        raise ValueError("image_size must be positive")
    for name, (first, last) in config.splits.items():
        if first > last or last - first + 1 != config.expected_counts[name]:
            raise ValueError(f"Invalid split specification for {name}")
    return config


def verify_archive(path: Path, expected_png_count: int) -> None:
    """Ensure an archive is readable and contains its expected HR image count."""
    with zipfile.ZipFile(path) as archive:
        invalid = archive.testzip()
        if invalid is not None:
            raise ValueError(f"Archive CRC failure in {path.name}: {invalid}")
        png_count = sum(info.filename.lower().endswith(".png") for info in archive.infolist())
    if png_count != expected_png_count:
        raise ValueError(f"{path.name} has {png_count} PNGs; expected {expected_png_count}")


def extract_archives(config: DatasetConfig) -> None:
    """Extract verified official archives once, preserving their original file names."""
    archives = (("DIV2K_train_HR.zip", 800), ("DIV2K_valid_HR.zip", 100))
    config.extracted_dir.mkdir(parents=True, exist_ok=True)
    for archive_name, count in archives:
        archive_path = config.archive_dir / archive_name
        if not archive_path.is_file():
            raise FileNotFoundError(f"Missing verified archive: {archive_path}")
        verify_archive(archive_path, count)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".png"):
                    continue
                target = config.extracted_dir / Path(member.filename).name
                if not target.exists() or target.stat().st_size != member.file_size:
                    with archive.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)


def split_for_image_id(image_id: int, splits: dict[str, tuple[int, int]]) -> str:
    for split, (first, last) in splits.items():
        if first <= image_id <= last:
            return split
    raise ValueError(f"Image ID {image_id:04d} belongs to no configured split")


def iter_raw_images(config: DatasetConfig) -> Iterable[tuple[str, int, Path]]:
    for image_path in sorted(config.extracted_dir.glob("*.png")):
        try:
            image_id = int(image_path.stem)
        except ValueError as error:
            raise ValueError(f"Unexpected DIV2K filename: {image_path.name}") from error
        yield split_for_image_id(image_id, config.splits), image_id, image_path


def image_record(image_path: Path, split: str, image_id: int) -> dict[str, object]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unreadable/non-RGB image: {image_path}")
    height, width = image.shape[:2]
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    return {
        "split": split,
        "image_id": image_id,
        "filename": image_path.name,
        "width": width,
        "height": height,
        "channels": 3,
        "file_bytes": image_path.stat().st_size,
        "sha256": digest,
    }


def process_dataset(config: DatasetConfig) -> list[dict[str, object]]:
    """Validate raw images and write deterministic square RGB PNGs by split."""
    records: list[dict[str, object]] = []
    for split, image_id, raw_path in iter_raw_images(config):
        record = image_record(raw_path, split, image_id)
        records.append(record)
        output_dir = config.processed_dir / split
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / raw_path.name
        image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        resized = cv2.resize(image, (config.image_size, config.image_size), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(output_path), resized):
            raise OSError(f"Could not write processed image: {output_path}")
    observed = {split: sum(record["split"] == split for record in records) for split in config.splits}
    if observed != config.expected_counts:
        raise ValueError(f"Split-count mismatch: {observed}, expected {config.expected_counts}")
    return records


def write_metadata(config: DatasetConfig, records: list[dict[str, object]]) -> Path:
    """Write image-level CSV and dataset summary JSON next to processed data."""
    metadata_dir = config.processed_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metadata_dir / "images.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    dimensions = np.array([(record["width"], record["height"]) for record in records], dtype=np.int64)
    summary = {
        "dataset": "DIV2K public HR re-split",
        "processed_image_size": config.image_size,
        "total_images": len(records),
        "split_counts": {split: sum(record["split"] == split for record in records) for split in config.splits},
        "raw_width": {"min": int(dimensions[:, 0].min()), "max": int(dimensions[:, 0].max()), "mean": float(dimensions[:, 0].mean())},
        "raw_height": {"min": int(dimensions[:, 1].min()), "max": int(dimensions[:, 1].max()), "mean": float(dimensions[:, 1].mean())},
        "leakage_check": "passed: each image ID appears in exactly one configured split",
    }
    (metadata_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return csv_path
