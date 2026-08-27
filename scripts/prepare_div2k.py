"""Prepare and validate the public DIV2K HR re-split."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dataset_pipeline import extract_archives, load_config, process_dataset, write_metadata


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "dataset.yaml")
    extract_archives(config)
    records = process_dataset(config)
    metadata = write_metadata(config, records)
    print(f"Prepared {len(records)} images; metadata: {metadata}")


if __name__ == "__main__":
    main()
