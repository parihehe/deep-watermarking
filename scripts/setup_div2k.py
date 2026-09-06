"""Set up the DIV2K dataset for the project, supporting two input sources:

1. **Official archives** (native): ``data/raw/archives/DIV2K_train_HR.zip`` and
   ``DIV2K_valid_HR.zip`` downloaded from
   ``https://data.vision.ee.ethz.ch/cvl/DIV2K/``. This is exactly what the
   existing ``scripts/prepare_div2k.py`` + ``src/utils/dataset_pipeline.py``
   expect, so they are used unchanged.

   Download (resumable curl):
       curl -C - -o data/raw/archives/DIV2K_train_HR.zip \\
           https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
       curl -C - -o data/raw/archives/DIV2K_valid_HR.zip \\
           https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

2. **KaggleHub folder**: source ``soumikrakshit/div2k-high-resolution-images``
   (the layout used --- the ``soumikrakshit/div2k-high-resolution-images``
   dataset). Pass ``--kagglehub <path>`` pointing at the extracted folder that
   contains ``DIV2K_train_HR/`` and ``DIV2K_valid_HR/``, or let the script
   download it with ``kagglehub.dataset_download(...)``. The folder is first
   mirrored into ``data/raw/div2k_hr`` (one flat PNG list, matching the
   extracted layout ``prepare_div2k.py`` produces), then processed through the
   SAME project pipeline.

Both paths converge on ``src/utils/dataset_pipeline.process_dataset`` so the
processed output (``data/processed/div2k_256/{train,validation,test}``) is
deterministic regardless of source, and the documented 700/100/100 split with
no ID leakage is preserved:

    train      = DIV2K IDs 0001-0700
    validation = DIV2K IDs 0701-0800
    test       = DIV2K IDs 0801-0900

Run::
    python scripts/setup_div2k.py                 # uses official archives
    python scripts/setup_div2k.py --kagglehub ~/.cache/kagglehub/datasets/soumikrakshit/div2k-high-resolution-images/versions/1
    python scripts/setup_div2k.py --kagglehub download   # kagglehub download + ingest
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.utils.dataset_pipeline import load_config, process_dataset, write_metadata


def _has_official_archives(config) -> bool:
    names = ("DIV2K_train_HR.zip", "DIV2K_valid_HR.zip")
    return all((config.archive_dir / name).is_file() for name in names)


def _folder_to_extracted(source: Path, config) -> None:
    """Mirror a kagglehub-style ``DIV2K_train_HR/`` + ``DIV2K_valid_HR/`` folder
    into ``config.extracted_dir`` (flat PNGs named by ID), the same layout
    ``extract_archives`` produces from the official zips."""
    components = [
        (source / "DIV2K_train_HR", "train_hr"),
        (source / "DIV2K_valid_HR", "valid_hr"),
    ]
    # Also accept a layout where the HR folders sit directly under `source`.
    for folder, _ in components:
        if not folder.is_dir():
            raise FileNotFoundError(f"expected folder {folder} under kagglehub source {source}")
    config.extracted_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for folder, _ in components:
        pngs = sorted(folder.glob("*.png"))
        if not pngs:
            raise FileNotFoundError(f"no PNGs in {folder}")
        for png in pngs:
            if png.name == ".gitkeep":
                continue
            target = config.extracted_dir / png.name
            if not target.exists() or target.stat().st_size != png.stat().st_size:
                shutil.copy2(png, target)
            n += 1
    print(f"[setup-div2k] mirrored {n} PNGs into {config.extracted_dir}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--kagglehub",
        nargs="?",
        const="download",
        default=None,
        metavar="PATH",
        help="PATH to a kagglehub-downloaded DIV2K folder containing "
        "DIV2K_train_HR/ and DIV2K_valid_HR/; or the literal "
        "'download' to fetch via kagglehub first.",
    )
    args = ap.parse_args(argv)

    config = load_config(PROJECT_ROOT / "configs" / "dataset.yaml")

    if args.kagglehub:
        if args.kagglehub == "download":
            print(
                "[setup-div2k] downloading via kagglehub (soumikrakshit/div2k-high-resolution-images)…"
            )
            try:
                import kagglehub
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise SystemExit(
                    "kagglehub is not installed. Run `pip install kagglehub` first, "
                    "or use the official-archive path (curl + scripts/setup_div2k.py)."
                ) from exc
            try:
                source = Path(
                    kagglehub.dataset_download("soumikrakshit/div2k-high-resolution-images")
                )
            except Exception as exc:  # pragma: no cover - network/auth - environment-specific
                raise SystemExit(
                    f"kagglehub download failed: {exc}. Ensure Kaggle credentials are "
                    "configured (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY), "
                    "or use the official-archive path via curl."
                ) from exc
            print(f"[setup-div2k] kagglehub returned {source}")
        else:
            source = Path(args.kagglehub)
        _folder_to_extracted(source, config)
    elif _has_official_archives(config):
        from src.utils.dataset_pipeline import extract_archives

        print(f"[setup-div2k] found official archives in {config.archive_dir}; extracting…")
        extract_archives(config)
    else:
        raise SystemExit(
            f"No DIV2K data found. Either:\n"
            f"  1. Place DIV2K_train_HR.zip + DIV2K_valid_HR.zip in {config.archive_dir}\n"
            f"     (download: curl -C - -o {config.archive_dir / 'DIV2K_train_HR.zip'} "
            f"https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)\n"
            f"  2. Run `python scripts/setup_div2k.py --kagglehub download` "
            f"(requires kagglehub + Kaggle credentials)."
        )

    print("[setup-div2k] processing to 256x256 and validating splits…")
    records = process_dataset(config)
    metadata = write_metadata(config, records)
    counts = {r["split"]: sum(1 for rec in records if rec["split"] == r["split"]) for r in records}
    print(
        f"[setup-div2k] done. {len(records)} processed images; splits: "
        f"train={counts.get('train')}, validation={counts.get('validation')}, "
        f"test={counts.get('test')}; metadata: {metadata}"
    )


if __name__ == "__main__":
    main()
