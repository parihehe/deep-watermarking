# DIV2K Dataset Protocol

## Source and licence

The host-image dataset is the official [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/). It is provided for academic research only; the original owners retain copyright. This project uses only the official public high-resolution training and validation archives.

## Why this split differs from the original challenge

Official DIV2K publishes 800 training HR and 100 validation HR images. The 100 official test HR images are not released. To obtain three disjoint host-image partitions without inventing unavailable data, this project makes the fixed, ID-based re-split below:

| Split | IDs | Count | Origin |
|---|---:|---:|---|
| Train | 0001–0700 | 700 | public training HR |
| Validation | 0701–0800 | 100 | public training HR |
| Test | 0801–0900 | 100 | public validation HR |

No image ID may occur in more than one split. The split is deterministic, configured in `configs/dataset.yaml`, and must not be changed after baseline development begins.

## Processing

All source PNGs are read as three-channel images, validated, and resized to 256×256 with area interpolation. Processing never overwrites raw images. Image-level source dimensions, SHA-256 digest and split assignment are saved in `data/processed/div2k_256/metadata/images.csv`; aggregate results are saved as JSON.

## Reproduction

After placing the two verified official archives under `data/raw/archives/`, run:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_div2k.py
```

The raw and processed directories are DVC-managed and ignored by Git. The loader at `src/training/dataset.py` returns normalized RGB tensors in `[0, 1]`.
