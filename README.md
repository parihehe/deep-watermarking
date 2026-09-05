# Deep Watermarking

Research framework for an invisible, blind image watermarking study based on a reproducible DWT–SVD–CNN baseline. This directory is intentionally independent from the Dayflow application in the parent workspace.

## Project status

Phase 0 is documented in [`docs/phase0`](docs/phase0). Phase 1 repository scaffolding is present, but the local machine must provide Python **3.12.x** before dependencies can be installed and validated. Python 3.14 is deliberately not used because the research protocol specifies 3.12.

## Known limitation: blind registry-ID reliability is not a mathematical guarantee

The web app's blind text extraction (`src/app/final_model.py`) reports a
registry-ID decode as reliable only above a calibrated confidence threshold
(`RELIABLE_REGISTRY_ID_CONFIDENCE = 0.78`). As of the current 5-message
registry, **zero wrong answers have ever been shown to a user** across all
real evaluation data - but this reflects the registry's current sparsity (5
of 14 usable ID slots populated), not a proof that the threshold makes wrong
answers impossible. Real measurements show an underlying ~0.5% rate of wrong
decodes at that confidence level; none have surfaced yet only because none
happened to land on one of the 5 currently-registered IDs. **Before
registering any message beyond this initial 5, re-run
`experiments/calibrate_registry_confidence.py` against the larger registry
and re-verify the threshold still holds** - see
[`docs/phase18_id_registry.md`](docs/phase18_id_registry.md) S5 for the full
data and required-check details.

## Planned structure

```text
configs/       reproducible experiment configurations
data/          DVC-managed datasets (not committed to Git)
docs/          research and operational documentation
src/           research implementation
tests/         automated tests
experiments/   experiment definitions
models/        ignored checkpoints
results/       ignored outputs
```

## Local setup (after Python 3.12 is available)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[training,tracking,dev]"
pytest
```

For PyTorch, select a CPU/CUDA wheel appropriate to the machine using the official PyTorch installer guidance before running training. Phase 1 will record the resulting Torch/CUDA configuration; it will not assume GPU availability.

## Reproducibility rules

- Python requirement: `>=3.12,<3.13`.
- No host dataset, checkpoints, result images, MLflow artifacts, or secrets are committed.
- Experiment code belongs in `src/`; the API and frontend, when introduced, call this code rather than duplicating its mathematics.
- Each future experiment will be run from a committed YAML configuration and fixed seed.
