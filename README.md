# Recommendation System with User Impression Feedback

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue.svg)](.github/workflows/ci.yml)

Recommendation-system experiment code for studying explicit user impression feedback in small-data settings. The public workflow implements a revised, scale-consistent review-response experiment with independent synthetic replicates and explicit train/test observation masks.

The manuscript source is maintained separately. The repository tracks one
compact canonical revised-result artifact; other generated outputs remain
local and reproducible from the tracked configuration.

## Included in this repository

- `src/`: recommendation models, evaluation logic, the revised experiment runner, and visualization scripts
- `tests/`: integration tests for the core workflow
- `data/`: committed input datasets and the fixed User A train/test split
- `.github/workflows/`: CI configuration for tests and quality checks

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the integration tests:

```bash
venv/bin/python -m pytest -q
```

Run the frozen 50-replicate revised experiment:

```bash
venv/bin/python -m src.experiments.run_revised_experiment \
  --config configs/revised_experiment.json \
  --output results/revised_experiment/canonical_results.json
```

Verify the canonical artifact, including deterministic regeneration of its
scientific payload:

```bash
venv/bin/python -m src.utils.verify_revised_results \
  --config configs/revised_experiment.json \
  --result results/revised_experiment/canonical_results.json \
  --reproduce
```

Generate the revised figures from the verified canonical artifact:

```bash
venv/bin/python -m src.visualization.plot_revised_experiment \
  --result results/revised_experiment/canonical_results.json \
  --output-dir results/revised_experiment/figures
```

Only the compact canonical result JSON is tracked. Reproduced artifacts and
figures are generated locally and remain ignored.

## Data

The committed dataset files below preserve the historical submitted-paper
scenario:

- `data/cosmetics_training.csv`
- `data/cosmetics_full.csv`
- `data/user_a_split.json`

They are not overwritten or consumed by the revised experiment. The revised
study independently generates 50 datasets and User A splits from the frozen
configuration in `configs/revised_experiment.json`. Its simulated feedback
noise is a controlled sensitivity analysis, not an empirically calibrated
model of real-user behavior.

## Repository structure

```text
.
├── .github/workflows/
├── configs/
├── data/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── experiments/       # tracked revised runner only
│   ├── models/
│   ├── utils/
│   └── visualization/
├── tests/
├── requirements.txt
└── requirements-dev.txt
```

## Notes

- `paper/` is excluded from this GitHub repository.
- `results/revised_experiment/canonical_results.json` is the sole tracked
  scientific result artifact; other result files remain generated locally.
- F1 is evaluated as post-feedback interactive reranking and therefore has no
  RMSE or MAE. UCF, Hybrid-CF, and F2 remain on the native 0–10 rating scale.
- The code targets Python 3.11+, and Python 3.12 has been validated in this workspace.

## License

This code is distributed under the MIT License. See [LICENSE](LICENSE).
