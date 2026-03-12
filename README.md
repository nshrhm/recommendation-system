# Recommendation System with User Impression Feedback

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue.svg)](.github/workflows/ci.yml)

Recommendation-system experiment code for studying explicit user impression feedback in small-data settings. This repository is published as a code-first, reproducible package: it includes the implementation, test suite, and input data needed to rerun the experiments locally.

The manuscript source and generated result artifacts are intentionally not included in this GitHub repository. They are maintained separately from the public code snapshot.

## Included in this repository

- `src/`: recommendation models, evaluation logic, experiment runners, and visualization scripts
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

Run the validated multi-seed experiment:

```bash
venv/bin/python src/experiments/run_statistical_validation.py
```

Regenerate figures locally:

```bash
venv/bin/python src/visualization/regenerate_all_paper_figures.py
```

These commands write outputs under `results/` when executed locally, but that directory is not tracked in this repository snapshot.

## Data

The committed dataset files used by the experiment are:

- `data/cosmetics_training.csv`
- `data/cosmetics_full.csv`
- `data/user_a_split.json`

This setup keeps the train/test scenario fixed so the reported workflow remains reproducible.

## Repository structure

```text
.
├── .github/workflows/
├── data/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── experiments/
│   ├── models/
│   ├── utils/
│   └── visualization/
├── tests/
├── requirements.txt
└── requirements-dev.txt
```

## Notes

- `paper/` is excluded from this GitHub repository.
- `results/` is generated locally and excluded from version control.
- The code targets Python 3.11+, and Python 3.12 has been validated in this workspace.

## License

This code is distributed under the MIT License. See [LICENSE](LICENSE).
