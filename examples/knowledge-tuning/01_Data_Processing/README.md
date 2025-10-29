
# Step 01 — Data Processing (Seed dataset generation)

## Navigation

- Overview: [Knowledge Tuning root](../README.md)
- Step 01 — Data Preprocessing (this page)
- Step 02 — Knowledge Generation: [../02_Knowledge_Generation/README.md](../02_Knowledge_Generation/README.md)
- Step 03 — Knowledge Mixing: [../03_Knowledge_Mixing/README.md](../03_Knowledge_Mixing/README.md)
- Step 04 — Model Training: [../04_Model_Training/README.md](../04_Model_Training/README.md)
- Step 05 — Evaluation: [../05_Evaluation/README.md](../05_Evaluation/README.md)

## Purpose

This step converts raw PDF documents into a small, curated seed dataset suitable for Synthetic Data Generation (SDG). The Jupyter notebook `Data_Preprocessing.ipynb` performs document conversion (via `docling`), chunking, selection of representative chunks, and generation of initial Q&A pairs.


## Prerequisites

- RHOAI workbench as described in the top-level README: Python 3.12 image with required ML tooling and a persistent volume mounted.
- `docling` Python package and other dependencies (see `pyproject.toml` for this step).

## Inputs

- Provide the list of urls where which you want to run the flow.

## Outputs

- `output/step_01/docling_output/` — docling JSON files
- `output/step_01/chunks.jsonl` — all chunks
- `output/step_01/seed_data.jsonl` — final seed dataset


## Install dependencies (pyproject)

Each step contains a `pyproject.toml` describing the Python dependencies. To install them in a workbench:

```bash
# from the step folder
pip install .
```

## How to run

1. Open `Data_Preprocessing.ipynb` in the workbench JupyterLab.
2. Confirm environment variables are set (via workbench secrets or a `.env` file).

## Next step

Proceed to [Knowledge Generation (step 02)](../02_Knowledge_Generation/README.md) once `seed_data.jsonl` is available.
