
# Step 01 — Base Model Evaluation

## Navigation

- Overview — [Knowledge Tuning](../README.md)
- Step 00 — [Setup](../00_Setup/00_Setup_README.md)
- Step 01 — Base Model Evaluation
- Step 02 — [Data Processing](../02_Data_Processing/02_Data_Processing_README.md)
- Step 03 — [Knowledge Generation](../03_Knowledge_Generation/03_Knowledge_Generation_README.md)
- Step 04 — [Knowledge Mixing](../04_Knowledge_Mixing/04_Knowledge_Mixing_README.md)
- Step 05 — [Model Training](../05_Model_Training/05_Model_Training_README.md)
- Step 06 — [Evaluation](../06_Evaluation/06_Evaluation_README.md)

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

1. Confirm environment variables are set via workbench secrets or `.env` file.
2. Open the [Base_Model_Evaluaion.ipynb](./Base_Model_Evaluaion.ipynb) file in JupyterLab and follow the instructions directly in the notebook.

## Next step

Proceed to [Data Processing](../02_Data_Processing/02_Data_Processing_README.md) after the base model has been evaluated.
