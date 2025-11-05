# Step 06 — Evaluation

## Navigation

- Overview — [Knowledge Tuning](../README.md)
- Step 00 — [Setup](../00_Setup/00_Setup_README.md)
- Step 01 — [Base Model Evaluation](../01_Base_Model_Evaluation/01_Base_Model_Evaluation_README.md)
- Step 02 — [Data Processing](../02_Data_Processing/02_Data_Processing_README.md)
- Step 03 — [Knowledge Generation](../03_Knowledge_Generation/03_Knowledge_Generation_README.md)
- Step 04 — [Knowledge Mixing](../04_Knowledge_Mixing/04_Knowledge_Mixing_README.md)
- Step 05 — [Model Training](../05_Model_Training/05_Model_Training_README.md)
- Step 06 — Evaluation

## Purpose

This step evaluates trained models and generated datasets against held-out test data. It computes metrics and produces human-readable reports for quality assessment.

## Prerequisites

- A trained model checkpoint produced in Step 04.
- Evaluation datasets (format depends on metric scripts).

## Inputs

- `output/step_05/base_model/` — The base model
- `output/step_04/fine_tuned_model/` — The fine tuned model

## Install dependencies (pyproject)

```bash
pip install .
```

## How to run

1. Confirm environment variables are set via workbench secrets or `.env` file.
2. Open the [Evaluation.ipynb](./Evaluation.ipynb) file in JupyterLab and follow the instructions directly in the notebook.

## Next steps

Review metrics and iterate on earlier steps (data, generation, mixing, or training) as needed.
