# Step 05 — Evaluation

## Navigation

- Overview: [Knowledge Tuning root](../README.md)
- Step 01 — Data Preprocessing: [../01_Data_Preprocessing/README.md](../01_Data_Processing/README.md)
- Step 02 — Knowledge Generation: [../02_Knowledge_Generation/README.md](../02_Knowledge_Generation/README.md)
- Step 03 — Knowledge Mixing: [../03_Knowledge_Mixing/README.md](../03_Knowledge_Mixing/README.md)
- Step 04 — Model Training: [../04_Model_Training/README.md](../04_Model_Training/README.md)
- Step 05 — Evaluation (this page)

## Purpose

This step evaluates trained models and generated datasets against held-out test data. It computes metrics and produces human-readable reports for quality assessment.


## Prerequisites

- A trained model checkpoint produced in Step 04.
- Evaluation datasets (format depends on metric scripts).

## Inputs

- Base Model in `output/step_04/base_model/`
- Fine tuned Model in `output/step_04/fine_tuned_model/`

## Install dependencies (pyproject)

```bash
pip install .
```

## How to run

1. Open `Evaluation.ipynb` and run the notebook cells in this folder.


## Next steps

- Review metrics and iterate on earlier steps (data, generation, mixing, or training) as needed.
