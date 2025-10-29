# Step 04 — Model Training

## Navigation

- Overview: [Knowledge Tuning root](../README.md)
- Step 01 — Data Preprocessing: [../01_Data_Preprocessing/README.md](../01_Data_Processing/README.md)
- Step 02 — Knowledge Generation: [../02_Knowledge_Generation/README.md](../02_Knowledge_Generation/README.md)
- Step 03 — Knowledge Mixing: [../03_Knowledge_Mixing/README.md](../03_Knowledge_Mixing/README.md)
- Step 04 — Model Training (this page)
- Step 05 — Evaluation: [../05_Evaluation/README.md](../05_Evaluation/README.md)

## Purpose

This step demonstrates how to fine-tune or instruction-tune a student model using the mixed datasets produced earlier. Training may be done on a GPU-enabled workbench or a training cluster.

## Prerequisites

- Completion of Steps 01–03 and availability of `combined_cut_*.jsonl` files.
- GPU-enabled workbench recommended for training (see top-level RHOAI specs).

## Inputs

- `output/step_03/combined_cut_*.jsonl`

## Outputs

- Model checkpoints and training logs (e.g. `output/step_04/checkpoints/`)

## Environment variables (common examples)

- `STUDENT_MODEL` - The model that is to be finetuned.

## Install dependencies (pyproject)

```bash
pip install .
```

## How to run

1. Open `Model_Training.ipynb` and run cells, or run your training script/entrypoint.

## Prerequisites from earlier steps

- Ensure combined datasets were built in Step 03 `combined_cut_*.jsonl` is available. Use different cut size to test the training.

## Debug & tips

- Monitor GPU memory if needed.

## Next step

Proceed to [Evaluation (step 05)](../05_Evaluation/README.md).
