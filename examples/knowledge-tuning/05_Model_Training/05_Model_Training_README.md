# Step 05 — Model Training

## Navigation

- Overview — [Knowledge Tuning](../README.md)
- Step 00 — [Setup](../00_Setup/00_Setup_README.md)
- Step 01 — [Base Model Evaluation](../01_Base_Model_Evaluation/01_Base_Model_Evaluation_README.md)
- Step 02 — [Data Processing](../02_Data_Processing/02_Data_Processing_README.md)
- Step 03 — [Knowledge Generation](../03_Knowledge_Generation/03_Knowledge_Generation_README.md)
- Step 04 — [Knowledge Mixing](../04_Knowledge_Mixing/04_Knowledge_Mixing_README.md)
- Step 05 — Model Training
- Step 06 — [Evaluation](../06_Evaluation/06_Evaluation_README.md)

## Purpose

This step demonstrates how to fine-tune or instruction-tune a student model using the mixed datasets produced earlier. Training may be done on a GPU-enabled workbench or a training cluster.

## Flow Diagram

![Model Training Flow Diagram](../../../assets/usecase/knowledge-tuning/Model%20Training.png)

## Prerequisites

- Completion of Steps 01–04 and availability of `combined_cut_*.jsonl` files.
- GPU-enabled workbench recommended for training (see top-level RHOAI specs).

## Inputs

- `output/step_04/combined_cut_*.jsonl`

## Outputs

- `output/step_05/checkpoints/` — Model checkpoints and training logs

## Environment variables (common examples)

- `STUDENT_MODEL` - The model that is to be fine tuned.

## Install dependencies (pyproject)

```bash
pip install .
```

## How to run

1. Confirm environment variables are set via workbench secrets or `.env` file.
2. Open the [Model_Training.ipynb](./Model_Training.ipynb) file in JupyterLab and follow the instructions directly in the notebook, or run your training script/entrypoint.
3. Review the output files in `output/step_05/`.

## Prerequisites from earlier steps

- Ensure combined datasets were built in Step 04 `combined_cut_*.jsonl` is available. Use different cut size to test the training.

## Debug & tips

- Monitor GPU memory if needed.

## Next step

Proceed to [Evaluation](../06_Evaluation/06_Evaluation_README.md).
