# Step 5: Model Training

## Navigation

- [Knowledge Tuning Overview](../README.md)
- [Setup](../00_Setup/00_Setup_README.md)
- [Step 1: Base Model Evaluation](../01_Base_Model_Evaluation/01_Base_Model_Evaluation_README.md)
- [Step 2: Data Processing](../02_Data_Processing/02_Data_Processing_README.md)
- [Step 3: Knowledge Generation](../03_Knowledge_Generation/03_Knowledge_Generation_README.md)
- [Step 4: Knowledge Mixing](../04_Knowledge_Mixing/04_Knowledge_Mixing_README.md)
- Step 5: Model Training
- [Step 6: Evaluation](../06_Evaluation/06_Evaluation_README.md)

## Fine tuning the model

This step demonstrates how to fine-tune or instruction-tune a student model using the mixed datasets produced earlier.
Training may be done on a GPU-enabled workbench or a training cluster.

![Model Training Flow Diagram](../../../assets/usecase/knowledge-tuning/Model%20Training.png)

### Prerequisites

- Previous sections successfully completed in order
- Environment variables are set via workbench secrets or `.env` file. See [.env.example](./.env.example) for reference.
- GPU-enabled workbench recommended for training. See [Setup](../00_Setup/00_Setup_README.md) for details.

#### Environment variables

- `STUDENT_MODEL` — Model to be fine tune

### Procedure

1. Open the [Model_Training.ipynb](./Model_Training.ipynb) file in JupyterLab and follow the instructions directly in
the notebook, or run your training script/entrypoint.

### Verification

After completing the notebook instructions, the following artifacts are generated.

- `output/step_05/checkpoints/` — Model checkpoints and training logs

## Debug & tips

- Monitor GPU memory if needed.

## Next step

Proceed to [Step 6: Evaluation](../06_Evaluation/06_Evaluation_README.md).
