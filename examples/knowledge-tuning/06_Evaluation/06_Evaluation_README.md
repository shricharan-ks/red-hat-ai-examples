# Step 6: Evaluation

## Navigation

- [Knowledge Tuning Overview](../README.md)
- [Setup](../00_Setup/00_Setup_README.md)
- [Step 1: Base Model Evaluation](../01_Base_Model_Evaluation/01_Base_Model_Evaluation_README.md)
- [Step 2: Data Processing](../02_Data_Processing/02_Data_Processing_README.md)
- [Step 3: Knowledge Generation](../03_Knowledge_Generation/03_Knowledge_Generation_README.md)
- [Step 4: Knowledge Mixing](../04_Knowledge_Mixing/04_Knowledge_Mixing_README.md)
- [Step 5: Model Training](../05_Model_Training/05_Model_Training_README.md)
- Step 6: Evaluation

## Evaluate the trained model

This step evaluates trained models and generated datasets against held-out test data. It computes metrics and produces
human-readable reports for quality assessment.

### Prerequisites

- Previous sections successfully completed in order
- Environment variables are set via workbench secrets or `.env` file. See [.env.example](./.env.example) for reference.

#### Environment variables

- `STUDENT_MODEL_NAME` — Model name

### Procedure

1. Open the [Evaluation.ipynb](./Evaluation.ipynb) file in JupyterLab and follow the instructions directly in the notebook.

### Verification

The trained model was successfully evaluated.

## Next steps

Review metrics and iterate on earlier steps (data, generation, mixing, or training) as needed.
