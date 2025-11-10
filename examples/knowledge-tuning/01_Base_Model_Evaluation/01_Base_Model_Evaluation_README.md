
# Step 1: Base Model Evaluation

## Navigation

- [Knowledge Tuning Overview](../README.md)
- [Setup](../00_Setup/00_Setup_README.md)
- Step 1: Base Model Evaluation
- [Step 2: Data Processing](../02_Data_Processing/02_Data_Processing_README.md)
- [Step 3: Knowledge Generation](../03_Knowledge_Generation/03_Knowledge_Generation_README.md)
- [Step 4: Knowledge Mixing](../04_Knowledge_Mixing/04_Knowledge_Mixing_README.md)
- [Step 5: Model Training](../05_Model_Training/05_Model_Training_README.md)
- [Step 6: Evaluation](../06_Evaluation/06_Evaluation_README.md)

## Evaluate the base model

Establish a baseline by evaluating the base model's performance on relevant tasks prior to any fine-tuning, enabling
objective comparison after knowledge tuning.

### Prerequisites

- Previous sections successfully completed in order
- Environment variables are set via workbench secrets or `.env` file. See [.env.example](./.env.example) for reference.

#### Environment variables

- `STUDENT_MODEL_NAME` — Model name

### Procedure

1. Open the [Base_Model_Evaluation.ipynb](./Base_Model_Evaluation.ipynb) file in JupyterLab and follow the instructions
directly in the notebook.

### Verification

The base model was successfully evaluated.

## Next step

Proceed to [Step 2: Data Processing](../02_Data_Processing/02_Data_Processing_README.md).
