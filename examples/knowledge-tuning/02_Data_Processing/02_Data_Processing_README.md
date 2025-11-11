
# Step 2: Data Processing

## Navigation

- [Knowledge Tuning Overview](../README.md)
- [Setup](../00_Setup/00_Setup_README.md)
- [Step 1: Base Model Evaluation](../01_Base_Model_Evaluation/01_Base_Model_Evaluation_README.md)
- Step 2: Data Processing
- [Step 3: Knowledge Generation](../03_Knowledge_Generation/03_Knowledge_Generation_README.md)
- [Step 4: Knowledge Mixing](../04_Knowledge_Mixing/04_Knowledge_Mixing_README.md)
- [Step 5: Model Training](../05_Model_Training/05_Model_Training_README.md)
- [Step 6: Evaluation](../06_Evaluation/06_Evaluation_README.md)

## Data processing and seed dataset generation

This step converts sources (e.g. URLs, PDF files) into a small, curated seed dataset suitable for Synthetic Data
Generation (SDG). The Jupyter notebook in this example performs document conversion using [docling](https://www.docling.ai/),
chunking, selection of representative chunks, and generation of initial Q&A pairs.

![Data Preprocessing Flow Diagram](../../../assets/usecase/knowledge-tuning/Data%20Preprocessing.png)

### Prerequisites

- Previous sections successfully completed in order

### Procedure

1. Open the [Data_Processing.ipynb](./Data_Processing.ipynb) file in JupyterLab and follow the instructions directly in the notebook.

### Verification

After completing the notebook instructions, the following artifacts are generated.

- `output/step_02/docling_output/` — Directory containing docling JSON files
- `output/step_02/chunks.jsonl` — File containing all chunks
- `output/step_02/seed_data.jsonl` — File containing the final seed dataset

## Next step

Proceed to [Step 3: Knowledge Generation](../03_Knowledge_Generation/03_Knowledge_Generation_README.md).
