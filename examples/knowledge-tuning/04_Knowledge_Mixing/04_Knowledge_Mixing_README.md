# Step 4: Knowledge Mixing

## Navigation

- [Knowledge Tuning Overview](../README.md)
- [Setup](../00_Setup/00_Setup_README.md)
- [Step 1: Base Model Evaluation](../01_Base_Model_Evaluation/01_Base_Model_Evaluation_README.md)
- [Step 2: Data Processing](../02_Data_Processing/02_Data_Processing_README.md)
- [Step 3: Knowledge Generation](../03_Knowledge_Generation/03_Knowledge_Generation_README.md)
- Step 4: Knowledge Mixing
- [Step 5: Model Training](../05_Model_Training/05_Model_Training_README.md)
- [Step 6: Evaluation](../06_Evaluation/06_Evaluation_README.md)

## Knowledge mixing and preparing training mixes

This step mixes generated Q&A, extractive/detailed summaries, and other artifacts into training-ready datasets.
It prepares different cut sizes and mixes (upsampling, downsampling) that are consumed by model training workflows.

![Knowledge Mixing Flow Diagram](../../../assets/usecase/knowledge-tuning/Knowledge%20Mixing.png)

### Prerequisites

- Previous sections successfully completed in order
- Environment variables are set via workbench secrets or `.env` file. See [.env.example](./.env.example) for reference.

## Inputs

- `output/step_03/*` — Summary folders

## Outputs

- `output/step_04/combined_cut_{N}x.jsonl` — Mixed datasets for each cut size

## Environment variables (common examples)

- `TOKENIZER_MODEL` — Tokenizer/model for token counting
- `SAVE_GPT_OSS_FORMAT` — Boolean value to save in GPT-OSS format (e.g. `false`)
- `CUT_SIZES` — Comma-separated list of cut sizes to generate (e.g. `10,20`)
- `QA_PER_DOC` — Number of Q&A pairs per document

### Procedure

1. Open the [Knowledge_Mixing.ipynb](./Knowledge_Mixing.ipynb) file in JupyterLab and follow the instructions directly in the notebook.

### Verification

After completing the notebook instructions, the following artifacts are generated.

- `output/step_03/combined_cut_{N}x.jsonl` — Mixed datasets for each cut size

## Debug & tips

- If token counting fails, ensure `TOKENIZER_MODEL` points to a valid tokenizer compatible with `transformers`.

## Next step

Perform subset selection and data verification before training using this [example notebook](https://github.com/opendatahub-io/data-processing/blob/main/notebooks/use-cases/subset-selection.ipynb). This step is required if the training dataset is too large for training (ex: 1 Million samples) and can be skipped for smaller datasets.

Proceed to [Step 5: Model Training](../05_Model_Training/05_Model_Training_README.md).
