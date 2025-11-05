# Step 04 — Knowledge Mixing (Prepare training mixes)

## Navigation

- Overview — [Knowledge Tuning](../README.md)
- Step 00 — [Setup](../00_Setup/00_Setup_README.md)
- Step 01 — [Base Model Evaluation](../01_Base_Model_Evaluation/01_Base_Model_Evaluation_README.md)
- Step 02 — [Data Processing](../02_Data_Processing/02_Data_Processing_README.md)
- Step 03 — [Knowledge Generation](../03_Knowledge_Generation/03_Knowledge_Generation_README.md)
- Step 04 — Knowledge Mixing
- Step 05 — [Model Training](../05_Model_Training/05_Model_Training_README.md)
- Step 06 — [Evaluation](../06_Evaluation/06_Evaluation_README.md)

## Purpose

This step mixes generated Q&A, extractive/detailed summaries, and other artifacts into training-ready datasets. It prepares different cut sizes and mixes (upsampling, downsampling) that are consumed by model training workflows.

## Flow Diagram

![Knowledge Mixing Flow Diagram](../../../assets/usecase/knowledge-tuning/Knowledge%20Mixing.png)

## Prerequisites

- Completion of Step 02 (seed dataset) and Step 03 (QnA generation).
- This step's Python dependencies installed (see `pyproject.toml` in this folder).

## Inputs

- `output/step_03/*` — Summary folders

## Outputs

- `output/step_03/combined_cut_{N}x.jsonl` — Mixed datasets for each cut size

## Environment variables (common examples)

- `TOKENIZER_MODEL` — Tokenizer/model for token counting
- `SAVE_GPT_OSS_FORMAT` - Boolean value to save in GPT-OSS format (e.g. `false`)
- `CUT_SIZES` — Comma-separated list of cut sizes to generate (e.g. `10,20`)
- `QA_PER_DOC` — Number of Q&A pairs per document

## Install dependencies (pyproject)

```bash
pip install .
```

## How to run

1. Confirm environment variables are set via workbench secrets or `.env` file.
2. Open the [Knowledge_Mixing.ipynb](./Knowledge_Mixing.ipynb) file in JupyterLab and follow the instructions directly in the notebook.
3. Confirm that `output/step_04/` contains `combined_cut_*.jsonl` files.

## Prerequisites from earlier steps

- Must have generated summary artifacts in Step 03.

## Debug & tips

- If token counting fails, ensure `TOKENIZER_MODEL` points to a valid tokenizer compatible with `transformers`.

## Next step

Proceed to [Model Training](../05_Model_Training/05_Model_Training_README.md).
