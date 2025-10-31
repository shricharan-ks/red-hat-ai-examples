# Step 03 — Knowledge Mixing (prepare training mixes)

## Navigation

- Overview: [Knowledge Tuning root](../README.md)
- Step 01 — Data Preprocessing: [../01_Data_Preprocessing/README.md](../01_Data_Processing/README.md)
- Step 02 — Knowledge Generation: [../02_Knowledge_Generation/README.md](../02_Knowledge_Generation/README.md)
- Step 03 — Knowledge Mixing (this page)
- Step 04 — Model Training: [../04_Model_Training/README.md](../04_Model_Training/README.md)
- Step 05 — Evaluation: [../05_Evaluation/README.md](../05_Evaluation/README.md)

## Purpose

This step mixes generated Q&A, extractive/detailed summaries, and other artifacts into training-ready datasets. It prepares different cut sizes and mixes (upsampling, downsampling) that are consumed by model training workflows.

## Flow Diagram

![Knowledge Mixing Flow Diagram](../../../assets/usecase/knowledge-tuning/Knowledge%20Mixing.png)

## Prerequisites

- Completion of Step 01 (seed dataset) and Step 02 (QnA generation).
- This step's Python dependencies installed (see `pyproject.toml` in this folder).

## Inputs

- `output/step_02/*` summary folders

## Outputs

- `output/step_03/combined_cut_{N}x.jsonl` — mixed datasets for each cut size

## Environment variables (common examples)

- `TOKENIZER_MODEL` — tokenizer/model for token counting
- `SAVE_GPT_OSS_FORMAT` - Boolean value to save in GPT-OSS format (e.g. `false`)
- `CUT_SIZES` — comma-separated list of cut sizes to generate (e.g. `10,20`)
- `QA_PER_DOC` — number of Q&A pairs per document

## Install dependencies (pyproject)

```bash
pip install .
```


# How to run

1. Open `Knowledge_Mixing.ipynb` in the workbench and run cells in order.
2. Confirm that `output/step_03/` contains `combined_cut_*.jsonl` files.

## Prerequisites from earlier steps

- Must have generated summary artifacts in Step 02.

## Debug & tips

- If token counting fails, ensure `TOKENIZER_MODEL` points to a valid tokenizer compatible with `transformers`.

## Next step

Proceed to [Model Training (step 04)](../04_Model_Training/README.md).
