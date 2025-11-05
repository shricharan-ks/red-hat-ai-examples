
# Step 03 — Knowledge Generation (Expand seed into QnA)

## Navigation

- Overview — [Knowledge Tuning](../README.md)
- Step 00 — [Setup](../00_Setup/00_Setup_README.md)
- Step 01 — [Base Model Evaluation](../01_Base_Model_Evaluation/01_Base_Model_Evaluation_README.md)
- Step 02 — [Data Processing](../02_Data_Processing/02_Data_Processing_README.md)
- Step 03 — Knowledge Generation
- Step 04 — [Knowledge Mixing](../04_Knowledge_Mixing/04_Knowledge_Mixing_README.md)
- Step 05 — [Model Training](../05_Model_Training/05_Model_Training_README.md)
- Step 06 — [Evaluation](../06_Evaluation/06_Evaluation_README.md)

## Purpose

This step expands the curated seed examples produced by Step 02 into a larger set of Q&A pairs using LLMs and local utilities. It can be used to produce synthetic training examples or to augment existing datasets.

## Flow Diagram

![Knowledge Generation Flow Diagram](../../../assets/usecase/knowledge-tuning/Knowledge%20Genertaion.png)

## Prerequisites

- `seed_data.jsonl` produced by Step 02 available under the step 02 output directory.
- Access to an LLM endpoint (API key and endpoint URL) pass it in `.env` file.
- Python environment with this step's `pyproject.toml` dependencies installed.

## Inputs

- `output/step_02/seed_data.jsonl` — Seed dataset

## Outputs

- `output/step_03/**/gen.jsonl` — Raw generation outputs
    Generates four different datasets
        - Extractive Summary
        - Detailed Summary
        - Key Facts Q&A
        - Document Based Q&A

## Environment variables (common examples)

- `TEACHER_MODEL_API_KEY` — LLM API key
- `TEACHER_MODEL_BASE_URL` — LLM HTTP endpoint
- `TEACHER_MODEL_NAME` — model to call for generation

## Install dependencies (pyproject)

From the `examples/knowledge-tuning/03_Knowledge_Generation` folder:

```bash
# from the step folder
pip install .
```

## How to run

1. Activate the venv for this step.
2. Confirm environment variables are set via workbench secrets or `.env` file.
3. Open the [Knowledge_Generation.ipynb](./Knowledge_Generation.ipynb) file in JupyterLab and follow the instructions directly in the notebook.
4. Review the generated dataset files in `output/step_03/`.

## Prerequisites from earlier steps

- Must have completed Step 02 and have `seed_data.jsonl` available.

## Debug & tips

- If generation fails: verify API key, endpoint URL, and request rate limits.

## Next step

Proceed to [Knowledge Mixing](../04_Knowledge_Mixing/04_Knowledge_Mixing_README.md) after you have all four datasets generated.
