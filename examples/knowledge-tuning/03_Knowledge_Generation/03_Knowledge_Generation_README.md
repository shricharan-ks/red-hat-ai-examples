
# Step 02 — Knowledge Generation (expand seed into QnA)

## Navigation

- Overview: [Knowledge Tuning root](../README.md)
- Step 01 — Data Preprocessing: [../01_Data_Preprocessing/README.md](../01_Data_Processing/README.md)
- Step 02 — Knowledge Generation (this page)
- Step 03 — Knowledge Mixing: [../03_Knowledge_Mixing/README.md](../03_Knowledge_Mixing/README.md)
- Step 04 — Model Training: [../04_Model_Training/README.md](../04_Model_Training/README.md)
- Step 05 — Evaluation: [../05_Evaluation/README.md](../05_Evaluation/README.md)

## Purpose

This step expands the curated seed examples produced by Step 01 into a larger set of Q&A pairs using LLMs and local utilities. It can be used to produce synthetic training examples or to augment existing datasets.

## Flow Diagram

![Knowledge Generation Flow Diagram](../../../assets/usecase/knowledge-tuning/Knowledge%20Genertaion.png)

## Prerequisites

- `seed_data.jsonl` produced by Step 01 available under the step 01 output directory.
- Access to an LLM endpoint (API key and endpoint URL) pass it in `.env` file.
- Python environment with this step's `pyproject.toml` dependencies installed.

## Inputs

- `output/step_01/seed_data.jsonl` (seed dataset)

## Outputs

- `output/step_02/**/gen.jsonl` — raw generation outputs 
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

From the `examples/knowledge-tuning/02_Knowledge_Generation` folder:

```bash
# from the step folder
pip install .
```

## How to run

1. Activate the venv for this step.
2. Open `Knowledge_Generation.ipynb` and run cells in order.
3. Review the generated dataset files in `output/step_02/`.

## Prerequisites from earlier steps

- Must have completed Step 01 and have `seed_data.jsonl` available.

## Debug & tips

- If generation fails: verify API key, endpoint URL, and request rate limits.

## Next step

Proceed to [Knowledge Mixing (step 03)](../03_Knowledge_Mixing/README.md) after you have all four datasets generated.
