
# Step 02 — Knowledge Generation (expand seed into QnA)

Navigation

- Overview: [Knowledge Tuning root](../README.md)
- Step 01 — Data Preprocessing: [../01_Data_Preprocessing/README.md](../01_Data_Preprocessing/README.md)
- Step 02 — Knowledge Generation (this page)
- Step 03 — Knowledge Mixing: [../03_Knowledge_Mixing/README.md](../03_Knowledge_Mixing/README.md)
- Step 04 — Model Training: [../04_Model_Training/README.md](../04_Model_Training/README.md)
- Step 05 — Evaluation: [../05_Evaluation/README.md](../05_Evaluation/README.md)

Purpose

This step expands the curated seed examples produced by Step 01 into a larger set of Q&A pairs using LLMs and local utilities. It can be used to produce synthetic training examples or to augment existing datasets.

End-to-end flow inside this step

- `final_seed_data.jsonl` → call LLM endpoint(s) → generate `qagen-*.json` artifacts → post-process → `qna.yaml`

Prerequisites

- `final_seed_data.jsonl` produced by Step 01 available under the step 01 output directory.
- Access to an LLM endpoint (API key and endpoint URL).
- Python environment with this step's `pyproject.toml` dependencies installed.

Inputs

- `output/step_01/final_seed_data.jsonl` (seed dataset)

Outputs

- `output/step_02/qagen-*.json` — raw generation outputs
- `output/step_02/qna.yaml` — consolidated QnA pairs ready for review

Environment variables (common examples)

- `API_KEY` — LLM API key
- `ENDPOINT` — LLM HTTP endpoint
- `MODEL_NAME` — model to call for generation
- `CUSTOMISATION_PROMPT` — optional prompt to tailor generation

Install dependencies (pyproject)

From the `examples/knowledge-tuning/02_Knowledge_Generation` folder:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If you need to use local helper utilities included in this repository (for example, the `ai_tools` package in `src/`), prefer one of these options instead of building wheels:

- Editable install from the repository root (recommended for development):

```bash
# from the step folder
uv pip install -e ../../
```

- Or add the repository `src/` directory to `PYTHONPATH` or `sys.path` in your notebooks (useful for ephemeral testing).

Avoid building and referencing local wheel files in examples — editable installs or workspace-based development are simpler and less error-prone.

How to run

1. Activate the venv for this step.
2. Open `Knowledge_Generation.ipynb` and run cells in order.
3. Review the generated `qagen-*.json` and `qna.yaml` files in `output/step_02/`.

Prerequisites from earlier steps

- Must have completed Step 01 and have `final_seed_data.jsonl` available.

Debug & tips

- If generation fails: verify API key, endpoint URL, and request rate limits.
- Keep the generation prompt small and iteratively improve `CUSTOMISATION_PROMPT`.

Next step

Proceed to [Knowledge Mixing (step 03)](../03_Knowledge_Mixing/README.md) after you have `qna.yaml` and generation artifacts.
