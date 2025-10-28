# Step 03 — Knowledge Mixing (prepare training mixes)

Navigation

- Overview: [Knowledge Tuning root](../README.md)
- Step 01 — Data Preprocessing: [../01_Data_Preprocessing/README.md](../01_Data_Preprocessing/README.md)
- Step 02 — Knowledge Generation: [../02_Knowledge_Generation/README.md](../02_Knowledge_Generation/README.md)
- Step 03 — Knowledge Mixing (this page)
- Step 04 — Model Training: [../04_Model_Training/README.md](../04_Model_Training/README.md)
- Step 05 — Evaluation: [../05_Evaluation/README.md](../05_Evaluation/README.md)

Purpose

This step mixes generated Q&A, extractive/detailed summaries, and other artifacts into training-ready datasets. It prepares different cut sizes and mixes (upsampling, downsampling) that are consumed by model training workflows.

End-to-end flow inside this step

- Input: `output/step_02/qna.yaml`, summary artifacts in `output/step_02/*`, or `final_seed_data.jsonl` → sampling and mixing utilities → `output/step_03/combined_cut_*.jsonl`

Prerequisites

- Completion of Step 01 (seed dataset) and Step 02 (QnA generation).
- This step's Python dependencies installed (see `pyproject.toml` in this folder).

Inputs

- `output/step_01/final_seed_data.jsonl` or `output/step_02/qna.yaml` and `output/step_02/*` summary folders

Outputs

- `output/step_03/combined_cut_{N}x.jsonl` — mixed datasets for each cut size

Environment variables (common examples)

- `OUTPUT_DATA_FOLDER` — experiment folder used by mixing scripts (default: `generated_output_data`)
- `STUDENT_MODEL` — tokenizer/model for token counting
- `CUT_SIZES` — comma-separated list of cut sizes to generate (e.g. `10,20`)
- `QA_PER_DOC` — number of Q&A pairs per document

Install dependencies (pyproject)

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

Avoid building and referencing local wheel files — editable installs or workspace-based development are simpler and less error-prone.

How to run

1. Activate the venv for this step.
2. Open `Knowledge_Mixing.ipynb` in the workbench and run cells in order.
3. Confirm that `output/step_03/` contains `combined_cut_*.jsonl` files.

Prerequisites from earlier steps

- Must have generated QnA (`qna.yaml`) and summary artifacts in Step 02.

Debug & tips

- If token counting fails, ensure `STUDENT_MODEL` points to a valid tokenizer compatible with `transformers`.

Next step

Proceed to [Model Training (step 04)](../04_Model_Training/README.md).
