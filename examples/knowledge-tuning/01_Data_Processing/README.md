
# Step 01 — Data Processing (Seed dataset generation)

Navigation

- Overview: [Knowledge Tuning root](../README.md)
- Step 01 — Data Preprocessing (this page)
- Step 02 — Knowledge Generation: [../02_Knowledge_Generation/README.md](../02_Knowledge_Generation/README.md)
- Step 03 — Knowledge Mixing: [../03_Knowledge_Mixing/README.md](../03_Knowledge_Mixing/README.md)
- Step 04 — Model Training: [../04_Model_Training/README.md](../04_Model_Training/README.md)
- Step 05 — Evaluation: [../05_Evaluation/README.md](../05_Evaluation/README.md)

Purpose

This step converts raw PDF documents into a small, curated seed dataset suitable for Synthetic Data Generation (SDG). The Jupyter notebook `Data_Preprocessing.ipynb` performs document conversion (via `docling`), chunking, selection of representative chunks, and generation of initial Q&A pairs.

End-to-end flow inside this step

- PDF files (placed in `source_documents/`) → docling JSON (`output/step_01/docling_output/`)
- docling JSON → `chunks.jsonl`
- `chunks.jsonl` → `selected_chunks.jsonl` (random sample used for seed examples)
- `selected_chunks.jsonl` → `qna.yaml` (LLM-generated QnA pairs)
- `qna.yaml` + `chunks.jsonl` → `final_seed_data.jsonl` (final seed dataset)

Prerequisites

- RHOAI workbench as described in the top-level README: Python 3.12 image with required ML tooling and a persistent volume mounted.
- `docling` Python package and other dependencies (see `pyproject.toml` for this step).
- Access to an LLM endpoint or API key (used for QnA generation).

Inputs

- Place your source PDFs in `source_documents/`.

Outputs

- `output/step_01/docling_output/` — docling JSON files
- `output/step_01/chunks.jsonl` — all chunks
- `output/step_01/selected_chunks.jsonl` — randomly selected chunks
- `output/step_01/qna.yaml` — generated QnA pairs
- `output/step_01/final_seed_data.jsonl` — final seed dataset

Environment variables (common examples)

- `API_KEY` — API key for LLMs
- `ENDPOINT` — LLM API endpoint
- `MODEL_NAME` — model to call for QnA generation
- `NUM_SEED_EXAMPLES` — number of seed examples to generate (optional override)

Install dependencies (pyproject)

Each step contains a `pyproject.toml` describing the Python dependencies. To install them in a workbench:

```bash
# from the step folder
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you need to use local helper utilities included in this repository (for example, the `ai_tools` package in `src/`), prefer one of these options instead of building wheels:

- Editable install from the repository root (recommended for development):

```bash
# from the step folder
pip install -e ../../
```

- Or add the repository `src/` directory to `PYTHONPATH` or `sys.path` in your notebooks (useful for ephemeral testing).

Avoid building and referencing local wheel files in examples — editable installs or workspace-based development are simpler and less error-prone.

How to run

1. Open `Data_Preprocessing.ipynb` in the workbench JupyterLab.
2. Confirm environment variables are set (via workbench secrets or a `.env` file).
3. Run the cells in order.

Debug & tips

- If `docling` fails on PDF conversion, check that the PDF is not password protected and has sufficient OCR quality.
- Monitor memory when processing large PDFs — split or process page ranges if needed.

Next step

Proceed to [Knowledge Generation (step 02)](../02_Knowledge_Generation/README.md) once `final_seed_data.jsonl` is available.
