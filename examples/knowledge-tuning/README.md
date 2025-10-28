
# Knowledge Tuning — High level overview

This repository contains a self-contained Knowledge Tuning example using the InstructLab methodology. It demonstrates how to convert domain documents into a seed dataset, generate Q&A knowledge, mix and prepare training data, train knowledge-aware models, and evaluate results — all in a reproducible workbench environment.

## Flow Diagram
Usecase Flow Diagram
![Usecase Flow Diagram](../../assets/usecase/knowledge-tuning/Overall%20Flow.png)


Detailed Flow Diagram
![Detailed Flow Diagram](../../assets/usecase/knowledge-tuning/Detailed%20Flow.png)

## Top-level flow (end-to-end)


1. Data Collection — gather domain documents (PDFs, manuals, etc.) into `examples/knowledge-tuning/source_documents`
2. Data Preprocessing — convert PDFs to structured JSON (docling), chunk text, and produce a small seed dataset (step 01)
3. Knowledge Generation — expand seed examples into more Q&A pairs using an LLM or endpoint (step 02)
4. Knowledge Mixing — combine generated content and summaries into training mixes (step 03)
5. Model Training — fine-tune or instruction-tune a model using the prepared mixes (step 04)
6. Evaluation — run evaluation notebooks and metrics on held-out data (step 05)

Each step lives in a subfolder under `examples/knowledge-tuning/` and contains a notebook, a `pyproject.toml` for dependencies, and a `README.md` describing that step.

RHOAI Workbench & platform specifications

These guidelines help you configure a Red Hat OpenShift AI (RHOAI) workbench that can run the notebooks reliably.

- Workbench image: a JupyterLab image with Python 3.12, CUDA and common ML tooling (tokenizers, transformers, polars). Example label: `jupyter/tensorflow-cuda-py3.12` (custom images often used internally).
- GPU: Optional for preprocessing and mixing. Required/strongly recommended for model training (step 04) — at least one NVIDIA A100/40GB or similar for fine-tuning large models; for smaller student models an 8–16 GB GPU may suffice.
- Persistent storage: A mounted persistent volume (PVC) for the workbench is required to store `examples/knowledge-tuning/output/` and large intermediate artifacts. Allocate 50+ GB for realistic datasets.
- Accelerators: If using on-node inference/quantized student models, preferred accelerators are NVIDIA GPUs with CUDA support. For CPU-only runs, ensure at least 16 vCPU and 64 GB RAM for heavier pipelines.
- Environment variables: Workbench should provide a `.env` or secret injection mechanism for API keys and configuration. Typical variables used across the notebooks include:
  - `API_KEY` — API key for LLM endpoints (if used)
  - `ENDPOINT` — LLM HTTP endpoint
  - `MODEL_NAME` — LLM model name for generation
  - `OUTPUT_DATA_FOLDER` — top-level folder for experiment outputs (default: `generated_output_data`)
  - `STUDENT_MODEL` — tokeniser/model to use for token counting

Security & access

Do not commit secrets. Use the workbench's secret management to inject API keys. The notebooks read environment variables via `python-dotenv` (if `.env` is present) or the process environment.

How to use this example

1. Create or choose a RHOAI workbench with the recommended image.
2. Attach a Persistent Volume with at least 50 GB.
3. Clone this repository inside the workbench.
4. Open `examples/knowledge-tuning/00_Setup.ipynb` and follow the setup steps (creating venvs and installing local packages as needed).
5. Work through the notebooks step-by-step, from `01_Data_Preprocessing` to `05_Evaluation`.


Next: start with [Data Preprocessing (step 01)](./01_Data_Preprocessing/README.md)
