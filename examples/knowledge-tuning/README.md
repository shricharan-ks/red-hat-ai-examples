
# Knowledge Tuning — High level overview

This repository contains a self-contained Knowledge Tuning example using the InstructLab methodology. It demonstrates how to convert domain documents into a seed dataset, generate Q&A knowledge, mix and prepare training data, train knowledge-aware models, and evaluate results — all in a reproducible workbench environment.

## Flow Diagram
Usecase Flow Diagram
![Usecase Flow Diagram](../../assets/usecase/knowledge-tuning/Overall%20Flow.png)


Detailed Flow Diagram
![Detailed Flow Diagram](../../assets/usecase/knowledge-tuning/Detailed%20Flow.png)

## Top-level flow (end-to-end)


1. Data Processing — convert PDFs to structured Md (docling), chunk text, and produce a small seed dataset (step 01)
2. Knowledge Generation — expand seed examples into more Q&A pairs using an LLM(teacher model) (step 02)
3. Knowledge Mixing — combine generated Q&A paris and summaries into training mixes (step 03)
4. Model Training — fine-tune or instruction-tune a model using the prepared mixes (step 04)
5. Evaluation — run evaluation notebooks and metrics on held-out data (step 05)

Each step lives in a subfolder under `examples/knowledge-tuning/` and contains a notebook, a `pyproject.toml` for dependencies, and a `README.md` describing that step.

## RHOAI Workbench & platform specifications

These guidelines help you configure a Red Hat OpenShift AI (RHOAI) workbench that can run the notebooks reliably.

- Workbench image: `Jupyter | Minimal | CUDA | Python 3.12`
- GPU: Optional for preprocessing and mixing. Required/strongly recommended for model training (step 04) — at least one NVIDIA A100/40GB or similar for fine-tuning large models; for smaller student models an 8–16 GB GPU may suffice.
- Persistent storage: A mounted persistent volume (PVC) for the workbench is required to store `examples/knowledge-tuning/output/` and large intermediate artifacts. Allocate 200+ GB for realistic datasets.
- Accelerators: If using on-node inference/quantized student models, preferred accelerators are NVIDIA GPUs with CUDA support.
- Environment variables: Workbench should provide a `.env` or secret injection mechanism for API keys and configuration. Example `.env.example` files are provided at usecase level and as well as step level.


## How to use this example

1. Create or choose a RHOAI workbench with the recommended image.
2. Attach a Persistent Volume with at least 200 GB.
3. Clone this repository inside the workbench.
4. Work through the notebooks step-by-step, from `01_Data_Preprocessing` to `05_Evaluation`. Each step has its own `pyproject.toml` file with required dependencies. The dependencies can be installed using `pip install .`


## Tests and Validation

This usecase has been tested on specific RHOAI version. 
RHOAI Version : 2.25
Last Test Run : October 2025

## Next Steps
start with [Data Preprocessing (step 01)](./01_Data_Processing/README.md)
