
## RHOAI workbench and platform specifications

Use the following guidelines to configure a Red Hat OpenShift AI (RHOAI) workbench that can run the example notebooks reliably.

- Workbench image: `Jupyter | Minimal | CUDA | Python 3.12`
- GPU: Optional for preprocessing and mixing. Required/strongly recommended for model training (step 04) — at least one NVIDIA A100/40GB or similar for fine-tuning large models; for smaller student models an 8–16 GB GPU may suffice.
- Persistent storage: A mounted persistent volume (PVC) for the workbench is required to store `examples/knowledge-tuning/output/` and large intermediate artifacts. Allocate 200+ GB for realistic datasets.
- Accelerators: If using on-node inference/quantized student models, preferred accelerators are NVIDIA GPUs with CUDA support.
- Environment variables: Workbench should provide a `.env` or secret injection mechanism for API keys and configuration. Example `.env.example` files are provided at usecase level and as well as step level.


## How to use this example

1. Create a RHOAI workbench with the recommended image.
2. Attach a Persistent Volume with at least 200 GB.
3. Clone this repository inside the workbench.
4. Work through the notebooks step-by-step, from `01_Data_Preprocessing` to `05_Evaluation`. Each step has its own `pyproject.toml` file with required dependencies. The dependencies can be installed using `pip install .`


## Tests and Validation

This example was tested on RHOAI Version 2.25
Last Test Run : October 2025

## Next Step

[01 Data Preprocessing](./01_Data_Preprocessing/README.md)