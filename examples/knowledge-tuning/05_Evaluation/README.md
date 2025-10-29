# Step 05 — Evaluation

Navigation

- Overview: [Knowledge Tuning root](../README.md)
- Step 01 — Data Preprocessing: [../01_Data_Preprocessing/README.md](../01_Data_Processing/README.md)
- Step 02 — Knowledge Generation: [../02_Knowledge_Generation/README.md](../02_Knowledge_Generation/README.md)
- Step 03 — Knowledge Mixing: [../03_Knowledge_Mixing/README.md](../03_Knowledge_Mixing/README.md)
- Step 04 — Model Training: [../04_Model_Training/README.md](../04_Model_Training/README.md)
- Step 05 — Evaluation (this page)

Purpose

This step evaluates trained models and generated datasets against held-out test data. It computes metrics and produces human-readable reports for quality assessment.

End-to-end flow inside this step

- Input: trained model checkpoints (`output/step_04/`) and evaluation datasets → run evaluation scripts → metrics and reports written to `output/step_05/`

Prerequisites

- A trained model checkpoint produced in Step 04.
- Evaluation datasets (format depends on metric scripts).

Inputs

- Model checkpoints in `output/step_04/`
- Evaluation datasets in `output/` or a specified path

Outputs

- Evaluation metrics and reports (e.g., accuracy, BLEU, human-review files) in `output/step_05/`

Environment variables (common examples)

- `OUTPUT_DATA_FOLDER` — experiment folder used by evaluation scripts

Install dependencies (pyproject)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

How to run

1. Activate the venv for this step.
2. Open `Evaluation.ipynb` and run the notebook cells, or run evaluation scripts in this folder.

Debug & tips

- Ensure the tokenizer and model used for evaluation match what was used during preprocessing and training.

Next steps

- Review metrics and iterate on earlier steps (data, generation, mixing, or training) as needed.
