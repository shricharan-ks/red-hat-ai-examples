
# Knowledge Generation for Seed Dataset Generation (SDG)

This directory contains the workflow and notebook for generating knowledge (QnA pairs and related data) as part of the Synthetic Data Generation (SDG) pipeline.

## Overview

This step takes the seed dataset (produced in the Data Preprocessing step) and generates additional knowledge artifacts, such as QnA pairs, using LLMs and custom utilities. The main steps are:

1. **Seed Dataset → QnA Generation**: Use the seed dataset and an LLM endpoint to generate new QnA pairs or expand existing ones.
2. **Review & Post-process**: Optionally review, filter, or post-process the generated QnA pairs for quality and relevance.
3. **Save Outputs**: Store the generated knowledge artifacts for downstream use.

All steps are implemented in the provided Jupyter Notebook: `Knowledge_Generation.ipynb`.

---

## Machine & Environment Requirements

- **RAM**: 16 GB or higher recommended
- **Disk Space**: At least 10 GB free
- **Python Version**: 3.12 (see `.python-version`)
- **Model Requirements**: Access to a suitable LLM endpoint for QnA generation (see notebook for API details)

---

## Setup Instructions

### 1. Check/Update the ai-tools Wheel Path

Before creating the virtual environment, ensure the path to the `ai-tools` wheel file in `pyproject.toml` is correct and points to the latest build. 
**Note:** Use an absolute path. Update the path if necessary.

### 2. Create a Virtual Environment (using [uv](https://github.com/astral-sh/uv))

```sh
uv venv .venv
source .venv/bin/activate
```

### 3. Install Required Packages

All dependencies are listed in `pyproject.toml` and `requirements.txt`.

```sh
uv pip install -r requirements.txt
uv pip install -e .
```

#### Key Dependencies
- `docling` (for document parsing and chunking)
- `ai-tools` (custom utilities for QnA generation and knowledge creation)

> **Note:** The `ai-tools` library is a local dependency. Ensure the wheel file or source is available as specified in `pyproject.toml`.

---

## Preparation Steps

1. **Ensure the seed dataset** (e.g., `final_seed_data.jsonl`) is available from the Data Preprocessing step.
2. **Run the notebook** `Knowledge_Generation.ipynb` step by step, following the flow described above.
3. **Outputs** will be saved in the `output/` directory:
    - `qagen-*.json` (generated QnA pairs)
    - `qna.yaml` (final QnA file)
    - Other intermediate or final artifacts as needed

---

## Additional Notes

- The notebook is modular; you can adjust parameters (e.g., number of QnA pairs, prompt) as needed.
- Ensure your API credentials and endpoint for QnA generation are set correctly in the notebook.
- For large datasets, monitor RAM and disk usage.

---

## References
- [docling documentation](https://pypi.org/project/docling/)
- [uv documentation](https://github.com/astral-sh/uv)
- [Jupyter Notebook](https://jupyter.org/)

---

For any issues or questions, please refer to the notebook comments or contact the repository maintainer.