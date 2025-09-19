
# Data Preprocessing for Seed Dataset Generation (SDG)

This directory contains the workflow and notebook for generating a **seed dataset** used in Synthetic Data Generation (SDG) as part of the Knowledge Tuning pipeline.

## Overview

The process transforms raw PDF documents into a structured seed dataset, ready for downstream machine learning or knowledge-tuning tasks. The main steps are:

1. **PDF file → Docling Output JSON**: Convert source PDF documents into structured JSON using the [docling](https://pypi.org/project/docling/) library.
2. **Docling Output JSON → Chunks**: Segment the converted documents into smaller, meaningful text chunks.
3. **Chunks → Selected Chunks**: Randomly select a subset of chunks to serve as seed examples.
4. **Selected Chunks → QnA.yaml File**: Generate question-answer (QnA) pairs for each selected chunk using an LLM API.
5. **Chunks + QnA.yaml → Seed Dataset**: Combine the chunks and QnA pairs into a final seed dataset for SDG.

All steps are implemented in the provided Jupyter Notebook: `Data_Preprocessing.ipynb`.

---

## Machine & Environment Requirements

- **RAM**: 16 GB or higher recommended (for processing large PDFs and running docling)
- **Disk Space**: At least 10 GB free (for intermediate and output files)
- **Python Version**: 3.12 (see `.python-version`)
- **Model Requirements**: Access to a suitable LLM endpoint for QnA generation (see notebook for API details)

---

## Setup Instructions

### 1. Create a Virtual Environment (using [uv](https://github.com/astral-sh/uv))

```sh
uv venv .venv
source .venv/bin/activate
```

### 2. Install Required Packages

All dependencies are listed in `pyproject.toml` and `requirements.txt`.

```sh
uv pip install -r requirements.txt
uv pip install -e .
```

#### Key Dependencies
- `docling` (for PDF parsing and chunking)
- `ai-tools` (custom utilities for QnA generation and seed dataset creation)

> **Note:** The `ai-tools` library is a local dependency. Ensure the wheel file or source is available as specified in `pyproject.toml`.

---

## Preparation Steps

1. **Place your PDF files** in the `source_documents/` directory.
2. **Run the notebook** `Data_Preprocessing.ipynb` step by step, following the flow described above.
3. **Outputs** will be saved in the `output/step_01/` directory:
    - `docling_output/` (JSONs)
    - `chunks.jsonl` (all chunks)
    - `selected_chunks.jsonl` (randomly selected chunks)
    - `qna.yaml` (QnA pairs)
    - `final_seed_data.jsonl` (final seed dataset)

---

## Additional Notes

- The notebook is modular; you can adjust parameters (e.g., number of chunks, QnA prompt) as needed.
- Ensure your API credentials and endpoint for QnA generation are set correctly in the notebook.
- For large documents or datasets, monitor RAM and disk usage.

---

## References
- [docling documentation](https://pypi.org/project/docling/)
- [uv documentation](https://github.com/astral-sh/uv)
- [Jupyter Notebook](https://jupyter.org/)

---

For any issues or questions, please refer to the notebook comments or contact the repository maintainer.
