# Data Preprocessing for Seed Dataset Generation (SDG)

The first step in tuning the knowledge pipeline is to transform raw documents into a structured seed dataset, ready for downstream learning or knowledge-tuning tasks. In our example we will process a PDF file using the [Docling](https://pypi.org/project/docling/) library and generate a seed dataset for SDG.

## Prerequisites

Ensure the following prerequisites are met before proceeding.

1. A Red Hat OpenShift AI cluster has been configured for this example and meets the minimum system requirements. See Knowledge Tuning Example [Requirements](../README.md#requirements) and [Getting Started](../README.md#getting-started) for more information.
2. An Open AI compatible endpoint for the model generating question and answer pairs, the model's API key, and the model's name.

## Inputs

A PDF file is required for processing. A sample file can be found in `../source_documents`.

## Process

The `Data_Preprocessing.ipynb` notebook will step though the following stages.

1. **PDF file → Docling Output JSON**: Convert source PDF documents into structured JSON using the [docling](https://pypi.org/project/docling/) library.
2. **Docling Output JSON → Chunks**: Segment the converted documents into smaller, meaningful text chunks.
3. **Chunks → Selected Chunks**: Randomly select a subset of chunks to serve as seed examples.
4. **Selected Chunks → QnA.yaml File**: Generate question-answer (QnA) pairs for each selected chunk using an LLM API.
5. **Chunks + QnA.yaml → Seed Dataset**: Combine the chunks and QnA pairs into a final seed dataset for SDG.

## Outputs

After executing all cells in the `Data_Preprocessing.ipynb` notebook, the following artifacts will be saved in the `../output/step_01` directory:

- `docling_output/`: JSONs
- `chunks.jsonl`: All chunks
- `selected_chunks.jsonl`: Randomly selected chunks
- `qna.yaml`: QnA pairs
- `final_seed_data.jsonl`: Final seed dataset for SDG

## Get Started

To get started, open the [Data_Preprocessing.ipynb](./Data_Preprocessing.ipynb) notebook within your workbench and follow the instructions directly in the notebook.

Once you are done, proceed to the next step, [Synthetic Data Processing](../02_Knowledge_Generation/README.md).

## Additional Notes

- The notebook is modular; you can adjust parameters (e.g., number of chunks, QnA prompt) as needed.
- Ensure your API credentials and endpoint for QnA generation are set correctly in the notebook.
- For large documents or datasets, monitor RAM and disk usage.

## References

- [docling documentation](https://pypi.org/project/docling/)
- [uv documentation](https://github.com/astral-sh/uv)
- [Jupyter Notebook](https://jupyter.org/)

For any issues or questions, please refer to the notebook comments or contact the repository maintainer.
