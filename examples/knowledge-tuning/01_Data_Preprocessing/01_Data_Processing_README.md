# Data preprocessing for Seed Dataset Generation (SDG)

The first step in the knowledge tuning workflow is to transform raw documents into a structured seed dataset that is ready for downstream learning or knowledge-tuning tasks. In this example, you process a PDF file by using the [Docling](https://pypi.org/project/docling/) library and then generate a seed dataset for SDG.

## Prerequisites

- You have access to a Red Hat OpenShift AI cluster that is configured for this example and meets the minimum system requirements. See Knowledge Tuning Example [Requirements](../README.md#requirements) and [Getting Started](../00_Set_up/00_Set_up_README.md) for more information.

- You have the following information for the model that generates the question and answer pairs:
   - An Open AI compatible endpoint
   - The model's API key
   - The model's name

- You have access to input data. For this example, a sample PDF file is located in `../source_documents`.
<!-- need to change to the URL?
-->

## Procedure

Follow the steps in the [Data_Preprocessing.ipynb](./Data_Preprocessing.ipynb) notebook. THe notebook steps guide you through the following stages:

1. **PDF file → Docling Output JSON**: Convert a source PDF document into structured JSON format by using the [Docling](https://pypi.org/project/docling/) library.
2. **Docling Output JSON → Chunks**: Segment the converted documents into smaller, meaningful text chunks.
3. **Chunks → Selected Chunks**: Randomly select a subset of chunks to serve as seed examples.
4. **Selected Chunks → QnA.yaml File**: Generate question and answer (QnA) pairs for each selected chunk using an LLM API.
5. **Chunks + QnA.yaml → Seed Dataset**: Combine the chunks and QnA pairs into a final seed dataset for SDG.

For any issues or questions, refer to the notebook comments or contact the Git repository maintainer.

## Verification

After you run all cells in the `Data_Preprocessing.ipynb` notebook, verify that the following artifacts are in the `../output/step_01` directory:

- `docling_output/`: JSONs
- `chunks.jsonl`: All chunks
- `selected_chunks.jsonl`: Randomly selected chunks
- `qna.yaml`: QnA pairs
- `final_seed_data.jsonl`: Final seed dataset for SDG

## Next step

[Synthetic Data Processing](../02_Knowledge_Generation/README.md)

## Additional resources

- [docling documentation](https://pypi.org/project/docling/)
- [uv documentation](https://github.com/astral-sh/uv)
- [Jupyter Notebook](https://jupyter.org/)

