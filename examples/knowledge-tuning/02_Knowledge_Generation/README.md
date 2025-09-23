# Knowledge Generation for Seed Dataset Generation (SDG)

The second step in tuning the knowledge pipeline is generating knowledge (QnA pairs and related data) as part of the Synthetic Data Generation (SDG) pipeline. In our example we will take the seed dataset (produced in the Data Preprocessing step) and generate additional knowledge artifacts, such as QnA pairs, using LLMs and custom utilities.

## Prerequisites

Ensure the following prerequisites are met before proceeding.

1. The previous example, [Data Preprocessing](../01_Data_Preprocessing/README.md), was successfully completed in this workbench.
2. An Open AI compatible endpoint for the model generating question and answer pairs, the model's API key, and the model's name.

## Inputs

The seed dataset from the Data Preprocessing step is available at `../output/step_1/final_seed_data.jsonl`

## Process

The `Knowledge Generation.ipynb` notebooks will step through the following stages.

1. **Seed Dataset → QnA Generation**: Use the seed dataset and an LLM endpoint to generate new QnA pairs or expand existing ones.
2. **Review & Post-process**: Optionally review, filter, or post-process the generated QnA pairs for quality and relevance.
3. **Save Outputs**: Store the generated knowledge artifacts for downstream use.

## Outputs

After executing all cells in the `Knowledge Generation.ipynb` notebook, the following artifacts will be saved in the `../output/step_02` directory:

- `qagen-*.json`: Generated QnA pairs
- `qna.yaml`: Final QnA file
- Other intermediate or final artifacts as needed

## Get Started

To get started, open the [Knowledge Generation.ipynb](./Knowledge_Generation.ipynb) notebook within your workbench and follow the instructions directly in the notebook.

Once you are done, proceed tot he next step...

## Additional Notes

- The notebook is modular; you can adjust parameters (e.g., number of QnA pairs, prompt) as needed.
- Ensure your API credentials and endpoint for QnA generation are set correctly in the notebook.
- For large datasets, monitor RAM and disk usage.

## References

- [docling documentation](https://pypi.org/project/docling/)
- [uv documentation](https://github.com/astral-sh/uv)
- [Jupyter Notebook](https://jupyter.org/)

For any issues or questions, please refer to the notebook comments or contact the repository maintainer.
