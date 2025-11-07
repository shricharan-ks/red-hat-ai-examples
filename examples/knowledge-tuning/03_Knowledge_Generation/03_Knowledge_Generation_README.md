
# Step 3: Knowledge Generation

## Navigation

- [Knowledge Tuning Overview](../README.md)
- [Setup](../00_Setup/00_Setup_README.md)
- [Step 1: Base Model Evaluation](../01_Base_Model_Evaluation/01_Base_Model_Evaluation_README.md)
- [Step 2: Data Processing](../02_Data_Processing/02_Data_Processing_README.md)
- Step 3: Knowledge Generation
- [Step 4: Knowledge Mixing](../04_Knowledge_Mixing/04_Knowledge_Mixing_README.md)
- [Step 5: Model Training](../05_Model_Training/05_Model_Training_README.md)
- [Step 6: Evaluation](../06_Evaluation/06_Evaluation_README.md)

## Knowledge generation and expanding seed dataset into Q&A

This step expands the curated seed examples produced in the previous step into a larger set of Q&A pairs using LLMs and
local utilities. It can be used to produce synthetic training examples or to augment existing datasets.

![Knowledge Generation Flow Diagram](../../../assets/usecase/knowledge-tuning/Knowledge%20Genertaion.png)

### Prerequisites

- Previous sections successfully completed in order
- Access to an LLM endpoint
- Environment variables are set via workbench secrets or `.env` file. See [.env.example](./.env.example) for reference.

#### Environment variables

- `TEACHER_MODEL_API_KEY` — LLM API key
- `TEACHER_MODEL_BASE_URL` — LLM HTTP endpoint
- `TEACHER_MODEL_NAME` — Model to call for generation

### Procedure

1. Open the [Knowledge_Generation.ipynb](./Knowledge_Generation.ipynb) file in JupyterLab and follow the instructions directly in the notebook.

### Verification

After completing the notebook instructions, the following artifacts are generated.

- `output/step_03/**/gen.jsonl` — Raw generation outputs for,
  - Extractive summary
  - Detailed summary
  - Key facts Q&A
  - Document based Q&A

## Debug & tips

- If generation fails, verify the API key, endpoint URL, and request rate limits.

## Next step

Proceed to [Step 4: Knowledge Mixing](../04_Knowledge_Mixing/04_Knowledge_Mixing_README.md).
