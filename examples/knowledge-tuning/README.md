# Knowledge Tuning Example

This example workflow demonstrates **knowledge tuning** using the InstructLab methodology on Red Hat OpenShift AI. Knowledge tuning enables you to inject domain-specific knowledge into language models, improving their accuracy and relevance for enterprise use cases.

In this example you will have the opportunity to run example notebooks that cover,

1. **Data Collection**  
    Gather domain-specific documents, FAQs, manuals, or other relevant knowledge sources.
2. **Data Preprocessing**  
    Clean, format, and chunk the data for efficient ingestion.
3. **Knowledge Base Creation**  
    Store the processed data in a vector database or knowledge store.
4. **Model Integration**  
    Connect the language model to the knowledge base using retrieval-augmented generation (RAG) or similar techniques.
5. **Query Handling**  
    When a user asks a question, retrieve relevant knowledge snippets and feed them to the model for context-aware answers.
6. **Evaluation & Iteration**  
    Test the system, gather feedback, and refine the knowledge base and model integration.

## Requirements

The knowledge tuning workflow is intended to be executed on a Red Hat OpenShift AI cluster meeting the following requirements.

- **RAM**: 16 GB or higher recommended (for processing large PDFs and running docling)
- **Disk Space**: At least 10 GB free (for intermediate and output files)
- **Model Requirements**: Access to a suitable LLM endpoint for QnA generation (see notebook for API details)

## Getting Started

To get started, we need to ensure our Red Hat OpenShift AI cluster and example-specific configuration is complete.

### Data Science Project

_Instructions on how to configure a Data Science Project for this example. See the [Fraud Detection Workshop](https://rh-aiservices-bu.github.io/fraud-detection/fraud-detection-workshop/setting-up-your-data-science-project.html) for inspiration..._

### Storage Data Connections

_Instructions on how to configure a Storage Data Connections for this example. See the [Fraud Detection Workshop](https://rh-aiservices-bu.github.io/fraud-detection/fraud-detection-workshop/storing-data-with-connections.html) for inspiration..._

### Workbench

This example includes running several JupyterLab Notebooks within a single workbench. To create a workbench,

1. _Instructions on how to create a `Jupyter | TensorFlow | CUDA | Python 3.12` Workbench for this example. See the [Fraud Detection Workshop](https://rh-aiservices-bu.github.io/fraud-detection/fraud-detection-workshop/storing-data-with-connections.html) for inspiration..._
2. Launch the workbench

### Clone Example Repository

1. _Instructions on how to clone the repository for this example. See the [Fraud Detection Workshop](https://rh-aiservices-bu.github.io/fraud-detection/fraud-detection-workshop/importing-files-into-jupyter.html) for inspiration..._
2. Clone <https://github.com/shricharan-ks/red-hat-ai-examples>

### Setup

In your JupyterLab workbench, open the [00_Setup.ipynb](./00_Setup.ipynb) notebook and follow the instructions within the notebook.

### Let's Begin

Congratulations! Your workbench is configured and ready for the knowledge training example. Throughout this example you will be guided through to a series of notebooks. Each notebook and supporting documentation will provide more hands-on details about each step in the pipeline.

Let's get started with [Data Preprocessing](./01_Data_Preprocessing/README.md)!
