# Knowledge tuning example

This example workflow demonstrates **knowledge tuning** using the InstructLab methodology on Red Hat OpenShift AI. Knowledge tuning enables you to inject domain-specific knowledge into language models, improving their accuracy and relevance for enterprise use cases.

This **Knowledge tuning** example workflow demonstrates how to inject domain-specific knowledge into language models, improving their accuracy and relevance for enterprise use cases. This **Knowledge tuning** examples uses the InstructLab methodology on Red Hat OpenShift AI. 

In this example, you can run example notebooks to learn about the following capabilities:

- **Data Collection**  
    Gather domain-specific documents, FAQs, manuals, or other relevant knowledge sources.
- **Data Preprocessing**  
    Clean, format, and chunk the data for efficient ingestion.
- **Knowledge Base Creation**  
    Store the processed data in a vector database or knowledge store.
- **Model Integration**  
    Connect the language model to the knowledge base using retrieval-augmented generation (RAG) or similar techniques.
- **Query Handling**  
    When a user asks a question, retrieve relevant knowledge snippets and feed them to the model for context-aware answers.
- **Evaluation & Iteration**  
    Test the system, gather feedback, and refine the knowledge base and model integration.

## Requirements

To run the knowledge tuning workflow, you must use a Red Hat OpenShift AI cluster that meets the following requirements.

- **RAM**: 16 GB or higher recommended for processing large PDFs and running docling
- **Disk Space**: At least 10 GB free for intermediate and output files
- **Model Requirements**: Access to an LLM endpoint for question and answer (QnA) generation.

## Table of Contents

[Set up](./00_Set_Up/README.md)
[Data preprocessing](./01_Data_Preprocessing/README.md)
[Knowledge generation](./02_Knowledge_Generation/README.md)
