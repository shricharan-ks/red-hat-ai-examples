
# Knowledge Tuning example

Welcome! This repository contains a self-contained Knowledge Tuning example that uses the InstructLab methodology.

By following the steps in this example, you learn how to inject domain-specific knowledge into an AI model, improving its accuracy and relevance for an example use case. 

## Overview of the example end-to-end workflow 

For this example workflow, you complete the following steps in your workbench environment, as illustrated in Figure 1 and Figure 2:

1. Data Preprocessing — Convert a URL page to structured Markdown (by using Docling), chunk text, and produce a small seed dataset.
2. Knowledge Generation — Expand the seed dataset and generate more Q&A pairs by using an LLM (teacher model)
3. Knowledge Mixing — Combine generated Q&A pairs and summaries into training mixes.
4. Model Training — Fine-tune a model by using the prepared mixes.
5. Evaluation — Run evaluation notebooks and metrics on held-out data.

*Figure 1. End-to-end workflow diagram*

![End-to-end workflow diagram](../../assets/usecase/knowledge-tuning/Overall%20Flow.png)

*Figure 2. End-to-end workflow diagram*

![End-to-end workflow diagram](../../assets/usecase/knowledge-tuning/Detailed%20Flow.png)

## About the example use case

A Canadian bank wants its employees to use the bank’s internal chatbot app to obtain accurate information about the client identification methods required by the Financial Transactions and Reports Analysis Centre of Canada (FINTRAC).

A general-purpose language model lacks the specific, nuanced knowledge of Canadian anti-money laundering (AML) regulations. When asked a detailed, specific question, it is likely to provide a generic or incorrect answer. 

In this example, your goal is to fine-tune a base model on the official FINTRAC guidance so that it provides accurate, context-specific answers that reflect the actual regulations.

## About the example Git repo structure

The files for each step in the workflow are organized in subfolders of this Git repository, under the `examples/knowledge-tuning/` folder. Each subfolder contains a notebook, a `pyproject.toml` file for dependencies, and a `README.md` file.

## Next step

[00_Setup](./00_Setup/00_Setup_README.md)