# Model Compression and Evaluationon on Red Hat OpenShift AI (RHOAI)

The compression and optimization of pretrained, off-the-shelf large language models (LLMs) is essential for organizations to reduce the hardware and energy requirements of their AI applications. This set of examples will introduce RHOAI users (Machine Learning Engineers and Data Scientists) to model compression using two tools in the open-source VLLM project: [`llm-compressor`](https://github.com/vllm-project/llm-compressor) for model compression and the [`vllm`](https://github.com/vllm-project/vllm) deployment engine for benchmarking of compressed models. 

While research in model compression is continually evolving and growing increasingly complex, the examples require only a basic understanding of Python and the [HuggingFace software ecosystem](https://huggingface.co/docs/transformers/index). By the end, users should know how to run compare the performance of different compression techniques, and how to customize to their own dataset or pretrained model.

> [!NOTE]  
> We also publish compressed versions of popular LLMs to HuggingFace that can be downloaded directly -- https://huggingface.co/RedHatAI

## Contents

Two pathways are provided:

1. [A Jupyter Notebook](workbench_example.ipynb) that can be used with the workbench image available at https://quay.io/repository/opendatahub/llmcompressor-workbench.

2. [An example pipeline](oneshot_pipeline.py) splits the notebook up into components that can be run as a Data Science Pipeline, with the runtime image available at https://quay.io/repository/opendatahub/llmcompressor-pipeline-runtime. The goal of this is to highlight how multiple compression algorithms can be compared in parallel, with compressed model artifacts and evaluation results easily shareable with stakeholders in a single web UI.

## Prerequisites

These examples assume the user has access to a Data Science Project on a Red Hat OpenShift AI cluster. If using the pipeline, Data Science Pipelines must be enabled. `HF_TOKEN` ...