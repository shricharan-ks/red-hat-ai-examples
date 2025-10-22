# OSFT Continual Learning on Red Hat OpenShift AI (RHOAI)

This example provides and overview on OSFT algorithm and an example on how to use it context of Red Hat OpenShift AI.

Our example will go through distributed training on two nodes with two GPUs each (2x48GB) however it can be tweaked to run on smaller configurations.

## Overview

Fine-tuning language models is hard—you need good data, lots of resources, and even small changes can cause problems. This makes it tough to add new abilities to a model. This problem is called continual learning and is what our new training technique, orthogonal subspace fine-tuning (OSFT), solves.

The OSFT algorithm implements Orthogonal Subspace Fine-Tuning based on Nayak et al. (2025), arXiv:2504.07097. This algorithm allows for continual training of pre-trained or instruction-tuned models without the need of a supplementary dataset to maintain the distribution of the original model/dataset that was trained.

**Key Benefits:**
- Enables continual learning without catastrophic forgetting
- No need for supplementary datasets to maintain original model distribution
- Significantly reduces data requirements for customizing instruction-tuned models
- Memory requirements similar to standard SFT

### Data Format Requirements

Training Hub's OSFT algorithm supports both **processed** and **unprocessed** data formats via the mini-trainer backend.

#### Option 1: Standard Messages Format (Recommended)

Your training data should be a **JSON Lines (.jsonl)** file containing messages data:

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there! How can I help you?"}]}
{"messages": [{"role": "user", "content": "What is OSFT?"}, {"role": "assistant", "content": "OSFT stands for Orthogonal Subspace Fine-Tuning..."}]}
```

#### Message Structure
- **`role`**: One of `"system"`, `"user"`, `"assistant"`, or `"pretraining"`
- **`content`**: The text content of the message
- **`reasoning_content`** (optional): Additional reasoning traces

#### Masking Control with `unmask_messages` Parameter

Control training behavior during data processing:

**Standard instruction tuning (default):**
```python
osft(..., unmask_messages=False)  # Only assistant responses used for loss
```

**Pretraining mode:**
```python
osft(..., unmask_messages=True)   # All content except system messages used for loss
```

#### Option 2: Pre-processed Dataset

If you have pre-processed data with `input_ids` and `labels` fields:

```json
{"input_ids": [1, 2, 3, ...], "labels": [1, 2, 3, ...]}
{"input_ids": [4, 5, 6, ...], "labels": [4, 5, 6, ...]}
```

Use with:
```python
osft(..., use_processed_dataset=True)
```
## Requirements

* An OpenShift cluster with OpenShift AI (RHOAI) installed:
  * The `dashboard`, `trainingoperator` and `workbenches` components enabled
* Sufficient worker nodes for your configuration(s). The example by default requires 2xL40/L40S (2x48GB) GPUs and a single GPU for workbench model evaluation. However the configuration can be tweaked to reduce the requirements.
* A dynamic storage provisioner supporting RWX PVC provisioning

## Setup

### Setup Workbench

* Access the OpenShift AI dashboard, for example from the top navigation bar menu:
![](./docs/01.png)
* Log in, then go to _Data Science Projects_ and create a project:
![](./docs/02.png)
* Once the project is created, click on _Create a workbench_:
![](./docs/03.png)
* Then create a workbench with the following settings:
    * Select the `Jupyter | Minimal | CPU | Python 3.12` notebook image if you want to run CPU based evaluation, `Jupyter | Minimal | CUDA | Python 3.12` for NVIDIA GPUs evaluation or `Jupyter | Minimal | ROCm | Python 3.12` for AMD GPUs evaluation and `Medium` container size:
    ![](./docs/04a.png)
    * Add an accelerator if you plan on evaluating your model on GPUs (faster):
    ![](./docs/04b.png)
        > [!NOTE]
        > Adding an accelerator is only needed to test the fine-tuned model from within the workbench so you can spare an accelerator if needed.
    * Create a storage that'll be shared between the workbench and the training pods.
    Make sure it uses a storage class with RWX capability and set it to 15GiB in size:
        ![](./docs/04c.png)
        > [!NOTE]
        > You can attach an existing shared storage if you already have one instead.
    * Review the storage configuration and click "Create workbench":
    ![](./docs/04d.png)
* From "Workbenches" page, click on _Open_ when the workbench you've just created becomes ready:
![](./docs/05.png)
* From the workbench, clone this repository, i.e., `https://red-hat-data-services/red-hat-ai-examples.git`
![](./docs/06.png)
* Navigate to the `examples/fine-tuning/osft` directory and open the `osft-example.ipynb` notebook
* The remaining part of this example is within the notebook itself

> [!IMPORTANT]
> * By default, the notebook requires 2xL40/L40S (2x48GB) but:
>   * The example goes through distributed training on two nodes with two GPUs but it can be changed
>   * If you want to do model evaluation part of the example, ideally an accelerator is attached to workbench
