from kfp import dsl

BASE_IMAGE = "quay.io/fabianofranz/docling-ubi9:2.54.0"


@dsl.component(
    base_image="quay.io/fabianofranz/docling-ubi9:2.54.0",
    packages_to_install=["transformers>=4.57.1"],
)
def save_student_model_locally(
    input_dataset: dsl.Input[dsl.Artifact],
    output_path: dsl.Output[dsl.Artifact],
    student_model_name: str = "RedHatAI/Llama-3.1-8B-Instruct",
):
    import os
    from pathlib import Path

    os.environ["HF_HOME"] = str(Path(output_path.path) / "HF_models")

    BASE_MODEL_NAME = student_model_name
    # SAVE THE MODEL LOCALLY
    BASE_MODEL_PATH = (
        Path(output_path.path) / "base_model" / BASE_MODEL_NAME.replace("/", "__")
    )

    if not BASE_MODEL_PATH.exists():
        print("Model not available locally, Downloading the model locally ")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Save the model
        print(f"Loading model {BASE_MODEL_NAME}")
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
        model.save_pretrained(BASE_MODEL_PATH)
        print(f"Model saved to {BASE_MODEL_PATH}")

        # Save the tokenizer
        print(f"Loading tokenizer {BASE_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        tokenizer.save_pretrained(BASE_MODEL_PATH)
        print(f"Tokenizer saved to {BASE_MODEL_PATH}")

        del model
        del tokenizer
    else:
        print(f"Model Available locally : {BASE_MODEL_PATH}")
