from components.document_processing import document_processing
from components.download_docling_models import download_docling_models
from components.ka import (
    generate_detailed_summaries,
    generate_document_based_qa,
    generate_extractive_summaries,
    generate_key_facts_summary,
    merge_all_outputs_component,
)
from components.knowledge_mixing import knowledge_mixing
from kfp import compiler, dsl, local

# from kfp_components.components.training import finetuning


@dsl.pipeline(
    name="knowledge-tuning-pipeline",
    description="A pipeline for knowledge tuning using document processing.",
)
def convert_pipeline(
    #  Dcocument processing parameters
    chunk_max_tokens: int = 512,
    chunk_overlap_tokens: int = 50,
    domain: str = "None",
    domain_outline: str = "None",
    icl_document: str = "None",
    icl_query1: str = "None",
    icl_query2: str = "None",
    icl_query3: str = "None",
    # Knowledge generation parameters
    model_name: str = "openai/gpt-oss-20b",
    api_key: str = "",
    api_base: str = "https://gpt-oss-20b-scharan.apps.rosa.scharan-1.0lts.p3.openshiftapps.com/v1",
    seed_data_subsample: int = 0,
    enable_reasoning: bool = True,
    number_of_summaries: int = 1,
    max_concurrency: int = 5,
    inference_timeout: int = 2500,
):

    caching_bool = False

    # Download models first - they'll be saved as an artifact
    artifacts = download_docling_models()
    artifacts.set_caching_options(True)

    # Document processing will use models from the artifact via DOCLING_CACHE_DIR env var
    document_processing_task = document_processing(
        artifacts_path=artifacts.outputs["output_path"],
        # output_path=output_path,
        chunk_max_tokens=chunk_max_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        domain=domain,
        document_outline=domain_outline,
        icl_document=icl_document,
        icl_query1=icl_query1,
        icl_query2=icl_query2,
        icl_query3=icl_query3,
    )
    document_processing_task.set_caching_options(False)

    detailed_summary_task = generate_detailed_summaries(
        input_dataset=document_processing_task.outputs["output_path"],
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        seed_data_subsample=seed_data_subsample,
        enable_reasoning=enable_reasoning,
        max_concurrency=max_concurrency,
        inference_timeout=inference_timeout,
        number_of_summaries=number_of_summaries,
    )
    detailed_summary_task.set_caching_options(True)

    extractive_summary_task = generate_extractive_summaries(
        input_dataset=document_processing_task.outputs["output_path"],
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        seed_data_subsample=seed_data_subsample,
        enable_reasoning=enable_reasoning,
        number_of_summaries=number_of_summaries,
        max_concurrency=max_concurrency,
        inference_timeout=inference_timeout,
    )
    extractive_summary_task.set_caching_options(False)

    key_facts_summary_task = generate_key_facts_summary(
        input_dataset=document_processing_task.outputs["output_path"],
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        seed_data_subsample=seed_data_subsample,
        enable_reasoning=enable_reasoning,
        max_concurrency=max_concurrency,
        inference_timeout=inference_timeout,
    )
    key_facts_summary_task.set_caching_options(True)

    document_based_qa_task = generate_document_based_qa(
        input_dataset=document_processing_task.outputs["output_path"],
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        seed_data_subsample=seed_data_subsample,
        enable_reasoning=enable_reasoning,
        max_concurrency=max_concurrency,
        inference_timeout=inference_timeout,
    )
    document_based_qa_task.set_caching_options(True)

    extractive_summary_task.after(
        document_based_qa_task, detailed_summary_task, key_facts_summary_task
    )

    merged_dataset_task = merge_all_outputs_component(
        extractive_data=extractive_summary_task.outputs["output_path"],
        detailed_data=detailed_summary_task.outputs["output_path"],
        key_facts_data=key_facts_summary_task.outputs["output_path"],
        doc_qa_data=document_based_qa_task.outputs["output_path"],
    )
    merged_dataset_task.set_caching_options(caching_bool)

    knowledge_mixing_task = knowledge_mixing(
        datasets_path=merged_dataset_task.outputs["merged_output"],
    )
    knowledge_mixing_task.set_caching_options(caching_bool)

    # student_model_artifact = save_student_model_locally(
    #     input_dataset=knowledge_mixing_task.outputs["output_path"],
    #     student_model_name="RedHatAI/Llama-3.1-8B-Instruct",
    # )
    # student_model_artifact.set_caching_options(caching_bool)

    # train_model(
    #     training_data=knowledge_mixing_task.outputs["output_path"],
    #     base_model_path=student_model_artifact.outputs["output_path"].path / "base_model" / "RedHatAI/Llama-3.1-8B-Instruct".replace("/", "__"),
    #     num_train_epochs=1,
    #     per_device_train_batch_size=1,
    # )


def run_local():
    local.init(runner=local.DockerRunner())
    convert_pipeline()


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=convert_pipeline,
        package_path="knowledge_tuning_pipeline.yaml",
    )

    # run_local()
    print("/n/n/n/n Compiling file")
