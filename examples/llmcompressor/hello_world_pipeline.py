import kfp
from kfp import dsl, kubernetes
from kfp.dsl.structures import ResourceSpec


@dsl.component(
    base_image="quay.io/opendatahub/llmcompressor-pipeline-runtime:main",
    packages_to_install=["llmcompressor~=0.5.0"],
)
def run_hello_gpu(name: str = "world") -> str:
    import torch
    result = f"Hello {name}! num_devices: {torch.cuda.device_count()}, available: {torch.cuda.is_available()}"
    print(result)

    return result


@dsl.pipeline(
    name="hello-world",
)
def pipeline(name: str = "Tom"):
    # task1 = run_hello_gpu()

    task2 = run_hello_gpu(
        name=name
    )
    task2.set_cpu_request("1400m").set_memory_request("320Mi")\
        .set_cpu_limit("1400m").set_memory_limit("320Mi")\
        .set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")
    kubernetes.add_toleration(task2, key="nvidia.com/gpu", operator="Equal", value="Tesla-T4-SHARED", effect="NoSchedule")
    

if __name__ == '__main__':
    COMPILE = True
    
    if COMPILE:
        kfp.compiler.Compiler().compile(
            pipeline_func=pipeline, package_path=__file__.replace(".py", ".yaml")
        )
    else:


        kubeflow_endpoint = "https://ds-pipeline-dspa-bdellabe.apps.prod.rhoai.rh-aiservices-bu.com"
        bearer_token = "<>"

        print(f'Connecting to Data Science Pipelines: {kubeflow_endpoint}')
        client = kfp.Client(
            host=kubeflow_endpoint,
            existing_token=bearer_token,
        )

        client.create_run_from_pipeline_func(
            pipeline,
            # arguments=metadata,
            experiment_name="Default",
            enable_caching=False
        )