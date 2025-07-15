from kfp.v2 import compiler
from kfp.v2.dsl import pipeline
import os
from kfp.components import load_component_from_file
from kfp import kubernetes


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")

@pipeline(
    name="Data-Quality-Pipeline",
    description="A pipeline for EDA, Data Preparation, and Data Validation with MLflow."
)
def data_quality_pipeline(
):
    """
    Defines the Kubeflow pipeline.

    Args:
        input_csv_uri (str): The URI of the input CSV file. In a real Kubeflow setup,
                             this might be a GCS/S3 path requiring a custom component
                             to download, or a path on a shared PVC. For simplicity,
                             this example assumes it's directly accessible or handled
                             by Kubeflow's artifact passing.
        mlflow_tracking_uri (str): The URI for the MLflow tracking server.
        wandb_api_key (str): The API key for Weights & Biases.
    """


    # pvc1 = kubernetes.CreatePVC(
    #     pvc_name = pvc_name,
    #     access_modes=['ReadWriteMany'],
    #     size='10Gi',
    #     storage_class_name='hostpath'
    # )

    pvc_name = "shared-pvc"
    download_dataset_op = load_component_from_file("components/download_dataset/download_dataset.yaml")
    download_dataset_op_task = download_dataset_op()
    kubernetes.mount_pvc(
        download_dataset_op_task,
        pvc_name = pvc_name,
        mount_path='/data/',
    )
    kubernetes.use_config_map_as_env(download_dataset_op_task, 'common-config', {"MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI"})
    kubernetes.set_image_pull_policy(download_dataset_op_task,"IfNotPresent")


    eda_op = load_component_from_file("components/eda/eda.yaml")

    eda_task = eda_op(
        input_csv=download_dataset_op_task.outputs["dataset_artifact_uri"],
        output_dir="/data/eda_output", # Kubeflow will map this to an output artifact path
    )

    kubernetes.mount_pvc(
        eda_task,
        pvc_name = pvc_name,
        mount_path='/data/',
    )

    kubernetes.use_config_map_as_env(eda_task, 'common-config', {"MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI"})
    kubernetes.set_image_pull_policy(eda_task,"IfNotPresent")


    data_preparation_op = load_component_from_file("components/data_prep/data_prep.yaml")
    # Step 2: Data Preparation
    data_prep_task = data_preparation_op(
        input_artifact_path=download_dataset_op_task.outputs["dataset_artifact_uri"],
    )

    kubernetes.set_image_pull_policy(data_prep_task,"IfNotPresent")

    kubernetes.mount_pvc(
    data_prep_task,
    pvc_name = pvc_name,
    mount_path='/data/',
    )

    kubernetes.use_config_map_as_env(data_prep_task, 'common-config', {"MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI"})
    # Ensure data preparation runs after EDA (logical flow, though not strict data dependency)
    data_prep_task.after(eda_task)

    # # Step 3: Data Validation
    # data_validation_task = data_validation_op(
    #     input_parquet=data_prep_task.outputs["output_parquet"], # Input from data_preparation_op
    #     output_validation_report="/tmp/validation_report.json",
    #     output_metrics_json="/tmp/data_quality_metrics.json",
    #     mlflow_tracking_uri=mlflow_tracking_uri,
    # )
    # data_validation_task.after(data_prep_task)
    
    train_test_split_op = load_component_from_file("components/train_test_split/train_test_split.yaml")
    train_test_split_task = train_test_split_op(input_artifact_path=download_dataset_op_task.outputs["dataset_artifact_uri"])


    kubernetes.set_image_pull_policy(train_test_split_task,"IfNotPresent")
    
    
    kubernetes.mount_pvc(
    train_test_split_task,
    pvc_name = pvc_name,
    mount_path='/data/',
    )

    kubernetes.use_config_map_as_env(train_test_split_task, 'common-config', {"MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI"})
    train_test_split_task.after(data_prep_task)


if __name__ == "__main__":
    # Compile the pipeline into a YAML file
    pipeline_filename = "data_quality_pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=data_quality_pipeline,
        package_path=pipeline_filename
    )
    print(f"Kubeflow pipeline compiled to {pipeline_filename}")