import os
import json
import argparse
from kfp.dsl import pipeline, component, OutputPath, PipelineTask
from kfp.components import load_component_from_file
from kfp import kubernetes
from components.training.katib_train_op import create_katib_experiment
from kfp_client.kfp_client_manager import KFPClientManager


class KubeflowOpsHelper:
    def __init__(self, pvc_name: str = "shared-pvc", config_map: str = "common-config"):
        self.pvc_name = pvc_name
        self.config_map = config_map

    def apply_common_settings(self, task: PipelineTask, env_var: str):
        kubernetes.mount_pvc(task, pvc_name=self.pvc_name, mount_path='/data/')
        kubernetes.use_config_map_as_env(task, self.config_map, {"MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI", "EXPERIMENT_NAME":"EXPERIMENT_NAME"})
        kubernetes.set_image_pull_policy(task, "IfNotPresent")

        # Using container with new tag
        image = os.getenv(env_var)
        task.set_container_image(image)


    def apply_gpu(self, task: PipelineTask, count: str):
        task.set_accelerator_type("nvidia.com/gpu")
        task.set_gpu_limit(count)
        kubernetes.add_toleration(task=task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")

    @staticmethod
    def load_op(path):
        return load_component_from_file(path)


@component(base_image="python:3.11.4-slim-buster")
def convert_katib_results(katib_results: str, lora_r: OutputPath(int), lora_dropout: OutputPath(float)): # type: ignore
    katib_results = json.loads(katib_results)
    for param in katib_results:
        if param["name"] == "r":
            with open(lora_r, "w") as f:
                f.write(str(param["value"]))
        elif param["name"] == "dropout":
            with open(lora_dropout, "w") as f:
                f.write(str(param["value"]))


@pipeline(name="kubeChat-pipeline")
def kube_chat_pipeline():
    helper = KubeflowOpsHelper()

    download_task = helper.load_op("components/download_dataset/download_dataset.yaml")()
    helper.apply_common_settings(download_task, "DOWNLOAD_DATASET_IMAGE")

    data_prep_task = helper.load_op("components/data_prep/data_prep.yaml")(
        input_artifact_path=download_task.outputs["dataset_artifact_uri"]
    )
    helper.apply_common_settings(data_prep_task, "DATA_PREP_IMAGE")

    split_task = helper.load_op("components/train_test_split/train_test_split.yaml")(
        input_artifact_path=data_prep_task.outputs["output_dataset_uri_path"]
    )
    helper.apply_common_settings(split_task, "TRAIN_TEST_SPLIT_IMAGE")



    
    # need to be added, as variable or argparse is not available during compile time.
    # repeated but required so that, for each commit new experiment will be created for
    # hyperparameter tunning
    base = os.getenv("EXPERIMENT_NAME")
    COMMIT_SHA = os.getenv("COMMIT_SHA")
    katib_experiment_name = f"{base}-{COMMIT_SHA}"
    katib_task = create_katib_experiment(
        experiment_name=katib_experiment_name,
        namespace=os.getenv("NAMESPACE"),
        training_dataset_path=split_task.outputs["train_artifact_path"],
        output_model_dir="/tmp/model/", # as model directory is not required for next training job
        base_yaml_path="katib.yaml",
        image=os.getenv("TRAINING_IMAGE")
    )
    kubernetes.set_image_pull_policy(katib_task, "IfNotPresent")
    helper.apply_gpu(katib_task, "1")

    convert_task = convert_katib_results(katib_results=katib_task.output)

    train_task = helper.load_op("components/training/train.yaml")(
        training_dataset_path=split_task.outputs["train_artifact_path"],
        lora_r=convert_task.outputs["lora_r"],
        lora_dropout=convert_task.outputs["lora_dropout"]
    )
    helper.apply_common_settings(train_task, "TRAINING_IMAGE")
    helper.apply_gpu(train_task, "1")

    test_task = helper.load_op("components/testing/test.yaml")(
        test_dataset_path=split_task.outputs["test_artifact_path"],
        model_path=train_task.outputs["output_model_dir"]
    )
    helper.apply_common_settings(test_task, "TESTING_IMAGE")
    helper.apply_gpu(test_task, "1")


class PipelineExecutor:
    def __init__(self):
        self.kfp_client_manager = KFPClientManager(
            api_url=os.getenv("KUBEFLOW_API_URL"),
            skip_tls_verify=True,
            dex_username=os.getenv("USER_NAME"),
            dex_password=os.getenv("PASSWORD"),
            dex_auth_type="local"
        )
        self.kfp_client = self.kfp_client_manager.create_kfp_client()
        self.pipeline_name = os.getenv("PIPELINE_NAME", "kubeChat-pipeline")
        self.namespace = os.getenv("NAMESPACE")
        self.experiment_name = os.getenv("EXPERIMENT_NAME")
        self.version_name = f"{self.pipeline_name}-{GITHUB_COMMIT_SHA}"

    def get_or_create_experiment(self):

        experiments = self.kfp_client.list_experiments(namespace=self.namespace).experiments
        if experiments is not None:
            for exp in experiments:
                if exp.display_name == self.experiment_name:
                    return exp
        return self.kfp_client.create_experiment(name=self.experiment_name, namespace=self.namespace)
            

    def find_existing_pipeline(self):
        pipelines = self.kfp_client.list_pipelines(namespace=self.namespace).pipelines
        if pipelines is not None:
            for p in pipelines:
                if p.display_name == self.pipeline_name:
                    return p
        return None

    def execute(self):
        experiment = self.get_or_create_experiment()
        existing_pipeline = self.find_existing_pipeline()

        if existing_pipeline:
            uploaded_version = self.kfp_client.upload_pipeline_version_from_pipeline_func(
                kube_chat_pipeline,
                pipeline_version_name=self.version_name,
                pipeline_id=existing_pipeline.pipeline_id
            )
            run = self.kfp_client.run_pipeline(
                experiment_id=experiment.experiment_id,
                pipeline_id=existing_pipeline.pipeline_id,
                version_id=uploaded_version.pipeline_version_id,
                job_name=self.version_name
            )
        else:
            uploaded_pipeline = self.kfp_client.upload_pipeline_from_pipeline_func(
                kube_chat_pipeline,
                pipeline_name=self.pipeline_name,
                namespace=self.namespace
            )
            version_id = self.kfp_client.list_pipeline_versions(pipeline_id=uploaded_pipeline.pipeline_id).pipeline_versions[0].pipeline_version_id
            run = self.kfp_client.run_pipeline(
                experiment_id=experiment.experiment_id,
                pipeline_id=uploaded_pipeline.pipeline_id,
                version_id=version_id,
                job_name=self.version_name
            )

        if run.error:
            raise RuntimeError("Pipeline failed to run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create KubeFlow pipeline and run")
    GITHUB_COMMIT_SHA = os.getenv("GITHUB_SHA")
    PipelineExecutor().execute()
