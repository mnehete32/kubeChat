from kfp.dsl import pipeline, component, OutputPath
from kfp.components import load_component_from_file
from kfp import kubernetes
from components.training.katib_train_op import create_katib_experiment
from datetime import datetime
from kfp_client.kfp_client_manager import KFPClientManager
import os
import json


class KubeflowOpsHelper:
    def __init__(self, pvc_name: str = "shared-pvc", config_map: str = "common-config"):
        self.pvc_name = pvc_name
        self.config_map = config_map

    def apply_common_settings(self, task):
        kubernetes.mount_pvc(task, pvc_name=self.pvc_name, mount_path='/data/')
        kubernetes.use_config_map_as_env(task, self.config_map, {"MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI"})
        kubernetes.set_image_pull_policy(task, "IfNotPresent")

    @staticmethod
    def load_op(path):
        return load_component_from_file(path)


@component(base_image="python:3.11.4-slim-buster")
def convert_katib_results(katib_results: str, lora_r: OutputPath(int), lora_dropout: OutputPath(float)):
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
    helper.apply_common_settings(download_task)

    data_prep_task = helper.load_op("components/data_prep/data_prep.yaml")(
        input_artifact_path=download_task.outputs["dataset_artifact_uri"]
    )
    helper.apply_common_settings(data_prep_task)

    split_task = helper.load_op("components/train_test_split/train_test_split.yaml")(
        input_artifact_path=data_prep_task.outputs["output_dataset_uri_path"]
    )
    helper.apply_common_settings(split_task)

    katib_task = create_katib_experiment(
        experiment_name="katib-train",
        namespace="kube-chat",
        training_dataset_path=split_task.outputs["train_artifact_path"],
        output_model_dir="/data/model/",
        base_yaml_path="katib.yaml"
    )
    kubernetes.set_image_pull_policy(katib_task, "IfNotPresent")

    convert_task = convert_katib_results(katib_results=katib_task.output)

    train_task = helper.load_op("components/training/train.yaml")(
        training_dataset_path=split_task.outputs["train_artifact_path"],
        lora_r=convert_task.outputs["lora_r"],
        lora_dropout=convert_task.outputs["lora_dropout"]
    )
    helper.apply_common_settings(train_task)

    test_task = helper.load_op("components/testing/test.yaml")(
        test_dataset_path=split_task.outputs["test_artifact_path"],
        model_path=train_task.outputs["output_model_dir"]
    )
    helper.apply_common_settings(test_task)


class PipelineExecutor:
    def __init__(self):
        self.kfp_client_manager = KFPClientManager(
            api_url="http://kubeflow.nehete.com/pipeline/",
            skip_tls_verify=True,
            dex_username=os.getenv("USER_NAME", "mayur@nehete.com"),
            dex_password=os.getenv("PASSWORD", "password"),
            dex_auth_type="local"
        )
        self.kfp_client = self.kfp_client_manager.create_kfp_client()
        self.pipeline_name = os.getenv("PIPELINE_NAME", "kubeChat-pipeline")
        self.namespace = os.getenv("NAMESPACE", "kube-chat")
        self.experiment_name = os.getenv("EXPERIEMENT_NAME", "KubeChat")
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.version_name = f"{self.pipeline_name} {self.timestamp}"

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
    PipelineExecutor().execute()
