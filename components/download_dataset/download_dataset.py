import pandas as pd
from datasets import load_dataset
import mlflow
import os
import io
import argparse


class HFDatasetLogger:
    def __init__(self, mlflow_tracking_uri: str, experiment_name: str):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        print(f"MLflow Tracking URI: {self.mlflow_tracking_uri}")
        mlflow.set_experiment(self.experiment_name)

    def download_and_log_dataset(self, dataset_name: str, output_dir: str = "./output/", artifact_path: str = "raw_data", output_file: str = "output.txt") -> str:
        """
        Downloads a dataset from Hugging Face, saves a specified split to a Parquet file,
        and logs it as an MLflow artifact.

        Args:
            dataset_name (str): The name of the dataset on Hugging Face (e.g., "glue").
            output_dir (str): Local directory to save the downloaded dataset temporarily.
            artifact_path (str): Path within the MLflow artifact store where the dataset
                                 will be logged.
            output_file (str): Path to the file where the MLflow artifact URI will be written.

        Returns:
            str: The MLflow artifact URI of the logged dataset.
        """
        os.makedirs(output_dir, exist_ok=True)

        dataset = load_dataset(dataset_name, split="train")
        print(f"Dataset '{dataset_name}' downloaded successfully.")

        df = dataset.to_pandas()
        file_name = f"{dataset_name.replace('/', '_')}.parquet"
        local_parquet_path = os.path.join(output_dir, file_name)
        df.to_parquet(local_parquet_path, index=False)
        print(f"Dataset saved locally to: {local_parquet_path}")

        with mlflow.start_run(run_name=f"Download_{dataset_name.replace('/', '_')}") as run:
            mlflow.log_param("dataset_name", dataset_name)
            mlflow.log_param("num_rows", df.shape[0])
            mlflow.log_param("num_columns", df.shape[1])
            mlflow.log_artifact(local_parquet_path, artifact_path=artifact_path)

            print(f"Dataset logged to MLflow as artifact at: '{artifact_path}/{file_name}'")
            artifact_uri = mlflow.get_artifact_uri(f"{artifact_path}/{file_name}")
            print(f"MLflow Artifact URI: {artifact_uri}")

            with open(output_file, "w") as f:
                f.write(artifact_uri)
            print(f"Output written to: {output_file}")

            return artifact_uri


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="File to store the MLflow artifact URI")
    args = parser.parse_args()

    dataset_logger = HFDatasetLogger(
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080/"),
        experiment_name=os.getenv("EXPERIMENT_NAME")
    )

    k8s_dataset_uri = dataset_logger.download_and_log_dataset(
        dataset_name="ComponentSoft/k8s-kubectl-cot-20k",
        output_dir="./output/",
        artifact_path="dataset",
        output_file=args.output
    )

    if k8s_dataset_uri:
        print(f"Successfully logged ComponentSoft/k8s-kubectl-cot-20k train split. URI: {k8s_dataset_uri}")
    else:
        print("Failed to log ComponentSoft/k8s-kubectl-cot-20k train split.")
