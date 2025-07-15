import pandas as pd
from datasets import load_dataset
import mlflow
import os
import io

# Set MLflow tracking URI from environment variable or default to local
# Make sure your MLflow server is running and accessible at this URI.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080/")
print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Set the MLflow experiment name
mlflow.set_experiment("kubeChat")


def download_and_log_hf_dataset(
    dataset_name: str,
    output_dir: str = "./output/",
    artifact_path: str = "raw_data"
) -> str:
    """
    Downloads a dataset from Hugging Face, saves a specified split to a Parquet file,
    and logs it as an MLflow artifact.

    Args:
        dataset_name (str): The name of the dataset on Hugging Face (e.g., "glue").
        output_dir (str): Local directory to save the downloaded dataset temporarily.
                          Defaults to "hf_datasets".
        artifact_path (str): The path within the MLflow artifact store where the dataset
                             will be logged. Defaults to "raw_data".

    Returns:
        str: The MLflow artifact URI of the logged dataset.
    """
    os.makedirs(output_dir, exist_ok=True)


    try:
        # Load the dataset from Hugging Face
        # if subset_name:
            # dataset = load_dataset(dataset_name, subset_name, split=split)
        # else:
        dataset = load_dataset(dataset_name,split="train")

        print(f"Dataset '{dataset_name}' downloaded successfully.")

        # Convert to pandas DataFrame for easy saving to Parquet
        df = dataset.to_pandas()

        # Define the local path for the Parquet file
        file_name = f"{dataset_name.replace('/', '_')}.parquet"
        local_parquet_path = os.path.join(output_dir, file_name)

        # Save the DataFrame to a Parquet file
        df.to_parquet(local_parquet_path, index=False)
        print(f"Dataset saved locally to: {local_parquet_path}")

        # Start an MLflow run to log the artifact
        with mlflow.start_run(run_name=f"Download_{dataset_name.replace('/', '_')}") as run:
            mlflow.log_param("dataset_name", dataset_name)
            mlflow.log_param("num_rows", df.shape[0])
            mlflow.log_param("num_columns", df.shape[1])

            # Log the Parquet file as an artifact
            mlflow.log_artifact(local_parquet_path, artifact_path=artifact_path)
            print(f"Dataset logged to MLflow as artifact at: '{artifact_path}/{file_name}'")

            # Construct the MLflow artifact URI
            artifact_uri = mlflow.get_artifact_uri(f"{artifact_path}/{file_name}")
            print(f"MLflow Artifact URI: {artifact_uri}")
            with open(args.output, "w") as f:
                f.write(artifact_uri)
            print(f"Output added to kubeflow")
            return artifact_uri

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    import argparse    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    k8s_dataset_uri = download_and_log_hf_dataset(
        dataset_name="ComponentSoft/k8s-kubectl-cot-20k",
        output_dir="/data/",
        artifact_path="dataset"
    )
    if k8s_dataset_uri:
        print(f"Successfully logged ComponentSoft/k8s-kubectl-cot-20k train split. URI: {k8s_dataset_uri}")
    else:
        print("Failed to log ComponentSoft/k8s-kubectl-cot-20k train split.")