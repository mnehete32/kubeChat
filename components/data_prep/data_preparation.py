import pandas as pd
import mlflow
import os

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("kubeChat")

def prepare_data(input_artifact_path, output_artifact_path):
    """
    Performs data preparation steps:
    - Loads data.
    - Handles missing values (simple imputation: median for numerical, mode for categorical).
    - Removes duplicate rows.
    - Logs processed data as an MLflow artifact.

    Args:
        input_artifact_path (str): Path to the input CSV file.
    """
    with mlflow.start_run(run_name="Data_Preparation_Phase"):
        mlflow.log_param("input_artifact_uri", input_artifact_path)

        print(f"Loading raw data from {input_artifact_path}...")
        local_path = mlflow.artifacts.download_artifacts(input_artifact_path)
        df = pd.read_parquet(local_path)
        print("Raw data loaded successfully.")

        initial_rows = df.shape[0]
        initial_cols = df.shape[1]
        mlflow.log_param("initial_rows", initial_rows)
        mlflow.log_param("initial_cols", initial_cols)

        # --- Data Cleaning and Preprocessing ---

        # Remove duplicate rows
        df = df.astype({col: "string" for col in df.select_dtypes(include=["object"]).columns})
        initial_duplicates = df.duplicated().sum()
        df.drop_duplicates(inplace=True)
        rows_after_dedup = df.shape[0]
        mlflow.log_param("rows_removed_by_deduplication", initial_rows - rows_after_dedup)
        print(f"Removed {initial_rows - rows_after_dedup} duplicate rows. New row count: {rows_after_dedup}")

        # Log final dataset shape
        mlflow.log_param("final_num_rows", df.shape[0])
        mlflow.log_param("final_num_columns", df.shape[1])

        # Save the processed data as Parquet
        output_dir = "processed_data/"
        os.makedirs(output_dir, exist_ok=True)
        dataset_path = os.path.join(output_dir, "dataset.parquet")
        df.to_parquet(dataset_path, index=False)
        # Log the processed data as an MLflow artifact
        mlflow.log_artifact(dataset_path, output_artifact_path)
        print("Processed data logged as MLflow artifact.")


        artifact_uri = mlflow.get_artifact_uri(f"{dataset_path}")
        print(f"MLflow Artifact URI: {artifact_uri}")
        with open(args.output_artifact_path, "w") as f:
                f.write(artifact_uri)
        print(f"Output added to kubeflow")
        return artifact_uri

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare data from a CSV file.")
    parser.add_argument("--input_artifact_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_artifact_path", type=str, required=True, help="Path to the input CSV file.")
    args = parser.parse_args()
    prepare_data(args.input_artifact_path, "processed_data")