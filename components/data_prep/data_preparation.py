import pandas as pd
import mlflow
import os


class DataPreparer:
    def __init__(self, input_artifact_path: str, output_dataset_uri_path: str, experiment_name: str = "kubeChat"):
        self.input_artifact_path = input_artifact_path
        self.output_dataset_uri_path = output_dataset_uri_path
        self.mlflow_experiment_name = os.getenv("EXPERIMENT_NAME",experiment_name)
        self.kf_run_id = os.getenv("KUBEFLOW_RUN_ID", "10000")
        self.df = None

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080"))
        mlflow.set_experiment(self.mlflow_experiment_name)

    def load_data(self):
        print(f"Loading raw data from {self.input_artifact_path}...")
        local_path = mlflow.artifacts.download_artifacts(self.input_artifact_path)
        self.df = pd.read_parquet(local_path)
        print("Raw data loaded successfully.")

    def clean_data(self):
        initial_rows, initial_cols = self.df.shape
        mlflow.log_param("initial_rows", initial_rows)
        mlflow.log_param("initial_cols", initial_cols)

        # Filter out classes with only one sample
        value_counts = self.df["command_name"].value_counts()
        self.df = self.df[self.df["command_name"].isin(value_counts[value_counts > 1].index)]

        # Convert object columns to string and remove duplicates
        self.df = self.df.astype({col: "string" for col in self.df.select_dtypes(include=["object"]).columns})
        self.df.drop_duplicates(inplace=True)
        rows_after_dedup = self.df.shape[0]
        mlflow.log_param("rows_removed_by_deduplication", initial_rows - rows_after_dedup)

        mlflow.log_param("final_num_rows", self.df.shape[0])
        mlflow.log_param("final_num_columns", self.df.shape[1])
        print(f"Removed {initial_rows - rows_after_dedup} duplicate rows. New row count: {rows_after_dedup}")

    def save_and_log_data(self):
        output_dir = "prepared_dataset/"
        os.makedirs(output_dir, exist_ok=True)
        dataset_file_name = "dataset.parquet"
        dataset_path = os.path.join(output_dir, dataset_file_name)
        self.df.to_parquet(dataset_path, index=False)

        # Log the processed dataset as an MLflow artifact
        mlflow.log_artifact(dataset_path, dataset_path)
        print("Processed data logged as MLflow artifact.")

        # Get the artifact URI
        artifact_uri = mlflow.get_artifact_uri(f"{dataset_path}")
        print(f"MLflow Artifact URI: {artifact_uri}")

        # Write the artifact URI to output path for Kubeflow downstream consumption
        with open(self.output_dataset_uri_path, "w") as f:
            f.write(artifact_uri)
        print(f"Artifact URI written to {self.output_dataset_uri_path}")

        return artifact_uri

    def run(self):
        with mlflow.start_run(run_name="Data_Preparation_Phase"):
            mlflow.log_param("input_artifact_uri", self.input_artifact_path)
            mlflow.log_param("kubeflow_run_id", self.kf_run_id)

            self.load_data()
            self.clean_data()
            return self.save_and_log_data()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare and clean dataset for ML pipeline.")
    parser.add_argument("--input_artifact_path", type=str, required=True, help="MLflow URI of the input artifact.")
    parser.add_argument("--output_dataset_uri_path", type=str, required=True, help="Local path to store output URI.")
    args = parser.parse_args()

    preparer = DataPreparer(
        input_artifact_path=args.input_artifact_path,
        output_dataset_uri_path=args.output_dataset_uri_path
    )
    preparer.run()