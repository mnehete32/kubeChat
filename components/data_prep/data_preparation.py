import pandas as pd
import mlflow
import os
from train_test_split import TrainTestSplit


class DataPreparer:
    def __init__(self, 
                input_artifact_path: str,
                train_artifact_path: str,
                test_artifact_path: str,
                experiment_name: str = "kubeChat"):
        self.input_artifact_path = input_artifact_path
        self.train_artifact_path = train_artifact_path
        self.test_artifact_path = test_artifact_path
        self.mlflow_experiment_name = os.getenv("EXPERIMENT_NAME",experiment_name)
        self.kf_run_id = os.getenv("KUBEFLOW_RUN_ID", "10000")
        self.df = None
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)

        self.__call__()
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
        return self.df

    def __call__(self):
        with mlflow.start_run(run_name="Data_Preparation_Phase") as mlf:
            mlflow.log_param("input_artifact_uri", self.input_artifact_path)
            mlflow.log_param("kubeflow_run_id", self.kf_run_id)

            self.load_data()
            self.clean_data()

            TrainTestSplit(
                df = self.df,
                train_artifact_path=self.train_artifact_path,
                test_artifact_path=self.test_artifact_path,
                mlflow_tracking_uri=self.mlflow_tracking_uri
            )
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare and clean dataset for ML pipeline.")
    parser.add_argument("--input_artifact_path", type=str, required=True, help="MLflow URI of the input artifact.")
    parser.add_argument("--train_artifact_path", type=str, required=True, help="Output file to store train artifact URI")
    parser.add_argument("--test_artifact_path", type=str, required=True, help="Output file to store test artifact URI")
    args = parser.parse_args()

    DataPreparer(
        input_artifact_path=args.input_artifact_path,
        train_artifact_path=args.train_artifact_path,
        test_artifact_path=args.test_artifact_path
    )