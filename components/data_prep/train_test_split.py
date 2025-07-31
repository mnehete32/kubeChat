import os
import argparse
import pandas as pd
import mlflow
from typing import Tuple
from sklearn.model_selection import train_test_split


class TrainTestSplit:
    def __init__(
        self,
        df: pd.DataFrame,
        train_artifact_path: str,
        test_artifact_path: str,
        target_col: str = "command_name",
        test_size: float = 0.2,
        random_state: int = 32,
        mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080"),
        experiment_name: str = os.getenv("EXPERIMENT_NAME"),

    ):
        self.df = df
        self.train_artifact_path = train_artifact_path
        self.test_artifact_path = test_artifact_path
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.experiment_name = experiment_name
        
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.__call__()

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataframe into train and test with fallback on stratification error"""
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )
            print("Stratified split successful.")
        except ValueError:
            print("Stratification failed. Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state
            )

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        return train_df, test_df

    def save_local(self, df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"Saved dataset to {path}")

    def log_to_mlflow(self, train_path: str, test_path: str) -> Tuple[str, str]:
        """Log datasets and parameters to MLflow, and return artifact URIs"""
        
        mlflow.log_params({
            "split_ratio": self.test_size,
            "random_state": self.random_state
        })


        # Define artifact relative paths
        train_name = os.path.basename(train_path)
        test_name = os.path.basename(test_path)
        base_output = "dataset"
        train_mlflow_artifact_path = os.path.join(base_output, train_name)
        test_mlflow_artifact_path = os.path.join(base_output, test_name)
        
        mlflow.log_artifact(train_path, base_output)
        mlflow.log_artifact(test_path, base_output)

        train_uri = mlflow.get_artifact_uri(train_mlflow_artifact_path)
        test_uri = mlflow.get_artifact_uri(test_mlflow_artifact_path)

        print(f"Train artifact URI: {train_uri}")
        print(f"Test artifact URI: {test_uri}")

        return train_uri, test_uri

    def write_output_paths(self, train_uri: str, test_uri: str):
        """Save the artifact URIs to given output files"""
        with open(self.train_artifact_path, "w") as f:
            f.write(train_uri)
        with open(self.test_artifact_path, "w") as f:
            f.write(test_uri)
        print("Artifact URIs written to output files.")

    def __call__(self):
        print("Starting data splitting pipeline...")
        train_df, test_df = self.split_data(self.df)

        train_path = "output/train.parquet"
        test_path = "output/test.parquet"

        self.save_local(train_df, train_path)
        self.save_local(test_df, test_path)

        train_uri, test_uri = self.log_to_mlflow(train_path, test_path)
        self.write_output_paths(train_uri, test_uri)
        print("Pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train-Test Split with MLflow Tracking")
    parser.add_argument("--input_artifact_path", type=str, required=False, help="MLflow artifact path to input dataset")
    parser.add_argument("--train_artifact_path", type=str, required=True, help="Output file to store train artifact URI")
    parser.add_argument("--test_artifact_path", type=str, required=True, help="Output file to store test artifact URI")
    args = parser.parse_args()
    
    df = pd.read_parquet(args.input_artifact_path)

    TrainTestSplit(
        df=df,
        train_artifact_path=args.train_artifact_path,
        test_artifact_path=args.test_artifact_path
    )