import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from typing import Tuple
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")
print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment("kubeChat")


def load_data_from_mlflow(artifact_path: str) -> pd.DataFrame:
    """Load preprocessed data from MLflow run"""
    try:

        local_path = mlflow.artifacts.download_artifacts(artifact_path)

        return pd.read_parquet(local_path)
    
    except Exception as e:
        print(f"Failed to load from MLflow: {str(e)}")
        raise

def get_input_data(artifact_path: str) -> pd.DataFrame:
    """Get input data trying MLflow first, then local fallback"""
    try:
        return load_data_from_mlflow(artifact_path)
    except Exception:
        print("Using local data fallback")
        raise

def perform_train_test_split(
    df: pd.DataFrame,
    target_col: str = 'command_name',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform train-test split with stratification fallback"""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        print("Stratified split successful")
    except ValueError:
        print("Stratification failed - performing standard split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
    
    return (
        pd.concat([X_train, y_train], axis=1),
        pd.concat([X_test, y_test], axis=1)
    )

def save_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_dir: str = '/data/',
    test_dir: str = '/data/',
    run_name: str = "Train_Test_Split",
    output_artifact_path: str = "dataset"
) -> None:
    """Save datasets to local files"""
    os.makedirs(os.path.dirname(train_dir), exist_ok=True)
    os.makedirs(os.path.dirname(test_dir), exist_ok=True)
    train_filename = "train.parquet"
    test_filename = "test.parquet"
    train_path = os.path.join(train_dir, train_filename)
    test_path = os.path.join(test_dir, test_filename)
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    print(f"Datasets saved to {train_dir} and {test_dir}")

    """Log datasets and parameters to MLflow"""
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({
            "split_ratio": 0.2,
            "random_state": 42
        })
        
        # Log tags
        mlflow.set_tags({
            "phase": "data_split",
            "parent_run": "Data_Preparation_Phase"
        })
        
        train_artifact_path = os.path.join(output_artifact_path, train_filename)
        test_artifact_path = os.path.join(output_artifact_path,test_filename)
        mlflow.log_artifact(train_path, train_artifact_path)
        mlflow.log_artifact(test_path, test_artifact_path)
        train_artifact_uri = mlflow.get_artifact_uri(train_artifact_path)
        test_artifact_uri = mlflow.get_artifact_uri(test_artifact_path)

        print(f"TRAIN MLflow Artifact URI: {train_artifact_uri}")
        print(f"TEST MLflow Artifact URI: {test_artifact_uri}")
        print("Datasets logged to MLflow")

        with open(args.train_artifact_path, "w") as f:
            f.write(train_artifact_uri)
        with open(args.test_artifact_path, "w") as f:
            f.write(test_artifact_uri)
        return train_artifact_uri, test_artifact_uri
        

if __name__ == "__main__":
    print("Starting data splitting process...")
    
    import argparse
    parser = argparse.ArgumentParser(description="Prepare data from a CSV file.")
    parser.add_argument("--input_artifact_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--train_artifact_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--test_artifact_path", type=str, required=True, help="Path to the input CSV file.")
    args = parser.parse_args()

    # Load data
    data = get_input_data(args.input_artifact_path)
    
    # Split data
    train_data, test_data = perform_train_test_split(data)
    
    # Save locally
    train_artifact_uri, test_artifact_uri = save_datasets(train_data, test_data)
    print("Process completed successfully")    
    
