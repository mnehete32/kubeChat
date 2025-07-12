import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from typing import Tuple
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")
print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment("kubeChat")


def load_data_from_mlflow() -> pd.DataFrame:
    """Load preprocessed data from MLflow run"""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("kubeChat")
        
        if not experiment:
            raise ValueError("Experiment 'kubeChat' not found")
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName = 'Data_Preparation_Phase'",
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        
        if not runs:
            raise ValueError("No matching runs found")
        
        run = runs[0]
        artifact_path = f"{run.info.artifact_uri}/processed_data/dataset.parquet"
        print(f"Loading data from MLflow run: {run.info.run_id}")
        local_path = mlflow.artifacts.download_artifacts(artifact_path)

        return pd.read_parquet(local_path)
    
    except Exception as e:
        print(f"Failed to load from MLflow: {str(e)}")
        raise

def load_local_data(file_path: str = 'dataset.parquet') -> pd.DataFrame:
    """Load data from local file as fallback"""
    return pd.read_parquet(file_path)

def get_input_data() -> pd.DataFrame:
    """Get input data trying MLflow first, then local fallback"""
    try:
        return load_data_from_mlflow()
    except Exception:
        print("Using local data fallback")
        return load_local_data()

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
    train_path: str = './data/k8s_kubectl_train.parquet',
    test_path: str = './data/k8s_kubectl_test.parquet'
) -> None:
    """Save datasets to local files"""
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    print(f"Datasets saved to {train_path} and {test_path}")

def log_to_mlflow(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_name: str = "Train_Test_Split"
) -> None:
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
        
        # Save and log artifacts
        temp_train = "temp_train.parquet"
        temp_test = "temp_test.parquet"
        train_df.to_parquet(temp_train, index=False)
        test_df.to_parquet(temp_test, index=False)
        
        mlflow.log_artifact(temp_train)
        mlflow.log_artifact(temp_test)
        print("Datasets logged to MLflow")

def main():
    """Main workflow function"""
    print("Starting data splitting process...")
    
    # Load data
    data = get_input_data()
    
    # Split data
    train_data, test_data = perform_train_test_split(data)
    
    # Save locally
    save_datasets(train_data, test_data)
    
    # Log to MLflow
    log_to_mlflow(train_data, test_data)
    
    print("Process completed successfully")

if __name__ == "__main__":
    main()