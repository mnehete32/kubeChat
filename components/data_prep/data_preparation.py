import pandas as pd
import mlflow
import os

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("kubeChat")

def prepare_data(input_csv_path, output_parquet_path):
    """
    Performs data preparation steps:
    - Loads data.
    - Handles missing values (simple imputation: median for numerical, mode for categorical).
    - Removes duplicate rows.
    - Logs processed data as an MLflow artifact.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_parquet_path (str): Path to save the processed Parquet file.
    """
    with mlflow.start_run(run_name="Data_Preparation_Phase"):
        mlflow.log_param("input_file", input_csv_path)
        mlflow.log_param("output_file", output_parquet_path)

        print(f"Loading raw data from {input_csv_path}...")
        df = pd.read_parquet(input_csv_path)
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
        output_dir = os.path.dirname(output_parquet_path)
        os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(output_parquet_path, index=False)
        print(f"Processed data saved to {output_parquet_path}")

        # Log the processed data as an MLflow artifact
        mlflow.log_artifact(output_parquet_path, "processed_data")
        print("Processed data logged as MLflow artifact.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare data from a CSV file.")
    parser.add_argument("--input_csv_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_parquet_path", type=str, required=True, help="Path to save the processed Parquet file.")
    args = parser.parse_args()
    prepare_data(args.input_csv_path, args.output_parquet_path)
