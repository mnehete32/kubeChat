import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import os
import io


# Set MLflow tracking URI from environment variable or default to local
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")
print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment("kubeChat")


def perform_eda(input_csv_path, output_dir):
    """
    Performs Exploratory Data Analysis (EDA) on the input CSV.
    Logs findings and visualizations to MLflow.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_dir (str): Directory to save EDA outputs (plots, summaries).
    """
    os.makedirs(output_dir, exist_ok=True)

    with mlflow.start_run(run_name="EDA_Phase"):
        mlflow.log_param("input_file", input_csv_path)
        print(f"Loading data from {input_csv_path}...")
        df = pd.read_parquet(input_csv_path)
        print("Data loaded successfully.")

        # Log basic dataset info
        buf = io.StringIO()
        df.info(buf=buf)
        info_str = buf.getvalue()
        mlflow.log_text(info_str, "data_info.txt")

        # Log missing values
        missing_summary = df.isnull().sum()
        mlflow.log_text(missing_summary.to_string(), "missing_values_summary.txt")

        # Log dataset shape
        mlflow.log_param("num_rows", df.shape[0])
        mlflow.log_param("num_columns", df.shape[1])

        print("Generating visualizations...")

        # Value counts for categorical columns
        categorical_cols = ["command_name"]
        for col in categorical_cols:
            value_counts = df[col].value_counts().head(10)
            mlflow.log_text(value_counts.to_string(), f"value_counts_{col}.txt")

            # Bar plot for top value counts
            plt.figure(figsize=(10, 6))
            sns.barplot(x=value_counts.values, y=value_counts.index)
            plt.title(f'Top 10 Value Counts of {col}')
            plt.xlabel("Count")
            plt.ylabel("Category")
            bar_path = os.path.join(output_dir, f'{col}_value_counts.png')
            plt.savefig(bar_path)
            mlflow.log_artifact(bar_path, "eda_plots")
            plt.close()

        print("EDA completed and results logged to MLflow.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Perform EDA on a CSV or Parquet file.")
    parser.add_argument("--input_csv_path", type=str, help="Path to the input file.")
    parser.add_argument("--output_dir", type=str, default="eda_output", help="Directory to save EDA outputs.")
    args = parser.parse_args()
    perform_eda(args.input_csv_path, args.output_dir)
