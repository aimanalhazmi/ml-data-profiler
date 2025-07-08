import os
from src.ingestion.loader import load_dataset
from src.analysis import stats
from src.preprocessing.preprocessing import Preprocessor as DQP
from src.model.registry import MODEL_REGISTRY
from helper import quality, fairness
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
import time


def main():
    print("===    Fairfluence   ===")

    # Dataset Input
    url = input("Enter dataset URL (OpenML, Kaggle, HuggingFace): ").strip()
    if not url:
        print("URL is required.")
        return

    print("Loading dataset...")
    df = load_dataset(url)
    print("Dataset loaded successfully!\n")

    # Dataset Summary
    print("=== Dataset Summary ===")
    summary = stats.get_dataset_summary(df)
    print(summary)

    # Column Types
    dqp = DQP(df, target_column="")
    col_summary = stats.get_column_type_summary(dqp)
    print("\n=== Column Types ===")
    print(col_summary)

    # Target Column Selection
    numeric_columns, categorical_columns = dqp.receive_categorized_columns()
    valid_columns = numeric_columns + categorical_columns

    print("\nAvailable columns:", list(df.columns))
    target_column = input(
        "Enter target column (or press Enter to auto-select): "
    ).strip()

    if not target_column:
        for col in reversed(df.columns):
            if col in valid_columns:
                target_column = col
                break
        if not target_column:
            print("No valid target column found.")
            return
        print(f"Auto-selected target column: {target_column}")
    elif target_column not in valid_columns:
        print(f"Invalid target column: {target_column}")
        return

    # Model Selection
    supported_models = list(MODEL_REGISTRY.keys())
    print("\nAvailable models:", supported_models)
    selected_model = input(
        f"Choose model_type (or press Enter for default '{supported_models[0]}'): "
    ).strip()
    if selected_model not in supported_models:
        print(f"No model_type selected. Using default: {supported_models[0]}")
        selected_model = supported_models[0]

    model_type = MODEL_REGISTRY[selected_model]
    print(f"Model selected: {selected_model} | Target column: {target_column}")

    # Run Quality and Fairness
    print("\nRunning quality and fairness analysis...")
    start_time = time.time()
    quality(df=df.copy(), model_type=model_type, target_column=target_column)
    fairness(df=df.copy(), model_type=model_type, target_column=target_column)
    print("Analysis complete.")
    end_time = time.time()
    print(f"run time:{(end_time - start_time)/60} minutes")

    # Generate Report
    report_path = "outputs/test_report.pdf"
    if os.path.exists(report_path):
        print(f"Report generated: {report_path}")
    else:
        print("No report file found (placeholder).")


if __name__ == "__main__":
    main()
