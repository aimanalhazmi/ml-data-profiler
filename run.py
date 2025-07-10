import os
from src.ingestion.loader import load_dataset
from src.analysis import stats
from src.preprocessing.preprocessing import Preprocessor as DQP
from src.model.registry import MODEL_REGISTRY
from pipeline import quality, fairness
import warnings
from sklearn.exceptions import ConvergenceWarning
from tabulate import tabulate
from src.utils.output import display_alerts

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
    overview_summary = stats.dataset_summary(df)
    print(tabulate(overview_summary, headers="keys", tablefmt="grid", showindex=False))

    # Column Types
    print("\n=== Column Type Summary ===")
    dqp = DQP(df, target_column="")
    col_summary = stats.print_column_type_summary_terminal(dqp)

    # Display alerts
    alerts = stats.get_alerts(df)
    display_alerts(alerts)

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

    is_numeric = target_column in numeric_columns
    if is_numeric:
        stats.plot_data_distribution_by_column(
            df,
            target_column,
            streamlit_mode=False,
            is_numeric=is_numeric,
            save=True,
            save_path=os.path.join(os.getcwd(), "outputs"),
        )
    else:
        stats.plot_data_distribution_by_column(
            df,
            target_column,
            streamlit_mode=False,
            is_numeric=is_numeric,
            save=True,
            save_path=os.path.join(os.getcwd(), "outputs"),
        )

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
    start_time = time.time()
    print("\nRunning analysis...")
    quality_results = quality(
        df=df.copy(), model_type=model_type, target_column=target_column
    )
    fairness_results = fairness(
        df=df.copy(), model_type=model_type, target_column=target_column
    )
    print("\nAnalysis complete.")

    # # Generate Report
    # report_path = "outputs/test_report.pdf"
    # if os.path.exists(report_path):
    #     print(f"Report generated: {report_path}")
    # else:
    #     print("No report file found (placeholder).")

    total_time = (time.time()) - start_time
    print("\n=== Program Complete ===")
    print(f"Total run time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")


if __name__ == "__main__":
    main()
