import sys
import warnings
import argparse
import json
from pipeline import quality, fairness
from sklearn.exceptions import ConvergenceWarning
from src.ingestion.loader import load_dataset
from src.analysis import stats
from src.preprocessing.preprocessing import Preprocessor
from src.model.registry import MODEL_REGISTRY
from src.utils.output import *
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


warnings.filterwarnings("ignore", category=ConvergenceWarning)
LOG_TO = os.path.join(os.getcwd(), "outputs")


def run_analysis(df, target_column, model_type):
    overview_summary = stats.dataset_summary(df)
    print(tabulate(overview_summary, headers="keys", tablefmt="grid", showindex=False))
    dqp = Preprocessor(df, target_column="")
    column_types = dqp.receive_number_of_columns()
    col_summary = print_column_type_summary(column_types)
    alerts = stats.get_alerts(df)
    display_alerts(alerts)

    numeric_columns, categorical_columns = dqp.receive_categorized_columns()
    valid_columns = numeric_columns + categorical_columns

    if target_column not in valid_columns:
        print(f"Invalid target column: {target_column}")
        return None

    stats.plot_data_distribution_by_column(
        df,
        target_column,
        streamlit_mode=False,
        is_numeric=target_column in numeric_columns,
        save=True,
        save_path=os.path.join(os.getcwd(), "outputs"),
    )

    (quality_results, quality_output), (fairness_results, fairness_output) = (
        run_quality_and_fairness(df.copy(), model_type, target_column)
    )
    print(quality_output)
    print(fairness_output)

    # quality_results = quality(df.copy(), model_type, target_column)
    # fairness_results = fairness(df.copy(), model_type, target_column)

    return {
        "overview_summary": overview_summary,
        "column_types": col_summary,
        "alerts": alerts,
        "quality_results": quality_results,
        "fairness_results": fairness_results,
        "target_column": target_column,
    }


def check_target_col(
    df: pd.DataFrame, target_column: str, valid_columns: list
) -> str | None:
    if target_column in valid_columns:
        return target_column
    if target_column:
        print(f"Target column '{target_column}' is not valid for this dataset.")
        return None
    for col in reversed(df.columns):
        if col in valid_columns:
            print(f"Auto-selected target column: {col}")
            return col
    print("No valid target column found.")
    return None


def run_quality_and_fairness(df, model_type, target_column):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_quality = executor.submit(
            run_and_capture_output, quality, df.copy(), model_type, target_column
        )
        future_fairness = executor.submit(
            run_and_capture_output, fairness, df.copy(), model_type, target_column
        )
        return future_quality.result(), future_fairness.result()


def run_pipeline_auto(datasets_path: str):
    all_results = []
    failed_datasets = []

    if not datasets_path or not os.path.exists(datasets_path):
        print(f"Dataset file '{datasets_path}' not found.")
        return

    with open(datasets_path, "r") as f:
        dataset_configs = json.load(f)

    if not isinstance(dataset_configs, dict):
        print("Dataset file must be a dictionary...")
        return

    for idx, (url, cfg) in enumerate(dataset_configs.items(), 1):
        target_column = cfg.get("target_column", "").strip()
        file_number = cfg.get("no_dataset", 1)

        print(f"\n=== Dataset {idx} ===")
        if not url or not target_column:
            print("Missing required fields. Skipping.")
            log_to_file(
                message=f"Missing required fields - Dataset {idx}: {url}", log_to=LOG_TO
            )

            failed_datasets.append((idx, url))
            continue

        supported_models = list(MODEL_REGISTRY.keys())
        selected_model = cfg.get("model_type", supported_models[0]).strip()
        if selected_model not in supported_models:
            print(f"Invalid model '{selected_model}'. Skipping.")
            log_to_file(message=f"Invalid model  - Dataset {idx}: {url}", log_to=LOG_TO)

            failed_datasets.append((idx, url))
            continue

        try:
            df = load_dataset(url, file_number)
            model_type = MODEL_REGISTRY[selected_model]
            print(f"Model selected: {selected_model} | Target column: {target_column}")

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_analysis, df, target_column, model_type)
                result = future.result(timeout=3600)

            if result:
                all_results.append((idx, url, result))
            else:
                log_to_file(
                    message=f"Invalid target column or failure in analysis pipeline - Dataset {idx}: {url}",
                    log_to=LOG_TO,
                )
                failed_datasets.append((idx, url))

        except FuturesTimeoutError:
            print(f"Timeout: Dataset {idx} exceeded 60 minutes.")
            log_to_file(message="TIMEOUT - Dataset {idx}: {url}", log_to=LOG_TO)
            failed_datasets.append((idx, url))
        except Exception as e:
            print(f"Error: {e}. Skipping dataset {idx}.")
            log_to_file(message="ERROR - Dataset {idx}: {url} | {e}", log_to=LOG_TO)
            failed_datasets.append((idx, url))

    if all_results:
        print("\n=== Generating Summary Report ===")
        for idx, url, result in all_results:
            report_path = f"outputs/final_report_dataset_{idx}.pdf"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            save_results_to_pdf(
                filepath=report_path,
                overview_summary=result["overview_summary"],
                column_types=result["column_types"],
                alerts=result["alerts"],
                quality_results=result["quality_results"],
                fairness_results=result["fairness_results"],
            )
            print(f"Report saved to: {report_path}")

    if failed_datasets:
        print("\n=== Failed Datasets ===")
        for idx, url in failed_datasets:
            print(f"Dataset {idx} failed: {url}")
    else:
        print("\nAll datasets processed successfully!")


def manual():
    print("===    Fairfluence   ===")

    url = input("Enter dataset URL (OpenML, Kaggle, HuggingFace): ").strip()
    if not url:
        print("URL is required.")
        return

    file_number_input = input(
        "Enter file number (or press Enter to use default 1): "
    ).strip()
    file_number = int(file_number_input) if file_number_input.isdigit() else 1

    print("Loading dataset...")
    df = load_dataset(url, file_number=file_number)
    print("Dataset loaded successfully!\n")

    print("=== Dataset Summary ===")
    overview_summary = stats.dataset_summary(df)
    print(tabulate(overview_summary, headers="keys", tablefmt="grid", showindex=False))

    print("\n=== Column Type Summary ===")
    dqp = Preprocessor(df, target_column="")
    column_types = dqp.receive_number_of_columns()
    col_summary = print_column_type_summary(column_types)

    alerts = stats.get_alerts(df)
    display_alerts(alerts)

    numeric_columns, categorical_columns = dqp.receive_categorized_columns()
    valid_columns = numeric_columns + categorical_columns

    print("\nAvailable columns:", list(df.columns))
    target_column = input(
        "Enter target column (or press Enter to auto-select): "
    ).strip()
    target_column = check_target_col(df, target_column, valid_columns)
    if not target_column:
        return

    stats.plot_data_distribution_by_column(
        df,
        target_column,
        streamlit_mode=False,
        is_numeric=target_column in numeric_columns,
        save=True,
        save_path=os.path.join(os.getcwd(), "outputs"),
    )

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

    start_time = time.time()
    print("\nRunning analysis...")
    (quality_results, quality_output), (fairness_results, fairness_output) = (
        run_quality_and_fairness(df.copy(), model_type, target_column)
    )
    print(quality_output)
    print(fairness_output)

    print("\nAnalysis complete.")

    report_path = "outputs/final_report.pdf"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    save_results_to_pdf(
        filepath=report_path,
        overview_summary=overview_summary,
        column_types=col_summary,
        alerts=alerts,
        quality_results=quality_results,
        fairness_results=fairness_results,
    )
    print(f"Report generated: {report_path}")

    total_time = (time.time()) - start_time
    print("\n=== Program Complete ===")
    print(f"Total run time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["manual", "auto"], default="manual", help="Run mode"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        help="Path to JSON file containing list of datasets for auto mode",
    )
    args = parser.parse_args()

    if args.mode == "auto" and not args.datasets:
        print("Error: --datasets is required when mode is 'auto'.", file=sys.stderr)
        sys.exit(1)

    if args.mode == "manual":
        manual()
    else:
        print("Run mode automatically!")
        run_pipeline_auto(args.datasets)
