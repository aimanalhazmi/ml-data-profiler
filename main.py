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
from multiprocessing import Process, Queue
from tqdm import tqdm
import random
import numpy as np
import config as cfg


warnings.filterwarnings("ignore", category=ConvergenceWarning)
LOG_TO = os.path.join(os.getcwd(), "outputs")
failed_datasets = []


def run_analysis(df, target_column, model_type, idx=None, url=None, timeout=3600):
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
        log_to_file(
            message=f"Invalid target column or failure in analysis pipeline - Dataset {idx}: {url}",
            log_to=LOG_TO,
        )
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
        run_quality_and_fairness(
            df=df.copy(),
            model_type=model_type,
            target_column=target_column,
            url=url,
            idx=idx,
            timeout=timeout,
        )
    )
    print(quality_output)
    print(fairness_output)

    if quality_results is None or fairness_results is None:
        return None

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


def run_quality_and_fairness(
    df, model_type, target_column, timeout=None, idx=1, url=None
):
    q1 = Queue()
    q2 = Queue()

    p1 = Process(
        target=run_with_output, args=(quality, q1, df.copy(), model_type, target_column)
    )
    p2 = Process(
        target=run_with_output,
        args=(fairness, q2, df.copy(), model_type, target_column),
    )

    p1.start()
    p2.start()

    if timeout is not None:
        start_time = time.time()
        p1.join(timeout)
        elapsed = time.time() - start_time
        remaining = max(0, timeout - elapsed)
        p2.join(remaining)
    else:
        p1.join()
        p2.join()

    if timeout is not None and p1.is_alive():
        print(
            f"Quality process exceeded {timeout // 60} minutes - Dataset {idx}: {url}"
        )
        log_to_file(message=f"TIMEOUT - QUALITY - Dataset {idx}: {url}", log_to=LOG_TO)
        p1.terminate()
        q1.put((None, "Quality process timed out."))

    if timeout is not None and p2.is_alive():
        print(
            f"Fairness process exceeded {timeout // 60} minutes - Dataset {idx}: {url}"
        )
        log_to_file(message=f"TIMEOUT - FAIRNESS - Dataset {idx}: {url}", log_to=LOG_TO)
        p2.terminate()
        q2.put((None, "Fairness process timed out."))

    try:
        result1 = q1.get(timeout=10)
    except Exception:
        msg = "Quality process failed to return results"
        log_to_file(message=f"{msg}- Dataset {idx}: {url}", log_to=LOG_TO)
        result1 = (None, f"{msg}.")

    try:
        result2 = q2.get(timeout=10)
    except Exception:
        msg = "Fairness process failed to return results"
        log_to_file(message=f"{msg}- Dataset {idx}: {url}", log_to=LOG_TO)
        result2 = (None, f"{msg}.")

    return result1, result2


def run_pipeline_auto(datasets_path: str):
    all_results = []

    if not datasets_path or not os.path.exists(datasets_path):
        print(f"Dataset file '{datasets_path}' not found.")
        return

    with open(datasets_path, "r") as f:
        dataset_configs = json.load(f)

    if not isinstance(dataset_configs, dict):
        print("Dataset file must be a dictionary...")
        return

    for idx, (url, cfg) in tqdm(
        enumerate(dataset_configs.items(), 1),
        total=len(dataset_configs),
        desc="Processing Datasets",
    ):
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
            timeout = cfg.get("timeout", 3600)
            print(
                f"\nModel selected: {selected_model} | Target column: {target_column}"
            )

            result = run_analysis(df, target_column, model_type, idx, url, timeout)

            if result:
                all_results.append((idx, url, result))
            else:
                failed_datasets.append((idx, url))

        except Exception as e:
            print(f"Error: {e}. Skipping dataset {idx}.")
            log_to_file(message=f"ERROR - Dataset {idx}: {url} | {e}", log_to=LOG_TO)
            failed_datasets.append((idx, url))

    if all_results:
        print("\n=== Generating Summary Report ===")
        for idx, url, result in all_results:
            report_path = f"outputs/final_report_dataset_{idx}.pdf"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            save_results_to_pdf(
                filepath=report_path,
                url=url,
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
        run_quality_and_fairness(
            df=df.copy(), model_type=model_type, target_column=target_column, url=url
        )
    )
    print(quality_output)
    print(fairness_output)

    if quality_results is None or fairness_results is None:
        print("\nNo results found.")
        return None

    print("\nAnalysis complete.")

    report_path = "outputs/final_report.pdf"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    save_results_to_pdf(
        filepath=report_path,
        url=url,
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
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
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
        print("Run Fairfluence automatically!")
        run_pipeline_auto(args.datasets)
