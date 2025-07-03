from src.ingestion.loader import load_dataset
from src.preprocessing.quality import preprocess_quality
from src.preprocessing.fairness import preprocess_fairness
from src.model.train import train_model
from src.influence.compute import compute_influence
from src.quality import with_influence as quality_with
from src.quality import no_influence as quality_no
from src.quality.clean import clean_data_quality
from src.fairness import with_influence as fairness_with
from src.fairness import no_influence as fairness_no
from src.analysis.compare import compare_results
from src.analysis.stats import generate_stats
from src.analysis.visual import generate_report
from sklearn.model_selection import train_test_split
import pandas as pd


def quality_with_influence(df: pd.DataFrame, model_type: str):
    # Data Split
    parameters = {}
    X_train, X_test, y_train, y_test = train_test_split(**parameters)

    # Train model
    model_q = train_model(X_train, y_train, model_type)

    # Calculate influence
    influence_q = compute_influence(model_q, X_train, y_train)

    # Quality With influence
    quality_results = quality_with.check_quality()

    # ToDO: Report findings and save in report_quality_check

    report_quality_check = None
    # ToDO: Clean Dataset based on influence
    cleaned_q_with = clean_data_quality()

    # ToDo Train model after cleaning the dataset
    parameters = {}
    (
        X_train_cleaned_q_with,
        X_test_cleaned_q_with,
        y_train_cleaned_q_witho,
        y_test_cleaned_q_with,
    ) = train_test_split(**parameters)

    # model_q_with = train_model(X_train_cleaned_q_with, y_train_cleaned_q_witho, model_type)

    # ToDo: report results (metrics) as dataframe
    results_after_cleaning = None

    return report_quality_check, results_after_cleaning


def quality_no_influence(df: pd.DataFrame, model_type: str):
    quality_results = quality_no.check_quality()

    # ToDO: Report findings and save in report_quality_check
    report_quality_check = None
    # ToDO: Clean Dataset
    cleaned_q_no = clean_data_quality()

    # ToDo Train model after cleaning the dataset
    parameters = {}
    (
        X_train_cleaned_q_no,
        X_test_cleaned_q_no,
        y_train_cleaned_q_no,
        y_test_cleaned_q_no,
    ) = train_test_split(**parameters)
    # model_q_no = train_model(X_train_cleaned_q_no, y_train_cleaned_q_no, model_type)

    # ToDo: report results (metrics) as dataframe
    results_after_cleaning = None

    return report_quality_check, results_after_cleaning


def run_quality_pipeline(df: pd.DataFrame, model_type: str):
    # ========== Data Quality Pipeline ==========
    df_q = preprocess_quality(df.copy())
    # Data Quality with Influence
    report_quality_check_with_influence, results_after_cleaning_with_influence = (
        quality_with_influence(df, model_type)
    )
    generate_report(report_quality_check_with_influence)
    generate_report(results_after_cleaning_with_influence)

    # Data Quality without Influence
    report_quality_check_without_influence, results_after_cleaning_without_influence = (
        quality_no_influence(df, model_type)
    )

    generate_report(report_quality_check_without_influence)
    generate_report(results_after_cleaning_without_influence)

    report_q = compare_results(
        results_after_cleaning_with_influence, results_after_cleaning_without_influence
    )
    generate_report(report_q)


def fairness_no_influence(df: pd.DataFrame):
    fairness_no_influence_results = fairness_no.check_fairness(df)
    return fairness_no_influence_results


def fairness_with_influence(df: pd.DataFrame, model_type: str):
    # ToDo Train model
    # Data Split
    parameters = {}
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(**parameters)

    model_f = train_model(X_train_f, y_train_f, model_type)

    influence_f = compute_influence(model_f, X_train_f, y_train_f)

    fairness_with_influence_results = fairness_with.check_fairness(df, influence_f)

    return fairness_with_influence_results


def run_fairness_pipeline(df: pd.DataFrame, model_type: str):
    # ========== Fairness Pipeline ==========
    df_f = preprocess_fairness(df)

    fairness_no_influence_results = fairness_no_influence(df_f)
    fairness_with_influence_results = fairness_with_influence(df_f, model_type)
    report_f = compare_results(
        fairness_no_influence_results, fairness_with_influence_results
    )

    # ToDo: report results (metrics) as dataframe
    fairness_report = {}

    generate_report(fairness_report)


def run_pipeline(dataset_link: str, model_type: str):
    # Step 1: Ingest dataset
    df = load_dataset(dataset_link)

    # Step 2: Generate basic stats
    generate_stats(df, "", "")

    # Data Quality Pipeline
    run_quality_pipeline(df.copy(), model_type)

    # Fairness Pipeline
    run_fairness_pipeline(df.copy(), model_type)
