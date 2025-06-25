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


def run_pipeline(dataset_link: str, model_type: str):
    # Step 1: Ingest dataset
    df = load_dataset(dataset_link)

    # Step 2: Generate basic stats
    generate_stats(df)

    # ========== Data Quality Pipeline ==========
    df_q = preprocess_quality(df.copy())

    # Data Split
    parameters = {}
    X_train, X_test, y_train, y_test = train_test_split(**parameters)

    # Train model
    model_q = train_model(X_train,y_train, model_type)

    influence_q = compute_influence(model_q, X_train, y_train)

    quality_no_results = quality_no.check_quality()
    quality_with_results = quality_with.check_quality()
    report_q = compare_results(quality_no_results, quality_with_results)
    generate_report(report_q)

    cleaned_q_no = clean_data_quality()
    parameters = {}
    X_train_cleaned_q_no, X_test_cleaned_q_no, y_train_cleaned_q_no, y_test_cleaned_q_no = train_test_split(**parameters)
    #model_q_no = train_model(X_train_cleaned_q_no, y_train_cleaned_q_no, model_type)

    parameters = {}
    X_train_cleaned_q_with, X_test_cleaned_q_with, y_train_cleaned_q_witho, y_test_cleaned_q_with = train_test_split(**parameters)
    cleaned_q_with = clean_data_quality()
    #model_q_with = train_model(X_train_cleaned_q_with, y_train_cleaned_q_witho, model_type)

    #compare_results()
    #generate_report()

    # ========== Fairness Pipeline ==========
    df_f = preprocess_fairness(df.copy())

    # Data Split
    parameters = {}
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(**parameters)

    model_f = train_model(X_train_f, y_train_f, model_type)

    influence_f = compute_influence(model_f, X_train_f, y_train_f)

    fairness_no_influence_results = fairness_no.check_fairness(df_f)


    fairness_with_influence_results = fairness_with.check_fairness(df_f, influence_f)

    report_f = compare_results(fairness_no_influence_results, fairness_with_influence_results)
    generate_report(report_f)
