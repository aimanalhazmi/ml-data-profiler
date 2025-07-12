from src.preprocessing.preprocessing import PreprocessorFactory
from sklearn.model_selection import train_test_split
import config as cfg
from src.model.train import train_model
from src.quality.compare import compare_outlier_removals
from src.quality.clean import summarize_outliers
from src.fairness.no_influence import (
    compute_fairness_classical_metrics,
    compute_fairness_influence_metrics,
)
from src.fairness.with_influence import evaluate_patterns
from src.influence.logistic_influence import LinearSVMInfluence, LogisticInfluence
from src.utils.output import *
import time


def preprocess_data(df, method, ohe, target_column, streamlit_active):
    preprocessor_factory = PreprocessorFactory(
        data=df, method=method, target_column=target_column
    )
    print_info(
        streamlit_active=streamlit_active,
        msg=f"Data preprocessing for the {method} pipeline completed.",
    )
    preprocessor = preprocessor_factory.create()
    processed_data_dict = preprocessor.process_data(ohe=ohe)
    return processed_data_dict


def calculate_influence(originalX, X, y, X_index, model_type, target_column):
    # X is a dataframe, we should convert it to a numpy to suit the LinearSVMInfluence and LogisticInfluence
    X = X.values.astype(float)
    y = y.values.astype(float)

    # If we split dataframe, it is still dataframe, if numpy then  numpy
    X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(
        X,
        y,
        X_index,
        test_size=cfg.TEST_SIZE,
        stratify=y,
        random_state=cfg.SEED,
    )

    model = train_model(X_train, y_train, model_type=model_type)
    if model_type == "svm":
        svm_influence = LinearSVMInfluence(model, X_train, y_train)
        # ToDo: change to frac
        X_train_infl = svm_influence.average_influence(X_test[:5], y_test[:5])
    else:
        logistic_influence = LogisticInfluence(model, X_train, y_train)
        # ToDo: change to frac
        X_train_infl = logistic_influence.average_influence(X_test[:10], y_test[:10])

    X_train_raw = originalX.loc[train_index].copy().reset_index(drop=True)

    X_train_raw["influence"] = X_train_infl

    return X_train_raw


def quality(df, model_type, target_column):
    # Quality
    streamlit_active = is_streamlit_active()
    print_step_start(name="Quality")
    quality_start = time.time()

    df_quality, numeric_columns_quality, _, _, _, target_column_quality = (
        preprocess_data(
            df,
            method="data quality",
            ohe=True,
            target_column=target_column,
            streamlit_active=streamlit_active,
        )
    )

    y = df_quality[target_column_quality]
    X = df_quality.drop(columns=[target_column_quality])

    # train split, model(Class),
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.SEED
    )
    model = train_model(X_train.values, y_train.values, model_type)

    report1 = compare_outlier_removals(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        num_cols=numeric_columns_quality,
        model=model,
        alpha=cfg.ALPHA,
        sigma_multiplier=cfg.SIGMA_MULTIPLIER,
        model_type=model_type,
    )

    report1_display = display_f1_report(report1)

    outlier_summary = summarize_outliers(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        num_cols=numeric_columns_quality,
        model=model,
        alpha=cfg.ALPHA,
        sigma_multiplier=cfg.SIGMA_MULTIPLIER,
    )

    summary_display = display_outlier_summary(outlier_summary)

    quality_results = [
        ("F1 Score Report", report1_display),
        ("Outlier Summary", summary_display),
    ]
    print_step_end(name="Quality", start_time=quality_start, streamlit_active=False)
    return quality_results


# Fairness
def fairness(df, model_type, target_column):

    streamlit_active = is_streamlit_active()
    print_step_start(name="Fairness")

    fairness_start = time.time()

    X_index = df.index
    df_fairness, _, _, _, sensitive_columns_fairness, target_column_fairness = (
        preprocess_data(
            df,
            method="fairness",
            ohe=True,
            target_column=target_column,
            streamlit_active=streamlit_active,
        )
    )

    y = df_fairness[target_column_fairness]
    X = df_fairness.drop(columns=[target_column_fairness])

    #  with influence
    X_train_raw = calculate_influence(
        originalX=df.copy(),
        X=X.copy(),
        y=y.copy(),
        X_index=X_index,
        model_type=model_type,
        target_column=target_column_fairness,
    )
    top_patterns = evaluate_patterns(
        X_train_raw,
        min_support=cfg.MIN_SUPPORT,
        top_k=cfg.TOP_K,
        max_predicates=cfg.MAX_PREDICATES,
    )

    top_patterns_display = display_top_patterns(top_patterns=top_patterns)

    #  without influence

    no_influence_top_patterns = []
    influence_top_patterns = []
    X_train, X_test, y_train, y_test, s_train_df, s_test_df = train_test_split(
        X.copy(),
        y.copy(),
        df[sensitive_columns_fairness],
        test_size=cfg.TEST_SIZE,
        stratify=y,
        random_state=cfg.SEED,
    )

    model = train_model(X_train.values, y_train.values, model_type)

    no_influence = compute_fairness_classical_metrics(
        X_test,
        y_test,
        s_test_df,
        df[sensitive_columns_fairness],
        model,
        dpd_tol=cfg.DPD_TOL,
        eod_tol=cfg.EOD_TOL,
        ppv_tol=cfg.PPV_TOL,
    )
    no_influence_top_patterns.append(no_influence)

    for i in range(len(top_patterns)):
        influence_group_col = top_patterns.at[i, "pattern_col_1"]
        positive_group = top_patterns.at[i, "pattern_val_1"]
        print(
            f"\nPattern [{i+1}]: Influence Group Column: {influence_group_col}, Positive Group: {positive_group}"
        )

        influence = compute_fairness_influence_metrics(
            X_train_raw,
            class_col=influence_group_col,
            positive_group=positive_group,
            d_tol=cfg.D_TOL,
        )

        influence_top_patterns.append(influence)

    no_influence_summary = print_no_influence_top_patterns(
        no_influence_top_patterns[0], streamlit_active
    )
    influence_summary = print_influence_top_patterns(
        influence_top_patterns, streamlit_active
    )

    fairness_results = [
        ("Top Patterns", top_patterns_display),
        ("Fairness (No Influence)", no_influence_summary),
        ("Fairness (With Influence)", influence_summary),
    ]

    print_step_end(name="Fairness", start_time=fairness_start, streamlit_active=False)
    return fairness_results
