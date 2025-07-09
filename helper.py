from src.preprocessing.preprocessing import PreprocessorFactory
from sklearn.model_selection import train_test_split
from src.model.train import train_model
from src.quality.compare import compare_outlier_removals
from src.quality.clean import summarize_outliers
from src.fairness.no_influence import compute_fairness_metrics
from src.fairness.with_influence import evaluate_patterns
from src.influence.logistic_influence import LinearSVMInfluence, LogisticInfluence
from src.utils.output import *
import time


def preprocess_data(df, method, ohe, target_column, streamlit_active):
    log_info(
        streamlit_active=streamlit_active,
        msg=f"Preprocessing for {method} pipeline...",
    )
    preprocessor_factory = PreprocessorFactory(
        data=df, method=method, target_column=target_column
    )
    preprocessor = preprocessor_factory.create()
    processed_data_dict = preprocessor.process_data(ohe=ohe)
    return processed_data_dict


# @st.cache_data
def quality(df, model_type, target_column):
    test_size = 0.2
    random_state = 42
    alpha = 0.01
    sigma_multiplier = 1.0

    # Quality
    streamlit_active = is_streamlit_active()
    log_step_start(streamlit_active=streamlit_active, name="Quality")
    quality_start = time.time()

    df_q = preprocess_data(
        df,
        method="data quality",
        ohe=True,
        target_column=target_column,
        streamlit_active=streamlit_active,
    )

    df_quality = df_q[0]
    numeric_columns_quality = df_q[1]
    categorical_columns_quality = df_q[2]
    text_columns_transformed_quality = df_q[3]
    sensitive_columns_quality = df_q[4]
    target_column_quality = df_q[5]

    y = df_quality[target_column_quality]
    X = df_quality.drop(columns=[target_column_quality])

    # train split, model(Class),
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    model = train_model(X_train.values, y_train.values, model_type)

    report1 = compare_outlier_removals(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        num_cols=numeric_columns_quality,
        model=model,
        alpha=alpha,
        sigma_multiplier=sigma_multiplier,
        model_type=model_type,
    )

    report1_display = display_f1_report(report1, streamlit_active=streamlit_active)

    outlier_summary = summarize_outliers(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        num_cols=numeric_columns_quality,
        model=model,
        alpha=alpha,
        sigma_multiplier=sigma_multiplier,
    )

    summary_display = display_outlier_summary(
        outlier_summary, streamlit_active=streamlit_active
    )

    quality_results = [
        ("F1 Score Report", report1_display),
        ("Outlier Summary", summary_display),
    ]
    log_step_end(
        streamlit_active=streamlit_active, name="Quality", start_time=quality_start
    )
    return quality_results


def prepare(orginal_X, X, y, X_index, model_type, target_column):
    random_state = 42
    test_size = 0.2

    # X is a dataframe, we should convert it to a numpy to suit the LinearSVMInfluence and LogisticInfluence
    X = X.values.astype(float)
    y = y.values.astype(float)

    # If we split dataframe, it is still dataframe, if numpy then  numpy
    X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(
        X,
        y,
        X_index,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
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

    X_train_raw = orginal_X.loc[train_index].copy().reset_index(drop=True)

    X_train_raw["influence"] = X_train_infl

    return X_train_raw


# Fairness
# @st.cache_data
def fairness(df, model_type, target_column):
    test_size = 0.2
    random_state = 42
    frac = 0.05
    dpd_tol = 0.1
    eod_tol = 0.1
    ppv_tol = 0.1
    influence_tol = 0.05

    streamlit_active = is_streamlit_active()
    log_step_start(streamlit_active=streamlit_active, name="Fairness")
    fairness_start = time.time()

    X_index = df.index
    df_f = preprocess_data(
        df,
        method="fairness",
        ohe=True,
        target_column=target_column,
        streamlit_active=streamlit_active,
    )
    df_fairness = df_f[0]
    numeric_columns_fairness = df_f[1]
    categorical_columns_fairness = df_f[2]
    text_columns_transformed_fairness = df_f[3]
    sensitive_columns_fairness = df_f[4]
    target_column_fairness = df_f[5]

    y = df_fairness[target_column_fairness]
    X = df_fairness.drop(
        columns=[target_column_fairness]
    )  # target_column or target_column_fairness

    # ToDO: with influence
    X_train_raw = prepare(
        orginal_X=df.copy(),
        X=X.copy(),
        y=y.copy(),
        X_index=X_index,
        model_type=model_type,
        target_column=target_column_fairness,
    )
    top_patterns = evaluate_patterns(
        X_train_raw, min_support=0.05, top_k=5, max_predicates=1
    )

    top_patterns_column_map = {
        "pattern_col_1": "Pattern Column 1",
        "pattern_val_1": "Pattern Value 1",
        "pattern_col_2": "Pattern Column 2",
        "pattern_val_2": "Pattern Value 2",
        "support": "Support",
        "responsibility": "Responsibility",
        "interestingness": "Interestingness",
    }

    top_patterns_display = top_patterns.rename(columns=top_patterns_column_map)

    if streamlit_active:
        if streamlit_active:
            st.markdown("Top Patterns")
            st.dataframe(
                top_patterns_display.style.format(
                    {
                        "Support": "{:.4f}",
                        "Responsibility": "{:.4f}",
                        "Interestingness": "{:.4f}",
                    }
                )
            )

    else:
        print("\n=== Top Patterns ===")
        print(
            tabulate(
                top_patterns_display.round(
                    {"Support": 4, "Responsibility": 4, "Interestingness": 4}
                ),
                headers="keys",
                tablefmt="grid",
                showindex=False,
            )
        )
        # print(print_pattern_table(top_patterns))

    # ToDO: without influence

    no_influence_top_patterns = []
    influence_top_patterns = []
    X_train, X_test, y_train, y_test, s_train_df, s_test_df = train_test_split(
        X.copy(),
        y.copy(),
        df[sensitive_columns_fairness],
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    model = train_model(X_train.values, y_train.values, model_type)

    for i in range(len(top_patterns)):
        influence_group_col = top_patterns.at[i, "pattern_col_1"]
        positive_group = top_patterns.at[i, "pattern_val_1"]
        print(
            f"\nPattern [{i+1}]: Influence Group Column: {influence_group_col}, Positive Group: {positive_group}"
        )
        _, _, _, _, group_train_df, _ = train_test_split(
            X.copy(),
            y.copy(),
            df[influence_group_col],
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
        no_influence, influence = compute_fairness_metrics(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            s_test_df=s_test_df,
            sens_cols=sensitive_columns_fairness,
            model=model,
            group_train_df=group_train_df,
            positive_group=positive_group,
            random_state=random_state,
            frac=frac,
            dpd_tol=dpd_tol,
            eod_tol=eod_tol,
            ppv_tol=ppv_tol,
            influence_tol=influence_tol,
        )

        no_influence_top_patterns.append(no_influence)
        influence_top_patterns.append(influence)

        # Flatten and summarize no_influence results
        print("\n=== Fairness (No Influence) ===")
        print_no_influence_top_patterns(no_influence)
        print("\n=== Fairness (With Influence) ===")
        print_one_influence_top_patterns(influence)

    # print("\n=== Fairness (No Influence) ===")
    # print_no_influence_top_patterns(no_influence_top_patterns[0])
    # print_influence_top_patterns(influence_top_patterns)

    fairness_results = []

    log_step_end(
        streamlit_active=streamlit_active, name="Fairness", start_time=fairness_start
    )
    return fairness_results
