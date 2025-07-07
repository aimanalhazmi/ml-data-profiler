from src.preprocessing.preprocessing import PreprocessorFactory
from streamlit import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from src.model.train import train_model
from src.model.registry import MODEL_REGISTRY
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from src.quality.with_influence import compute_influence
from src.quality.compare import compare_outlier_removals
from src.quality.clean import summarize_outliers
from src.fairness.no_influence import compute_fairness_metrics
from src.fairness.with_influence import evaluate_patterns, print_pattern_table
from src.influence.logistic_influence import LinearSVMInfluence, LogisticInfluence
import random


def preprocess_data(df, method, ohe, target_column):
    preprocessor_factory = PreprocessorFactory(
        data=df, method=method, target_column=target_column
    )
    preprocessor = preprocessor_factory.create()
    processed_data_dict = preprocessor.process_data(ohe=ohe)
    return processed_data_dict


def quality_with_influence(df: pd.DataFrame, target_column, model_type: str):
    random_state = 42
    test_size = 0.2
    frac = 0.01
    st.info(model_type)
    st.info(target_column)

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(df[target_column]), index=df.index)
    st.dataframe(y)

    X = df.drop(columns=[target_column])
    st.dataframe(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if frac < 1.0:
        X_te = X_test.sample(frac=frac, random_state=42)
        y_te = y_test.loc[X_te.index]
    else:
        X_te, y_te = X_test, y_test

    # Train model
    model = train_model(X_train, y_train, model_type)

    # Calculate influence
    influence_q = compute_influence(model, df, X_train, y_train, X_te, y_te)
    st.dataframe(influence_q)

    # Quality With influence
    # quality_results = quality_with.check_quality()

    # ToDO: Report findings and save in report_quality_check

    report_quality_check = None
    # ToDO: Clean Dataset based on influence
    # cleaned_q_with = clean_data_quality()

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


def fairness_with_influence(df: pd.DataFrame, model_type: str):

    pass


# @st.cache_data
def quality(df, model_type, target_column):
    # Quality
    df_q = preprocess_data(
        df, method="data quality", ohe=False, target_column=target_column
    )
    df_quality = df_q[0]
    numeric_columns_quality = df_q[1]
    categorical_columns_quality = df_q[2]
    text_columns_transformed_quality = df_q[3]
    sensitive_columns_quality = df_q[4]
    target_column_quality = df_q[5]

    positive_class = random.choice(df[target_column].unique())
    # train split, model(Class),
    report1 = compare_outlier_removals(
        df=df_quality,
        num_cols=numeric_columns_quality,
        target_col=target_column,
        positive_class=positive_class,
        model=model_type,
    )
    st.dataframe(report1)
    print(report1)
    # train split, model(Class),
    outlier_summary = summarize_outliers(
        df=df_quality,
        num_cols=numeric_columns_quality,
        target_col=target_column,
        positive_class=positive_class,
        alpha=0.01,
        sigma_multiplier=3.0,
    )
    st.dataframe(outlier_summary)
    print(outlier_summary)


def prepare(df, categorical_columns, model_type, target_column):
    random_state = 42
    test_size = 0.2

    st.dataframe(df[target_column])
    le = LabelEncoder()
    y = le.fit_transform(df[target_column])
    st.dataframe(y)

    X = df.drop(columns=[target_column])
    st.info(f"data before")
    st.dataframe(X)
    df_encoded = pd.get_dummies(X, columns=categorical_columns)
    scaler = StandardScaler()
    X_transformeed = scaler.fit_transform(df_encoded)
    st.info(f"data after")
    st.dataframe(X_transformeed)

    X_index = df.index
    X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(
        X_transformeed,
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

    X_train_raw = df.loc[train_index].copy().reset_index(drop=True)

    print(X_train_raw)

    X_train_raw["influence"] = X_train_infl

    return X_train_raw


# Fairness
# @st.cache_data
def fairness(df, model_type, target_column):
    df_f = preprocess_data(
        df, method="fairness", ohe=False, target_column=target_column
    )
    df_fairness = df_f[0]
    numeric_columns_fairness = df_f[1]
    categorical_columns_fairness = df_f[2]
    text_columns_transformed_fairness = df_f[3]
    sensitive_columns_fairness = df_f[4]
    target_column_fairness = df_f[5]
    positive_class = random.choice(df[target_column].unique())

    # ToDO: with influence
    X_train_raw = prepare(
        df=df_fairness.copy(),
        categorical_columns=categorical_columns_fairness,
        model_type=model_type,
        target_column=target_column_fairness,
    )
    top_patterns = evaluate_patterns(
        X_train_raw, min_support=0.05, top_k=5, max_predicates=1
    )
    st.info(f"top_patterns")
    st.dataframe(top_patterns)
    print(print_pattern_table(top_patterns))

    # ToDo: influence_group_col positive_group

    # ToDO: without influence

    no_influence_top_patterns = []
    influence_top_patterns = []
    for i in range(len(top_patterns)):
        influence_group_col = top_patterns.at[i, "pattern_col_1"]
        positive_group = top_patterns.at[i, "pattern_val_1"]

        no_influence, influence = compute_fairness_metrics(
            df=df_fairness.copy(),
            target_col=target_column_fairness,
            sens_cols=sensitive_columns_fairness,
            model=model_type,
            positive_class=positive_class,
            influence_group_col=influence_group_col,
            positive_group=positive_group,
            test_size=0.2,
            random_state=912,
            frac=0.05,
            dpd_tol=0.1,
            eod_tol=0.1,
            ppv_tol=0.1,
            influence_tol=0.05,
        )
        no_influence_top_patterns.append(no_influence)
        influence_top_patterns.append(influence)

    st.markdown("no_influence")
    st.dataframe(no_influence_top_patterns)
    st.markdown("influence")
    st.dataframe(influence_top_patterns)
    print("no_influence")
    print(no_influence_top_patterns)
    print("influence")
    print(influence_top_patterns)
