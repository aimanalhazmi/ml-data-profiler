import streamlit as st
import pandas as pd
import os
import random
import numpy as np
import config as cfg
from src.ingestion.loader import load_dataset
from src.profiling import stats
from src.preprocessing.preprocessing import Preprocessor as DQP
from src.model.registry import MODEL_REGISTRY
from src.utils.output import *
from pipeline import quality, fairness
import time


def clear_outputs():
    keys_to_clear = [
        "df",
        "summary",
        "column_types",
        "alerts",
        "target_column",
        "model",
        "sample_frac",
        "reduced_df",
        "quality_results",
        "fairness_results",
        "report_path",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


@st.cache_data
def run_fairness_analysis(data, model_type, target_col):
    return fairness(data, model_type, target_col)


def display_result_section(title, results):
    with st.expander(title, expanded=True):
        for section_title, df_section in results:
            st.subheader(section_title)
            st.dataframe(df_section)


random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
st.set_page_config(page_title="Fairfluence App", layout="wide")
st.title("Fairfluence")
# Dataset Input
input_method = st.radio("Choose dataset input method", ("URL", "Upload CSV"))

if input_method == "URL":
    url = st.text_input("Enter dataset URL (OpenML, Kaggle, HuggingFace)")
    st.session_state.url = url
    if st.button("Load Dataset") and url:
        clear_outputs()
        with st.spinner("Loading dataset..."):
            df = load_dataset(url)
            st.session_state.df = df
        st.success("Dataset loaded successfully!")

elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    st.session_state.url = "Uploaded file"
    if uploaded_file is not None:
        clear_outputs()
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("CSV file loaded successfully!")


# Show dataset and stats
if "df" in st.session_state:
    df = st.session_state.df
    st.dataframe(df)
    dqp = DQP(df, target_column="")
    tab1, tab2 = st.tabs(["Overview", "Alerts"])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Dataset Summary")
            summary = stats.dataset_summary(df)
            st.session_state.summary = summary
            st.markdown(markdown_dataset_summary(summary))

        with col2:
            st.subheader("Column Analysis")
            selected_col = st.selectbox("Select a column", df.columns)
            column_statistics = stats.get_column_statistics(dqp, selected_col)
            st.markdown(column_statistics)

        with col3:
            st.subheader("Column Types")
            column_types = dqp.receive_number_of_columns()
            column_types_summary = make_table_column_type_summary(
                column_types=column_types
            )
            st.session_state.column_types = column_types
            st.markdown(column_types_summary)

    with tab2:
        st.subheader("Alerts")
        alerts = stats.get_alerts(df)
        st.session_state.alerts = alerts
        if alerts:
            for alert in alerts:
                if alert["level"] == "info":
                    st.info(alert["message"])
                elif alert["level"] == "warning":
                    st.warning(alert["message"])
                else:
                    st.error(alert["message"])
        else:
            st.info("No alerts found.")

    st.markdown("---")

    st.subheader("Target Column Selection & Distribution")
    numeric_columns, categorical_columns = dqp.receive_categorized_columns()
    valid_columns = numeric_columns + categorical_columns
    selected_col = st.selectbox(
        "Select the target column", df.columns, index=None, placeholder="(auto-select)"
    )
    target = None

    if selected_col:
        if selected_col in valid_columns:
            target = selected_col
        else:
            st.error(
                f'"{selected_col}" is not a valid numerical or categorical column.'
            )
    else:
        # Try from last column backward until a valid one is found
        for col in reversed(df.columns):
            if col in valid_columns:
                target = col
                break
        if not target:
            st.error("No valid numerical or categorical column found in the dataset.")

    is_numeric = target in numeric_columns
    if target:
        st.success(f"Selected target column: **{target}**")
        st.session_state.target_column = target
        is_numeric = target in numeric_columns
        if is_numeric:
            stats.plot_data_distribution_by_column(
                df, target, streamlit_mode=True, st=st, is_numeric=is_numeric
            )
        else:
            stats.plot_data_distribution_by_column(
                df, target, streamlit_mode=True, st=st, is_numeric=is_numeric
            )


@st.cache_data
def run_quality_analysis(data, model_type, target_col):
    return quality(data, model_type, target_col)


# Model Selection
if "df" in st.session_state and st.session_state.target_column:
    st.markdown("---")
    df = st.session_state.df
    target_column = st.session_state.target_column
    supported_models = list(MODEL_REGISTRY.keys())

    selected_model = st.selectbox(
        "Choose model",
        supported_models,
        index=None,
        placeholder=(supported_models[0]),
    )
    if selected_model:
        st.success(f"Model selected: {selected_model} | Target column: {target_column}")
        st.session_state.model = MODEL_REGISTRY[selected_model]
    else:
        st.info(
            f"No model selected. Using default: Logistic Regression | Target column: {target_column}"
        )
        st.session_state.model = MODEL_REGISTRY[supported_models[0]]

    # User-defined sampling
    use_sample = st.checkbox("Sample dataset before training", value=False)

    if use_sample:
        default_sample_frac = st.session_state.get("sample_frac", 0.25)
        default_sample_percent = int(default_sample_frac * 100)

        sample_percent = st.slider(
            "Sampling fraction (%)",
            min_value=5,
            max_value=100,
            value=default_sample_percent,
            step=5,
        )

        sample_frac = sample_percent / 100
        st.session_state.sample_frac = sample_frac
        if sample_frac < 1.0:
            st.session_state.reduced_df = df.sample(
                frac=sample_frac, random_state=cfg.SEED
            )
        else:
            st.session_state.reduced_df = df
        st.info(f"Using {sample_percent}% of the dataset for training and analysis.")
    else:
        st.session_state.reduced_df = df

    st.session_state.setdefault("quality_results", None)
    st.session_state.setdefault("fairness_results", None)

    if st.button("Run Quality Analysis"):
        st.session_state.quality_results = None
        with st.spinner("Running quality pipeline..."):
            quality_start = time.time()
            st.session_state.quality_results = run_quality_analysis(
                data=st.session_state.reduced_df.copy(),
                model_type=st.session_state.model,
                target_col=target_column,
            )
            messages = print_step_end(
                name="Quality", start_time=quality_start, streamlit_active=True
            )
            st.success(messages)

    if st.session_state.quality_results:
        display_result_section("📊 Quality Results", st.session_state.quality_results)

    if st.button("Run Fairness Analysis"):
        st.session_state.fairness_results = None
        with st.spinner("Running fairness pipeline..."):
            fairness_start = time.time()
            st.session_state.fairness_results = run_fairness_analysis(
                data=st.session_state.reduced_df.copy(),
                model_type=st.session_state.model,
                target_col=target_column,
            )
            messages = print_step_end(
                name="Fairness", start_time=fairness_start, streamlit_active=True
            )
            st.success(messages)

    if st.session_state.fairness_results:
        display_result_section("⚖️ Fairness Results", st.session_state.fairness_results)

    # Final Report Viewer
    if st.session_state.quality_results or st.session_state.fairness_results:
        st.markdown("---")
        if st.button("Generate Final Report"):
            with st.spinner("Generating report..."):
                report_path = "outputs/2/final_report.pdf"
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                column_types = print_column_type_summary(st.session_state.column_types)
                save_results_to_pdf(
                    filepath=report_path,
                    url=st.session_state.url,
                    overview_summary=st.session_state.summary,
                    column_types=column_types,
                    alerts=st.session_state.alerts,
                    quality_results=st.session_state.quality_results,
                    fairness_results=st.session_state.fairness_results,
                )
                st.session_state.report_path = report_path
            st.success("Report generated!")

        if "report_path" in st.session_state and os.path.exists(
            st.session_state.report_path
        ):
            with open(st.session_state.report_path, "rb") as f:
                st.download_button(
                    label="Download Final Report",
                    data=f,
                    file_name=os.path.basename(st.session_state.report_path),
                    mime="application/pdf",
                )
