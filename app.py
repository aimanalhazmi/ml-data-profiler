import streamlit as st
import pandas as pd
import os
from src.ingestion.loader import load_dataset
from src.analysis import stats
from src.preprocessing.preprocessing import Preprocessor as DQP
from src.model.registry import MODEL_REGISTRY
from src.utils.output import save_results_to_pdf
from helper import quality, fairness

st.set_page_config(page_title="Fairfluence App", layout="wide")
st.title("Fairfluence")

# Dataset Input
input_method = st.radio("Choose dataset input method", ("URL", "Upload CSV"))

if input_method == "URL":
    url = st.text_input("Enter dataset URL (OpenML, Kaggle, HuggingFace)")
    if st.button("Load Dataset") and url:
        with st.spinner("Loading dataset..."):
            df = load_dataset(url)
            st.session_state.df = df
        st.success("Dataset loaded successfully!")

elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
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
            st.markdown(stats.get_markdown_dataset_summary(summary))

        with col2:
            st.subheader("Column Analysis")
            selected_col = st.selectbox("Select a column", df.columns)
            column_statistics = stats.get_column_statistics(dqp, selected_col)
            st.markdown(column_statistics)

        with col3:
            st.subheader("Column Types")
            column_types_summary = stats.get_column_type_summary(dqp=dqp)
            st.markdown(column_types_summary)

    with tab2:
        st.subheader("Alerts")
        alerts = stats.get_alerts(df)
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
            st.error(f'"{selected_col}" is not a numerical or categorical column.')
    else:
        # Try from last column backward until a valid one is found
        for col in reversed(df.columns):
            if col in valid_columns:
                target = col
                break
        if not target:
            st.error("No numerical or categorical column found in the dataset.")

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
def run_quality_analysis(df, model_type, target_column):
    return quality(df.copy(), model_type, target_column)


@st.cache_data
def run_fairness_analysis(df, model_type, target_column):
    return fairness(df.copy(), model_type, target_column)


def display_result_section(title, results):
    with st.expander(title, expanded=True):
        for section_title, df_section in results:
            st.subheader(section_title)
            st.dataframe(df_section)


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
            f"No model selected. Using default: Logistic Regression | Target column: '{target_column}'"
        )
        st.session_state.model = MODEL_REGISTRY[supported_models[0]]

    # User-defined sampling
    use_sample = st.checkbox("Sample dataset before training", value=True)
    sample_frac = 0.25  # default

    if use_sample:
        sample_frac = (
            st.slider(
                "Sampling fraction (%)", min_value=5, max_value=100, value=25, step=5
            )
            / 100
        )
        reduced_df = df.sample(frac=sample_frac, random_state=42)
        st.info(
            f"Using {int(sample_frac * 100)}% of the dataset for training and analysis."
        )
    else:
        reduced_df = df

    st.session_state.setdefault("quality_results", None)
    st.session_state.setdefault("fairness_results", None)
    if st.button("Train Model & Run Quality"):
        st.session_state.quality_results = None
        with st.spinner("Running quality pipeline..."):
            st.session_state.quality_results = run_quality_analysis(
                reduced_df.copy(), st.session_state.model, target_column
            )
        # st.success("✅ Quality pipeline completed.")
    if st.session_state.quality_results:
        display_result_section("📊 Quality Results", st.session_state.quality_results)

    if st.session_state.quality_results:
        if st.button("Run Fairness Analysis"):
            st.session_state.fairness_results = None
            with st.spinner("Running fairness pipeline..."):
                st.session_state.fairness_results = run_fairness_analysis(
                    reduced_df.copy(), st.session_state.model, target_column
                )
            # st.success("✅ Fairness analysis completed.")

    # Show fairness results
    if st.session_state.fairness_results:
        display_result_section("⚖️ Fairness Results", st.session_state.fairness_results)

# Final Report Viewer
if "quality_results" in st.session_state and "fairness_results" in st.session_state:
    st.markdown("---")
    if st.button("Generate Final Report"):
        with st.spinner("Generating report..."):
            report_path = "outputs/final_report.pdf"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            save_results_to_pdf(
                filepath=report_path,
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
