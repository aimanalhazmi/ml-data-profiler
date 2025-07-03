import streamlit as st
import pandas as pd
import os
from src.ingestion.loader import load_dataset

st.set_page_config(page_title="Fairfluence App", layout="wide")
st.title("Fairfluence")

# Dataset Input
url = st.text_input("Enter dataset URL (OpenML, Kaggle, HuggingFace)")

if st.button("Load Dataset") and url:
    with st.spinner("Loading dataset..."):
        # example usage
        # df = pd.DataFrame({"example_column": [1, 2, 3]})

        df = load_dataset(url)

        st.session_state.df = df
    st.success("Dataset loaded successfully!")
    st.dataframe(df)

# Model Selection
if "df" in st.session_state:
    st.markdown("---")
    model_type = st.selectbox("Choose model", ["Logistic Regression", "SVM"])

    if st.button("Train Model and Compute Influence"):
        with st.spinner("Training model and computing influence scores..."):
            # Replace with training and influence pipeline
            influence_df = pd.DataFrame(
                {"index": [0, 1, 2], "influence_score": [0.5, 0.3, 0.1]}
            )
            st.session_state.influence_df = influence_df
        st.success("Influence scores computed!")
        st.dataframe(influence_df.sort_values("influence_score", ascending=False))

# Quality & Fairness
if "influence_df" in st.session_state:
    st.markdown("---")
    if st.button("Run Quality and Fairness Checks"):
        with st.spinner("Running analysis..."):
            # Replace with quality and fairness modules
            quality = {"missing_values": "None", "outliers": "Detected"}
            fairness = {"demographic_parity": 0.12, "equal_opportunity": 0.08}
            st.session_state.quality = quality
            st.session_state.fairness = fairness
        st.success("Analysis complete!")
        st.subheader("Data Quality Summary")
        st.json(quality)
        st.subheader("Fairness Metrics")
        st.json(fairness)

# Final Report Viewer
if "quality" in st.session_state and "fairness" in st.session_state:
    st.markdown("---")
    if st.button("Generate Final Report"):
        with st.spinner("Generating report..."):
            # Replace this with real report generation
            report_path = "outputs/test_report.pdf"
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
