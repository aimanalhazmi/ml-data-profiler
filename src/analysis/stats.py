import pandas as pd
import matplotlib.pyplot as plt
import re


def generate_stats(df: pd.DataFrame, target_col, sensitive_col):
    pass


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return general dataset statistics."""
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    duplicated_rows = df.duplicated().sum()
    memory = df.memory_usage(deep=True).sum()

    return {
        "number_of_variables": df.shape[1],
        "number_of_observations": df.shape[0],
        "missing_cells": int(missing_cells),
        "missing_percent": round(missing_cells / total_cells * 100, 1),
        "duplicate_rows": int(duplicated_rows),
        "duplicate_percent": round(duplicated_rows / df.shape[0] * 100, 1),
        "total_memory_kb": round(memory / 1024, 1),
        "avg_record_size_bytes": round(memory / df.shape[0], 1),
    }


def get_variable_type_summary(df: pd.DataFrame) -> dict:
    """Return summary of variable types."""
    type_map = df.dtypes.apply(
        lambda x: (
            "Text"
            if x == "object"
            else "Categorical" if str(x).startswith("category") else str(x)
        )
    )
    return type_map.value_counts().to_dict()


def get_column_statistics(df: pd.DataFrame, column_name: str) -> dict:
    """Return column specific statistics."""
    col = df[column_name]
    return {
        "distinct": int(col.nunique()),
        "distinct_percent": round(col.nunique() / df.shape[0] * 100, 1),
        "number_of_missing_values": int(col.isna().sum()),
        "missing_percent": round(col.isna().mean() * 100, 1),
        "memory_kb": round(col.memory_usage(deep=True) / 1024, 1),
    }


def get_alerts(df: pd.DataFrame) -> list:
    """Generate basic data quality alerts."""
    alerts = []
    for col in df.columns:
        if df[col].nunique() == 1:
            alerts.append(f'Column "{col}" has constant value "{df[col].iloc[0]}"')
        if df[col].isna().sum() > 0:
            alerts.append(
                f'Column "{col}" has {df[col].isna().sum()} missing values '
                f"({df[col].isna().mean() * 100:.1f}%)"
            )
        if df[col].isna().sum() == 0 and df[col].nunique() == df.shape[0]:
            alerts.append(f'Column "{col}" has all unique values')
    return alerts


def plot_data_distribution_by_column(
    df, column_name, save=False, save_path="", streamlit_mode=False, st=None
):
    """Plot the distribution of a column (histogram for numeric, bar plot for categorical)."""
    data = df[column_name]
    title = re.sub(r"[_\-.]", " ", column_name).title()

    plt.figure(figsize=(8, 4))

    if pd.api.types.is_numeric_dtype(data):
        plt.hist(data.dropna(), bins="auto", edgecolor="black")
        plt.xlabel(title)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {title}")
    else:
        counts = data.value_counts()
        total = counts.sum()
        ax = counts.plot(kind="bar")
        for i, (label, count) in enumerate(counts.items()):
            pct = count / total * 100
            ax.text(
                i,
                count + total * 0.01,
                f"{count} ({pct:.1f}%)",
                ha="center",
                fontsize=9,
            )
        plt.xlabel(title)
        plt.ylabel("Number of Instances")
        plt.title(f"Distribution of {title}")

    if save and save_path:
        plt.savefig(f"{save_path}/{column_name}_distribution.png", bbox_inches="tight")

    if streamlit_mode and st is not None:
        st.pyplot(plt.gcf())
    else:
        plt.show()

    plt.close()
