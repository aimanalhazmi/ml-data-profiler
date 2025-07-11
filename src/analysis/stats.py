import pandas as pd
import matplotlib.pyplot as plt
import re
from src.preprocessing.preprocessing import Preprocessor as DQP
from tabulate import tabulate
import os


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return general dataset statistics."""
    num_rows, num_cols = df.shape
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    duplicated_rows = df.duplicated().sum()
    memory_bytes = df.memory_usage(deep=True).sum()

    data = [
        ["Number of Columns", num_cols],
        ["Number of Records", num_rows],
        ["Total Cells", int(total_cells)],
        ["Missing Cells", int(missing_cells)],
        [
            "Missing Percentage",
            f"{(missing_cells / total_cells * 100):.2f}%" if total_cells else "0.00%",
        ],
        ["Duplicate Records", int(duplicated_rows)],
        [
            "Duplicate Percentage",
            f"{(duplicated_rows / num_rows * 100):.2f}%" if num_rows else "0.00%",
        ],
        ["Memory Usage", f"{memory_bytes / 1024:.2f} KB"],
        [
            "Average Record Size",
            f"{(memory_bytes / num_rows):.2f} Bytes" if num_rows else "0.00 Bytes",
        ],
    ]

    return pd.DataFrame(data, columns=["Metric", "Value"])


def column_statistics(dqp: DQP, column_name: str) -> dict:
    """Return column specific statistics."""
    df = dqp.data
    numeric_columns, categorical_columns = dqp.receive_categorized_columns()
    _, text_columns, _ = dqp.categorize_columns()
    col = df[column_name]
    if column_name in numeric_columns:
        col_type = "numeric"
    elif column_name in categorical_columns:
        col_type = "categorical"
    elif column_name in text_columns:
        col_type = "text"
    else:
        col_type = "other"

    return {
        "Column_Type": col_type,
        "Distinct_Values": int(col.nunique()),
        "Distinct_Percentage": round(col.nunique() / df.shape[0] * 100, 2),
        "Missing_Values": int(col.isna().sum()),
        "Missing_Percentage": round(col.isna().mean() * 100, 2),
        "Memory_Usage_KB": round(col.memory_usage(deep=True) / 1024, 2),
    }


def get_column_statistics(dqp: DQP, column_name: str) -> str:
    """Return column specific statistics as a table."""
    statistics = column_statistics(dqp, column_name)
    table = "| Metric  | Value |\n"
    table += "|-------------|--------|\n"
    for key, value in statistics.items():
        table += f"|**{key.replace('_', ' ')}** | {value}|\n"
    return table


import pandas as pd


def get_alerts(df: pd.DataFrame, missing_threshold: float = 0.7) -> list[dict]:
    """Generate data quality alerts for a DataFrame."""
    alerts = []
    for col in df.columns:
        series = df[col]
        total = len(series)
        missing = series.isna().sum()
        n_unique = series.nunique(dropna=True)
        dtype = series.dtype

        col_alerts = []
        error_flag = False
        warning_flag = False

        # Error alerts
        if missing == total:
            col_alerts.append(
                {"level": "error", "message": f'Column "{col}" has all values missing'}
            )
            error_flag = True
        elif missing / total > missing_threshold:
            col_alerts.append(
                {
                    "level": "error",
                    "message": f'Column "{col}" has more than {missing_threshold * 100:.0f}% missing values ({missing / total * 100:.2f}%)',
                }
            )
            error_flag = True
        if n_unique == 1 and missing == 0 and total > 1:
            col_alerts.append(
                {
                    "level": "error",
                    "message": f'Column "{col}" is entirely constant and may be irrelevant',
                }
            )
            error_flag = True
        if (
            dtype == "object"
            and series.dropna().apply(lambda x: isinstance(x, str)).all()
        ):
            max_len = series.dropna().map(len).max()
            if max_len > 275:
                col_alerts.append(
                    {
                        "level": "error",
                        "message": f'Column "{col}" contains long strings (up to {max_len} characters)',
                    }
                )
                error_flag = True

        # Warning alerts (only if no error)
        if not error_flag:
            if 0 < missing < total:
                col_alerts.append(
                    {
                        "level": "warning",
                        "message": f'Column "{col}" has {missing} missing values ({missing / total * 100:.2f}%)',
                    }
                )
                warning_flag = True
            if 0.9 < n_unique / total < 1.0:
                col_alerts.append(
                    {
                        "level": "warning",
                        "message": f'Column "{col}" has high cardinality ({n_unique} unique values)',
                    }
                )
                warning_flag = True
            if dtype == "object":
                inferred_types = series.dropna().map(type).nunique()
                if inferred_types > 1:
                    col_alerts.append(
                        {
                            "level": "warning",
                            "message": f'Column "{col}" has mixed data types',
                        }
                    )
                    warning_flag = True
                elif series.dropna().apply(lambda x: isinstance(x, str)).all():
                    max_len = series.dropna().map(len).max()
                    if max_len > 100:
                        col_alerts.append(
                            {
                                "level": "warning",
                                "message": f'Column "{col}" contains long strings (up to {max_len} characters)',
                            }
                        )
                        warning_flag = True

        # Info alerts (only if no warning or error)
        if not error_flag and not warning_flag:
            if missing == 0:
                col_alerts.append(
                    {
                        "level": "info",
                        "message": f'Column "{col}" has no missing values',
                    }
                )
            if n_unique == total:
                col_alerts.append(
                    {
                        "level": "info",
                        "message": f'Column "{col}" has all unique values',
                    }
                )
            if n_unique == 1 and missing < total:
                val = series.dropna().iloc[0] if not series.dropna().empty else "N/A"
                col_alerts.append(
                    {
                        "level": "info",
                        "message": f'Column "{col}" has constant value "{val}"',
                    }
                )

        alerts.extend(col_alerts)

    alerts.sort(key=lambda x: {"error": 0, "warning": 1, "info": 2}[x["level"]])
    return alerts


def plot_data_distribution_by_column(
    df,
    column_name,
    is_numeric: bool,
    save=False,
    save_path="",
    streamlit_mode=False,
    st=None,
    max_categories=20,
):
    """Plot the distribution of a column (histogram if numeric, bar plot if categorical)."""
    data = df[column_name]
    title = re.sub(r"[_\-.]", " ", column_name).title()

    plt.figure(figsize=(14, 6))

    if is_numeric:
        plt.hist(data.dropna(), bins="auto", edgecolor="black")
        plt.xlabel(title)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {title}")
    else:
        counts = data.value_counts(dropna=False)

        if len(counts) > max_categories:
            top_counts = counts.iloc[:max_categories]
            other_sum = counts.iloc[max_categories:].sum()
            top_counts["Other"] = other_sum
            counts = top_counts

        total = counts.sum()
        ax = counts.plot(kind="bar", edgecolor="black")

        for bar, count in zip(ax.patches, counts):
            height = bar.get_height()
            pct = count / total * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (0.001 * total),
                f"{count} ({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.xlabel(title)
        plt.ylabel("Count")
        plt.title(
            f"Top {max_categories} Categories in {title}"
            if len(counts) >= max_categories
            else f"Distribution of {title}"
        )
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if save and save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{column_name}_distribution.png", bbox_inches="tight")

    if streamlit_mode and st is not None:
        st.pyplot(plt.gcf())
    elif not streamlit_mode and not save:
        plt.show()

    plt.close()
