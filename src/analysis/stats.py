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


def get_markdown_dataset_summary(summary_df: pd.DataFrame) -> str:
    """Return general dataset statistics as a markdown table string."""
    lines = [
        "| Metric                  | Value               |",
        "|------------------------|----------------------|",
    ]
    for _, row in summary_df.iterrows():
        metric = str(row["Metric"]).strip()
        value = str(row["Value"]).strip()
        lines.append(f"| {metric:<23} | {value:<20} |")
    return "\n".join(lines)


def get_column_type_summary(dqp: DQP) -> str:
    """Return summary of variable types as a markdown table (for Streamlit)."""
    column_types = dqp.receive_number_of_columns()

    table = "| Column Type | Count |\n"
    table += "|-------------|--------|\n"
    for key, value in column_types.items():
        table += f"| {key.replace('_', ' ').title()} | {value} |\n"
    return table


def print_column_type_summary_terminal(dqp):
    """Print summary of variable types using tabulate (for terminal)."""
    column_types = dqp.receive_number_of_columns()

    df = pd.DataFrame(
        list(column_types.items()), columns=["Column Type", "Count"]
    ).sort_values(by="Column Type")

    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    return df


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


def get_alerts(df: pd.DataFrame) -> list[dict]:
    """Generate basic data quality alerts."""
    alerts = []
    for col in df.columns:
        series = df[col]
        n_unique = series.nunique(dropna=True)
        total = len(series)
        missing = series.isna().sum()
        dtype = series.dtype

        # Info alerts
        if missing == 0:
            alerts.append(
                {"level": "info", "message": f'Column "{col}" has no missing values'}
            )
        if n_unique == total:
            alerts.append(
                {"level": "info", "message": f'Column "{col}" has all unique values'}
            )
        if n_unique == 1:
            alerts.append(
                {
                    "level": "info",
                    "message": f'Column "{col}" has constant value "{series.dropna().iloc[0]}"',
                }
            )

        # Warning alerts
        if 0 < missing < total:
            alerts.append(
                {
                    "level": "warning",
                    "message": f'Column "{col}" has {missing} missing values ({missing/total*100:.2f}%)',
                }
            )
        if n_unique / total > 0.9 and n_unique < total:
            alerts.append(
                {
                    "level": "warning",
                    "message": f'Column "{col}" has high cardinality ({n_unique} unique values)',
                }
            )
        if dtype == "object":
            inferred_types = series.dropna().map(type).nunique()
            if inferred_types > 1:
                alerts.append(
                    {
                        "level": "warning",
                        "message": f'Column "{col}" has mixed data types',
                    }
                )
        if (
            dtype == "object"
            and series.dropna().apply(lambda x: isinstance(x, str)).all()
        ):
            max_len = series.dropna().map(len).max()
            if max_len > 100:
                alerts.append(
                    {
                        "level": "warning",
                        "message": f'Column "{col}" contains long strings (up to {max_len} characters)',
                    }
                )

        # Error alerts
        if missing == total:
            alerts.append(
                {"level": "error", "message": f'Column "{col}" has all values missing'}
            )
        if n_unique == 1 and missing == 0 and total > 1:
            alerts.append(
                {
                    "level": "error",
                    "message": f'Column "{col}" is entirely constant and may be irrelevant',
                }
            )

    order = {"error": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda alert: order[alert["level"]])
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
