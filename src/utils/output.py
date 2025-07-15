import streamlit as st
import time
from tabulate import tabulate
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from datetime import datetime
import os
import sys
import io
from contextlib import redirect_stdout

BLUE = "\033[34m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def print_step_start(name):
    """Prints the start of a named analysis step."""
    print(f"\n=== Running {name} Analysis ===")


def print_info(streamlit_active, msg):
    """Prints info message and shows Streamlit success if active."""
    if streamlit_active:
        st.success(msg)
    print(msg)


def log_to_file(message: str, log_to: str = ""):
    timestamp = datetime.now()
    filename_time = timestamp.strftime("%m-%d_%H")
    log_path = os.path.join(log_to, f"{filename_time}_log.txt")

    entry_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"[{entry_time}] {message}\n")


def run_and_capture_output(func, *args, **kwargs):
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = func(*args, **kwargs)
    output = buf.getvalue()
    return result, output


def run_with_output(func, queue, *args):
    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer

    try:
        result = func(*args)
        output = buffer.getvalue()
        queue.put((result, output))
    except Exception as e:
        queue.put((None, f"Error: {e}"))
    finally:
        sys.stdout = sys_stdout
        buffer.close()


def format_bool_label(val: bool, streamlit_active: bool = False) -> str | bool:
    """
    Returns 'Yes' or 'No' for display in PDF/terminal,
    or the raw boolean value if used in Streamlit.
    """
    if streamlit_active:
        return val
    return "Yes" if val else "No"


def is_streamlit_active():
    """Checks if Streamlit is running."""
    try:
        return st.runtime.exists()
    except (ImportError, AttributeError):
        return False


def print_step_end(name, start_time, streamlit_active):
    """Prints or returns elapsed time for a named step."""
    elapsed = time.time() - start_time
    minutes = elapsed / 60
    message = f"\n{name} completed in {elapsed:.2f} seconds ({minutes:.2f} minutes)."
    if streamlit_active:
        return message
    else:
        print(message)
        return None


def display_alerts(alerts: list[dict]):
    """Prints alert messages with colored level tags."""
    if not alerts:
        print(f"{BLUE}[INFO]{RESET} No alerts found.")
        return

    for alert in alerts:
        level = alert.get("level", "").lower()
        msg = alert.get("message", "")

        if level == "error":
            print(f"{RED}[ERROR]{RESET}: {msg}")
        elif level == "warning":
            print(f"{YELLOW}[WARNING]{RESET}:  {msg}")
        else:
            print(f"{BLUE}[INFO]{RESET}: {msg}")


def markdown_dataset_summary(summary_df: pd.DataFrame) -> str:
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


def make_table_column_type_summary(column_types: dict) -> str:
    """Returns column type summary as markdown table string."""
    table = "| Column Type | Count |\n"
    table += "|-------------|--------|\n"
    for key, value in column_types.items():
        table += f"| {key.replace('_', ' ').title()} | {value} |\n"
    return table


def print_column_type_summary(column_types: dict) -> pd.DataFrame:
    """Prints column type summary as a tabulated table (terminal)."""
    mapper = {
        "numeric_columns": "Numeric Columns",
        "categorical_columns": "Categorical Columns",
        "text_columns": "Text Columns",
    }

    df = pd.DataFrame(list(column_types.items()), columns=["Column Type", "Count"])
    df["Column Type"] = df["Column Type"].map(mapper).fillna(df["Column Type"])
    df = df.sort_values(by="Column Type")

    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    return df


def display_f1_report(report_df):
    """Displays F1 score report in tabular format."""
    column_map = {
        "f1_orig": "F1 Score (Original)",
        "f1_statistic": "F1 Score (Statistic)",
        "f1_influence": "F1 Score (Influence)",
    }
    display_df = report_df.round(4).rename(columns=column_map)

    print("\n=== F1 Score Report ===")
    print(tabulate(display_df, headers="keys", tablefmt="grid", showindex=False))
    return display_df


def display_outlier_summary(summary_df):
    """Displays outlier detection summary in tabular format."""
    column_map = {
        "Influence_outliers_count": "Influence Outliers (Count)",
        "Influence_outliers_%": "Influence Outliers (%)",
        "Mahalanobis_outliers_count": "Mahalanobis Outliers (Count)",
        "Mahalanobis_outliers_%": "Mahalanobis Outliers (%)",
        "Overlap_count": "Overlap (Count)",
    }
    display_df = summary_df.rename(columns=column_map)

    print("\n=== Outlier Summary ===")
    print(tabulate(display_df, headers="keys", tablefmt="grid", showindex=False))
    return display_df


def display_top_patterns(top_patterns):
    """Displays top discovered patterns in a tabular format."""
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
    if (
        top_patterns["pattern_col_2"].isnull().all()
        and top_patterns["pattern_val_2"].isnull().all()
    ):
        top_patterns_display = top_patterns_display.drop(
            columns=["Pattern Column 2", "Pattern Value 2"]
        )

    print("\n=== Top Patterns ===")
    print(
        tabulate(
            top_patterns_display,
            headers="keys",
            tablefmt="grid",
            showindex=False,
        )
    )
    return top_patterns_display


def print_influence_top_patterns(influence_top_patterns: list[dict], streamlit_active):
    """Prints influence-based fairness patterns and returns as DataFrame."""
    print("\n=== Fairness (With Influence) ===")
    influence_summary = []
    for pattern in influence_top_patterns:
        influence_summary.append(
            {
                "Influence Group": pattern["group_col"],
                "Positive Group": pattern["positive_group"],
                "Mean (Positive)": f"{float(pattern['mean_positive_group']):.8f}",
                "Mean (Other)": f"{float(pattern['mean_other']):.8f}",
                "Mean Difference": f"{float(pattern['Inf_mean_diff']):.8f}",
                "Influence Fair": format_bool_label(
                    bool(pattern["Inf_fair"]), streamlit_active
                ),
                "Cohen Fair": format_bool_label(
                    bool(pattern["Cohen_fair"]), streamlit_active
                ),
            }
        )
    print(tabulate(influence_summary, headers="keys", tablefmt="grid"))
    return pd.DataFrame(influence_summary)


def print_one_influence_top_patterns(influence_top_patterns: dict, streamlit_active):
    """Prints a single influence-based fairness pattern."""
    influence_summary = [
        {
            "Influence Group": influence_top_patterns["group_col"],
            "Positive Group": influence_top_patterns["positive_group"],
            "Mean (Positive)": f"{float(influence_top_patterns['mean_positive_group']):.8f}",
            "Mean (Other)": f"{float(influence_top_patterns['mean_other']):.8f}",
            "Mean Difference": f"{float(influence_top_patterns['Inf_mean_diff']):.8f}",
            "Influence Fair": format_bool_label(
                bool(influence_top_patterns["Inf_fair"]), streamlit_active
            ),
            "Cohen Fair": format_bool_label(
                bool(influence_top_patterns["Cohen_fair"]), streamlit_active
            ),
        }
    ]
    print(
        tabulate(
            influence_summary, headers="keys", tablefmt="grid", disable_numparse=True
        )
    )


def print_no_influence_top_patterns(
    no_influence_top_patterns: list[dict], streamlit_active
):
    """Prints fairness metrics without influence."""
    print("\n=== Fairness (No Influence) ===")
    no_influence_summary = []
    for pattern in no_influence_top_patterns:
        no_influence_summary.append(
            {
                "Sensitive Feature": pattern["sensitive_feature"],
                "DPD": round(float(pattern["DPD"]), 4),
                "DPD Fair": format_bool_label(
                    bool(pattern["DPD_fair"]), streamlit_active
                ),
                "EOD": round(float(pattern["EOD"]), 4),
                "EOD Fair": format_bool_label(
                    bool(pattern["EOD_fair"]), streamlit_active
                ),
                "PPV Diff": round(float(pattern["PPV_diff"]), 4),
                "PPV Fair": format_bool_label(
                    bool(pattern["PPV_fair"]), streamlit_active
                ),
            }
        )
    print(tabulate(no_influence_summary, headers="keys", tablefmt="grid"))
    return pd.DataFrame(no_influence_summary)


def print_pattern_table(df_patterns):
    """Prints a condensed table of top patterns (manual formatting)."""
    header = f"{'Pattern':<65} | {'Support':>8} | {'Respons.':>10} | {'Interest.':>11}"
    print(header)
    print("-" * len(header))

    for _, row in df_patterns.iterrows():
        cond1 = f"{row['pattern_col_1']} = {row['pattern_val_1']}"
        cond2 = (
            f" AND {row['pattern_col_2']} = {row['pattern_val_2']}"
            if pd.notna(row["pattern_col_2"])
            else ""
        )
        pattern_str = (cond1 + cond2)[:65]

        print(
            f"{pattern_str:<65} | {row['support']:>8.4f} | {row['responsibility']:>10.2f} | {row['interestingness']:>11.2f}"
        )


def pattern_to_readable(pattern, columns):
    """Converts a pattern dict to a list of readable condition strings."""
    readable = []
    for col_idx, val in pattern.items():
        col_name = columns[col_idx] if isinstance(col_idx, int) else col_idx
        if isinstance(val, pd.Interval):
            readable.append(f"{col_name} ∈ {val}")
        else:
            readable.append(f"{col_name} = {val}")
    return readable


def create_section_header(title, styles, align="left"):
    """Creates a formatted paragraph title for a section header."""
    alignment = {"left": 0, "center": 1}.get(align, 0)
    style = ParagraphStyle(name=title, parent=styles["Heading2"], alignment=alignment)
    return Paragraph(f"<b>{title}</b>", style)


def create_table(data, styles, col_widths, h_align="CENTER"):
    """Builds a styled ReportLab table from data and column widths."""
    table = Table(data, colWidths=col_widths, hAlign=h_align)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def add_section(
    title, results, elements, styles, max_table_width, align="center", is_alerts=False
):
    """Adds a content section to the PDF report, optionally styling alerts."""
    elements.append(Spacer(1, 12))
    elements.append(create_section_header(title, styles, align=align))

    if is_alerts and not results:
        elements.append(Paragraph("No alerts found.", styles["Normal"]))
        return

    subsection_style = ParagraphStyle(
        name="subsection-title", parent=styles["Heading4"], alignment=1
    )
    wrap_center = ParagraphStyle(
        "wrap-center",
        parent=styles["Normal"],
        alignment=1,
        fontSize=10,
        leading=11,
        wordWrap="CJK",
    )
    wrap_left = ParagraphStyle(
        "wrap-left",
        parent=styles["Normal"],
        alignment=0,
        fontSize=10,
        leading=11,
        wordWrap="CJK",
    )

    for section_title, content in results:
        elements.append(Spacer(1, 6))
        if section_title:
            elements.append(Paragraph(section_title, subsection_style))

        if is_alerts:
            data = [
                [
                    Paragraph("<b>Level</b>", wrap_center),
                    Paragraph("<b>Message</b>", wrap_center),
                ]
            ]
            for alert in content:
                level = alert.get("level", "").capitalize()
                message = alert.get("message", "")
                color = {"error": "red", "warning": "orange", "info": "blue"}.get(
                    level.lower(), "black"
                )
                level_para = Paragraph(
                    f'<font color="{color}">{level}</font>', wrap_center
                )
                message_para = Paragraph(
                    f'<font color="{color}">{message}</font>', wrap_center
                )
                data.append([level_para, message_para])

        elif isinstance(content, pd.DataFrame) and not content.empty:
            data = [content.columns.tolist()] + content.astype(str).values.tolist()
            data = [[Paragraph(cell, wrap_left) for cell in row] for row in data]
        else:
            elements.append(Paragraph("No data.", styles["Normal"]))
            continue

        col_count = len(data[0])
        col_widths = [max_table_width / col_count] * col_count
        elements.append(create_table(data, styles, col_widths))
        elements.append(Spacer(1, 6))


def save_results_to_pdf(
    filepath,
    url,
    overview_summary,
    column_types,
    alerts,
    quality_results,
    fairness_results,
):
    """Generates and saves the final PDF report with all analysis results."""
    doc = SimpleDocTemplate(filepath, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    page_width = A4[0]

    main_title = Paragraph(
        "<b>Fairfluence Report</b>",
        ParagraphStyle(
            name="MainTitle",
            parent=styles["Heading1"],
            alignment=1,
            fontSize=18,
            spaceAfter=8,
        ),
    )
    elements.append(main_title)

    url_paragraph = Paragraph(
        f'<font size="10">The dataset used to create this report: <a href="{url}">{url}</a></font>',
        ParagraphStyle(
            name="URLStyle",
            parent=styles["Normal"],
            alignment=1,
            spaceAfter=12,
        ),
    )

    elements.append(url_paragraph)

    add_section(
        "Dataset Summary",
        [("", overview_summary)],
        elements,
        styles,
        page_width - 6 * cm,
    )
    add_section(
        "Column Type Summary",
        [("", column_types)],
        elements,
        styles,
        page_width - 6 * cm,
    )
    add_section(
        "Alerts", [("", alerts)], elements, styles, page_width - 2 * cm, is_alerts=True
    )
    if quality_results:
        add_section(
            "Quality Results", quality_results, elements, styles, page_width - 2 * cm
        )
    if fairness_results:
        add_section(
            "Fairness Results", fairness_results, elements, styles, page_width - 2 * cm
        )

    doc.build(elements)
