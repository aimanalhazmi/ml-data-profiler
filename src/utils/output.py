import streamlit as st
import time
from tabulate import tabulate
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

BLUE = "\033[34m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def print_step_start(name):
    print(f"\n=== Running {name} Analysis ===")


def print_info(streamlit_active, msg):
    if streamlit_active:
        st.success(msg)
    print(msg)


def color_bool(val, streamlit_active=False):
    if streamlit_active:
        return val
    else:
        return f"\033[92m{val}\033[0m" if val else f"\033[91m{val}\033[0m"


def is_streamlit_active():
    try:
        return st.runtime.exists()
    except (ImportError, AttributeError):
        return False


def print_step_end(name, start_time, streamlit_active):
    elapsed = time.time() - start_time
    minutes = elapsed / 60
    message = f"\n{name} completed in {elapsed:.2f} seconds ({minutes:.2f} minutes)."
    if streamlit_active:
        return message
    else:
        print(message)
        return None


def display_alerts(alerts: list[dict]):
    if not alerts:
        print(f"{BLUE}[INFO]{RESET} No alerts found.")
        return

    for alert in alerts:
        level = alert.get("level", "").lower()
        msg = alert.get("message", "")

        if level == "error":
            print(f"{RED}[ERROR]{RESET}: {msg}")
        elif level == "warning":
            print(f"{YELLOW}: [WARNING]{RESET}:  {msg}")
        else:
            print(f"{BLUE}[INFO]{RESET}: {msg}")


def display_f1_report(report_df):
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
    print("\n=== Fairness (With Influence) ===")
    influence_summary = []
    for pattern in influence_top_patterns:
        influence_summary.append(
            {
                "Influence Group": pattern["group_col"],
                "Positive Group": pattern["positive_group"],
                "Mean (Positive)": round(float(pattern["mean_pos"]), 8),
                "Mean (Other)": round(float(pattern["mean_other"]), 8),
                "Mean Difference": round(float(pattern["Inf_mean_diff"]), 8),
                "Influence Fair": color_bool(
                    bool(pattern["Inf_fair"]), streamlit_active
                ),
            }
        )
    print(tabulate(influence_summary, headers="keys", tablefmt="grid"))
    return pd.DataFrame(influence_summary)


def print_one_influence_top_patterns(influence_top_patterns: dict, streamlit_active):
    influence_summary = [
        {
            "Influence Group": influence_top_patterns["group_col"],
            "Positive Group": influence_top_patterns["positive_group"],
            "Mean (Positive)": f"{float(influence_top_patterns['mean_pos']):.8f}",
            "Mean (Other)": f"{float(influence_top_patterns['mean_other']):.8f}",
            "Mean Difference": f"{float(influence_top_patterns['Inf_mean_diff']):.8f}",
            "Influence Fair": color_bool(
                bool(influence_top_patterns["Inf_fair"]), streamlit_active
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
    print("\n=== Fairness (No Influence) ===")
    no_influence_summary = []
    for pattern in no_influence_top_patterns:
        no_influence_summary.append(
            {
                "Sensitive Feature": pattern["sensitive_feature"],
                "DPD": round(float(pattern["DPD"]), 4),
                "DPD Fair": color_bool(bool(pattern["DPD_fair"]), streamlit_active),
                "EOD": round(float(pattern["EOD"]), 4),
                "EOD Fair": color_bool(bool(pattern["EOD_fair"]), streamlit_active),
                "PPV Diff": round(float(pattern["PPV_diff"]), 4),
                "PPV Fair": color_bool(bool(pattern["PPV_fair"]), streamlit_active),
            }
        )
    print(tabulate(no_influence_summary, headers="keys", tablefmt="grid"))
    return pd.DataFrame(no_influence_summary)


def print_pattern_table(df_patterns):
    print("\n=== Top Patterns (Condensed View) ===")
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
    readable = []
    for col_idx, val in pattern.items():
        col_name = columns[col_idx] if isinstance(col_idx, int) else col_idx
        if isinstance(val, pd.Interval):
            readable.append(f"{col_name} ∈ {val}")
        else:
            readable.append(f"{col_name} = {val}")
    return readable


def add_section(title, results, elements, styles):
    elements.append(Spacer(1, 12))

    # Section title: LEFT-aligned
    section_style = ParagraphStyle(
        name="section-title", parent=styles["Heading2"], alignment=0  # 0 = left
    )
    elements.append(Paragraph(f"<b>{title}</b>", section_style))

    page_width = A4[0]
    max_table_width = page_width - 2 * cm

    wrap_style = ParagraphStyle(
        "wrap",
        parent=styles["Normal"],
        fontSize=10,
        leading=11,
        wordWrap="CJK",
    )

    # Subsection title style: LEFT-aligned
    subsection_style = ParagraphStyle(
        name="subsection-title", parent=styles["Heading4"], alignment=1
    )

    for section_title, df in results:
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(section_title, subsection_style))

        if isinstance(df, pd.DataFrame) and not df.empty:
            data = [df.columns.tolist()] + df.astype(str).values.tolist()
            data = [[Paragraph(cell, wrap_style) for cell in row] for row in data]

            num_cols = len(data[0])
            base_col_width = max_table_width / num_cols
            col_widths = [base_col_width] * num_cols

            table = Table(data, colWidths=col_widths, hAlign="CENTER")
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
            elements.append(table)
        else:
            elements.append(Paragraph("No data.", styles["Normal"]))

        elements.append(Spacer(1, 6))


def save_results_to_pdf(filepath, quality_results, fairness_results):
    doc = SimpleDocTemplate(filepath, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    add_section("1. Quality Results", quality_results, elements, styles)
    add_section("2. Fairness Results", fairness_results, elements, styles)

    doc.build(elements)
