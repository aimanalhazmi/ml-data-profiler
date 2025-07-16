import numpy as np
import pandas as pd
import re
from sklearn.metrics import precision_score
from scipy import stats
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame,
)

import os, sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

def compute_fairness_influence_metrics(df, class_col, positive_group, d_tol=0.2):
    """
    Compute group‐level fairness metrics based on influence scores using Cohen’s d.

    This function supports both discrete class labels and interval‐based groups
    (e.g., pd.Interval or an interval string like "(low, high]"). It calculates
    the mean influence for the specified positive group versus the rest, the
    difference of means, and Cohen’s d effect size. Two fairness flags are set:
    one based on a relative‐difference tolerance (mean difference / mean_other)
    and one based on the Cohen’s d threshold.

    Args:
        df (pd.DataFrame):
            DataFrame containing at least the 'influence' column and the
            column specified by `class_col`.
        class_col (str):
            Name of the column in `df` holding group labels or numeric values
            to be binned via interval logic.
        positive_group (str | pd.Interval):
            The subgroup to evaluate. May be:
            - A discrete label matching values in `class_col`
            - A pandas Interval instance
            - A string in interval notation, e.g. "(0.0, 1.0]"
        d_tol (float, optional):
            Maximum absolute Cohen’s d for which the group is considered fair
            (default: 0.2).

    Returns:
        dict:
            A dictionary containing:
            - 'group_col':        the name of `class_col`
            - 'positive_group':   the provided `positive_group` value
            - 'mean_positive_group': mean influence of the positive subgroup
            - 'mean_other':       mean influence of the complement subgroup
            - 'Inf_mean_diff':    difference of subgroup means (mean_pos − mean_other)
            - 'cohen_d':          Cohen’s d effect size
            - 'Inf_fair':         bool flag if |mean difference / mean_other| ≤ 3
            - 'Cohen_fair':       bool flag if |Cohen’s d| ≤ d_tol
    """
    influences = df['influence']
    groups = df[class_col]

    if isinstance(positive_group, pd.Interval):
        mask_pos = groups == positive_group

    elif isinstance(positive_group, str):
        m = re.match(r'([\[\(])\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*([\]\)])', 
                     positive_group)
        if m:
            lb, lo, hi, rb = m.groups()
            lo, hi = float(lo), float(hi)
            lower_ok = (groups >= lo) if lb == '[' else (groups > lo)
            upper_ok = (groups <= hi) if rb == ']' else (groups < hi)
            mask_pos = lower_ok & upper_ok
        else:
            mask_pos = groups.astype(str) == positive_group

    else:
        mask_pos = groups == positive_group

    pos_vals   = influences[mask_pos]
    other_vals = influences[~mask_pos]

    n1, n2 = len(pos_vals), len(other_vals)
    mean1, mean2 = pos_vals.mean(), other_vals.mean()
    std1, std2 = pos_vals.std(ddof=1), other_vals.std(ddof=1)
    delta = mean1 - mean2

    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / max(n1+n2-2, 1))
    cohens_d = delta / pooled_std if pooled_std > 0 else 0.0

    inf_fair   = abs(delta / mean2) <= 3
    cohen_fair = abs(cohens_d) <= d_tol

    return {
        'group_col': class_col,
        'positive_group': positive_group,
        'mean_positive_group': mean1,
        'mean_other': mean2,
        'Inf_mean_diff': delta,
        'cohen_d': cohens_d,
        'Inf_fair': inf_fair,
        'Cohen_fair': cohen_fair
    }



# Influence threshold is a parameter because we have to find out which number should be the threshold.
# Receives target column and sensitive coolumns and caluclates DPD and EOD for each sensitive column, with yes or no flag.
# Computes PPV for each group (and flags whether each group is within tolerance).
# Computes the global mean‐influence for positive vs. negative classes and flags it too.
# The model parameter must be trained!
def compute_fairness_classical_metrics(
    X_test,
    y_test,
    s_test_df,
    sens_cols,
    model,
    dpd_tol=0.1,
    eod_tol=0.1,
    ppv_tol=0.1,
):
    """
    Compute classical fairness metrics (DPD, EOD, PPV) for a trained classifier.

    Args:
        X_test (pd.DataFrame or np.ndarray):
            Features for the test set.
        y_test (pd.Series or np.ndarray):
            True labels for the test set.
        s_test_df (pd.DataFrame):
            DataFrame containing sensitive attributes for the test set.
        sens_cols (list[str]):
            Column names in `s_test_df` to evaluate (e.g., ["race", "gender"]).
        model:
            A trained classifier with a `.predict()` method.
        dpd_tol (float):
            Tolerance threshold for Demographic Parity Difference.
        eod_tol (float):
            Tolerance threshold for Equalized Odds Difference.
        ppv_tol (float):
            Tolerance threshold for Predictive-Value Parity gap.

    Returns:
        list[dict]:
            A list of records, one per sensitive feature, each containing:
            - "sensitive_feature": feature name
            - "DPD": demographic parity difference
            - "DPD_fair": whether |DPD| ≤ dpd_tol
            - "EOD": equalized odds difference
            - "EOD_fair": whether |EOD| ≤ eod_tol
            - "PPV_diff": precision gap between best and worst groups
            - "PPV_fair": whether |PPV_diff| ≤ ppv_tol
            - "PPV_by_group": dict of per-group precision scores
            - "PPV_group_fairness": dict of per-group flags indicating
              whether each group's precision is within `ppv_tol` of the best group.
    """
    y_pred = model.predict(X_test)

    # Get DPD and EOD per sensitive column. Then get PPV for each category inside the columns.
    records = []
    for sens in sens_cols:
        sf_test = s_test_df[sens].astype(str)
        dpd = demographic_parity_difference(
            y_true=y_test, y_pred=y_pred, sensitive_features=sf_test
        )
        eod = equalized_odds_difference(
            y_true=y_test, y_pred=y_pred, sensitive_features=sf_test
        )
        mf = MetricFrame(
            metrics=lambda y_true, y_pred: precision_score(
                y_true, y_pred, zero_division=0
            ),
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sf_test,
        )
        ppv_by_group = mf.by_group.to_dict()
        ppv_diff = mf.difference()

        records.append(
            {
                "sensitive_feature": sens,
                "DPD": dpd,
                "DPD_fair": abs(dpd) <= dpd_tol,
                "EOD": eod,
                "EOD_fair": abs(eod) <= eod_tol,
                "PPV_diff": ppv_diff,
                "PPV_fair": abs(ppv_diff) <= ppv_tol,
                "PPV_by_group": ppv_by_group,
                "PPV_group_fairness": {
                    g: (max(ppv_by_group.values()) - ppv) <= ppv_tol
                    for g, ppv in ppv_by_group.items()
                },
            }
        )

    return records
