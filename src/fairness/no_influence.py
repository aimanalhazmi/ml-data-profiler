import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame
)

import os, sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

from influence.logistic_influence import LogisticInfluence

def compute_influence_group_diff(df, target_col, group_col, positive_group, model, test_size=0.2, random_state=912, frac=0.001, influence_tol=0.001):
    y = (df[target_col] == df[target_col].unique().max()).astype(int)
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    if frac < 1.0:
        X_te = X_test.sample(frac=frac, random_state=random_state)
        y_te = y_test.loc[X_te.index]
    else:
        X_te, y_te = X_test, y_test

    model.fit(X_train.values, y_train.values)
    infl = LogisticInfluence(model, X_train.values.astype(float), y_train.values.astype(float))
    avg_inf = infl.average_influence(X_te.values.astype(float), y_te.values.astype(float))
    grp = df[group_col].loc[X_train.index]
    mean_pos = avg_inf[grp == positive_group].mean()
    mean_other = avg_inf[grp != positive_group].mean()
    diff = abs(mean_pos - mean_other)
    fair = diff <= influence_tol
    return {'group_col': group_col, 'positive_group': positive_group, 'mean_pos': mean_pos, 'mean_other': mean_other, 'Inf_mean_diff': diff, 'Inf_fair': fair}


# Influence threshold is a parameter because we have to find out which number should be the threshold.
# Receives target column and sensitive coolumns and caluclates DPD and EOD for each sensitive column, with yes or no flag.
# Computes PPV for each group (and flags whether each group is within tolerance).
# Computes the global mean‐influence for positive vs. negative classes and flags it too.

def compute_fairness_metrics(df, target_col, sens_cols, model, positive_class, influence_group_col, positive_group, test_size=0.2, random_state=912, frac=0.001, dpd_tol=0.1, eod_tol=0.1, ppv_tol=0.1, influence_tol=0.001):
    # first compute your influence‐group statistics
    infl_group = compute_influence_group_diff(df, target_col, influence_group_col, positive_group, model, test_size, random_state, frac, influence_tol)

    # now compute fairness metrics per sensitive feature
    y = (df[target_col] == positive_class).astype(int)
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    X_train, X_test, y_train, y_test, s_train_df, s_test_df = train_test_split(
        X, y, df[sens_cols],
        test_size=test_size, stratify=y, random_state=random_state
    )

    if frac < 1.0:
        X_te = X_test.sample(frac=frac, random_state=random_state)
        y_te = y_test.loc[X_te.index]
    else:
        X_te, y_te = X_test, y_test

    model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)

    records = []
    for sens in sens_cols:
        sf_test = s_test_df[sens]
        dpd = demographic_parity_difference(y_true=y_test, y_pred=y_pred, sensitive_features=sf_test)
        eod = equalized_odds_difference(y_true=y_test, y_pred=y_pred, sensitive_features=sf_test)
        mf = MetricFrame(metrics=precision_score, y_true=y_test, y_pred=y_pred, sensitive_features=sf_test)
        ppv_by_group = mf.by_group.to_dict()
        ppv_diff = mf.difference()

        records.append({
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
            }
        })

    return records, infl_group