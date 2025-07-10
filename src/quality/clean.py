import numpy as np
import pandas as pd

from src.quality.no_influence import mahalanobis_outliers
from src.quality.with_influence import influence_outliers
import os, sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)


# For Report #1
# Model must be trained
def summarize_outliers(
    X_train, X_test, y_train, y_test, model, num_cols, alpha=0.01, sigma_multiplier=1.0
):
    m_mask = mahalanobis_outliers(X_train, X_test, num_cols, alpha=alpha)
    i_mask = influence_outliers(
        X_train,
        X_test,
        y_train,
        y_test,
        model,
        frac=0.001,
        random_state=912,
        sigma_multiplier=sigma_multiplier,
    )
    total = len(X_train) + len(X_test)

    inf_count = i_mask.sum()
    inf_pct = inf_count / total * 100
    mah_count = m_mask.sum()
    mah_pct = mah_count / total * 100
    overlap = np.logical_and(m_mask, i_mask).sum()

    summary = pd.DataFrame(
        [
            {
                "Influence_outliers_count": inf_count,
                "Influence_outliers_%": round(inf_pct, 4),
                "Mahalanobis_outliers_count": mah_count,
                "Mahalanobis_outliers_%": round(mah_pct, 4),
                "Overlap_count": overlap,
            }
        ]
    )
    return summary


# return df without rows flagged by Mahalanobis
def drop_statistic_outliers(X_train, X_test, num_cols, alpha=0.01):
    full = pd.concat([X_train, X_test])
    mask = mahalanobis_outliers(X_train, X_test, num_cols, alpha=alpha)
    return full.loc[~mask].copy()


# return df without rows flagged by Influence‐based method
# Model must be trained
def drop_influence_outliers(
    X_train, X_test, y_train, y_test, model, sigma_multiplier=1.0
):
    full = pd.concat([X_train, X_test])
    mask = influence_outliers(
        X_train,
        X_test,
        y_train,
        y_test,
        model,
        frac=0.001,
        random_state=912,
        sigma_multiplier=sigma_multiplier,
    )
    return full.loc[~mask].copy()
