import numpy as np
import pandas as pd

from src.quality.no_influence import mahalanobis_outliers
from src.quality.with_influence import influence_outliers
import os, sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)


# For Report #1
def summarize_outliers(
    df,
    num_cols,
    target_col,
    positive_class,
    alpha=0.01,
    sigma_multiplier=3.0,
    model="logistic",
):
    m_mask = mahalanobis_outliers(df, num_cols, alpha=alpha)
    i_mask = influence_outliers(
        df,
        target_col=target_col,
        positive_class=positive_class,
        frac=0.01,
        test_size=0.2,
        random_state=912,
        sigma_multiplier=sigma_multiplier,
        model=model,
    )
    total = len(df)

    inf_count = i_mask.sum()
    inf_pct = inf_count / total * 100
    mah_count = m_mask.sum()
    mah_pct = mah_count / total * 100
    overlap = np.logical_and(m_mask, i_mask).sum()

    summary = pd.DataFrame(
        [
            {
                "Influence_outliers_count": inf_count,
                "Influence_outliers_%": inf_pct,
                "Mahalanobis_outliers_count": mah_count,
                "Mahalanobis_outliers_%": mah_pct,
                "Overlap_count": overlap,
            }
        ]
    )
    return summary


# return df without rows flagged by Mahalanobis
def drop_statistic_outliers(df, num_cols, alpha=0.01):
    mask = mahalanobis_outliers(df, num_cols, alpha=alpha)
    return df.loc[~mask].copy()


# return df without rows flagged by Influence‐based method
def drop_influence_outliers(
    df, target_col, positive_class, sigma_multiplier=3.0, model="logistic"
):
    mask = influence_outliers(
        df,
        target_col=target_col,
        positive_class=positive_class,
        frac=0.01,
        test_size=0.2,
        random_state=912,
        sigma_multiplier=sigma_multiplier,
        model=model,
    )
    return df.loc[~mask].copy()
