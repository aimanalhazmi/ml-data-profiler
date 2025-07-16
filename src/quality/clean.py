import numpy as np
import pandas as pd

from src.quality.no_influence import mahalanobis_outliers
from src.quality.with_influence import influence_outliers
import os, sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)


def summarize_outliers(
    X_train, X_test, y_train, y_test, model, num_cols, alpha=0.01, sigma_multiplier=1.0
):
    
    """
    Summarize outlier counts and overlap for Mahalanobis and influence-based methods.

    Args:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Test feature set.
        y_train (pd.Series): Training target labels.
        y_test (pd.Series): Test target labels.
        model: A trained model implementing .predict().
        num_cols (list[str]): List of numeric column names to use for Mahalanobis.
        alpha (float): Significance level for Mahalanobis outlier cutoff.
        sigma_multiplier (float): Multiplier for influence outlier threshold.

    Returns:
        pd.DataFrame: A single-row DataFrame with counts and percentages of
        influence-based and Mahalanobis outliers, plus their overlap.
    """
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
    """
    Remove rows flagged by Mahalanobis distance outlier detection.

    Args:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Test feature set.
        num_cols (list[str]): List of numeric column names to use for Mahalanobis.
        alpha (float): Significance level for Mahalanobis outlier cutoff.

    Returns:
        pd.DataFrame: Concatenated DataFrame of X_train and X_test with
        Mahalanobis outliers removed.
    """
    full = pd.concat([X_train, X_test])
    mask = mahalanobis_outliers(X_train, X_test, num_cols, alpha=alpha)
    return full.loc[~mask].copy()


# return df without rows flagged by Influence‐based method
# Model must be trained
def drop_influence_outliers(
    X_train, X_test, y_train, y_test, model, sigma_multiplier=1.0
):
    """
    Remove rows flagged by influence-based outlier detection.

    Args:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Test feature set.
        y_train (pd.Series): Training target labels.
        y_test (pd.Series): Test target labels.
        model: A trained model implementing .predict().
        sigma_multiplier (float): Multiplier for influence outlier threshold.

    Returns:
        pd.DataFrame: Concatenated DataFrame of X_train and X_test with
        influence-based outliers removed.
    """
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
