import numpy as np
import pandas as pd

import os, sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

from src.influence.logistic_influence import LogisticInfluence


def influence_outliers(
    X_train,
    X_test,
    y_train,
    y_test,
    model,
    frac=0.001,
    random_state=912,
    sigma_multiplier=3.0,
):
    """
    Flag outliers based on model‐aware influence scores using logistic influence functions.

    This function computes the average influence of each training point on a
    (sub)sample of test points, and flags those whose influence exceeds a
    threshold defined by mean + sigma_multiplier·std (clamped at zero).

    Args:
        X_train (pd.DataFrame or np.ndarray):
            Training feature set.
        X_test (pd.DataFrame or np.ndarray):
            Test feature set.
        y_train (pd.Series or np.ndarray):
            Training labels.
        y_test (pd.Series or np.ndarray):
            Test labels.
        model:
            A trained model instance (e.g., scikit learn estimator) compatible
            with LogisticInfluence.
        frac (float):
            Fraction of X_test to sample for influence computation (default=0.001).
        random_state (int):
            Random seed for sampling test subset (default=912).
        sigma_multiplier (float):
            Multiplier for the standard deviation when setting the positive threshold
            (default=3.0).

    Returns:
        np.ndarray of bool, shape=(n_train + n_test,):
            Boolean mask where True indicates a training point whose average
            influence on the sampled test set exceeds the threshold.
    """
    # Optionally subsample the test set to limit computation
    if frac < 1.0:
        X_te = X_test.sample(frac=frac, random_state=random_state)
        y_te = y_test.loc[X_te.index]
    else:
        X_te, y_te = X_test, y_test

    infl = LogisticInfluence(
        model, X_train.values.astype(np.float64), y_train.values.astype(np.float64)
    )
    avg_inf = infl.average_influence(
        X_te.values.astype(np.float64), y_te.values.astype(np.float64)
    )

    # Determine threshold: mean + multiplier·std, clamped at zero
    mu = avg_inf.mean()
    sigma = avg_inf.std()
    thresh = max(mu + sigma_multiplier * sigma, 0.0)  # Keep only positive points!

    flagged_idxs = X_train.index[avg_inf > thresh]
    full_index = X_train.index.append(X_test.index)
    full_mask = pd.Series(False, index=full_index)
    full_mask.loc[flagged_idxs] = True

    return full_mask.values
