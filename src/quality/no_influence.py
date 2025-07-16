import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from pyod.models.kpca import KPCA


def mahalanobis_outliers(X_train, X_test, num_cols, alpha = 0.01):
    """
    Flag outliers based on Mahalanobis distance under a multivariate normal assumption.

    This method concatenates the training and test sets, computes the empirical
    mean and covariance on the specified numeric columns, applies a small ridge
    regularization to ensure invertibility, and then computes each point’s
    squared Mahalanobis distance. Points with distance exceeding the χ² cutoff
    for significance level α are marked as outliers.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        num_cols (list[str]): List of numeric column names to include.
        alpha (float): Significance level for the χ² threshold (default=0.01).

    Returns:
        np.ndarray of bool, shape=(n_train + n_test,):
            Boolean mask where True indicates an outlier.
    """
    full = pd.concat([X_train, X_test])
    X = full[num_cols].values
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    # Ensure 2D covariance
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    eps = 1e-6
    cov_reg = cov + eps * np.eye(cov.shape[0])
    cov_inv = np.linalg.inv(cov_reg)
    diff = X - mu
    m2 = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
    thresh = chi2.ppf(1 - alpha, df=X.shape[1])
    return m2 > thresh


def dbscan_outliers(X_train, X_test, num_cols, eps = 0.5, min_samples = 5):
    """
    Flag outliers using the DBSCAN clustering algorithm.

    This method concatenates the training and test sets, standardizes the
    specified numeric columns, and applies DBSCAN. Points assigned to the
    “noise” label (-1) are marked as outliers.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        num_cols (list[str]): List of numeric column names to include.
        eps (float): Neighborhood radius for DBSCAN (default=0.5).
        min_samples (int): Minimum number of neighbors to form a cluster (default=5).

    Returns:
        np.ndarray of bool, shape=(n_train + n_test,):
            Boolean mask where True indicates an outlier.
    """
    full = pd.concat([X_train, X_test])
    X = full[num_cols].values
    Xs = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(Xs)
    return db.labels_ == -1


def kpca_outliers(df, num_cols, frac=1, contamination=0.05):
    """
    Flag outliers using Kernel PCA from the PyOD library.

    This method samples a fraction of the full DataFrame to fit the KPCA
    detector (to avoid memory issues), obtains a boolean mask of outliers on
    that sample, and then projects those labels back onto the full index.

    Args:
        df (pd.DataFrame): Full dataset containing numeric features.
        num_cols (list[str]): List of numeric column names to include.
        frac (float): Fraction of rows to sample for detector training (default=1.0).
        contamination (float): Expected proportion of outliers in the data (default=0.05).

    Returns:
        np.ndarray of bool, shape=(n_rows,):
            Boolean mask where True indicates an outlier.
    """
    # Make a subset of the dataset
    sample_df = df.sample(frac=frac, random_state=912)
    Xs = sample_df[num_cols].values
    detector = KPCA(contamination=contamination)
    detector.fit(Xs)
    mask_sample = detector.labels_.astype(bool)
    # Build full‐length mask
    full_mask = pd.Series(False, index=df.index)
    full_mask.loc[sample_df.index] = mask_sample
    return full_mask.values
