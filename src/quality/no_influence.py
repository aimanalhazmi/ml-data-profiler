import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from pyod.models.kpca import KPCA

# Statistical outliers: using mahalanobis distance
#  α = 0.01 means we flag any point that lies outside the
#  99% confidence ellipsoid under a multivariate normal
#  assumption like sayun p-value < 0.01.

# Just the X_train and X_test, maybe the df

def mahalanobis_outliers(X_train, X_test, num_cols, alpha = 0.01):
    full = pd.concat([X_train, X_test])
    X = full[num_cols].values
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    eps = 1e-6
    cov_reg = cov + eps * np.eye(cov.shape[0])
    cov_inv = np.linalg.inv(cov_reg)
    diff = X - mu
    m2 = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
    thresh = chi2.ppf(1 - alpha, df=X.shape[1])
    return m2 > thresh

# Statistical outliers: using DBSCAN algorithm
#  eps is the neighborhood radius. So the points
#  within eps of each other are considered neighbors.
#  min_samples ist minimum number of neighbors required to
#  form a cluster. Anything not in a cluster is outlier.
def dbscan_outliers(X_train, X_test, num_cols, eps = 0.5, min_samples = 5):
    full = pd.concat([X_train, X_test])
    X = full[num_cols].values
    Xs = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(Xs)
    return db.labels_ == -1

# Anomaly detection method: using Kernel PCA form PyOD
#  contamination is the fraction of the sampled data to flag as outliers.
# I have created the frac parameter because my computer runs out of memory using 100% of the datapoints so i had to use just 10% to test it.
# I leave it because it might happen the same for you. In the future we should remoive it.
def kpca_outliers(df, num_cols, frac=1, contamination=0.05):
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
