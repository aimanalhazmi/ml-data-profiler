import os, sys
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from pyod.models.kpca import KPCA

import os, sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

from ingestion.ingestorFactory import IngestorFactory

# Import data

# Future: Preprocessor
link = "https://huggingface.co/datasets/scikit-learn/adult-census-income"
ingestor_factory = IngestorFactory(link, 0)
ingestor = ingestor_factory.create()
df = ingestor.load_data()

# Select numeric columns, in the future, all but the target wil be numeric.
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()


# Statistical outliers: using mahalanobis distance
#  α = 0.01 means we flag any point that lies outside the
#  99% confidence ellipsoid under a multivariate normal
#  assumption like sayun p-value < 0.01.
def mahalanobis_outliers(df, num_cols, alpha = 0.01):
    X = df[num_cols].values
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov_inv = np.linalg.inv(cov)
    diff = X - mu
    m2 = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
    thresh = chi2.ppf(1 - alpha, df=X.shape[1])
    return m2 > thresh

# Statistical outliers: using DBSCAN algorithm
#  eps is the neighborhood radius. So the points
#  within eps of each other are considered neighbors.
#  min_samples ist minimum number of neighbors required to
#  form a cluster. Anything not in a cluster is outlier.
def dbscan_outliers(df, num_cols, eps = 0.5, min_samples = 5):
    X = df[num_cols].values
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


# Example usage
m_mask = mahalanobis_outliers(df, num_cols, alpha=0.01)
d_mask = dbscan_outliers(df, num_cols, eps=0.5, min_samples=5)
k_mask = kpca_outliers(df, num_cols, frac=0.1)


print(f"Total rows: {len(df)}")
print(f"Mahalanobis outliers: {m_mask.sum()}")
print(f"DBSCAN outliers: {d_mask.sum()}")
print(f"KPCA outliers: {k_mask.sum()}\n")

print("Sample Mahalanobis outliers:")
print(df.loc[m_mask, num_cols].head(), "\n")
print("Sample DBSCAN outliers:")
print(df.loc[d_mask, num_cols].head(), "\n")
print("Sample KPCA outliers:")
print(df.loc[k_mask, num_cols].head())

# Create asks to see overlap.
m_and_d = np.logical_and(m_mask, d_mask)
m_and_k = np.logical_and(m_mask, k_mask)
d_and_k = np.logical_and(d_mask, k_mask)
all_three = m_mask & d_mask & k_mask

# In case you want to see the overlap between the methods.
print("Overlap counts:")
print(f"Mahalanobis and DBSCAN: {m_and_d.sum()}")
print(f"Mahalanobis and KPCA: {m_and_k.sum()}")
print(f"DBSCAN and KPCA: {d_and_k.sum()}")
print(f"All three methods: {all_three.sum()}")