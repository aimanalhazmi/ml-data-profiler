import numpy as np
import pandas as pd

import os, sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

from ingestion.ingestorFactory import IngestorFactory
from no_influence import mahalanobis_outliers, dbscan_outliers, kpca_outliers
from with_influence import influence_outliers
# Import data

# Future: Preprocessor
link = "https://huggingface.co/datasets/scikit-learn/adult-census-income"
ingestor_factory = IngestorFactory(link, 0)
ingestor = ingestor_factory.create()
df = ingestor.load_data()

# Select numeric columns, in the future, all but the target wil be numeric.
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ------ No Influence Outliers ------ #
# Example usage
m_mask = mahalanobis_outliers(df, num_cols, alpha=0.01)
d_mask = dbscan_outliers(df, num_cols, eps=0.5, min_samples=5)
k_mask = kpca_outliers(df, num_cols, frac=0.1)

# ------ Influence Outliers ------ #

# Future: get from ingestor, i guess.
target_col = "income"
positive_class=">50K"

inf_mask = influence_outliers(df, target_col=target_col, positive_class=positive_class, frac=0.01, test_size=0.2, random_state=912, sigma_multiplier=1.0)

# ------ Analysis ------ #

mask_df = pd.DataFrame({
    "Mahalanobis": m_mask,
    "DBSCAN": d_mask,
    "KPCA": k_mask,
    "Influence": inf_mask
}, index=df.index)

# Print individual outlier counts
print("\n--- Outlier counts per method ---")
print(mask_df.sum())

# Sample view of detected outliers from each method
print("\n--- Sample Mahalanobis outliers ---")
print(df.loc[m_mask, num_cols].head())
print("\n--- Sample DBSCAN outliers ---")
print(df.loc[d_mask, num_cols].head())
print("\n--- Sample KPCA outliers ---")
print(df.loc[k_mask, num_cols].head())
print("\n--- Sample Influence outliers ---")
print(df.loc[inf_mask, num_cols].head())


def jaccard(a, b):
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return inter / uni if uni else 0.0

print("\n--- Pairwise Jaccard similarities ---")
for col1 in mask_df.columns:
    for col2 in mask_df.columns:
        if col1 >= col2:
            continue
        score = jaccard(mask_df[col1], mask_df[col2])
        print(f"{col1} vs {col2}: Jaccard = {score:.3f}")

print("\n--- Raw Overlap Counts ---")
m_and_d = m_mask & d_mask
m_and_k = m_mask & k_mask
m_and_i = m_mask & inf_mask
d_and_k = d_mask & k_mask
d_and_i = d_mask & inf_mask
k_and_i = k_mask & inf_mask
all_four = m_mask & d_mask & k_mask & inf_mask

print(f"Mahalanobis & DBSCAN: {m_and_d.sum()}")
print(f"Mahalanobis & KPCA: {m_and_k.sum()}")
print(f"Mahalanobis & Influence: {m_and_i.sum()}")
print(f"DBSCAN & KPCA: {d_and_k.sum()}")
print(f"DBSCAN & Influence: {d_and_i.sum()}")
print(f"KPCA & Influence: {k_and_i.sum()}")
print(f"All four methods: {all_four.sum()}")