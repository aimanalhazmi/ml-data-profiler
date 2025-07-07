import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os, sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

from ingestion.ingestorFactory import IngestorFactory
from no_influence import mahalanobis_outliers, dbscan_outliers
from with_influence import influence_outliers
from model.train import train_model
# Import data

# Future: Preprocessor
link = "https://huggingface.co/datasets/scikit-learn/adult-census-income"
ingestor_factory = IngestorFactory(link, 0)
ingestor = ingestor_factory.create()
df = ingestor.load_data()

# Select numeric columns, in the future, all but the target will be numeric.
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

target_col = "income"
positive_class=">50K"

y = (df[target_col] == positive_class).astype(int)
X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=912)
model = train_model(X_train.values, y_train.values, 'logistic')

# ------ No Influence Outliers ------ #

m_mask = mahalanobis_outliers(X_train, X_test, num_cols, alpha=0.01)
d_mask = dbscan_outliers(X_train, X_test, num_cols, eps=0.5, min_samples=5)

# ------ Influence Outliers ------ #

inf_mask = influence_outliers(X_train, X_test, y_train, y_test, model, frac = 0.001, random_state= 912, sigma_multiplier = 1.0)

# ------ Analysis ------ #

mask_df = pd.DataFrame({
    "Mahalanobis": m_mask,
    "DBSCAN": d_mask,
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
print("\n--- Sample Influence outliers ---")
print(df.loc[inf_mask, num_cols].head())

print("\n--- Raw Overlap Counts ---")
m_and_d = m_mask & d_mask
m_and_i = m_mask & inf_mask
d_and_i = d_mask & inf_mask
all_three = m_mask & d_mask & inf_mask

print(f"Mahalanobis & DBSCAN: {m_and_d.sum()}")
print(f"Mahalanobis & Influence: {m_and_i.sum()}")
print(f"DBSCAN & Influence: {d_and_i.sum()}")
print(f"All three methods: {all_three.sum()}")