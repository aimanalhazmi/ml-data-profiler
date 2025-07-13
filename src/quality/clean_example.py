import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os, sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

from src.ingestion.ingestorFactory import IngestorFactory
from src.quality.clean import (
    summarize_outliers,
    drop_influence_outliers,
    drop_statistic_outliers,
)
from src.model.train import train_model

link = "https://huggingface.co/datasets/scikit-learn/adult-census-income"
ingestor_factory = IngestorFactory(link, 0)
ingestor = ingestor_factory.create()
df = ingestor.load_data()

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
target_col = "income"
positive_class = ">50K"

y = (df[target_col] == positive_class).astype(int)
X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=912
)
model = train_model(X_train.values, y_train.values, "logistic")

# How to get the summary dataframe of outlier metrics
outlier_summary = summarize_outliers(
    X_train, X_test, y_train, y_test, model, num_cols, alpha=0.01, sigma_multiplier=1.0
)
print(outlier_summary)

# How to remove statistic outliers
cleaned_mahal = drop_statistic_outliers(X_train, X_test, num_cols)
print(f"\nRows after dropping Mahalanobis outliers: {len(cleaned_mahal)}")

# How to remove influence outliers
cleaned_infl = drop_influence_outliers(
    X_train, X_test, y_train, y_test, model, sigma_multiplier=1.0
)
print(f"Rows after dropping Influence outliers: {len(cleaned_infl)}")
