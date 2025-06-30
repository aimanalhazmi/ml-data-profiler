import numpy as np
import pandas as pd

import os, sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

from ingestion.ingestorFactory import IngestorFactory
from clean import summarize_outliers, drop_influence_outliers, drop_statistic_outliers

link = "https://huggingface.co/datasets/scikit-learn/adult-census-income"
ingestor_factory = IngestorFactory(link, 0)
ingestor = ingestor_factory.create()
df = ingestor.load_data()

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
target_col = "income"
positive_class = ">50K"

# How to get the summary dataframe of outlier metrics
outlier_summary = summarize_outliers(df, num_cols, target_col, positive_class, alpha=0.01, sigma_multiplier = 3.0)
print(outlier_summary)

# How to remove statistic outliers
cleaned_mahal = drop_statistic_outliers(df, num_cols)
print(f"\nRows after dropping Mahalanobis outliers: {len(cleaned_mahal)}")

# How to remove influence outliers
cleaned_infl = drop_influence_outliers(df, target_col, positive_class, sigma_multiplier=1.0) #dont forget you can use model = 'logistic', 'svm' here
print(f"Rows after dropping Influence outliers: {len(cleaned_infl)}")