import numpy as np

import os, sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

from ingestion.ingestorFactory import IngestorFactory
from compare import compare_outlier_removals

# load full dataset
df = IngestorFactory("https://huggingface.co/datasets/scikit-learn/adult-census-income", 0).create().load_data()

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
target_col = "income"
positive_class = ">50K"


result_df = compare_outlier_removals(df, num_cols, target_col, positive_class, model='logistic')
print(result_df)

print(f"F1 on original data: {result_df['f1_orig']:.4f}")
print(f"F1 after dropping statistic outliers: {result_df['f1_statistic']:.4f}")
print(f"F1 after dropping influence outliers: {result_df['f1_influence']:.4f}")