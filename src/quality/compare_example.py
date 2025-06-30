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
