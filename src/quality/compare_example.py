import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os, sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

from ingestion.ingestorFactory import IngestorFactory
from compare import compare_outlier_removals
from model.train import train_model

# load full dataset
link = "https://huggingface.co/datasets/scikit-learn/adult-census-income"
ingestor_factory = IngestorFactory(link, 0)
ingestor = ingestor_factory.create()
df = ingestor.load_data()

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
target_col = "income"
positive_class = ">50K"

y = (df[target_col] == positive_class).astype(int)
X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=912)
model = train_model(X_train.values, y_train.values, 'logistic')

result_df = compare_outlier_removals(X_train, X_test, y_train, y_test, num_cols, model, model_type='logistic')
print(result_df)
