import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import os, sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

from src.ingestion.ingestorFactory import IngestorFactory
from src.fairness.no_influence import compute_fairness_metrics

# load data
link = "https://huggingface.co/datasets/scikit-learn/adult-census-income"
ingestor_factory = IngestorFactory(link, 0)
ingestor = ingestor_factory.create()
df = ingestor.load_data()

# specify target and sensitive columns
target_col = "income"
positive_class = ">50K"
sens_cols = ["sex", "race"]
influence_group_col = "education"
positive_group = "HS-grad"


# initialize model
y = (df[target_col] == positive_class).astype(int)
X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)

X_train, X_test, y_train, y_test, _, s_test_df, group_train, _ = train_test_split(
    X,
    y,
    df[sens_cols],
    df[influence_group_col],
    test_size=0.2,
    stratify=y,
    random_state=912,
)

model = LogisticRegression(fit_intercept=False, max_iter=100)
model.fit(X_train.values, y_train.values)

# compute all metrics including influence‐based fairness
no_influence, influence = compute_fairness_metrics(
    X_train,
    X_test,
    y_train,
    y_test,
    s_test_df,
    sens_cols,
    model,
    group_train,
    positive_group,
)

print("Influence-group summary:")
for key, val in influence.items():
    print(f"{key}: {val}")
print()

# then print the per-feature fairness metrics
print("Per-sensitive-feature fairness metrics:")
for rec in no_influence:
    for k, v in rec.items():
        print(f"{k}: {v}")
    print()
