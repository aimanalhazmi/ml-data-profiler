import pandas as pd
from sklearn.linear_model import LogisticRegression

import os, sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

from src.ingestion.ingestorFactory import IngestorFactory
from src.fairness.no_influence import (
    compute_fairness_metrics,
)  # our new combined function

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
model = LogisticRegression(fit_intercept=False, max_iter=1000)

# compute all metrics including influence‐based fairness
no_influence, influence = compute_fairness_metrics(
    df,
    target_col=target_col,
    sens_cols=sens_cols,
    model=model,
    positive_class=positive_class,
    influence_group_col=influence_group_col,
    positive_group=positive_group,
    test_size=0.2,
    random_state=912,
    frac=0.05,
    dpd_tol=0.1,
    eod_tol=0.1,
    ppv_tol=0.1,
    influence_tol=0.05,
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
