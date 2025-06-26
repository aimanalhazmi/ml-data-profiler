import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame
)

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

# Future: Target column
y = 'income'
X = pd.get_dummies(df.drop(columns=[y]), drop_first=True)

# Future: Sensitive Columns
sens_cols = ['sex', 'race']

# Future: import model not to be wasteful!
model = LogisticRegression(max_iter=1000)


def compute_fairness_metrics_no_influence(df, target_col, sens_cols, model, test_size: float = 0.2, random_state: int = 912) -> pd.DataFrame:
    y = (df[target_col] == df[target_col].unique().max()).astype(int)
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)

    # SHould we doe the test/tran split here? Probably its better to receive the data as a parameter.
    X_train, X_test, y_train, y_test, s_train_df, s_test_df = train_test_split(
        X, y, df[sens_cols],
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # The model could already be fitted
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute DPD, and EOD for each sensitive column. Compute PPV for each category inside each sensitive column.
    records = []
    for sens in sens_cols:
        sf_test = s_test_df[sens]

        dpd = demographic_parity_difference(y_true=y_test, y_pred=y_pred, sensitive_features=sf_test)
        eod = equalized_odds_difference(y_true=y_test, y_pred=y_pred, sensitive_features=sf_test)
        mf = MetricFrame(metrics=precision_score, y_true=y_test, y_pred=y_pred, sensitive_features=sf_test)

        ppv_by_group = mf.by_group.to_dict()
        ppv_diff = mf.difference()

        records.append({
            "sensitive_feature": sens,
            "DPD": dpd,
            "EOD": eod,
            "PPV_diff": ppv_diff,
            "PPV_by_group": ppv_by_group
        })

    return records

# Example usage
results = compute_fairness_metrics_no_influence(df, y, sens_cols, model)

for r in results:
    print(f"Demographic Parity Difference: {r['DPD']:.4f}")
    print(f"Equalized Odds Difference:    {r['EOD']:.4f}")
    print("Positive Predictive Value by group:", r['PPV_by_group'])