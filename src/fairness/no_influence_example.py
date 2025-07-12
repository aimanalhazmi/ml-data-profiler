import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os, sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, SRC)

from src.ingestion.ingestorFactory import IngestorFactory
from src.fairness.no_influence import compute_fairness_classical_metrics, compute_fairness_influence_metrics
from src.influence.logistic_influence import LogisticInfluence
from src.model.train import train_model


# load data
link = "https://huggingface.co/datasets/scikit-learn/adult-census-income"
ingestor_factory = IngestorFactory(link, 0)
ingestor = ingestor_factory.create()
df = ingestor.load_data()
df = df.replace("?", np.nan).dropna()

# specify target and sensitive columns
target_col = "income"
sens_cols = ["sex", "race"]
influence_group_col = "race"
positive_group = "Black"

categorical_cols = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]
numerical_cols = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]

y = (df["income"] == ">50K").astype(int).values

# scaling
df.drop("income", axis=1, inplace=True)
df_encoded = pd.get_dummies(df, columns=categorical_cols)
scaler = StandardScaler()
X_numerical = scaler.fit_transform(df_encoded)

X_index = df.index

X_train, X_test, y_train, y_test, _, s_test_df, train_index, _= train_test_split(
    X_numerical,
    y,
    df[sens_cols],
    X_index,
    test_size=0.2,
    stratify=y,
    random_state=912,
)

model = train_model(X_train, y_train, "logistic")

no_influence = compute_fairness_classical_metrics(
    X_test,
    y_test,
    s_test_df,
    sens_cols,
    model,
)

print("Per-sensitive-feature fairness metrics:")
for rec in no_influence:
    for k, v in rec.items():
        print(f"{k}: {v}")
    print()

logistic_influence = LogisticInfluence(model, X_train, y_train)
X_train_infl = logistic_influence.average_influence(X_test[:10], y_test[:10])
X_train_raw = df.loc[train_index].copy().reset_index(drop=True)
print(X_train_raw)
X_train_raw["influence"] = X_train_infl

influence = compute_fairness_influence_metrics(X_train_raw, influence_group_col, positive_group)


print("Influence-group summary:")
for key, val in influence.items():
    print(f"{key}: {val}")
print()


