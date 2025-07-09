import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.fairness.with_influence import evaluate_patterns
from src.utils.output import print_pattern_table
from src.influence.logistic_influence import LinearSVMInfluence
from src.model.train import train_model

# === 1. read data ===
current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, "..", "..", "tests", "Dataset", "adult.csv")
df = pd.read_csv(csv_path)

# === 2. Data cleaning ===
df = df.replace("?", np.nan).dropna()

# select attribute
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


# labeling data
y = (df["income"] == ">50K").astype(int).values

# scaling
df.drop("income", axis=1, inplace=True)
df_encoded = pd.get_dummies(df, columns=categorical_cols)
scaler = StandardScaler()
X_numerical = scaler.fit_transform(df_encoded)

X_index = df.index
X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(
    X_numerical, y, X_index, test_size=0.2, random_state=42
)


model = train_model(X_train, y_train, "svm")
# logistic_influence = LogisticInfluence(model, X_train, y_train)
svm_influence = LinearSVMInfluence(model, X_train, y_train)

# X_train_infl = logistic_influence.average_influence(X_test[:10], y_test[:10])
X_train_infl = svm_influence.average_influence(X_test[:5], y_test[:5])

X_train_raw = df.loc[train_index].copy().reset_index(drop=True)

print(X_train_raw)

X_train_raw["influence"] = X_train_infl
top_patterns = evaluate_patterns(
    X_train_raw, min_support=0.05, top_k=5, max_predicates=1
)

print_pattern_table(top_patterns)
