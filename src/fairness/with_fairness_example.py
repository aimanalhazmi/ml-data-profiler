import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.fairness.with_influence import evaluate_patterns
from src.influence import LogisticInfluence


def pattern_to_readable(pattern, columns):
    readable = []
    for col_idx, val in pattern.items():
        col_name = columns[col_idx] if isinstance(col_idx, int) else col_idx
        if isinstance(val, pd.Interval):
            readable.append(f"{col_name} ∈ {val}")
        else:
            readable.append(f"{col_name} = {val}")
    return readable


# === 1. read data ===
current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, '..', '..', 'tests', 'Dataset', 'adult.csv')
df = pd.read_csv(csv_path)

# === 2. Data cleaning ===
df = df.replace('?', np.nan).dropna()

# select attribute
categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']


# labeling data
y = (df['income'] == '>50K').astype(int).values

# scaling
scaler = StandardScaler()
X_numerical = scaler.fit_transform(df[numerical_cols])
X_index = df.index
X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X_numerical, y, X_index, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=500).fit(X_train, y_train)

logistic_influence = LogisticInfluence(model, X_train, y_train)
X_train_infl = logistic_influence.average_influence(X_test[:5], y_test[:5])

X_train_raw = df.loc[train_index].copy().reset_index(drop=True)
print(X_train_raw)
X_train_raw['influence'] = X_train_infl
X_train_raw = X_train_raw.drop(columns=["income"])

X_train_raw['predicted_label'] = model.predict(X_train)
X_train_raw['true_label'] = y_train

top_patterns = evaluate_patterns(X_train_raw, min_support=0.05, top_k=14)

for i, p in enumerate(top_patterns):

    print(f"Pattern {i + 1}:")
    readable = pattern_to_readable(p['pattern'], df.columns)
    for cond in readable:
        print("  -", cond)
    print(
        f"  Support: {p['support']:.2%}, Responsibility: {p['responsibility']:.3f}, Interestingness: {p['interestingness']:.3f}")