import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from src.quality.clean import drop_statistic_outliers, drop_influence_outliers
from src.model.train import train_model

import os, sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)


def evaluate_f1(df, target_col, positive_class):
    y = (df[target_col] == positive_class).astype(int)
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=912
    )
    model = train_model(X_train.values, y_train.values, "logistic")
    return f1_score(y_test, model.predict(X_test))


def compare_outlier_removals(
    df, num_cols, target_col, positive_class, model="logistic"
):
    df_stat = drop_statistic_outliers(df, num_cols)
    # NO TRAIN
    df_infl = drop_influence_outliers(df, target_col, positive_class, model=model)

    f1_orig = evaluate_f1(df, target_col, positive_class)
    f1_statistic = evaluate_f1(df_stat, target_col, positive_class)
    f1_influence = evaluate_f1(df_infl, target_col, positive_class)

    return pd.DataFrame(
        [
            {
                "f1_orig": f1_orig,
                "f1_statistic": f1_statistic,
                "f1_influence": f1_influence,
            }
        ]
    )
