import os, sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)
import pandas as pd
from sklearn.metrics import f1_score
from src.quality.no_influence import mahalanobis_outliers
from src.quality.with_influence import influence_outliers
from src.model.train import train_model



def compare_outlier_removals(
    X_train,
    X_test,
    y_train,
    y_test,
    num_cols,
    model,
    alpha=0.01,
    sigma_multiplier=1.0,
    model_type="logistic",
):
    """
    Compare F₁ performance of a baseline model versus
    models retrained after statistical and influence-based outlier removal.

    Args:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Test feature set.
        y_train (pd.Series): Training target labels.
        y_test (pd.Series): Test target labels.
        num_cols (list[str]): Names of numeric columns for Mahalanobis analysis.
        model: A model instance used by the influence_outliers function to compute influence.
        alpha (float): Significance level for the χ² cutoff in Mahalanobis outlier detection.
        sigma_multiplier (float): Multiplier for the influence‐based outlier threshold.
        model_type (str): Identifier passed to `train_model` to select and configure the classifier.

    Returns:
        pd.DataFrame: Single-row DataFrame with rounded F₁ scores for:
            - "f1_orig": baseline model on original data
            - "f1_statistic": model after removing Mahalanobis outliers
            - "f1_influence": model after removing influence‐based outliers
    """
    n_classes = y_test.nunique()
    average = "binary" if n_classes == 2 else "weighted"

    # train baseline on original data and get f1 score
    baseline = train_model(X_train, y_train, model_type)
    f1_orig = f1_score(y_test, baseline.predict(X_test), average=average)

    # train without statistical outliers and get f1
    stat_mask_full = mahalanobis_outliers(X_train, X_test, num_cols, alpha=alpha)
    stat_train_mask = pd.Series(stat_mask_full[: len(X_train)], index=X_train.index)
    X_train_stat = X_train.loc[~stat_train_mask]
    y_train_stat = y_train.loc[~stat_train_mask]
    stat_clf = train_model(X_train_stat, y_train_stat, model_type)
    f1_statistic = f1_score(y_test, stat_clf.predict(X_test), average=average)

    # train without influence outliers and get f1
    infl_mask_full = influence_outliers(
        X_train,
        X_test,
        y_train,
        y_test,
        model,
        frac=0.001,
        random_state=912,
        sigma_multiplier=sigma_multiplier,
    )
    infl_train_mask = pd.Series(infl_mask_full[: len(X_train)], index=X_train.index)
    X_train_infl = X_train.loc[~infl_train_mask]
    y_train_infl = y_train.loc[~infl_train_mask]
    infl_clf = train_model(X_train_infl.values, y_train_infl.values, model_type)
    f1_influence = f1_score(y_test, infl_clf.predict(X_test.values), average=average)

    return pd.DataFrame(
        [
            {
                "f1_orig": round(f1_orig, 4),
                "f1_statistic": round(f1_statistic, 4),
                "f1_influence": round(f1_influence, 4),
            }
        ]
    )
