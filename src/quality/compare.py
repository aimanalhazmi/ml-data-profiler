import pandas as pd
from sklearn.metrics import f1_score

import os, sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

from no_influence import mahalanobis_outliers
from with_influence import influence_outliers
from model.train import train_model

# Trains and evaluates on original data, then again without statistical outliers and then once more without influence based outliers
# model must be trained before. model_type is the model on which we calculate f1
def compare_outlier_removals(X_train, X_test, y_train, y_test, num_cols, model, alpha = 0.01, sigma_multiplier = 1.0, model_type='logistic'):
    # train baseline on original data and get f1 score
    baseline = train_model(X_train.values, y_train.values, model_type)
    f1_orig = f1_score(y_test, baseline.predict(X_test.values))

    # train without statistical outliers and get f1
    stat_mask_full = mahalanobis_outliers(X_train, X_test, num_cols, alpha=alpha)
    stat_train_mask = pd.Series(stat_mask_full[:len(X_train)], index=X_train.index)
    X_train_stat = X_train.loc[~stat_train_mask]
    y_train_stat = y_train.loc[~stat_train_mask]
    stat_clf = train_model(X_train_stat.values, y_train_stat.values, model_type)
    f1_statistic = f1_score(y_test, stat_clf.predict(X_test.values))

    # train without influence outliers and get f1
    infl_mask_full = influence_outliers(X_train, X_test, y_train, y_test, model, frac = 0.001, random_state= 912, sigma_multiplier = sigma_multiplier)
    infl_train_mask = pd.Series(infl_mask_full[:len(X_train)], index=X_train.index)
    X_train_infl = X_train.loc[~infl_train_mask]
    y_train_infl = y_train.loc[~infl_train_mask]
    infl_clf = train_model(X_train_infl.values, y_train_infl.values, model_type)
    f1_influence = f1_score(y_test, infl_clf.predict(X_test.values))

    return pd.DataFrame([{
        'f1_orig': f1_orig,
        'f1_statistic': f1_statistic,
        'f1_influence': f1_influence
    }])