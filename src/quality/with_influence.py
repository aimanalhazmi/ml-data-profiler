import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os, sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

from influence.logistic_influence import LogisticInfluence
from model.train import train_model

# This flags the really influential rows in your dataset based on a logistic‐influence score.
# It slects those that deviate "sigma_multiplier" standard deviations from the mean.
# frac is just the amount of test points used to calculate influence, preferably low.
def influence_outliers(df, target_col, positive_class, frac = 0.01, test_size = 0.2, random_state= 912, sigma_multiplier = 3.0):
    # Turn the target column to a binary column
    y = (df[target_col] == positive_class).astype(int)
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)

    # split into train vs. test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    if frac < 1.0:
        X_te = X_test.sample(frac=frac, random_state=random_state)
        y_te = y_test.loc[X_te.index]
    else:
        X_te, y_te = X_test, y_test

    model = train_model(X_train.values,  y_train.values, "logistic")

    # Outlier calculation
    infl = LogisticInfluence(model, X_train.values.astype(np.float64), y_train.values.astype(np.float64))
    avg_inf = infl.average_influence(X_te.values.astype(np.float64), y_te.values.astype(np.float64))

    mu = avg_inf.mean()
    sigma = avg_inf.std()
    thresh = max(mu + sigma_multiplier * sigma, 0.0) # Kepp only positive points!

    full_mask = pd.Series(False, index=X_train.index)
    flagged_idxs = X_train.index[np.where(avg_inf > thresh)[0]]
    full_mask.loc[flagged_idxs] = True

    return full_mask.reindex(df.index, fill_value=False).values