import numpy as np
import pandas as pd

import os, sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

from influence.logistic_influence import LogisticInfluence

# This flags the really influential rows in your dataset based on a logistic‐influence score.
# It slects those that deviate "sigma_multiplier" standard deviations from the mean.
# frac is just the amount of test points used to calculate influence, preferably low.
# Recieves the split data and the TRAINED model as a parameter
def influence_outliers(X_train, X_test, y_train, y_test, model, frac = 0.001, random_state= 912, sigma_multiplier = 3.0):
    if frac < 1.0:
        X_te = X_test.sample(frac=frac, random_state=random_state)
        y_te = y_test.loc[X_te.index]
    else:
        X_te, y_te = X_test, y_test

    infl = LogisticInfluence(model, X_train.values.astype(np.float64), y_train.values.astype(np.float64))
    avg_inf = infl.average_influence(X_te.values.astype(np.float64), y_te.values.astype(np.float64))

    mu = avg_inf.mean()
    sigma = avg_inf.std()
    thresh = max(mu + sigma_multiplier * sigma, 0.0) # Keep only positive points!

    flagged_idxs = X_train.index[avg_inf > thresh]
    full_index = X_train.index.append(X_test.index)
    full_mask = pd.Series(False, index=full_index)
    full_mask.loc[flagged_idxs] = True

    return full_mask.values