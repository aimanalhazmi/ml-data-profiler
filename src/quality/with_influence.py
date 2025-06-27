import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import os, sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC)

from ingestion.ingestorFactory import IngestorFactory
from influence.logistic_influence import LogisticInfluence
# Import data

# Future: Preprocessor
link = "https://huggingface.co/datasets/scikit-learn/adult-census-income"
ingestor_factory = IngestorFactory(link, 0)
ingestor = ingestor_factory.create()
df = ingestor.load_data()


# Future: Target column
y = (df["income"] == ">50K").astype(int)
X = pd.get_dummies(df.drop(columns=["income"]), drop_first=True)

# Test/Train split will be passed as a parameter
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=912)

# Future: import model not to be wasteful! Or pass it as a parameter.
model = LogisticRegression(max_iter=1000, fit_intercept=False)
model.fit(X_train, y_train)

# Detect outliers as the points that deviate by 3 standard deviations from the mean average influence.
# X_train, y_train, X_test and y_test should be passed as parameters.
# sigma_multiplier is how many standard deviations our outliers deviate.
# frac now controls what fraction of the *test* set we sample (to speed up the influence loop).
def influence_outliers(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    frac: float = 1.0,             # fraction of the TEST data to sample
    sigma_multiplier: float = 3.0  # how many σ above the mean to flag
):
    # optionally subsample the *test* set to save memory/time
    if frac < 1.0:
        X_te = X_test.sample(frac=frac, random_state=912)
        y_te = y_test.loc[X_te.index]
    else:
        X_te = X_test
        y_te = y_test

    # Calculate logistic influence on the FULL training set
    X_vals = X_train.values.astype(np.float64)
    y_vals = y_train.values.astype(np.float64)
    print("Number of test points for influence:", len(X_te), '\n')
    infl = LogisticInfluence(model, X_vals, y_vals)

    # Compute average influence only over the (sampled) TEST set
    avg_inf = infl.average_influence(
        X_te.values.astype(np.float64),
        y_te.values.astype(np.float64)
    )

    # Establish the statistical threshold
    mu = avg_inf.mean()
    sigma = avg_inf.std()
    thresh = max(mu + sigma_multiplier * sigma, 0.0) # Keeping just the positive tail

    # Create the mask of train‐indices whose influence exceeds threshold
    mask = avg_inf > thresh
    idxs = np.where(mask)[0]
    orig_idxs = X_train.index[idxs]

    return mask, idxs, orig_idxs, mu, sigma, thresh

# Example usage:
# here frac=0.1 samples 10% of the TEST set when computing influence
mask, idxs, orig_idxs, mu, sigma, thresh = influence_outliers(
    model, X_train, y_train, X_test, y_test,
    frac=0.001, sigma_multiplier=3.0
)

print(f"Mean influence = {mu:.4e}, σ = {sigma:.4e}")
print(f"Threshold (μ + 3σ) = {thresh:.4e}")
print(f"Found {len(idxs)} influence‐based outliers\n")

print("Sample of influence‐based outliers from X_train:")
print(X_train.loc[orig_idxs].head(), "\n")

