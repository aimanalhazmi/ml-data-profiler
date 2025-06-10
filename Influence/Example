from sklearn.linear_model import LogisticRegression
from Influence.logistic_influence import LogisticInfluence
import numpy as np

X_train = np.random.rand(20, 5)
y_train = np.random.randint(0, 2, size=20)
X_test = X_train[0:4]
y_test = y_train[0:4]

model = LogisticRegression().fit(X_train, y_train)

influencer = LogisticInfluence(model, X_train, y_train)
influences1 = influencer.get_influence(X_test[0], y_test[0])
influences2 = influencer.average_influence(X_test, y_test)
print(influences1)
print(influences2)

top_k_idx1 = np.argsort(influences1)[-5:]
top_k_idx2 = np.argsort(influences2)[-5:]

print("Top 5 influential samples:", top_k_idx1)
print("Top 5 batch influential samples:", top_k_idx2)