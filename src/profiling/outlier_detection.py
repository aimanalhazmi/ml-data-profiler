## basic outlier detection method without using influence function. Support methods: zscore, iqr
import numpy as np


class OutlierDetector:
    def __init__(self, method="zscore", threshold=3):
        self.method = method
        self.threshold = threshold

    def detect(self, X):
        if self.method == "zscore":
            z = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
            return np.where(z > self.threshold)
        elif self.method == "iqr":
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            mask = (X < (Q1 - self.threshold * IQR)) | (X > (Q3 + self.threshold * IQR))
            return np.where(mask)
        else:
            raise ValueError("Unknown method")
