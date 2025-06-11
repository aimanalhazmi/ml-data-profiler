import pandas as pd
import numpy as np
## basic empty detection method without using influence function.
class empty_detection:
    def __init__(self):
        pass

    def report(self, df: pd.DataFrame):
        return df.isnull().sum() / len(df)

    def detect(self, df: pd.DataFrame, threshold=0.2):
        return df.columns[df.isnull().mean() > threshold].tolist()

