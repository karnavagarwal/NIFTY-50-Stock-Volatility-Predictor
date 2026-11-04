import pandas as pd
import numpy as np

#The first baseline, naive persistence
#This predicts that the volatility of the next five days is simply the volatility of the previous five days.
class NaivePersistence:
    def __init__(self, feature_name = 'Vol_5d'):
        self.feature_name = feature_name
    def fit(self, X, y = None):
        return self
    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas dataframe.")
        return X[self.feature_name].values