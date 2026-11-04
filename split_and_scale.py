import pandas as pd
import numpy as np

#We will use the object oriented approach here.
class StandardScaler:
    def __init__(self):
        #Did this to mimic scikit learn's NotFitterError
        self.mean_ = None
        self.scale_ = None
    def fit(self, X):
        X_array = np.array(X, dtype = float)
        self.mean_ = np.mean(X_array, axis = 0)
        self.scale_ = np.std(X_array, axis = 0)
        self.scale_[self.scale_ == 0] = 1.0
        return self
    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("The scaler has not been fitted yet!")
        X_array = np.array(X, dtype = float)
        return (X_array - self.mean_ )/self.scale_
    def fit_transform(self, X):
        return self.fit(X).transform(X)


df = pd.read_csv("processed_data.csv", parse_dates=['Date'])

train_df = df[df['Date'].dt.year <= 2018].copy()
val_df = df[df['Date'].dt.year == 2019].copy()
test_df = df[df['Date'].dt.year > 2019].copy()

train_df = train_df.groupby('Ticker').head(-5)
val_df = val_df.groupby('Ticker').head(-5)

train_df.to_csv("train_raw.csv", index = False)
val_df.to_csv("val_raw.csv", index = False)
test_df.to_csv("test_raw.csv", index = False)
print("Raw CSVs exported.")

features = ['DailyLogReturn', '5dayCumReturn', 'Vol_5d', 'Vol_21d', 'IntradayRange', 'VolumeRatio']
target = 'Target_Vol_5d'

scaler = StandardScaler()
train_scaled_df = train_df.copy()
val_scaled_df = val_df.copy()
test_scaled_df = test_df.copy()

train_scaled_df[features] = scaler.fit_transform(train_df[features])
val_scaled_df[features] = scaler.transform(val_df[features])
test_scaled_df[features] = scaler.transform(test_df[features])

train_scaled_df.to_csv("train_scaled.csv", index = False)
val_scaled_df.to_csv("val_scaled.csv", index = False)
test_scaled_df.to_csv("test_scaled.csv", index = False)
print("Scaled CSVs exported.")