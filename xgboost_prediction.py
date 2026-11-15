import numpy as np
import pandas as pd
from xg_boost import XGBoost
#values chosen from the validation graph. chosen depth and lambda are the argmin values, but the number of trees is the elbow in the graph.
CHOSEN_DEPTH = 3
CHOSEN_TREES = 30
CHOSEN_LAMBDA = 10.0

print(f"Loading data for final test run (Depth: {CHOSEN_DEPTH}, Trees: {CHOSEN_TREES}, Lambda: {CHOSEN_LAMBDA})...")
train_df = pd.read_csv("train_raw.csv", parse_dates=['Date'])
val_df = pd.read_csv("val_raw.csv", parse_dates=['Date']) # Added this
test_df = pd.read_csv("test_raw.csv", parse_dates=['Date'])

feature_cols = ['DailyLogReturn', '5dayCumReturn', 'Vol_5d', 'Vol_21d', 'IntradayRange', 'VolumeRatio']
target_col = 'Target_Vol_5d'

# Extract arrays
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_val = val_df[feature_cols].values       # Added this
y_val = val_df[target_col].values         # Added this
X_test = test_df[feature_cols].values

# Concatenate the time-series
X_train_full = np.vstack((X_train, X_val))
y_train_full = np.concatenate((y_train, y_val))

print("\nTraining final XGBoost model on combined 2008-2019 data...")
final_model = XGBoost(
    n_estimators=CHOSEN_TREES,
    learning_rate=0.1,
    max_depth=CHOSEN_DEPTH,
    lambda_=CHOSEN_LAMBDA,
    gamma=0.1,
    delta=0.1
)

# Train on the concatenated data
final_model.fit(X_train_full, y_train_full)

print("\nGenerating predictions on test data...")
predictions = final_model.predict(X_test)

output_df = test_df[['Date', 'Ticker', 'Target_Vol_5d']].copy()
output_df.rename(columns={'Target_Vol_5d': 'Actual_Volatility'}, inplace=True)
output_df['Predicted_Volatility'] = predictions

output_file = "xgboost_output.csv"
output_df.to_csv(output_file, index=False)
print(f"Output successfully saved to {output_file}! Number of rows: {len(output_df)}")