import pandas as pd
import numpy as np
from random_forest import RandomForest

def mse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    residual = y_true - y_pred
    mse = np.sum(np.square(residual)) / np.size(residual)
    return mse


CHOSEN_DEPTH = 5   # Placeholder, update after checking the graph
CHOSEN_SPLIT = 2   # Placeholder, update after checking the graph


print(f"Loading raw train, validation and test data (Depth: {CHOSEN_DEPTH}, Min Split: {CHOSEN_SPLIT})...")
train_df = pd.read_csv("train_raw.csv", parse_dates=['Date'])
val_df = pd.read_csv("val_raw.csv", parse_dates=['Date'])
test_df = pd.read_csv("test_raw.csv", parse_dates=['Date'])

feature_cols = ['DailyLogReturn', '5dayCumReturn', 'Vol_5d', 'Vol_21d', 'IntradayRange', 'VolumeRatio']
target_col = 'Target_Vol_5d'

# Extract X and y
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_val = val_df[feature_cols].values
y_val = val_df[target_col].values
X_test = test_df[feature_cols].values
y_test = test_df[target_col].values

print("\nRetraining final Random Forest model on full 2008-2019 dataset...")
# Concatenating for ES 335 defensibility
X_train_full = np.vstack((X_train, X_val))
y_train_full = np.concatenate((y_train, y_val))

# Upping n_estimators to 30 for the final stable prediction
final_model = RandomForest(
    n_estimators=30, 
    max_depth=CHOSEN_DEPTH, 
    min_sample_split=CHOSEN_SPLIT, 
    random_state=42
)
final_model.fit(X_train_full, y_train_full)

# Test Evaluation & Predictions
print("\nGenerating predictions on test data with best model...")
test_predictions = final_model.predict(X_test)
test_score = mse(y_test, test_predictions)
print(f"Test Set MSE: {test_score:.6f} (RMSE: {np.sqrt(test_score):.6f})")

# Save results
output_df = test_df[['Date', 'Ticker', 'Target_Vol_5d']].copy()
output_df.rename(columns={'Target_Vol_5d': 'Actual_Volatility'}, inplace=True)
output_df['Predicted_Volatility'] = test_predictions

output_file = "random_forest_output.csv"
output_df.to_csv(output_file, index=False)
print(f"Predictions successfully generated and saved to {output_file}")