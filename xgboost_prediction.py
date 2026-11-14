import pandas as pd
from xg_boost import XGBoost
#From the xgboost_tuning_curve.jpg file in the outputs folder, we have selected the following parameters.
#Even though no of trees were 50 in the lowest one, 30 was the point of elbow.
#Which means that after 30 trees, adding more trees did not add have much impact in our model.
CHOSEN_DEPTH = 3
CHOSEN_TREES = 30
CHOSEN_LAMBDA = 10.0

print(f"Loading data for final test run (Depth: {CHOSEN_DEPTH}, Trees: {CHOSEN_TREES}, Lambda: {CHOSEN_LAMBDA})...")
train_df = pd.read_csv("train_raw.csv", parse_dates=['Date'])
test_df = pd.read_csv("test_raw.csv", parse_dates=['Date'])

feature_cols = ['DailyLogReturn', '5dayCumReturn', 'Vol_5d', 'Vol_21d', 'IntradayRange', 'VolumeRatio']
target_col = 'Target_Vol_5d'

X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test = test_df[feature_cols].values

print("\nTraining final XGBoost model...")
final_model = XGBoost(
    n_estimators=CHOSEN_TREES,
    learning_rate=0.1,
    max_depth=CHOSEN_DEPTH,
    lambda_=CHOSEN_LAMBDA,
    gamma=0.1,
    delta=0.1
)

final_model.fit(X_train, y_train)

print("\nGenerating predictions on test data...")
predictions = final_model.predict(X_test)

output_df = test_df[['Date', 'Ticker', 'Target_Vol_5d']].copy()
output_df.rename(columns={'Target_Vol_5d': 'Actual_Volatility'}, inplace=True)
output_df['Predicted_Volatility'] = predictions

output_file = "xgboost_output.csv"
output_df.to_csv(output_file, index=False)
print(f"Output successfully saved to {output_file}! Number of rows: {len(output_df)}")