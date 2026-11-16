import pandas as pd
import numpy as np
from xg_boost import XGBoost

CHOSEN_DEPTH = 3
CHOSEN_TREES = 30
CHOSEN_LAMBDA = 10.0

print("Loading data for Walk-Forward Optimization...")
train_df = pd.read_csv("train_raw.csv", parse_dates=['Date'])
val_df = pd.read_csv("val_raw.csv", parse_dates=['Date'])
test_df = pd.read_csv("test_raw.csv", parse_dates=['Date'])

feature_cols = ['DailyLogReturn', '5dayCumReturn', 'Vol_5d', 'Vol_21d', 'IntradayRange', 'VolumeRatio']
target_col = 'Target_Vol_5d'

# Combine everything chronologically for the rolling window
full_df = pd.concat([train_df, val_df, test_df]).sort_values('Date').reset_index(drop=True)

test_start_date = pd.to_datetime('2020-01-01')
max_date = test_df['Date'].max()

current_start = test_start_date
predictions_list = []

print(f"\nStarting 6-Month Rolling Predictions for Custom XGBoost...")
while current_start <= max_date:
    # 6-month test window boundary
    current_end = current_start + pd.DateOffset(months=6) - pd.Timedelta(days=1)
    
    train_mask = full_df['Date'] < current_start
    test_mask = (full_df['Date'] >= current_start) & (full_df['Date'] <= current_end)
    
    if not test_mask.any():
        current_start += pd.DateOffset(months=6)
        continue
        
    X_train = full_df.loc[train_mask, feature_cols].values
    y_train = full_df.loc[train_mask, target_col].values
    X_test = full_df.loc[test_mask, feature_cols].values
    
    print(f"Training on data up to {current_start.date()} | Predicting: {current_start.date()} to {current_end.date()} (N={len(X_test)})")
    
    model = XGBoost(
        n_estimators=CHOSEN_TREES,
        learning_rate=0.1,
        max_depth=CHOSEN_DEPTH,
        lambda_=CHOSEN_LAMBDA,
        gamma=0.1,
        delta=0.1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    temp_out = full_df.loc[test_mask, ['Date', 'Ticker', 'Target_Vol_5d']].copy()
    temp_out.rename(columns={'Target_Vol_5d': 'Actual_Volatility'}, inplace=True)
    temp_out['Predicted_Volatility'] = preds
    predictions_list.append(temp_out)
    
    current_start += pd.DateOffset(months=6)

output_df = pd.concat(predictions_list).sort_values('Date').reset_index(drop=True)
output_file = "xgboost_rolling_output.csv"
output_df.to_csv(output_file, index=False)
print(f"\nWalk-forward predictions successfully saved to {output_file}! Total rows: {len(output_df)}")