import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random_forest import RandomForest

def mse(y_true,y_pred):
    y_true = np.array(y_true,dtype = float)
    y_pred = np.array(y_pred,dtype = float)
    residual = y_true - y_pred
    mse = np.sum(np.square(residual)) / np.size(residual)
    return mse

print("Loading raw train, validation and test data...")
train_df = pd.read_csv("train_raw.csv",parse_dates=['Date'])
val_df = pd.read_csv("val_raw.csv",parse_dates=['Date'])
test_df = pd.read_csv("test_raw.csv",parse_dates=['Date'])

feature_cols = ['DailyLogReturn','5dayCumReturn','Vol_5d','Vol_21d','IntradayRange','VolumeRatio']
target_col = 'Target_Vol_5d'

# Extract X and y
X_train = train_df[feature_cols].values
X_val = val_df[feature_cols].values
X_test = test_df[feature_cols].values
y_train = train_df[target_col].values
y_val = val_df[target_col].values
y_test = test_df[target_col].values

# Grid search parameters
depths_for_val = [3,5,7]
splits_for_val = [2,5,10]

best_depth = None
best_split = None
best_val_mse = float('inf')
best_model = None

# Store dictionary of hyperparameters with MSE
mse_history = {}

print("Tuning max_depth and min_sample_split on validation data...")
# Grid search
for depth in depths_for_val:
    for split in splits_for_val:

        # Train model with current hyperparameters
        model = RandomForest(n_estimators=10,max_depth=depth,min_sample_split=split,random_state=42)
        model.fit(X_train, y_train)

        # Prediction on validation data
        val_predictions = model.predict(X_val)

        # Calculate error and update dictionary
        val_mse = mse(y_val,val_predictions)
        mse_history[(depth,split)] = val_mse
        print(f"Depth: {depth:2d} | Min Split: {split:2d} | Val MSE: {val_mse:.6f}")

        # keep track for best model
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_depth = depth
            best_split = split
            best_model = model

print(f"\nSelected Best Model - Depth: {best_depth}, Split: {best_split} with MSE: {best_val_mse:.6f}")

# Test Evaluation & Predictions
print("Generating predictions on test data with best model...")
test_predictions = best_model.predict(X_test)
test_score = mse(y_test, test_predictions)
print(f"Test Set MSE: {test_score:.6f} (RMSE: {np.sqrt(test_score):.6f})")

# Save results
output_df = test_df[['Date', 'Ticker', 'Target_Vol_5d']].copy()
output_df.rename(columns={'Target_Vol_5d': 'Actual_Volatility'}, inplace=True)
output_df['Predicted_Volatility'] = test_predictions

output_df.to_csv("random_forest_output.csv", index=False)
print("Predictions successfully generated and saved to random_forest_output.csv")

# Plotting the 2D Validation Curve
plt.figure(figsize=(10, 6))

# Plot a separate line for each min_samples_split value
for split in splits_for_val:
    # Extract the MSEs for this specific split value across all depths
    mses_for_this_split = [mse_history[(d, split)] for d in depths_for_val]
    plt.plot(depths_for_val, mses_for_this_split, marker='o', label=f'Min Split: {split}')

# Highlight the absolute best combination with a red star
plt.plot(best_depth, best_val_mse, marker='*', color='red', markersize=14, 
         label=f'Best (Depth {best_depth}, Split {best_split})', linestyle='None')

plt.title('Hyperparameter Tuning: Tree Depth vs. Min Samples Split')
plt.xlabel('Max Depth')
plt.ylabel('Validation MSE')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save the plot to the output folder ignored by Git
plt.savefig("Outputs/rf_2d_tuning_curve.png")

# Display the plot on screen
plt.show()