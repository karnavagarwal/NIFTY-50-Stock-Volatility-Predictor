import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linear_regression_baseline import Ridge

def mean_square_error(y_true,y_pred):
    y_true = np.array(y_true,dtype = float)
    y_pred = np.array(y_pred,dtype = float)

    residuals = y_true - y_pred
    r_square = np.square(residuals)
    mse = np.sum(r_square)/np.size(r_square)

    return mse

# We need to train the linear the linear regression model
print("Loading scaled train, validation and test data...")
train_df = pd.read_csv("train_scaled.csv",parse_dates=['Date'])
val_df = pd.read_csv("val_scaled.csv",parse_dates=['Date'])
test_df = pd.read_csv("test_scaled.csv", parse_dates=['Date'])

feature_cols = ['Vol_5d','IntradayRange','VolumeRatio']
target_col = 'Target_Vol_5d'

# Extract X and y
X_train = train_df[feature_cols]
X_val = val_df[feature_cols]
X_test = test_df[feature_cols]
y_train = train_df[target_col].values
y_val = val_df[target_col].values
y_test = test_df[target_col].values

# Grid search for hyperparameter as we want to only tune one hyperparameter
lambda_for_val = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0,5000.0,6000.0,7000.0,10000.0]
best_lambda = None
best_val_mse = float('inf')
best_model = None
mse_history = [] # To store different MSE values for different lambda

# Validation loop
for lam in lambda_for_val:
    # Train model with current lambda
    model = Ridge(l2_penalty=lam)
    model.fit(X_train,y_train)

    # Predictions on validation set
    val_predictions = model.predict(X_val)

    # Calculate error
    val_mse = mean_square_error(y_val,val_predictions)
    mse_history.append(val_mse)
    print(f"Lambda: {lam:7.3f} | Validation MSE: {val_mse:.6f}") 

    # Keep track of best performing model
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_lambda = lam
        best_model = model

print(f"\nSelected Best Lambda: {best_lambda} with MSE: {best_val_mse:.6f}")

print("Generating predictions on test data with best model decided based on validation loop")
predictions = best_model.predict(X_test)

# Save results
output_df = test_df[['Date', 'Ticker', 'Target_Vol_5d']].copy()
output_df.rename(columns = {'Target_Vol_5d': 'Actual_Volatility'}, inplace = True)
output_df['Predicted_Volatility'] = predictions

output_df.to_csv("linear_regression_output.csv", index = False)
print("Predictions successfully generated and saved to linear_regression_output.csv")
print("Learned Weights (Intercept first, then features):")
print(best_model.theta_)

plt.figure(figsize=(8,5))
plt.xscale('log')
plt.plot(lambda_for_val,mse_history,marker = 'o',linestyle = '-',color = 'b',linewidth = 2)

# Highlight the best lambda with a red dot
plt.plot(best_lambda, best_val_mse, marker='o', color='red', markersize=8, label=f'Best: {best_lambda}')

plt.title('Hyperparameter Tuning: Validation MSE vs. Lambda')
plt.xlabel('Lambda (L2 Penalty)')
plt.ylabel('Validation Mean Squared Error (MSE)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save the plot as an image
plt.savefig("Outputs/lambda_tuning_curve.png")

# Display the plot on screen
plt.show()
