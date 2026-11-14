import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xg_boost import XGBoost
import os

def pseudo_huber_loss(y_true, y_pred, delta = 1.0):
    y_true = np.array(y_true, dtype = float)
    y_pred = np.array(y_pred, dtype = float)
    a = y_true - y_pred
    return np.mean(delta**2 * (np.sqrt(1 + (a/delta)**2)-1))

print("Loading data...")
train_df = pd.read_csv("train_raw.csv", parse_dates = ['Date'])
val_df = pd.read_csv("val_raw.csv", parse_dates = ['Date'])
test_df = pd.read_csv("test_raw.csv", parse_dates = ['Date'])

feature_cols = ['DailyLogReturn', '5dayCumReturn', 'Vol_5d', 'Vol_21d', 'IntradayRange', 'VolumeRatio']
target_col = 'Target_Vol_5d'

X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_val = val_df[feature_cols].values
y_val = val_df[target_col].values
X_test = test_df[feature_cols].values

estimators_grid = [10, 20, 30, 40, 50]
depth_grid = [2, 3, 4, 5, 6]
lambda_grid = [0.1, 1.0, 10.0, 100.0, 1000.0]

best_val_loss = float('inf')
best_model = None
best_params = {}
results = []

print("Starting Hyperparameter Tuning...")
print("\n")

for d in depth_grid:
    for n in estimators_grid:
        for lam in lambda_grid:
            model = XGBoost(
                n_estimators=n,
                learning_rate=0.1,
                max_depth = d,
                lambda_ = lam,
                gamma = 0.1,
                delta = 0.1
            )
            model.fit(X_train, y_train)
            val_predictions = model.predict(X_val)
            val_loss = pseudo_huber_loss(y_val, val_predictions, delta = 0.1)
            print(f"Depth: {d} | Trees: {n} | Lambda: {lam:4.1f} ||| Val. Huber Loss: {val_loss:.6f}")
            results.append({'depth': d, 'trees': n, 'lambda': lam, 'loss': val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {'max_depth': d, 'n_estimators': n, 'lambda': lam}
                best_model = model

print("\n")
print(f"Selected the best paramters: {best_params} with validation loss: {best_val_loss:.6f}")
print("\n")

print("Generating visuals")
results_df = pd.DataFrame(results)

fig, axes = plt.subplots(1, len(depth_grid), figsize=(18, 6), sharey=True)
fig.suptitle('Validation: Pseudo-Huber Loss vs. Trees & Lambda', fontsize=14)

for i, d in enumerate(depth_grid):
    ax = axes[i]
    subset_d = results_df[results_df['depth'] == d]
    for lam in lambda_grid:
        subset_lam = subset_d[subset_d['lambda'] == lam]
        ax.plot(subset_lam['trees'], subset_lam['loss'], marker='o', linestyle='-', label=f'Lambda: {lam}')
    ax.set_title(f'Max Depth: {d}')
    ax.set_xlabel('Number of trees')
    ax.set_xticks(estimators_grid)
    ax.grid(True, linestyle='--', alpha=0.7)
    if i == 0:
        ax.set_ylabel('Validation Pseudo-Huber Loss')
    ax.legend()

best_ax_idx = depth_grid.index(best_params['max_depth'])
axes[best_ax_idx].plot(
    best_params['n_estimators'], best_val_loss, 
    marker='o', color='red', markersize=10, 
    label=f"Best Model", zorder=5
)
axes[best_ax_idx].legend()
plt.tight_layout()

os.makedirs("Outputs", exist_ok=True)
plt.savefig("Outputs/xgboost_tuning_curve.png")
print("Saved image to Outputs/xgboost_tuning_curve.png")

print('Generating predictions on testing data...')
predictions = best_model.predict(X_test)
output_df = test_df[['Date', 'Ticker', 'Target_Vol_5d']].copy()
#Renaming for convenience
output_df.rename(columns={'Target_Vol_5d': 'Actual_Volatility'}, inplace = True)
output_df['Predicted_Volatility'] = predictions
output_file = "xgboost_output.csv"
output_df.to_csv(output_file, index = False)
print(f"Ouput succesfully saved to {output_file}! Number of rows: {len(output_df)}")