import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk") 

def pseudo_huber_loss(y_true, y_pred, delta=0.1):
    a = y_true - y_pred
    return np.mean(delta**2 * (np.sqrt(1 + (a/delta)**2) - 1))

def mean_directional_accuracy(y_true, y_pred):
    actual_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    return np.mean(actual_direction == pred_direction)

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

print("Loading model predictions...")
models = {}
color_map = {}

# 1. Static ML Models
try:
    models['XGBoost (Static)'] = pd.read_csv("xgboost_output.csv", parse_dates=['Date'])
    color_map['XGBoost (Static)'] = '#1f77b4' 
except FileNotFoundError: pass

try:
    models['Random Forest (Static)'] = pd.read_csv("random_forest_output.csv", parse_dates=['Date'])
    color_map['Random Forest (Static)'] = '#ff7f0e' 
except FileNotFoundError: pass

# 2. Rolling (Walk-Forward) ML Models
try:
    models['XGBoost (Rolling)'] = pd.read_csv("xgboost_rolling_output.csv", parse_dates=['Date'])
    color_map['XGBoost (Rolling)'] = '#aec7e8' 
except FileNotFoundError: print("Warning: xgboost_rolling_output.csv not found.")

try:
    models['Random Forest (Rolling)'] = pd.read_csv("random_forest_rolling_output.csv", parse_dates=['Date'])
    color_map['Random Forest (Rolling)'] = '#ffbb78' 
except FileNotFoundError: print("Warning: random_forest_rolling_output.csv not found.")

# 3. Baselines
try:
    models['Ridge Regression'] = pd.read_csv("linear_regression_output.csv", parse_dates=['Date'])
    color_map['Ridge Regression'] = '#2ca02c' 
except FileNotFoundError: pass

try:
    models['Naive Persistence'] = pd.read_csv("naive_persistence_output.csv", parse_dates=['Date'])
    color_map['Naive Persistence'] = '#d62728' 
except FileNotFoundError: pass

results = []

for name, df in models.items():
    y_true = df['Actual_Volatility'].values
    y_pred = df['Predicted_Volatility'].values
    
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    huber = pseudo_huber_loss(y_true, y_pred, delta=0.1)
    mda = mean_directional_accuracy(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    
    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'Pseudo-Huber': huber,
        'MDA (%)': mda * 100,
        'Out-of-Sample R2': r2
    })

results_df = pd.DataFrame(results).set_index('Model')
print("\n" + "="*85)
print("FINAL OUT-OF-SAMPLE EVALUATION (2020-2026)")
print("="*85)
print(results_df.round(4).to_string())
print("="*85)

# Organize Outputs neatly
os.makedirs("outputs/Metrics", exist_ok=True)
plot_colors = [color_map[m] for m in results_df.index]

# ---------------------------------------------------------
# Visual 1 & 2: Bar Charts (Errors and Predictive Power)
# ---------------------------------------------------------
fig1, axes1 = plt.subplots(1, 3, figsize=(22, 7))
fig1.suptitle('Model Error Metrics (Global Portfolio)', fontsize=18, fontweight='bold', y=1.05)
for i, metric in enumerate(['RMSE', 'MAE', 'Pseudo-Huber']):
    ax = axes1[i]
    bars = ax.bar(results_df.index, results_df[metric], color=plot_colors, edgecolor='black', linewidth=1.5)
    ax.set_title(metric, fontsize=14, fontweight='bold')
    ax.set_ylabel('Error')
    ax.tick_params(axis='x', rotation=25)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig("outputs/Metrics/model_comparison_errors.png", dpi=300, bbox_inches='tight')
plt.close()

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
fig2.suptitle('Model Predictive Power (Global Portfolio)', fontsize=18, fontweight='bold', y=1.05)
for i, metric in enumerate(['MDA (%)', 'Out-of-Sample R2']):
    ax = axes2[i]
    bars = ax.bar(results_df.index, results_df[metric], color=plot_colors, edgecolor='black', linewidth=1.5)
    ax.set_title(metric, fontsize=14, fontweight='bold')
    if metric == 'MDA (%)':
        ax.set_ylabel('Accuracy (%)')
        ax.axhline(50, color='black', linestyle='--', alpha=0.7, label='Coin Flip Baseline')
        ax.legend()
    else:
        ax.set_ylabel('R-Squared')
        ax.axhline(0, color='black', linestyle='--', alpha=0.7, label='Mean Baseline')
        ax.legend()
    ax.tick_params(axis='x', rotation=25)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig("outputs/Metrics/model_comparison_power.png", dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Visuals 3+: Isolated Plots for EVERY Ticker and EVERY Model
# ---------------------------------------------------------
if not models:
    print("No model outputs found! Exiting graphing loop.")
else:
    ref_df = list(models.values())[0]
    tickers = ref_df['Ticker'].unique()

    print(f"\nGenerating dozens of isolated time-series plots for {len(tickers)} tickers across {len(models)} models...")

    start_date_covid = pd.to_datetime('2020-01-01')
    end_date_covid = pd.to_datetime('2020-07-31')

    for ticker in tickers:
        safe_ticker = str(ticker).replace(".", "_")
        print(f"Creating subfolders and visuals for {ticker}...")
        
        # Create dedicated subfolders for this specific ticker
        os.makedirs(f"outputs/Overall/{safe_ticker}", exist_ok=True)
        os.makedirs(f"outputs/Covid_Crash/{safe_ticker}", exist_ok=True)
        
        for name, df in models.items():
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
            model_color = color_map[name]
            
            mask_all = df['Ticker'] == ticker
            mask_covid = (df['Date'] >= start_date_covid) & (df['Date'] <= end_date_covid) & (df['Ticker'] == ticker)
            
            if not mask_all.any():
                continue
            
            # 1. Overall Plot (Isolated)
            plt.figure(figsize=(16, 7))
            plt.plot(df[mask_all]['Date'], df[mask_all]['Actual_Volatility'], label='Actual Volatility', color='black', linewidth=2.5, alpha=0.8)
            plt.plot(df[mask_all]['Date'], df[mask_all]['Predicted_Volatility'], label=name, color=model_color, linewidth=2, alpha=0.9)
            
            plt.title(f'Overall Predictions (2020-2026) - {name} ({ticker})', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Target 5-Day Volatility', fontsize=14)
            plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=12, loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            # Save into the ticker's subfolder
            plt.savefig(f"outputs/Overall/{safe_ticker}/overall_{safe_name}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Covid Crash Plot (Isolated)
            if mask_covid.any():
                plt.figure(figsize=(14, 7))
                plt.plot(df[mask_covid]['Date'], df[mask_covid]['Actual_Volatility'], label='Actual Volatility', color='black', linewidth=3)
                plt.plot(df[mask_covid]['Date'], df[mask_covid]['Predicted_Volatility'], label=name, color=model_color, linewidth=2.5, linestyle='--')
                
                plt.title(f'Volatility Shock Response (March 2020) - {name} ({ticker})', fontsize=16, fontweight='bold')
                plt.xlabel('Date', fontsize=14)
                plt.ylabel('Target 5-Day Volatility', fontsize=14)
                plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                
                # Save into the ticker's subfolder
                plt.savefig(f"outputs/Covid_Crash/{safe_ticker}/covid_{safe_name}.png", dpi=300, bbox_inches='tight')
                plt.close()

    print("\nEvaluation complete! Isolated visuals are organized by ticker in Outputs/Overall/ and Outputs/Covid_Crash/.")