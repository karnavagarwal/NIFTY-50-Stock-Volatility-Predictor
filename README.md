# NIFTY-50 Stock Volatility Prediction

> **ES335: Machine Learning Course Project**  
> **Authors:** Karnav Agarwal (25110158) & Shubham Ruparel (25110278)  
> **Interactive Report & Visualizations:** [GitHub Pages Web App](https://karnavagarwal.github.io/NIFTY-50-Stock-Volatility-Predictor/)

---

## 1. Project Overview & Motivation

This project investigates whether tree-based machine learning models (Random Forest and XGBoost) and regularized linear baselines (Ridge Regression), **engineered entirely from scratch in NumPy without `scikit-learn`**, can accurately forecast 5-day forward annualized equity volatility on major constituents of the Indian **NIFTY-50 Index**.

We benchmark static models against **Walk-Forward (rolling window) retrained models** to evaluate how effectively adaptive machine learning handles real-world non-stationary regime shifts—most notably during the unprecedented **March 2020 COVID-19 crash** and the **2022 global inflationary interest-rate shock**.

---

## 2. Setup and Rebuild Steps

### Prerequisites
- Python 3.10+ (tested on Python 3.10 and 3.11)
- `pip` package manager

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/karnavagarwal/NIFTY-50-Stock-Volatility-Predictor.git
cd NIFTY-50-Stock-Volatility-Predictor
pip install -r requirements.txt
```

The environment strictly requires:
- `numpy==2.4.3`
- `pandas==2.3.3`
- `yfinance==1.7.0`
- `matplotlib==3.11.2`
- `seaborn==0.13.2`

### One-Command Full Pipeline Rebuild
To run the full end-to-end extraction, feature engineering, data purging/scaling, model training, rolling retrains, and visual evaluations in exact chronological order:

```bash
python run_pipeline.py
```

This single master command automatically executes:
1. Historical market data extraction (`fetch_data.py`).
2. Feature engineering and rolling return computations (`preprocess_data.py`).
3. 5-day temporal boundary purging and zero-leakage standard scaling (`split_and_scale.py`).
4. Baseline model generation: Naive Persistence and Ridge Regression.
5. Static Random Forest and XGBoost training (2008–2019 train $\to$ 2020–2026 test).
6. Walk-Forward rolling window retrained models (6-month adaptation intervals).
7. Evaluation metrics calculation (RMSE, MAE, Pseudo-Huber loss, MDA, $R^2$) and loss plot figure generation (`evaluate_models.py`).

---

## 3. Code Location & Repository Architecture

All algorithm implementations, preprocessing transformations, and evaluation logic are organized in modular, transparent scripts:

```
NIFTY-50-Stock-Volatility-Predictor/
├── fetch_data.py                      # Data extraction from NSE via yfinance API
├── preprocess_data.py                 # Feature engineering (Target_Vol_5d, Vol_21d, Volume ratios)
├── split_and_scale.py                 # 5-day boundary purging & zero-leakage StandardScaler
├── linear_regression_baseline.py      # Closed-form L2 Ridge Regression solver
├── linear_regression_prediction.py    # Ridge model training and test predictions
├── naive_persistence.py               # Naive persistence baseline definition
├── naive_persistence_predictions.py   # Baseline martingale evaluation (Vol_5d t -> t+5)
├── random_forest.py                   # From-scratch Regression Trees & Random Forest ensemble
├── random_forest_prediction.py        # Static Random Forest pipeline
├── random_forst_rolling_prediction.py # Walk-forward rolling retrained Random Forest
├── random_forest_validation.py        # Hyperparameter grid-search on validation partition
├── xg_boost.py                        # From-scratch XGBoost with Pseudo-Huber loss
├── xgboost_prediction.py              # Static XGBoost pipeline
├── xgboost_rolling_prediction.py      # Walk-forward rolling retrained XGBoost
├── xgboost_validation.py              # XGBoost depth, tree count, and L2 hyperparameter tuning
├── evaluate_models.py                 # Metric evaluation and high-resolution chart exporter
├── export_webdata.py                  # Downsamples real test-set predictions for GitHub Pages
├── run_pipeline.py                    # Master orchestrator script
├── requirements.txt                   # Explicit package dependencies
├── docs/                              # GitHub Pages frontend
│   ├── index.html                     # Full interactive web application & study report
│   ├── results_data.json              # Real evaluation metrics and prediction time series
│   └── images/                        # High-resolution validation and regime loss figures
└── README.md                          # Repository documentation, setup, and disclosures
```

---

## 4. Data Access & Security Statement

### Data Access
- **Source:** Public equity market daily trading records from the **National Stock Exchange (NSE) of India**.
- **Equities Covered:** Five major index-weighted stocks across sectors:
  - Reliance Industries (`RELIANCE.NS`)
  - Tata Consultancy Services (`TCS.NS`)
  - ITC Limited (`ITC.NS`)
  - Maruti Suzuki (`MARUTI.NS`)
  - HDFC Bank (`HDFCBANK.NS`)
- **Time Horizon:** January 1, 2008 to October 23, 2026.
- **Acquisition:** Data is downloaded via the open-source `yfinance` library (`fetch_data.py`).

### Security, Secrets & Privacy Disclosure
- **No Private Data:** All financial records used in this project are public historical market data.
- **No Credentials / API Keys:** The pipeline requires **no authentication tokens, private passwords, paid API keys, or cloud secrets**.
- **Self-Contained & Reproducible:** Datasets are saved as relative local CSV files, and all code can be run independently on any standard machine without external accounts.

---

## 5. AI-Use Disclosure

In accordance with academic integrity guidelines, the use of AI tools in this project was strictly scoped to the following:

1. **Script Checking and Debugging:** Generative AI was used to review Python syntax, verify variable scopes, and assist in debugging script errors.
2. **Graph and Visualization Generation:** Code templates used to render publication-style charts using Matplotlib and Seaborn were created with AI assistance.
3. **Web Interface Development:** The interactive GitHub Pages interface (`docs/index.html` and styling) was developed with AI assistance to present the course project findings and interactive data explorer.
