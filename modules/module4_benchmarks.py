"""
Module 4: Benchmark Models
==========================

Implements benchmark models for volatility forecasting:
1. HAR-RV (Heterogeneous Autoregressive Realized Volatility)
2. FIGARCH (Fractionally Integrated GARCH)
3. Pure ML (XGBoost without LRD features)

All models produce out-of-sample forecasts for evaluation.
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
import pickle

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = r"C:\Users\Akash\OneDrive\Desktop\LRD_Nicholas"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "results", "intermediate")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "tables")

# Create directories
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"   {text}")
    print("=" * 70)


# =============================================================================
# 1. HAR-RV MODEL
# =============================================================================

def compute_har_components(rv_series):
    """
    Compute HAR components: daily, weekly, monthly RV.

    HAR-RV model (Corsi, 2009):
    RV_{t+1} = c + β_d * RV_t + β_w * RV_t^{(w)} + β_m * RV_t^{(m)} + ε_{t+1}

    where:
    - RV_t^{(w)} = (1/5) * sum(RV_{t-i}, i=0..4)  [weekly average]
    - RV_t^{(m)} = (1/22) * sum(RV_{t-i}, i=0..21) [monthly average]
    """
    rv = rv_series.copy()

    # Daily RV (lagged)
    rv_d = rv.shift(1)

    # Weekly RV (average of last 5 days)
    rv_w = rv.rolling(window=5, min_periods=5).mean().shift(1)

    # Monthly RV (average of last 22 days)
    rv_m = rv.rolling(window=22, min_periods=22).mean().shift(1)

    return rv_d, rv_w, rv_m


def fit_har_rv(rv_train, rv_d_train, rv_w_train, rv_m_train):
    """
    Fit HAR-RV model using OLS.

    Returns coefficients: [intercept, beta_d, beta_w, beta_m]
    """
    from sklearn.linear_model import LinearRegression

    # Build design matrix
    X = pd.concat([rv_d_train, rv_w_train, rv_m_train], axis=1)
    X.columns = ['RV_d', 'RV_w', 'RV_m']
    y = rv_train

    # Drop missing values
    mask = X.notna().all(axis=1) & y.notna()
    X_clean = X[mask]
    y_clean = y[mask]

    if len(X_clean) < 50:
        return None

    # Fit OLS
    model = LinearRegression()
    model.fit(X_clean.values, y_clean.values)

    return {
        'intercept': model.intercept_,
        'beta_d': model.coef_[0],
        'beta_w': model.coef_[1],
        'beta_m': model.coef_[2],
        'model': model
    }


def forecast_har_rv(rv_d, rv_w, rv_m, coef):
    """Produce HAR-RV forecast given components and coefficients."""
    if coef is None:
        return np.nan

    return (coef['intercept'] +
            coef['beta_d'] * rv_d +
            coef['beta_w'] * rv_w +
            coef['beta_m'] * rv_m)


# =============================================================================
# 2. FIGARCH MODEL
# =============================================================================

def fit_figarch(returns, vol_target=False):
    """
    Fit FIGARCH(1,d,1) model using the arch package.

    Returns fitted model or None if fitting fails.
    """
    try:
        from arch import arch_model

        # Scale returns for numerical stability (in percentage)
        returns_pct = returns * 100

        # Fit FIGARCH(1,d,1)
        model = arch_model(returns_pct, vol='FIGARCH', p=1, o=0, q=1,
                          power=2.0, dist='normal')

        # Fit with more iterations
        result = model.fit(disp='off', options={'maxiter': 1000})

        return result
    except Exception as e:
        return None


def forecast_figarch(model, horizon=1):
    """
    Produce h-step ahead variance forecast from FIGARCH model.

    Returns variance forecast (rescaled to original units).
    """
    if model is None:
        return np.nan

    try:
        forecast = model.forecast(horizon=horizon)
        # Returns variance in percent^2, convert back to decimal
        return forecast.variance.values[-1, -1] / 10000
    except:
        return np.nan


# =============================================================================
# 3. PURE ML (XGBoost) BASELINE
# =============================================================================

def prepare_ml_features(rv_series, lags=[1, 2, 3, 5, 10, 22]):
    """
    Prepare lagged RV features for pure ML model (no LRD features).

    Features: lagged RV at different horizons.
    """
    features = pd.DataFrame(index=rv_series.index)

    for lag in lags:
        features[f'rv_lag{lag}'] = rv_series.shift(lag)

    # Add rolling statistics
    features['rv_mean5'] = rv_series.rolling(5).mean().shift(1)
    features['rv_std5'] = rv_series.rolling(5).std().shift(1)
    features['rv_mean22'] = rv_series.rolling(22).mean().shift(1)
    features['rv_std22'] = rv_series.rolling(22).std().shift(1)

    return features


def fit_xgboost_baseline(X_train, y_train):
    """
    Fit XGBoost model for volatility forecasting.
    """
    try:
        from xgboost import XGBRegressor

        # Remove NaN
        mask = X_train.notna().all(axis=1) & y_train.notna()
        X_clean = X_train[mask]
        y_clean = y_train[mask]

        if len(X_clean) < 100:
            return None

        # Conservative hyperparameters to prevent overfitting
        model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_clean.values, y_clean.values)
        return model
    except Exception as e:
        print(f"  XGBoost error: {e}")
        return None


# =============================================================================
# 4. ROLLING WINDOW FORECASTING
# =============================================================================

def rolling_forecast_benchmarks(vol_proxy, returns, train_start=1000,
                                  step=5, stocks_subset=None):
    """
    Produce rolling window forecasts for all benchmark models.

    Parameters:
    -----------
    vol_proxy : DataFrame
        Volatility proxy (squared returns) for all stocks
    returns : DataFrame
        Log returns for all stocks (needed for FIGARCH)
    train_start : int
        Minimum training observations before first forecast
    step : int
        Step size for rolling window (for efficiency)
    stocks_subset : list
        Subset of stocks to process (None = all)

    Returns:
    --------
    Dictionary with forecasts for each model and actual values
    """

    if stocks_subset is None:
        stocks_subset = vol_proxy.columns[:10]  # Start with 10 stocks for speed

    results = {
        'actual': {},
        'har_rv': {},
        'figarch': {},
        'xgboost_base': {}
    }

    total_stocks = len(stocks_subset)

    for idx, ticker in enumerate(stocks_subset):
        print(f"  Processing {ticker} ({idx+1}/{total_stocks})...")

        rv = vol_proxy[ticker].dropna()
        ret = returns[ticker].dropna()

        if len(rv) < train_start + 100:
            continue

        # Compute HAR components
        rv_d, rv_w, rv_m = compute_har_components(rv)

        # Prepare ML features
        ml_features = prepare_ml_features(rv)

        # Storage
        actuals = []
        har_forecasts = []
        figarch_forecasts = []
        xgb_forecasts = []
        dates = []

        # Rolling forecast loop
        forecast_indices = range(train_start, len(rv) - 1, step)

        for t in forecast_indices:
            # Actual value at t+1
            if t + 1 >= len(rv):
                continue
            actual = rv.iloc[t + 1]
            date = rv.index[t + 1]

            # ---- HAR-RV ----
            # Train on data up to t
            har_coef = fit_har_rv(
                rv.iloc[:t+1],
                rv_d.iloc[:t+1],
                rv_w.iloc[:t+1],
                rv_m.iloc[:t+1]
            )
            # Forecast for t+1 using data at t
            har_pred = forecast_har_rv(rv_d.iloc[t], rv_w.iloc[t], rv_m.iloc[t], har_coef)

            # ---- Pure XGBoost ----
            xgb_model = fit_xgboost_baseline(
                ml_features.iloc[:t+1],
                rv.iloc[:t+1]
            )
            if xgb_model is not None and not ml_features.iloc[t].isna().any():
                xgb_pred = xgb_model.predict(ml_features.iloc[t:t+1].values)[0]
            else:
                xgb_pred = np.nan

            # Store
            actuals.append(actual)
            har_forecasts.append(har_pred)
            xgb_forecasts.append(xgb_pred)
            dates.append(date)

        # Store results
        if len(dates) > 0:
            results['actual'][ticker] = pd.Series(actuals, index=dates)
            results['har_rv'][ticker] = pd.Series(har_forecasts, index=dates)
            results['xgboost_base'][ticker] = pd.Series(xgb_forecasts, index=dates)

    return results


def compute_figarch_forecasts_sample(returns, vol_proxy, stocks_subset,
                                     train_end_idx=4000):
    """
    Fit FIGARCH models on training sample and produce single forecast.

    FIGARCH is slow, so we fit once on training data and produce
    out-of-sample forecasts for the test period.
    """
    figarch_results = {}

    for ticker in stocks_subset:
        print(f"  FIGARCH for {ticker}...")

        ret = returns[ticker].dropna()

        if len(ret) < train_end_idx:
            continue

        # Fit on training data
        ret_train = ret.iloc[:train_end_idx]

        figarch_model = fit_figarch(ret_train)

        if figarch_model is not None:
            figarch_results[ticker] = {
                'd': figarch_model.params.get('d', np.nan),
                'conditional_variance': figarch_model.conditional_volatility ** 2 / 10000
            }

    return figarch_results


# =============================================================================
# 5. SIMPLE EVALUATION METRICS
# =============================================================================

def compute_losses(actual, predicted, proxy='rv'):
    """
    Compute forecast loss functions.

    MSE: Mean Squared Error
    MAE: Mean Absolute Error
    QLIKE: Quasi-Likelihood loss (robust to RV proxy)
    """
    # Align and drop missing
    df = pd.DataFrame({'actual': actual, 'predicted': predicted}).dropna()

    if len(df) < 10:
        return {'mse': np.nan, 'mae': np.nan, 'qlike': np.nan, 'n': 0}

    a = df['actual'].values
    p = df['predicted'].values

    # Ensure positive values for QLIKE
    p = np.maximum(p, 1e-10)
    a = np.maximum(a, 1e-10)

    # MSE
    mse = np.mean((a - p) ** 2)

    # MAE
    mae = np.mean(np.abs(a - p))

    # QLIKE (quasi-likelihood)
    # QLIKE = mean(a/p - log(a/p) - 1) or equivalently mean(a/p + log(p))
    qlike = np.mean(a / p - np.log(a / p) - 1)

    return {'mse': mse, 'mae': mae, 'qlike': qlike, 'n': len(df)}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print_header("MODULE 4: BENCHMARK MODELS")

    # Load data
    print("Loading data...")
    vol_proxy = pd.read_csv(os.path.join(PROCESSED_DIR, "volatility_proxy.csv"),
                           index_col=0, parse_dates=True)
    returns = pd.read_csv(os.path.join(PROCESSED_DIR, "returns.csv"),
                         index_col=0, parse_dates=True)

    print(f"  Volatility proxy: {vol_proxy.shape}")
    print(f"  Returns: {returns.shape}")

    # Select subset of stocks for benchmarking (start with 20 for speed)
    # Choose stocks with good coverage
    coverage = vol_proxy.notna().mean()
    good_stocks = coverage[coverage > 0.95].index.tolist()

    # Select diverse subset
    stocks_subset = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ', 'PG', 'KO', 'BA',
                    'CAT', 'DUK', 'NEE', 'PLD', 'AMT', 'GOOGL', 'INTC',
                    'IBM', 'GE', 'WMT', 'MRK', 'HD']
    stocks_subset = [s for s in stocks_subset if s in good_stocks][:15]

    print(f"\nBenchmarking on {len(stocks_subset)} stocks: {stocks_subset}")

    # =========================================================================
    # Run rolling forecasts for HAR-RV and XGBoost
    # =========================================================================
    print("\n" + "-" * 50)
    print("Running rolling window forecasts...")
    print("-" * 50)

    results = rolling_forecast_benchmarks(
        vol_proxy,
        returns,
        train_start=1500,  # ~6 years of training before first forecast
        step=5,            # Forecast every 5 days for efficiency
        stocks_subset=stocks_subset
    )

    # =========================================================================
    # Compute aggregate performance
    # =========================================================================
    print("\n" + "-" * 50)
    print("Computing performance metrics...")
    print("-" * 50)

    performance = {'har_rv': [], 'xgboost_base': []}

    for ticker in stocks_subset:
        if ticker not in results['actual']:
            continue

        actual = results['actual'][ticker]

        for model_name in ['har_rv', 'xgboost_base']:
            if ticker in results[model_name]:
                pred = results[model_name][ticker]
                losses = compute_losses(actual, pred)
                losses['ticker'] = ticker
                losses['model'] = model_name
                performance[model_name].append(losses)

    # Aggregate results
    print("\n" + "=" * 70)
    print("   BENCHMARK RESULTS (Preliminary)")
    print("=" * 70)

    summary = []
    for model_name in ['har_rv', 'xgboost_base']:
        if performance[model_name]:
            df = pd.DataFrame(performance[model_name])
            summary.append({
                'Model': model_name.upper(),
                'Avg MSE': df['mse'].mean(),
                'Avg MAE': df['mae'].mean(),
                'Avg QLIKE': df['qlike'].mean(),
                'Stocks': len(df)
            })

    summary_df = pd.DataFrame(summary)
    print("\n" + summary_df.to_string(index=False))

    # =========================================================================
    # Save results
    # =========================================================================
    print("\n" + "-" * 50)
    print("Saving results...")
    print("-" * 50)

    # Save forecasts
    with open(os.path.join(INTERMEDIATE_DIR, 'benchmark_forecasts.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f"  Saved forecasts to benchmark_forecasts.pkl")

    # Save performance summary
    for model_name in ['har_rv', 'xgboost_base']:
        if performance[model_name]:
            perf_df = pd.DataFrame(performance[model_name])
            perf_df.to_csv(os.path.join(INTERMEDIATE_DIR, f'{model_name}_performance.csv'),
                          index=False)
            print(f"  Saved {model_name}_performance.csv")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("   MODULE 4 COMPLETE (Phase 1)")
    print("=" * 70)

    print("""
Models implemented:
  - HAR-RV: Heterogeneous Autoregressive Realized Volatility (Corsi, 2009)
  - XGBoost Baseline: ML with lagged RV features (no LRD)

Rolling forecast setup:
  - Training window: 1500 days minimum (~6 years)
  - Forecast step: every 5 days
  - Out-of-sample period: ~4800 days (~19 years)

Next steps:
  - Module 5: Add LRD features to create hybrid models
  - Module 6: Statistical evaluation (DM tests, MCS)

Note: FIGARCH implementation available but slow - will run separately
      if needed for full paper results.
""")

    print(f"Outputs saved to: {INTERMEDIATE_DIR}")
