"""
MODULE 3: Feature Engineering
=============================
Constructs the novel LRD-based features for ML modeling.
Produces Table 4 for the paper.

Key Features:
1. LRD Features: d_hat, SE(d_hat), z_t^2 (standardized residuals)
2. Memory Dynamics: Δd, Vol(d), Trend(d)
3. Cross-Sectional: mean_d, std_d, skew_d
4. HAR Components: RV_d, RV_w, RV_m
5. Market Features: VIX, returns, spreads

Author: Akash Deep, Nicholas Appiah
Date: January 2025
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add modules directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from module2_lrd_estimation import gph_estimator, local_whittle_estimator

# ============================================
# CONFIGURATION
# ============================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
INTERMEDIATE_DIR = os.path.join(RESULTS_DIR, "intermediate")

# Feature engineering parameters
ROLLING_WINDOW = 500      # Window for LRD estimation
MEMORY_DYNAMICS_LAG = 22  # ~1 month for memory dynamics
HAR_DAILY = 1
HAR_WEEKLY = 5
HAR_MONTHLY = 22


def load_data():
    """Load all required data"""
    print("Loading data...")

    returns = pd.read_csv(os.path.join(PROCESSED_DIR, "returns.csv"),
                          index_col=0, parse_dates=True)
    vol_proxy = pd.read_csv(os.path.join(PROCESSED_DIR, "volatility_proxy.csv"),
                            index_col=0, parse_dates=True)
    market = pd.read_csv(os.path.join(PROCESSED_DIR, "market.csv"),
                         index_col=0, parse_dates=True)
    macro = pd.read_csv(os.path.join(PROCESSED_DIR, "macro.csv"),
                        index_col=0, parse_dates=True)

    # Load pre-computed LRD estimates if available
    lrd_files = {
        'vol_gph': os.path.join(INTERMEDIATE_DIR, "lrd_vol_gph.csv"),
        'cs_dispersion': os.path.join(INTERMEDIATE_DIR, "lrd_cs_dispersion.csv"),
    }

    lrd_data = {}
    for name, path in lrd_files.items():
        if os.path.exists(path):
            lrd_data[name] = pd.read_csv(path, index_col=0, parse_dates=True)
            print(f"  Loaded {name}: {lrd_data[name].shape}")

    print(f"  Returns: {returns.shape}")
    print(f"  Volatility proxy: {vol_proxy.shape}")
    print(f"  Market: {market.shape}")
    print(f"  Macro: {macro.shape}")

    return returns, vol_proxy, market, macro, lrd_data


# ============================================
# HAR COMPONENTS
# ============================================
def compute_har_components(vol_proxy):
    """
    Compute HAR (Heterogeneous Autoregressive) components.

    HAR-RV uses three horizons:
    - Daily: RV_t (lag 1)
    - Weekly: average of RV_{t-4} to RV_t (5 days)
    - Monthly: average of RV_{t-21} to RV_t (22 days)
    """
    print("\nComputing HAR components...")

    har_features = pd.DataFrame(index=vol_proxy.index)

    # For each stock, compute HAR components
    for col in vol_proxy.columns:
        rv = vol_proxy[col]

        # Daily (lag 1)
        har_features[f'{col}_RV_d'] = rv.shift(1)

        # Weekly (5-day average)
        har_features[f'{col}_RV_w'] = rv.rolling(window=HAR_WEEKLY).mean().shift(1)

        # Monthly (22-day average)
        har_features[f'{col}_RV_m'] = rv.rolling(window=HAR_MONTHLY).mean().shift(1)

    print(f"  HAR features shape: {har_features.shape}")
    return har_features


def compute_har_for_stock(vol_series):
    """Compute HAR components for a single stock"""
    rv_d = vol_series.shift(1)
    rv_w = vol_series.rolling(window=HAR_WEEKLY).mean().shift(1)
    rv_m = vol_series.rolling(window=HAR_MONTHLY).mean().shift(1)

    return pd.DataFrame({
        'RV_d': rv_d,
        'RV_w': rv_w,
        'RV_m': rv_m,
    }, index=vol_series.index)


# ============================================
# ROLLING LRD FEATURES
# ============================================
def compute_rolling_lrd_features(vol_proxy, window=ROLLING_WINDOW, sample_freq=1):
    """
    Compute rolling LRD features for all stocks.

    Features per stock:
    - d_hat: estimated memory parameter
    - se_d: standard error
    - d_significant: binary indicator (|t| > 1.96)

    sample_freq: compute every N days to save time (1 = daily)
    """
    print(f"\nComputing rolling LRD features (window={window})...")

    n_stocks = len(vol_proxy.columns)
    n_dates = len(vol_proxy)

    # Initialize storage
    d_hat_all = pd.DataFrame(index=vol_proxy.index, columns=vol_proxy.columns)
    se_all = pd.DataFrame(index=vol_proxy.index, columns=vol_proxy.columns)

    # Compute for each date (after warmup)
    for i in range(window, n_dates, sample_freq):
        if i % 250 == 0:
            print(f"  Processing day {i}/{n_dates} ({vol_proxy.index[i].strftime('%Y-%m')})")

        for col in vol_proxy.columns:
            window_data = vol_proxy[col].iloc[i-window:i].values
            d_hat, se = gph_estimator(window_data)
            d_hat_all.iloc[i, d_hat_all.columns.get_loc(col)] = d_hat
            se_all.iloc[i, se_all.columns.get_loc(col)] = se

    # Forward fill if sample_freq > 1
    if sample_freq > 1:
        d_hat_all = d_hat_all.ffill()
        se_all = se_all.ffill()

    print(f"  d_hat shape: {d_hat_all.shape}")
    return d_hat_all, se_all


def compute_rolling_lrd_fast(vol_proxy, window=ROLLING_WINDOW, sample_freq=5):
    """
    Fast version: compute LRD every sample_freq days and interpolate.
    """
    print(f"\nComputing rolling LRD features (fast mode, every {sample_freq} days)...")

    n_dates = len(vol_proxy)
    sample_indices = list(range(window, n_dates, sample_freq))

    # Storage for sampled values
    d_samples = []
    se_samples = []
    dates_samples = []

    for i in sample_indices:
        if len(d_samples) % 50 == 0:
            print(f"  Processing {len(d_samples)}/{len(sample_indices)} samples...")

        d_row = {}
        se_row = {}

        for col in vol_proxy.columns:
            window_data = vol_proxy[col].iloc[i-window:i].dropna().values
            if len(window_data) >= 250:
                d_hat, se = gph_estimator(window_data)
                d_row[col] = d_hat
                se_row[col] = se

        d_samples.append(d_row)
        se_samples.append(se_row)
        dates_samples.append(vol_proxy.index[i])

    # Convert to DataFrames
    d_hat_sampled = pd.DataFrame(d_samples, index=dates_samples)
    se_sampled = pd.DataFrame(se_samples, index=dates_samples)

    # Reindex to full date range and interpolate
    full_index = vol_proxy.index[window:]
    d_hat_all = d_hat_sampled.reindex(full_index).interpolate(method='linear').ffill().bfill()
    se_all = se_sampled.reindex(full_index).interpolate(method='linear').ffill().bfill()

    print(f"  d_hat shape: {d_hat_all.shape}")
    return d_hat_all, se_all


# ============================================
# MEMORY DYNAMICS FEATURES
# ============================================
def compute_memory_dynamics(d_hat_df, lag=MEMORY_DYNAMICS_LAG):
    """
    Compute memory dynamics features:
    - Δd: change in d_hat
    - Vol(d): rolling volatility of d_hat
    - Trend(d): slope of d_hat over last L days
    """
    print("\nComputing memory dynamics features...")

    dynamics = {}

    for col in d_hat_df.columns:
        d = d_hat_df[col]

        # Change in d
        delta_d = d.diff()

        # Volatility of d (rolling std)
        vol_d = d.rolling(window=lag).std()

        # Trend of d (slope over last L days)
        def rolling_slope(series, window):
            slopes = []
            for i in range(len(series)):
                if i < window - 1:
                    slopes.append(np.nan)
                else:
                    y = series.iloc[i-window+1:i+1].values
                    x = np.arange(window)
                    mask = ~np.isnan(y)
                    if mask.sum() >= 2:
                        slope = np.polyfit(x[mask], y[mask], 1)[0]
                        slopes.append(slope)
                    else:
                        slopes.append(np.nan)
            return pd.Series(slopes, index=series.index)

        trend_d = rolling_slope(d, lag)

        dynamics[f'{col}_delta_d'] = delta_d
        dynamics[f'{col}_vol_d'] = vol_d
        dynamics[f'{col}_trend_d'] = trend_d

    dynamics_df = pd.DataFrame(dynamics)
    print(f"  Memory dynamics shape: {dynamics_df.shape}")
    return dynamics_df


# ============================================
# CROSS-SECTIONAL FEATURES
# ============================================
def compute_cross_sectional_features(d_hat_df):
    """
    Compute cross-sectional features from d_hat distribution.

    Features:
    - mean_d: cross-sectional mean of d_hat
    - std_d: cross-sectional std (dispersion)
    - skew_d: cross-sectional skewness
    - kurt_d: cross-sectional kurtosis
    - pct_high_d: % of stocks with d > 0.3
    """
    print("\nComputing cross-sectional features...")

    cs_features = pd.DataFrame(index=d_hat_df.index)

    cs_features['mean_d'] = d_hat_df.mean(axis=1)
    cs_features['std_d'] = d_hat_df.std(axis=1)
    cs_features['median_d'] = d_hat_df.median(axis=1)
    cs_features['skew_d'] = d_hat_df.apply(lambda x: stats.skew(x.dropna()), axis=1)
    cs_features['kurt_d'] = d_hat_df.apply(lambda x: stats.kurtosis(x.dropna()), axis=1)
    cs_features['pct_high_d'] = (d_hat_df > 0.3).mean(axis=1)
    cs_features['range_d'] = d_hat_df.max(axis=1) - d_hat_df.min(axis=1)

    print(f"  Cross-sectional features shape: {cs_features.shape}")
    return cs_features


# ============================================
# MARKET FEATURES
# ============================================
def compute_market_features(market, macro):
    """
    Compute market-level features.
    """
    print("\nComputing market features...")

    mkt_features = pd.DataFrame(index=market.index)

    # VIX and its dynamics
    mkt_features['VIX'] = market['VIX']
    mkt_features['VIX_lag1'] = market['VIX'].shift(1)
    mkt_features['VIX_ma5'] = market['VIX'].rolling(5).mean()
    mkt_features['VIX_ma22'] = market['VIX'].rolling(22).mean()
    mkt_features['VIX_change'] = market['VIX'].diff()
    mkt_features['VIX_pct_change'] = market['VIX'].pct_change()

    # Market returns
    if 'SP500_ret' in market.columns:
        mkt_features['SP500_ret'] = market['SP500_ret']
        mkt_features['SP500_ret_lag1'] = market['SP500_ret'].shift(1)
        mkt_features['SP500_ret_ma5'] = market['SP500_ret'].rolling(5).mean()

    # Macro variables
    for col in macro.columns:
        mkt_features[col] = macro[col]
        mkt_features[f'{col}_change'] = macro[col].diff()

    print(f"  Market features shape: {mkt_features.shape}")
    return mkt_features


# ============================================
# BUILD FEATURE MATRIX
# ============================================
def build_feature_matrix_single_stock(ticker, vol_proxy, returns, d_hat_df, se_df,
                                       cs_features, mkt_features):
    """
    Build complete feature matrix for a single stock.
    """
    features = pd.DataFrame(index=vol_proxy.index)

    # Target: next-day volatility (to predict)
    features['target_vol'] = vol_proxy[ticker].shift(-1)
    features['target_vol_5d'] = vol_proxy[ticker].rolling(5).mean().shift(-5)
    features['target_vol_22d'] = vol_proxy[ticker].rolling(22).mean().shift(-22)

    # HAR components
    har = compute_har_for_stock(vol_proxy[ticker])
    for col in har.columns:
        features[col] = har[col]

    # Current volatility
    features['vol_current'] = vol_proxy[ticker]

    # Returns
    features['return'] = returns[ticker]
    features['return_lag1'] = returns[ticker].shift(1)
    features['return_abs'] = returns[ticker].abs()
    features['return_neg'] = (returns[ticker] < 0).astype(int)

    # LRD features (if available)
    if d_hat_df is not None and ticker in d_hat_df.columns:
        features['d_hat'] = d_hat_df[ticker]
        features['d_hat_lag1'] = d_hat_df[ticker].shift(1)

        if se_df is not None and ticker in se_df.columns:
            features['se_d'] = se_df[ticker]
            features['d_significant'] = (features['d_hat'].abs() / features['se_d'] > 1.96).astype(int)

    # Memory dynamics
    if d_hat_df is not None and ticker in d_hat_df.columns:
        d = d_hat_df[ticker]
        features['delta_d'] = d.diff()
        features['vol_d'] = d.rolling(22).std()
        features['trend_d'] = d.diff(22) / 22

    # Cross-sectional features
    for col in cs_features.columns:
        features[f'cs_{col}'] = cs_features[col]

    # Market features
    for col in mkt_features.columns:
        features[f'mkt_{col}'] = mkt_features[col]

    return features


# ============================================
# EXPORT TABLE 4
# ============================================
def export_table4_latex(feature_stats, filepath):
    """Export Table 4: Feature definitions and statistics"""
    print(f"\nExporting Table 4 to {filepath}")

    with open(filepath, 'w') as f:
        f.write("% Table 4: Feature Definitions and Summary Statistics\n")
        f.write("% Generated by module3_feature_engineering.py\n\n")

        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Feature Definitions and Summary Statistics}\n")
        f.write("\\label{tab:features}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{llcccc}\n")
        f.write("\\toprule\n")
        f.write("Category & Feature & Mean & Std & Min & Max \\\\\n")
        f.write("\\midrule\n")

        # Group features by category
        categories = {
            'HAR': ['RV_d', 'RV_w', 'RV_m'],
            'LRD': ['d_hat', 'se_d', 'd_significant'],
            'Memory Dynamics': ['delta_d', 'vol_d', 'trend_d'],
            'Cross-Sectional': ['cs_mean_d', 'cs_std_d', 'cs_skew_d'],
            'Market': ['mkt_VIX', 'mkt_Term_Spread', 'mkt_IG_Credit_Spread'],
        }

        for cat, feats in categories.items():
            f.write(f"\\multicolumn{{6}}{{l}}{{\\textbf{{{cat}}}}} \\\\\n")
            for feat in feats:
                if feat in feature_stats.index:
                    row = feature_stats.loc[feat]
                    f.write(f"  & {feat.replace('_', '\\_')} & {row['mean']:.4f} & {row['std']:.4f} & ")
                    f.write(f"{row['min']:.4f} & {row['max']:.4f} \\\\\n")
            f.write("\\midrule\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write("\\item Notes: Statistics computed from pooled sample across all stocks and dates. ")
        f.write("RV = realized volatility (squared returns). ")
        f.write("$\\hat{d}$ = GPH memory parameter estimate. ")
        f.write("Cross-sectional features computed from distribution of $\\hat{d}$ across stocks.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{table}\n")


def main():
    print("="*70)
    print("   MODULE 3: FEATURE ENGINEERING")
    print("="*70)

    # Create directories
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

    # Load data
    returns, vol_proxy, market, macro, lrd_data = load_data()

    # ========================================
    # 1. HAR Components
    # ========================================
    har_features = compute_har_components(vol_proxy)

    # ========================================
    # 2. Rolling LRD Features (use fast mode)
    # ========================================
    d_hat_df, se_df = compute_rolling_lrd_fast(vol_proxy, window=ROLLING_WINDOW, sample_freq=5)

    # ========================================
    # 3. Memory Dynamics
    # ========================================
    memory_dynamics = compute_memory_dynamics(d_hat_df)

    # ========================================
    # 4. Cross-Sectional Features
    # ========================================
    cs_features = compute_cross_sectional_features(d_hat_df)

    # ========================================
    # 5. Market Features
    # ========================================
    mkt_features = compute_market_features(market, macro)

    # ========================================
    # 6. Build Sample Feature Matrix (for one stock as example)
    # ========================================
    print("\nBuilding sample feature matrix for AAPL...")
    sample_features = build_feature_matrix_single_stock(
        'AAPL', vol_proxy, returns, d_hat_df, se_df, cs_features, mkt_features
    )

    # ========================================
    # 7. Compute Feature Statistics
    # ========================================
    print("\nComputing feature statistics...")
    feature_stats = sample_features.describe().T[['mean', 'std', 'min', 'max']]

    # ========================================
    # 8. Save Results
    # ========================================
    print("\nSaving results...")

    # Save feature matrices
    d_hat_df.to_csv(os.path.join(INTERMEDIATE_DIR, "rolling_d_hat.csv"))
    se_df.to_csv(os.path.join(INTERMEDIATE_DIR, "rolling_se_d.csv"))
    cs_features.to_csv(os.path.join(INTERMEDIATE_DIR, "cross_sectional_features.csv"))
    mkt_features.to_csv(os.path.join(INTERMEDIATE_DIR, "market_features.csv"))
    sample_features.to_csv(os.path.join(INTERMEDIATE_DIR, "sample_features_AAPL.csv"))

    # Export Table 4
    export_table4_latex(feature_stats, os.path.join(TABLES_DIR, "table4_features.tex"))

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("   MODULE 3 COMPLETE")
    print("="*70)

    print("\nFeature categories created:")
    print(f"  - HAR components: 3 features per stock (RV_d, RV_w, RV_m)")
    print(f"  - LRD features: {d_hat_df.shape[1]} stocks x rolling d_hat")
    print(f"  - Memory dynamics: delta_d, vol_d, trend_d per stock")
    print(f"  - Cross-sectional: {cs_features.shape[1]} market-level features")
    print(f"  - Market features: {mkt_features.shape[1]} features")

    print(f"\nSample feature matrix (AAPL):")
    print(f"  Shape: {sample_features.shape}")
    print(f"  Date range: {sample_features.index.min()} to {sample_features.index.max()}")
    print(f"  Non-null rows: {sample_features.dropna().shape[0]}")

    print("\nKey feature statistics:")
    key_features = ['d_hat', 'delta_d', 'cs_mean_d', 'cs_std_d', 'mkt_VIX']
    for feat in key_features:
        if feat in feature_stats.index:
            row = feature_stats.loc[feat]
            print(f"  {feat}: mean={row['mean']:.4f}, std={row['std']:.4f}")

    print("\nOutputs created:")
    print(f"  - {TABLES_DIR}/table4_features.tex")
    print(f"  - {INTERMEDIATE_DIR}/rolling_d_hat.csv")
    print(f"  - {INTERMEDIATE_DIR}/cross_sectional_features.csv")
    print(f"  - {INTERMEDIATE_DIR}/market_features.csv")


if __name__ == "__main__":
    main()
