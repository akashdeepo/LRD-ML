"""
LRD-ML Research: Data Preprocessing Pipeline
Cleans and prepares all data for LRD estimation and ML modeling
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")

# Winsorization bounds for returns (percentiles)
WINSORIZE_LOWER = 0.1  # 0.1%
WINSORIZE_UPPER = 99.9  # 99.9%

# Minimum data coverage required (fraction of total days)
MIN_COVERAGE = 0.7  # 70%


def load_original_stocks():
    """Load and fix the original 30 stocks data"""
    print("\n[1] Loading original 30 stocks...")

    df = pd.read_csv(os.path.join(DATA_DIR, "LRD ML DATA(PX_LAST).csv"))

    # Parse dates and set as index
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df.set_index('Date')

    # Reverse order (data comes newest-first)
    df = df.sort_index()

    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()

    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    return df


def load_additional_stocks():
    """Load additional stocks data"""
    print("\n[2] Loading additional stocks...")

    df = pd.read_csv(os.path.join(DATA_DIR, "additional_stocks.csv"),
                     index_col=0, parse_dates=True)
    df = df.sort_index()

    print(f"  Shape: {df.shape}")

    return df


def load_market_data():
    """Load VIX and market indices"""
    print("\n[3] Loading market data...")

    # VIX
    vix = pd.read_csv(os.path.join(DATA_DIR, "vix_data.csv"),
                      index_col=0, parse_dates=True)

    # Market indices
    indices = pd.read_csv(os.path.join(DATA_DIR, "market_indices.csv"),
                          index_col=0, parse_dates=True)

    # Combine
    market = pd.concat([vix, indices], axis=1)
    market = market.sort_index()

    print(f"  Shape: {market.shape}")
    print(f"  Columns: {list(market.columns)}")

    return market


def load_macro_data():
    """Load macro variables"""
    print("\n[4] Loading macro data...")

    df = pd.read_csv(os.path.join(DATA_DIR, "macro_data.csv"),
                     index_col=0, parse_dates=True)
    df = df.sort_index()

    print(f"  Shape: {df.shape}")

    return df


def load_volume_data():
    """Load volume data"""
    print("\n[5] Loading volume data...")

    df = pd.read_csv(os.path.join(DATA_DIR, "volume_data.csv"),
                     index_col=0, parse_dates=True)
    df = df.sort_index()

    print(f"  Shape: {df.shape}")

    return df


def compute_returns(prices, method='log'):
    """
    Compute returns from prices

    Parameters:
    -----------
    prices : DataFrame
        Price data
    method : str
        'log' for log returns, 'simple' for arithmetic returns

    Returns:
    --------
    DataFrame of returns
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()

    return returns


def winsorize_returns(returns, lower_pct=0.1, upper_pct=99.9):
    """
    Winsorize returns to handle extreme values

    Parameters:
    -----------
    returns : DataFrame
        Return data
    lower_pct, upper_pct : float
        Percentile bounds

    Returns:
    --------
    DataFrame of winsorized returns
    """
    lower = np.nanpercentile(returns.values.flatten(), lower_pct)
    upper = np.nanpercentile(returns.values.flatten(), upper_pct)

    print(f"  Winsorizing returns: [{lower:.4f}, {upper:.4f}]")

    winsorized = returns.clip(lower=lower, upper=upper)

    n_clipped = ((returns < lower) | (returns > upper)).sum().sum()
    print(f"  Clipped {n_clipped} values ({100*n_clipped/(returns.shape[0]*returns.shape[1]):.3f}%)")

    return winsorized


def filter_by_coverage(df, min_coverage=0.7):
    """
    Remove columns with insufficient data coverage

    Parameters:
    -----------
    df : DataFrame
        Data with potential missing values
    min_coverage : float
        Minimum fraction of non-null values required

    Returns:
    --------
    DataFrame with low-coverage columns removed
    """
    coverage = df.notna().sum() / len(df)
    good_cols = coverage[coverage >= min_coverage].index.tolist()
    removed = len(df.columns) - len(good_cols)

    if removed > 0:
        print(f"  Removed {removed} columns with <{100*min_coverage:.0f}% coverage")
        removed_cols = coverage[coverage < min_coverage].index.tolist()
        print(f"  Removed: {removed_cols[:5]}{'...' if len(removed_cols) > 5 else ''}")

    return df[good_cols]


def align_datasets(*dfs):
    """
    Align multiple DataFrames to common dates

    Returns:
    --------
    List of aligned DataFrames
    """
    # Find common dates
    common_idx = dfs[0].index
    for df in dfs[1:]:
        common_idx = common_idx.intersection(df.index)

    print(f"  Common dates: {len(common_idx)}")

    return [df.loc[common_idx] for df in dfs]


def compute_volatility_proxy(returns):
    """
    Compute daily volatility proxy (squared returns)

    Parameters:
    -----------
    returns : DataFrame
        Log returns

    Returns:
    --------
    DataFrame of volatility proxy (annualized)
    """
    # Squared returns as volatility proxy
    vol_proxy = returns ** 2

    # Annualize (252 trading days)
    vol_proxy_annual = vol_proxy * 252

    return vol_proxy, vol_proxy_annual


def create_summary_stats(prices, returns, vol_proxy):
    """Create summary statistics for the processed data"""
    stats = pd.DataFrame({
        'mean_price': prices.mean(),
        'std_price': prices.std(),
        'mean_return': returns.mean() * 252,  # Annualized
        'std_return': returns.std() * np.sqrt(252),  # Annualized
        'skew_return': returns.skew(),
        'kurt_return': returns.kurtosis(),
        'mean_vol_proxy': vol_proxy.mean() * 252,
        'coverage': prices.notna().sum() / len(prices),
        'start_date': prices.apply(lambda x: x.first_valid_index()),
        'end_date': prices.apply(lambda x: x.last_valid_index()),
    })
    return stats


def main():
    print("="*70)
    print("   LRD-ML DATA PREPROCESSING PIPELINE")
    print("="*70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # ========================================
    # LOAD DATA
    # ========================================
    original_stocks = load_original_stocks()
    additional_stocks = load_additional_stocks()
    market_data = load_market_data()
    macro_data = load_macro_data()
    volume_data = load_volume_data()

    # ========================================
    # COMBINE STOCK PRICES
    # ========================================
    print("\n[6] Combining stock prices...")

    # Merge original and additional (avoid duplicates)
    all_tickers = list(original_stocks.columns) + \
                  [c for c in additional_stocks.columns if c not in original_stocks.columns]
    all_prices = pd.concat([original_stocks, additional_stocks], axis=1)
    all_prices = all_prices.loc[:, ~all_prices.columns.duplicated()]

    print(f"  Combined: {all_prices.shape[1]} unique stocks")

    # ========================================
    # FILTER BY COVERAGE
    # ========================================
    print("\n[7] Filtering by coverage...")
    all_prices_filtered = filter_by_coverage(all_prices, MIN_COVERAGE)

    # ========================================
    # COMPUTE RETURNS
    # ========================================
    print("\n[8] Computing log returns...")
    returns = compute_returns(all_prices_filtered, method='log')

    # Drop first row (NaN from differencing)
    returns = returns.iloc[1:]
    prices_aligned = all_prices_filtered.iloc[1:]

    print(f"  Returns shape: {returns.shape}")

    # ========================================
    # WINSORIZE EXTREME RETURNS
    # ========================================
    print("\n[9] Winsorizing extreme returns...")
    returns_winsorized = winsorize_returns(returns, WINSORIZE_LOWER, WINSORIZE_UPPER)

    # ========================================
    # COMPUTE VOLATILITY PROXY
    # ========================================
    print("\n[10] Computing volatility proxy...")
    vol_proxy, vol_proxy_annual = compute_volatility_proxy(returns_winsorized)
    print(f"  Mean annualized vol proxy: {vol_proxy_annual.mean().mean():.4f}")

    # ========================================
    # ALIGN ALL DATASETS
    # ========================================
    print("\n[11] Aligning datasets...")
    returns_aligned, vol_aligned, market_aligned, macro_aligned = align_datasets(
        returns_winsorized, vol_proxy, market_data, macro_data
    )

    # Forward fill macro data (small gaps from holidays)
    macro_aligned = macro_aligned.ffill().bfill()

    # ========================================
    # COMPUTE MARKET RETURNS
    # ========================================
    print("\n[12] Computing market returns...")
    market_returns = compute_returns(market_aligned[['SP500', 'DJIA', 'NASDAQ', 'Russell2000']], method='log')
    market_aligned = pd.concat([market_aligned, market_returns.add_suffix('_ret')], axis=1)

    # ========================================
    # SUMMARY STATISTICS
    # ========================================
    print("\n[13] Computing summary statistics...")
    summary_stats = create_summary_stats(prices_aligned, returns_aligned, vol_aligned)

    # ========================================
    # SAVE PROCESSED DATA
    # ========================================
    print("\n[14] Saving processed data...")

    # Stock returns (main dataset for LRD estimation)
    returns_aligned.to_csv(os.path.join(OUTPUT_DIR, "returns.csv"))
    print(f"  returns.csv: {returns_aligned.shape}")

    # Volatility proxy
    vol_aligned.to_csv(os.path.join(OUTPUT_DIR, "volatility_proxy.csv"))
    print(f"  volatility_proxy.csv: {vol_aligned.shape}")

    # Stock prices (cleaned)
    prices_aligned.to_csv(os.path.join(OUTPUT_DIR, "prices.csv"))
    print(f"  prices.csv: {prices_aligned.shape}")

    # Market data (VIX, indices, returns)
    market_aligned.to_csv(os.path.join(OUTPUT_DIR, "market.csv"))
    print(f"  market.csv: {market_aligned.shape}")

    # Macro data (forward filled)
    macro_aligned.to_csv(os.path.join(OUTPUT_DIR, "macro.csv"))
    print(f"  macro.csv: {macro_aligned.shape}")

    # Summary statistics
    summary_stats.to_csv(os.path.join(OUTPUT_DIR, "summary_stats.csv"))
    print(f"  summary_stats.csv: {summary_stats.shape}")

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*70)
    print("   PREPROCESSING COMPLETE")
    print("="*70)

    print(f"\nProcessed data saved to: {OUTPUT_DIR}")
    print(f"\nDataset summary:")
    print(f"  Stocks: {returns_aligned.shape[1]}")
    print(f"  Trading days: {returns_aligned.shape[0]}")
    print(f"  Date range: {returns_aligned.index.min()} to {returns_aligned.index.max()}")
    print(f"\nFiles created:")
    for f in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath) / 1024
        print(f"  - {f} ({size:.1f} KB)")

    # ========================================
    # DATA QUALITY REPORT
    # ========================================
    print("\n" + "="*70)
    print("   DATA QUALITY REPORT")
    print("="*70)

    print(f"\nReturn statistics (annualized):")
    print(f"  Mean: {returns_aligned.mean().mean() * 252:.2%}")
    print(f"  Std:  {returns_aligned.std().mean() * np.sqrt(252):.2%}")
    print(f"  Min:  {returns_aligned.min().min():.2%}")
    print(f"  Max:  {returns_aligned.max().max():.2%}")

    print(f"\nMissing values:")
    print(f"  Returns: {returns_aligned.isnull().sum().sum()} ({100*returns_aligned.isnull().sum().sum()/(returns_aligned.shape[0]*returns_aligned.shape[1]):.2f}%)")
    print(f"  Market:  {market_aligned.isnull().sum().sum()}")
    print(f"  Macro:   {macro_aligned.isnull().sum().sum()}")

    print(f"\nStocks with highest volatility:")
    top_vol = returns_aligned.std().sort_values(ascending=False).head(5)
    for ticker, vol in top_vol.items():
        print(f"  {ticker}: {vol * np.sqrt(252):.2%}")

    print(f"\nStocks with lowest volatility:")
    low_vol = returns_aligned.std().sort_values().head(5)
    for ticker, vol in low_vol.items():
        print(f"  {ticker}: {vol * np.sqrt(252):.2%}")


if __name__ == "__main__":
    main()
