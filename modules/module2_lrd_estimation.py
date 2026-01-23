"""
MODULE 2: LRD Estimation
========================
Implements GPH and Local Whittle estimators for the memory parameter d.
Produces Table 3 and Figure 2 for the paper.

Outputs:
- Table 3: LRD estimates by sector
- Figure 2: (a) Distribution of d-hat, (b) Rolling d-hat, (c) Cross-sectional dispersion

Author: Akash Deep, Nicholas Appiah
Date: January 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
INTERMEDIATE_DIR = os.path.join(RESULTS_DIR, "intermediate")

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['font.family'] = 'serif'

# LRD estimation parameters
BANDWIDTH_POWER = 0.65  # m = T^0.65 (common choice between 0.5 and 0.8)
ROLLING_WINDOW = 500    # Days for rolling estimation
MIN_OBS = 250           # Minimum observations required

# Sector map (same as Module 1)
SECTOR_MAP = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology', 'GOOGL': 'Technology',
    'INTC': 'Technology', 'CSCO': 'Technology', 'ORCL': 'Technology', 'ADBE': 'Technology',
    'TXN': 'Technology', 'QCOM': 'Technology', 'AMD': 'Technology', 'IBM': 'Technology',
    'JPM': 'Financials', 'BAC': 'Financials', 'GS': 'Financials', 'WFC': 'Financials',
    'C': 'Financials', 'MS': 'Financials', 'AXP': 'Financials', 'BLK': 'Financials',
    'SCHW': 'Financials', 'USB': 'Financials', 'PNC': 'Financials', 'TFC': 'Financials',
    'COF': 'Financials',
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'MRK': 'Healthcare',
    'LLY': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare', 'DHR': 'Healthcare',
    'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare', 'CVS': 'Healthcare',
    'AMZN': 'Consumer Disc.', 'HD': 'Consumer Disc.', 'MCD': 'Consumer Disc.',
    'NKE': 'Consumer Disc.', 'SBUX': 'Consumer Disc.', 'TJX': 'Consumer Disc.',
    'LOW': 'Consumer Disc.', 'TGT': 'Consumer Disc.', 'BKNG': 'Consumer Disc.',
    'MAR': 'Consumer Disc.', 'F': 'Consumer Disc.',
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
    'WMT': 'Consumer Staples', 'COST': 'Consumer Staples', 'PM': 'Consumer Staples',
    'MO': 'Consumer Staples', 'CL': 'Consumer Staples', 'KMB': 'Consumer Staples',
    'GIS': 'Consumer Staples', 'K': 'Consumer Staples', 'SYY': 'Consumer Staples',
    'ADM': 'Consumer Staples',
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'EOG': 'Energy', 'VLO': 'Energy', 'OXY': 'Energy', 'HAL': 'Energy', 'WMB': 'Energy',
    'BA': 'Industrials', 'CAT': 'Industrials', 'UNP': 'Industrials', 'HON': 'Industrials',
    'GE': 'Industrials', 'RTX': 'Industrials', 'LMT': 'Industrials', 'MMM': 'Industrials',
    'DE': 'Industrials', 'EMR': 'Industrials', 'ITW': 'Industrials', 'ETN': 'Industrials',
    'FDX': 'Industrials',
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
    'AEP': 'Utilities', 'EXC': 'Utilities', 'SRE': 'Utilities', 'XEL': 'Utilities',
    'PEG': 'Utilities', 'ED': 'Utilities', 'WEC': 'Utilities', 'ES': 'Utilities',
    'FCX': 'Materials', 'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials',
    'ECL': 'Materials', 'NEM': 'Materials', 'NUE': 'Materials', 'VMC': 'Materials',
    'MLM': 'Materials', 'DD': 'Materials', 'PPG': 'Materials',
    'AMT': 'Real Estate', 'PLD': 'Real Estate', 'SPG': 'Real Estate', 'EQIX': 'Real Estate',
    'PSA': 'Real Estate', 'O': 'Real Estate', 'WELL': 'Real Estate', 'AVB': 'Real Estate',
    'EQR': 'Real Estate', 'DLR': 'Real Estate', 'VTR': 'Real Estate', 'BXP': 'Real Estate',
    'DIS': 'Communication', 'CMCSA': 'Communication', 'NFLX': 'Communication',
    'T': 'Communication', 'VZ': 'Communication', 'CHTR': 'Communication', 'EA': 'Communication',
}


# ============================================
# GPH ESTIMATOR
# ============================================
def gph_estimator(x, m=None, return_details=False):
    """
    Geweke-Porter-Hudak (GPH) log-periodogram regression estimator.

    Parameters:
    -----------
    x : array-like
        Time series (should be stationary or weakly dependent)
    m : int, optional
        Number of Fourier frequencies to use. Default: T^0.65
    return_details : bool
        If True, return additional estimation details

    Returns:
    --------
    d_hat : float
        Estimated memory parameter
    se : float
        Standard error of the estimate
    details : dict (if return_details=True)
        Additional information including t-stat, p-value
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    T = len(x)

    if T < MIN_OBS:
        return np.nan, np.nan

    # Bandwidth selection
    if m is None:
        m = int(np.floor(T ** BANDWIDTH_POWER))

    # Demean the series
    x_demean = x - np.mean(x)

    # Compute periodogram at Fourier frequencies
    # I(lambda_j) = (1/2*pi*T) * |sum_{t=1}^T x_t * exp(i*lambda_j*t)|^2
    fft_x = np.fft.fft(x_demean)
    freq_idx = np.arange(1, m + 1)
    I = np.abs(fft_x[freq_idx]) ** 2 / (2 * np.pi * T)

    # Fourier frequencies
    lambda_j = 2 * np.pi * freq_idx / T

    # Log-periodogram regression
    # ln(I(lambda_j)) = c - 2d * ln(lambda_j) + error
    # Equivalent: ln(I) = c + d * ln(4*sin^2(lambda/2)) + error

    y = np.log(I + 1e-10)  # Add small constant for numerical stability
    X = np.log(4 * np.sin(lambda_j / 2) ** 2)

    # OLS regression
    X_demean = X - np.mean(X)
    y_demean = y - np.mean(y)

    beta = np.sum(X_demean * y_demean) / np.sum(X_demean ** 2)
    d_hat = -beta / 2

    # Standard error: se(d) = pi / sqrt(24 * m)
    se = np.pi / np.sqrt(24 * m)

    if return_details:
        t_stat = d_hat / se
        p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))
        residuals = y - (np.mean(y) + beta * X_demean)

        details = {
            'd_hat': d_hat,
            'se': se,
            't_stat': t_stat,
            'p_value': p_value,
            'm': m,
            'T': T,
            'residuals': residuals,
        }
        return d_hat, se, details

    return d_hat, se


# ============================================
# LOCAL WHITTLE ESTIMATOR
# ============================================
def local_whittle_estimator(x, m=None, return_details=False):
    """
    Local Whittle (LW) semiparametric estimator.

    More efficient than GPH (variance 1/4 vs pi^2/24).

    Parameters:
    -----------
    x : array-like
        Time series
    m : int, optional
        Bandwidth. Default: T^0.65

    Returns:
    --------
    d_hat : float
        Estimated memory parameter
    se : float
        Standard error (asymptotic: 1/(2*sqrt(m)))
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    T = len(x)

    if T < MIN_OBS:
        return np.nan, np.nan

    if m is None:
        m = int(np.floor(T ** BANDWIDTH_POWER))

    # Demean
    x_demean = x - np.mean(x)

    # Periodogram
    fft_x = np.fft.fft(x_demean)
    freq_idx = np.arange(1, m + 1)
    I = np.abs(fft_x[freq_idx]) ** 2 / (2 * np.pi * T)

    # Fourier frequencies
    lambda_j = 2 * np.pi * freq_idx / T

    # Log of frequencies for numerical stability
    log_lambda = np.log(lambda_j)

    # Local Whittle objective function (corrected)
    def Q(d):
        """Local Whittle objective to minimize"""
        # G_hat(d) = (1/m) * sum(lambda_j^{2d} * I(lambda_j))
        G_hat = np.mean(lambda_j ** (2 * d) * I)
        if G_hat <= 0:
            return np.inf
        return np.log(G_hat) - 2 * d * np.mean(log_lambda)

    # Grid search + refinement for robustness
    d_grid = np.linspace(-0.4, 0.8, 50)
    Q_values = [Q(d) for d in d_grid]
    d_init = d_grid[np.argmin(Q_values)]

    # Refine with bounded optimization
    try:
        result = minimize_scalar(Q, bounds=(max(-0.49, d_init - 0.2), min(0.99, d_init + 0.2)),
                                 method='bounded')
        d_hat = result.x
    except:
        d_hat = d_init

    # Standard error: se(d) = 1 / (2 * sqrt(m))
    se = 1 / (2 * np.sqrt(m))

    if return_details:
        t_stat = d_hat / se
        p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))

        details = {
            'd_hat': d_hat,
            'se': se,
            't_stat': t_stat,
            'p_value': p_value,
            'm': m,
            'T': T,
            'objective': Q(d_hat),
        }
        return d_hat, se, details

    return d_hat, se


# ============================================
# ROLLING WINDOW ESTIMATION
# ============================================
def rolling_lrd_estimate(x, window=ROLLING_WINDOW, method='gph'):
    """
    Compute rolling window LRD estimates.

    Parameters:
    -----------
    x : pd.Series
        Time series with datetime index
    window : int
        Rolling window size
    method : str
        'gph' or 'lw' (local whittle)

    Returns:
    --------
    pd.DataFrame with columns: d_hat, se
    """
    estimator = gph_estimator if method == 'gph' else local_whittle_estimator

    results = []
    dates = []

    for i in range(window, len(x)):
        window_data = x.iloc[i-window:i].values
        d_hat, se = estimator(window_data)
        results.append({'d_hat': d_hat, 'se': se})
        dates.append(x.index[i])

    df = pd.DataFrame(results, index=dates)
    return df


# ============================================
# CROSS-SECTIONAL ESTIMATION
# ============================================
def cross_sectional_lrd(data, method='gph'):
    """
    Estimate d for all stocks in the cross-section.

    Parameters:
    -----------
    data : pd.DataFrame
        Returns or volatility proxy (columns = tickers)
    method : str
        'gph' or 'lw'

    Returns:
    --------
    pd.DataFrame with d_hat, se, t_stat for each stock
    """
    estimator = gph_estimator if method == 'gph' else local_whittle_estimator

    results = {}
    for col in data.columns:
        x = data[col].dropna()
        if len(x) >= MIN_OBS:
            d_hat, se, details = estimator(x, return_details=True)
            results[col] = {
                'd_hat': d_hat,
                'se': se,
                't_stat': details['t_stat'],
                'p_value': details['p_value'],
                'T': details['T'],
                'm': details['m'],
            }

    return pd.DataFrame(results).T


# ============================================
# MAIN ANALYSIS
# ============================================
def load_data():
    """Load processed data"""
    print("Loading data...")

    returns = pd.read_csv(os.path.join(PROCESSED_DIR, "returns.csv"),
                          index_col=0, parse_dates=True)
    vol_proxy = pd.read_csv(os.path.join(PROCESSED_DIR, "volatility_proxy.csv"),
                            index_col=0, parse_dates=True)
    market = pd.read_csv(os.path.join(PROCESSED_DIR, "market.csv"),
                         index_col=0, parse_dates=True)

    print(f"  Returns: {returns.shape}")
    print(f"  Volatility proxy: {vol_proxy.shape}")

    return returns, vol_proxy, market


def estimate_all_lrd(returns, vol_proxy):
    """
    Estimate LRD parameters for returns and volatility.
    """
    print("\n" + "="*60)
    print("   ESTIMATING LRD PARAMETERS")
    print("="*60)

    # 1. Returns - GPH
    print("\n[1/4] GPH estimates for returns...")
    returns_gph = cross_sectional_lrd(returns, method='gph')
    returns_gph['Sector'] = returns_gph.index.map(lambda x: SECTOR_MAP.get(x, 'Other'))

    # 2. Returns - Local Whittle
    print("[2/4] Local Whittle estimates for returns...")
    returns_lw = cross_sectional_lrd(returns, method='lw')
    returns_lw['Sector'] = returns_lw.index.map(lambda x: SECTOR_MAP.get(x, 'Other'))

    # 3. Volatility proxy - GPH
    print("[3/4] GPH estimates for volatility...")
    vol_gph = cross_sectional_lrd(vol_proxy, method='gph')
    vol_gph['Sector'] = vol_gph.index.map(lambda x: SECTOR_MAP.get(x, 'Other'))

    # 4. Volatility proxy - Local Whittle
    print("[4/4] Local Whittle estimates for volatility...")
    vol_lw = cross_sectional_lrd(vol_proxy, method='lw')
    vol_lw['Sector'] = vol_lw.index.map(lambda x: SECTOR_MAP.get(x, 'Other'))

    return {
        'returns_gph': returns_gph,
        'returns_lw': returns_lw,
        'vol_gph': vol_gph,
        'vol_lw': vol_lw,
    }


def compute_rolling_estimates(returns, vol_proxy, sample_tickers=None):
    """
    Compute rolling window LRD estimates for selected stocks.
    """
    if sample_tickers is None:
        sample_tickers = ['AAPL', 'JPM', 'XOM', 'JNJ', 'PG']

    print("\n" + "="*60)
    print("   COMPUTING ROLLING LRD ESTIMATES")
    print("="*60)

    rolling_results = {}

    for ticker in sample_tickers:
        if ticker in vol_proxy.columns:
            print(f"  {ticker}...")
            rolling_results[ticker] = rolling_lrd_estimate(
                vol_proxy[ticker].dropna(),
                window=ROLLING_WINDOW,
                method='gph'
            )

    return rolling_results


def compute_cross_sectional_dispersion(vol_proxy, window=ROLLING_WINDOW):
    """
    Compute time series of cross-sectional d-hat dispersion.
    """
    print("\nComputing cross-sectional dispersion over time...")

    # We'll compute d for each stock in rolling windows, then cross-sectional stats
    dates = vol_proxy.index[window:]
    n_dates = len(dates)

    # Sample every 22 days (monthly) for computational efficiency
    sample_freq = 22
    sample_dates = dates[::sample_freq]

    results = []

    for i, date in enumerate(sample_dates):
        if i % 10 == 0:
            print(f"  Processing {date.strftime('%Y-%m')} ({i+1}/{len(sample_dates)})")

        # Get window ending at this date
        end_idx = vol_proxy.index.get_loc(date)
        start_idx = end_idx - window

        window_data = vol_proxy.iloc[start_idx:end_idx]

        # Estimate d for each stock
        d_estimates = []
        for col in window_data.columns:
            x = window_data[col].dropna()
            if len(x) >= MIN_OBS:
                d_hat, _ = gph_estimator(x)
                if not np.isnan(d_hat):
                    d_estimates.append(d_hat)

        if len(d_estimates) > 10:
            results.append({
                'date': date,
                'mean_d': np.mean(d_estimates),
                'std_d': np.std(d_estimates),
                'median_d': np.median(d_estimates),
                'skew_d': stats.skew(d_estimates),
                'n_stocks': len(d_estimates),
            })

    return pd.DataFrame(results).set_index('date')


def print_summary(estimates):
    """Print summary of LRD estimates"""
    print("\n" + "="*60)
    print("   LRD ESTIMATION SUMMARY")
    print("="*60)

    for name, df in estimates.items():
        print(f"\n{name.upper()}:")
        print(f"  Mean d-hat: {df['d_hat'].mean():.4f}")
        print(f"  Std d-hat:  {df['d_hat'].std():.4f}")
        print(f"  Min d-hat:  {df['d_hat'].min():.4f}")
        print(f"  Max d-hat:  {df['d_hat'].max():.4f}")
        print(f"  % significant (5%): {100*(df['p_value'] < 0.05).mean():.1f}%")


def export_table3_latex(estimates, filepath):
    """Export Table 3: LRD estimates by sector"""
    print(f"\nExporting Table 3 to {filepath}")

    vol_gph = estimates['vol_gph']
    vol_lw = estimates['vol_lw']

    # Sector aggregation - compute separately to avoid lambda naming issues
    sector_d_stats = vol_gph.groupby('Sector')['d_hat'].agg(['mean', 'std', 'count']).round(3)

    # Compute % significant separately
    def pct_significant(group):
        return (group['p_value'] < 0.05).mean() * 100

    sector_sig = vol_gph.groupby('Sector').apply(pct_significant, include_groups=False)

    sector_lw = vol_lw.groupby('Sector')['d_hat'].agg(['mean', 'std']).round(3)

    with open(filepath, 'w') as f:
        f.write("% Table 3: Long-Range Dependence Estimates by Sector\n")
        f.write("% Generated by module2_lrd_estimation.py\n\n")

        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Long-Range Dependence Estimates for Volatility by Sector}\n")
        f.write("\\label{tab:lrd_estimates}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lccccccc}\n")
        f.write("\\toprule\n")
        f.write(" & \\multicolumn{3}{c}{GPH Estimator} & \\multicolumn{2}{c}{Local Whittle} & \\\\\n")
        f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-6}\n")
        f.write("Sector & $\\bar{d}$ & Std($d$) & \\% Sig. & $\\bar{d}$ & Std($d$) & N \\\\\n")
        f.write("\\midrule\n")

        for sector in sector_d_stats.index:
            gph_mean = sector_d_stats.loc[sector, 'mean']
            gph_std = sector_d_stats.loc[sector, 'std']
            gph_sig = sector_sig.loc[sector]
            gph_n = int(sector_d_stats.loc[sector, 'count'])
            lw_mean = sector_lw.loc[sector, 'mean']
            lw_std = sector_lw.loc[sector, 'std']

            f.write(f"{sector} & {gph_mean:.3f} & {gph_std:.3f} & {gph_sig:.0f}\\% ")
            f.write(f"& {lw_mean:.3f} & {lw_std:.3f} & {gph_n} \\\\\n")

        # Overall
        f.write("\\midrule\n")
        overall_gph_mean = vol_gph['d_hat'].mean()
        overall_gph_std = vol_gph['d_hat'].std()
        overall_gph_sig = (vol_gph['p_value'] < 0.05).mean() * 100
        overall_lw_mean = vol_lw['d_hat'].mean()
        overall_lw_std = vol_lw['d_hat'].std()
        n_total = len(vol_gph)

        f.write(f"\\textbf{{Overall}} & {overall_gph_mean:.3f} & {overall_gph_std:.3f} & {overall_gph_sig:.0f}\\% ")
        f.write(f"& {overall_lw_mean:.3f} & {overall_lw_std:.3f} & {n_total} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write("\\item Notes: $d$ is the fractional differencing parameter estimated from squared returns (volatility proxy). ")
        f.write("GPH is the Geweke-Porter-Hudak log-periodogram estimator; Local Whittle is the semiparametric estimator. ")
        f.write(f"Bandwidth $m = T^{{{BANDWIDTH_POWER}}}$. \\% Sig. is the percentage of stocks with $d$ significantly different from zero at 5\\%.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{table}\n")


def create_figure2(estimates, rolling_results, cs_dispersion, market, filepath):
    """
    Create Figure 2: LRD Estimation Results
    (a) Distribution of d-hat
    (b) Rolling d-hat for sample stocks
    (c) Cross-sectional dispersion over time
    """
    print(f"\nCreating Figure 2...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Distribution of d-hat (volatility)
    ax = axes[0, 0]
    vol_d = estimates['vol_gph']['d_hat'].dropna()
    ax.hist(vol_d, bins=30, density=True, alpha=0.7, color='darkred', edgecolor='black')
    ax.axvline(x=vol_d.mean(), color='black', linestyle='--', linewidth=2,
               label=f'Mean = {vol_d.mean():.3f}')
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('$\\hat{d}$ (Memory Parameter)')
    ax.set_ylabel('Density')
    ax.set_title('(a) Distribution of $\\hat{d}$ for Volatility (GPH)', fontweight='bold')
    ax.legend()
    ax.set_xlim(-0.1, 0.7)

    # (b) Rolling d-hat for sample stocks
    ax = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(rolling_results)))
    for (ticker, df), color in zip(rolling_results.items(), colors):
        ax.plot(df.index, df['d_hat'], label=ticker, alpha=0.8, linewidth=1)

    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, label='d=0.5 (nonstationary)')
    ax.set_xlabel('')
    ax.set_ylabel('$\\hat{d}$')
    ax.set_title(f'(b) Rolling $\\hat{{d}}$ (Window = {ROLLING_WINDOW} days)', fontweight='bold')
    ax.legend(loc='upper right', ncol=2)
    ax.set_xlim(list(rolling_results.values())[0].index.min(),
                list(rolling_results.values())[0].index.max())

    # Add recession shading
    ax.axvspan(pd.Timestamp('2007-12-01'), pd.Timestamp('2009-06-30'), alpha=0.2, color='gray')
    ax.axvspan(pd.Timestamp('2020-02-01'), pd.Timestamp('2020-04-30'), alpha=0.2, color='gray')

    # (c) Cross-sectional dispersion over time
    ax = axes[1, 0]
    ax.plot(cs_dispersion.index, cs_dispersion['mean_d'], 'b-', linewidth=1.5, label='Mean $\\hat{d}$')
    ax.fill_between(cs_dispersion.index,
                    cs_dispersion['mean_d'] - cs_dispersion['std_d'],
                    cs_dispersion['mean_d'] + cs_dispersion['std_d'],
                    alpha=0.3, color='blue', label='$\\pm$ 1 Std')
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('')
    ax.set_ylabel('Cross-sectional $\\hat{d}$')
    ax.set_title('(c) Cross-Sectional Memory Dispersion Over Time', fontweight='bold')
    ax.legend()
    ax.axvspan(pd.Timestamp('2007-12-01'), pd.Timestamp('2009-06-30'), alpha=0.2, color='gray')
    ax.axvspan(pd.Timestamp('2020-02-01'), pd.Timestamp('2020-04-30'), alpha=0.2, color='gray')

    # (d) d-hat vs VIX correlation
    ax = axes[1, 1]
    # Use cross-sectional mean d
    merged = cs_dispersion[['mean_d', 'std_d']].copy()
    merged['VIX'] = market['VIX'].reindex(merged.index).values

    ax.scatter(merged['VIX'], merged['mean_d'], alpha=0.5, s=20, c='darkred')
    # Add regression line
    mask = ~(merged['VIX'].isna() | merged['mean_d'].isna())
    if mask.sum() > 10:
        z = np.polyfit(merged.loc[mask, 'VIX'], merged.loc[mask, 'mean_d'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(merged['VIX'].min(), merged['VIX'].max(), 100)
        ax.plot(x_line, p(x_line), 'b--', linewidth=2)
        corr = merged.loc[mask, ['VIX', 'mean_d']].corr().iloc[0, 1]
        ax.text(0.05, 0.95, f'Corr = {corr:.3f}', transform=ax.transAxes,
                fontsize=11, verticalalignment='top')

    ax.set_xlabel('VIX')
    ax.set_ylabel('Cross-sectional Mean $\\hat{d}$')
    ax.set_title('(d) Memory Parameter vs VIX', fontweight='bold')

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.savefig(filepath.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filepath}")


def main():
    print("="*70)
    print("   MODULE 2: LRD ESTIMATION")
    print("="*70)

    # Create directories
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

    # Load data
    returns, vol_proxy, market = load_data()

    # Estimate LRD parameters
    estimates = estimate_all_lrd(returns, vol_proxy)

    # Print summary
    print_summary(estimates)

    # Rolling estimates for sample stocks
    rolling_results = compute_rolling_estimates(returns, vol_proxy)

    # Cross-sectional dispersion
    cs_dispersion = compute_cross_sectional_dispersion(vol_proxy)

    # Export Table 3
    export_table3_latex(estimates, os.path.join(TABLES_DIR, "table3_lrd_estimates.tex"))

    # Create Figure 2
    create_figure2(estimates, rolling_results, cs_dispersion, market,
                   os.path.join(FIGURES_DIR, "fig2_lrd_estimates.pdf"))

    # Save intermediate results
    print("\nSaving intermediate results...")
    for name, df in estimates.items():
        df.to_csv(os.path.join(INTERMEDIATE_DIR, f"lrd_{name}.csv"))

    cs_dispersion.to_csv(os.path.join(INTERMEDIATE_DIR, "lrd_cs_dispersion.csv"))

    # Save rolling results
    for ticker, df in rolling_results.items():
        df.to_csv(os.path.join(INTERMEDIATE_DIR, f"lrd_rolling_{ticker}.csv"))

    print("\n" + "="*70)
    print("   MODULE 2 COMPLETE")
    print("="*70)
    print("\nKey findings:")
    print(f"  - Mean d (volatility, GPH): {estimates['vol_gph']['d_hat'].mean():.3f}")
    print(f"  - Mean d (volatility, LW):  {estimates['vol_lw']['d_hat'].mean():.3f}")
    print(f"  - Mean d (returns, GPH):    {estimates['returns_gph']['d_hat'].mean():.3f}")
    print(f"  - % significant (vol):      {100*(estimates['vol_gph']['p_value'] < 0.05).mean():.1f}%")

    print("\nOutputs created:")
    print(f"  - {TABLES_DIR}/table3_lrd_estimates.tex")
    print(f"  - {FIGURES_DIR}/fig2_lrd_estimates.pdf")


if __name__ == "__main__":
    main()
