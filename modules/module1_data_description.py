"""
MODULE 1: Data Description
==========================
Produces Table 1-2 and Figure 1 for the paper.

Outputs:
- Table 1: Summary statistics for returns
- Table 2: Data description (sectors, macro variables)
- Figure 1: (a) Sample series, (b) ACF comparison, (c) VIX and market

Author: Akash Deep, Nicholas Appiah
Date: January 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
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

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['font.family'] = 'serif'

# Sector classifications (GICS-style)
SECTOR_MAP = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology', 'GOOGL': 'Technology',
    'INTC': 'Technology', 'CSCO': 'Technology', 'ORCL': 'Technology', 'ADBE': 'Technology',
    'TXN': 'Technology', 'QCOM': 'Technology', 'AMD': 'Technology', 'IBM': 'Technology',
    # Financials
    'JPM': 'Financials', 'BAC': 'Financials', 'GS': 'Financials', 'WFC': 'Financials',
    'C': 'Financials', 'MS': 'Financials', 'AXP': 'Financials', 'BLK': 'Financials',
    'SCHW': 'Financials', 'USB': 'Financials', 'PNC': 'Financials', 'TFC': 'Financials',
    'COF': 'Financials',
    # Healthcare
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'MRK': 'Healthcare',
    'LLY': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare', 'DHR': 'Healthcare',
    'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare', 'CVS': 'Healthcare',
    # Consumer Discretionary
    'AMZN': 'Consumer Disc.', 'HD': 'Consumer Disc.', 'MCD': 'Consumer Disc.',
    'NKE': 'Consumer Disc.', 'SBUX': 'Consumer Disc.', 'TJX': 'Consumer Disc.',
    'LOW': 'Consumer Disc.', 'TGT': 'Consumer Disc.', 'BKNG': 'Consumer Disc.',
    'MAR': 'Consumer Disc.', 'F': 'Consumer Disc.',
    # Consumer Staples
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
    'WMT': 'Consumer Staples', 'COST': 'Consumer Staples', 'PM': 'Consumer Staples',
    'MO': 'Consumer Staples', 'CL': 'Consumer Staples', 'KMB': 'Consumer Staples',
    'GIS': 'Consumer Staples', 'K': 'Consumer Staples', 'SYY': 'Consumer Staples',
    'ADM': 'Consumer Staples',
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'EOG': 'Energy', 'VLO': 'Energy', 'OXY': 'Energy', 'HAL': 'Energy', 'WMB': 'Energy',
    # Industrials
    'BA': 'Industrials', 'CAT': 'Industrials', 'UNP': 'Industrials', 'HON': 'Industrials',
    'GE': 'Industrials', 'RTX': 'Industrials', 'LMT': 'Industrials', 'MMM': 'Industrials',
    'DE': 'Industrials', 'EMR': 'Industrials', 'ITW': 'Industrials', 'ETN': 'Industrials',
    'FDX': 'Industrials',
    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
    'AEP': 'Utilities', 'EXC': 'Utilities', 'SRE': 'Utilities', 'XEL': 'Utilities',
    'PEG': 'Utilities', 'ED': 'Utilities', 'WEC': 'Utilities', 'ES': 'Utilities',
    # Materials
    'FCX': 'Materials', 'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials',
    'ECL': 'Materials', 'NEM': 'Materials', 'NUE': 'Materials', 'VMC': 'Materials',
    'MLM': 'Materials', 'DD': 'Materials', 'PPG': 'Materials',
    # Real Estate
    'AMT': 'Real Estate', 'PLD': 'Real Estate', 'SPG': 'Real Estate', 'EQIX': 'Real Estate',
    'PSA': 'Real Estate', 'O': 'Real Estate', 'WELL': 'Real Estate', 'AVB': 'Real Estate',
    'EQR': 'Real Estate', 'DLR': 'Real Estate', 'VTR': 'Real Estate', 'BXP': 'Real Estate',
    # Communication Services
    'DIS': 'Communication', 'CMCSA': 'Communication', 'NFLX': 'Communication',
    'T': 'Communication', 'VZ': 'Communication', 'CHTR': 'Communication', 'EA': 'Communication',
}


def load_data():
    """Load all processed data"""
    print("Loading processed data...")

    returns = pd.read_csv(os.path.join(PROCESSED_DIR, "returns.csv"),
                          index_col=0, parse_dates=True)
    prices = pd.read_csv(os.path.join(PROCESSED_DIR, "prices.csv"),
                         index_col=0, parse_dates=True)
    vol_proxy = pd.read_csv(os.path.join(PROCESSED_DIR, "volatility_proxy.csv"),
                            index_col=0, parse_dates=True)
    market = pd.read_csv(os.path.join(PROCESSED_DIR, "market.csv"),
                         index_col=0, parse_dates=True)
    macro = pd.read_csv(os.path.join(PROCESSED_DIR, "macro.csv"),
                        index_col=0, parse_dates=True)

    print(f"  Returns: {returns.shape}")
    print(f"  Prices: {prices.shape}")
    print(f"  Market: {market.shape}")
    print(f"  Macro: {macro.shape}")

    return returns, prices, vol_proxy, market, macro


def compute_summary_stats(returns):
    """
    Compute summary statistics for returns

    Returns Table 1 data
    """
    print("\nComputing summary statistics...")

    # Annualization factors
    ann_mean = 252
    ann_std = np.sqrt(252)

    stats_dict = {}
    for col in returns.columns:
        r = returns[col].dropna()
        stats_dict[col] = {
            'N': len(r),
            'Mean (%)': r.mean() * ann_mean * 100,
            'Std (%)': r.std() * ann_std * 100,
            'Skewness': r.skew(),
            'Kurtosis': r.kurtosis(),
            'Min (%)': r.min() * 100,
            'Max (%)': r.max() * 100,
            'JB Stat': stats.jarque_bera(r)[0],
            'JB p-val': stats.jarque_bera(r)[1],
        }

    stats_df = pd.DataFrame(stats_dict).T

    # Add sector
    stats_df['Sector'] = stats_df.index.map(lambda x: SECTOR_MAP.get(x, 'Other'))

    return stats_df


def compute_panel_stats(returns, vol_proxy, market):
    """Compute panel-level statistics"""
    print("\nComputing panel statistics...")

    # Returns panel stats
    all_returns = returns.values.flatten()
    all_returns = all_returns[~np.isnan(all_returns)]

    # Volatility proxy stats
    all_vol = vol_proxy.values.flatten()
    all_vol = all_vol[~np.isnan(all_vol)]

    panel_stats = {
        'Returns': {
            'Mean (% ann.)': np.mean(all_returns) * 252 * 100,
            'Std (% ann.)': np.std(all_returns) * np.sqrt(252) * 100,
            'Skewness': stats.skew(all_returns),
            'Kurtosis': stats.kurtosis(all_returns),
            'Min (%)': np.min(all_returns) * 100,
            'Max (%)': np.max(all_returns) * 100,
            '1st Percentile (%)': np.percentile(all_returns, 1) * 100,
            '99th Percentile (%)': np.percentile(all_returns, 99) * 100,
        },
        'Volatility Proxy': {
            'Mean (% ann.)': np.sqrt(np.mean(all_vol) * 252) * 100,
            'Std (% ann.)': np.sqrt(np.std(all_vol) * 252) * 100,
            'Skewness': stats.skew(all_vol),
            'Kurtosis': stats.kurtosis(all_vol),
        },
        'VIX': {
            'Mean': market['VIX'].mean(),
            'Std': market['VIX'].std(),
            'Min': market['VIX'].min(),
            'Max': market['VIX'].max(),
        }
    }

    return panel_stats


def compute_sector_breakdown(stats_df):
    """Compute sector-level statistics"""
    print("\nComputing sector breakdown...")

    sector_stats = stats_df.groupby('Sector').agg({
        'N': 'count',
        'Mean (%)': 'mean',
        'Std (%)': 'mean',
        'Skewness': 'mean',
        'Kurtosis': 'mean',
    }).round(2)

    sector_stats.columns = ['Count', 'Mean Return (%)', 'Mean Vol (%)',
                            'Avg Skewness', 'Avg Kurtosis']

    return sector_stats.sort_values('Count', ascending=False)


def compute_acf(x, nlags=50):
    """Compute autocorrelation function"""
    x = x.dropna()
    n = len(x)
    x_demean = x - x.mean()
    acf = np.correlate(x_demean, x_demean, mode='full')[n-1:] / (x.var() * n)
    return acf[:nlags+1]


def export_table1_latex(stats_df, panel_stats, filepath):
    """Export Table 1: Summary Statistics"""
    print(f"\nExporting Table 1 to {filepath}")

    with open(filepath, 'w') as f:
        f.write("% Table 1: Summary Statistics for Stock Returns\n")
        f.write("% Generated by module1_data_description.py\n\n")

        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Summary Statistics for Daily Stock Returns}\n")
        f.write("\\label{tab:summary_stats}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write(" & Mean & Std & Skewness & Kurtosis & Min & Max \\\\\n")
        f.write(" & (\\% ann.) & (\\% ann.) & & & (\\%) & (\\%) \\\\\n")
        f.write("\\midrule\n")

        # Panel A: Overall
        f.write("\\multicolumn{7}{l}{\\textbf{Panel A: Pooled Sample}} \\\\\n")
        ps = panel_stats['Returns']
        f.write(f"All Stocks (N=125) & {ps['Mean (% ann.)']:.2f} & {ps['Std (% ann.)']:.2f} & ")
        f.write(f"{ps['Skewness']:.2f} & {ps['Kurtosis']:.2f} & ")
        f.write(f"{ps['Min (%)']:.2f} & {ps['Max (%)']:.2f} \\\\\n")
        f.write("\\midrule\n")

        # Panel B: By Sector
        f.write("\\multicolumn{7}{l}{\\textbf{Panel B: By Sector}} \\\\\n")
        sector_stats = stats_df.groupby('Sector').agg({
            'Mean (%)': 'mean',
            'Std (%)': 'mean',
            'Skewness': 'mean',
            'Kurtosis': 'mean',
            'Min (%)': 'mean',
            'Max (%)': 'mean',
        })

        for sector in sector_stats.index:
            row = sector_stats.loc[sector]
            n_stocks = (stats_df['Sector'] == sector).sum()
            f.write(f"{sector} (n={n_stocks}) & {row['Mean (%)']:.2f} & {row['Std (%)']:.2f} & ")
            f.write(f"{row['Skewness']:.2f} & {row['Kurtosis']:.2f} & ")
            f.write(f"{row['Min (%)']:.2f} & {row['Max (%)']:.2f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write("\\item Notes: Sample period is January 2000 to January 2025. ")
        f.write("Mean and Std are annualized. Skewness and Kurtosis are excess values. ")
        f.write("Returns are log returns, winsorized at 0.1\\% and 99.9\\%.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{table}\n")


def export_table2_latex(sector_stats, macro, filepath):
    """Export Table 2: Data Description"""
    print(f"\nExporting Table 2 to {filepath}")

    with open(filepath, 'w') as f:
        f.write("% Table 2: Data Description\n")
        f.write("% Generated by module1_data_description.py\n\n")

        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Data Description}\n")
        f.write("\\label{tab:data_description}\n")
        f.write("\\small\n")

        # Panel A: Sample composition
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("\\multicolumn{3}{l}{\\textbf{Panel A: Sample Composition}} \\\\\n")
        f.write("Sector & N Stocks & \\% of Sample \\\\\n")
        f.write("\\midrule\n")

        total = sector_stats['Count'].sum()
        for sector in sector_stats.index:
            count = sector_stats.loc[sector, 'Count']
            pct = 100 * count / total
            f.write(f"{sector} & {count} & {pct:.1f}\\% \\\\\n")

        f.write(f"\\midrule\n")
        f.write(f"Total & {total} & 100.0\\% \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n\n")

        f.write("\\vspace{0.5cm}\n\n")

        # Panel B: Macro variables
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("\\multicolumn{6}{l}{\\textbf{Panel B: Macro Variables}} \\\\\n")
        f.write("Variable & Mean & Std & Min & Max & N \\\\\n")
        f.write("\\midrule\n")

        for col in macro.columns:
            m = macro[col]
            f.write(f"{col.replace('_', ' ')} & {m.mean():.2f} & {m.std():.2f} & ")
            f.write(f"{m.min():.2f} & {m.max():.2f} & {m.notna().sum()} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write("\\item Notes: Sample period January 2000 - January 2025. ")
        f.write("Macro variables from FRED. Treasury rates and spreads in percentage points.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{table}\n")


def create_figure1(returns, vol_proxy, market, filepath):
    """
    Create Figure 1: Data Overview
    (a) Sample return series
    (b) ACF of returns vs squared returns
    (c) VIX and S&P 500
    """
    print(f"\nCreating Figure 1...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Sample return series - AAPL
    ax = axes[0, 0]
    sample_stock = 'AAPL'
    returns[sample_stock].plot(ax=ax, linewidth=0.5, alpha=0.8, color='steelblue')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title(f'(a) Daily Log Returns: {sample_stock}', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Log Return')
    ax.set_xlim(returns.index.min(), returns.index.max())

    # (b) Squared returns (volatility clustering)
    ax = axes[0, 1]
    vol_proxy[sample_stock].plot(ax=ax, linewidth=0.5, alpha=0.8, color='darkred')
    ax.set_title(f'(b) Squared Returns (Volatility Proxy): {sample_stock}', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Squared Return')
    ax.set_xlim(returns.index.min(), returns.index.max())

    # (c) ACF comparison
    ax = axes[1, 0]
    sample_returns = returns[sample_stock].dropna()
    sample_squared = (sample_returns ** 2).dropna()

    acf_returns = compute_acf(sample_returns, 50)
    acf_squared = compute_acf(sample_squared, 50)

    lags = np.arange(len(acf_returns))
    width = 0.35

    ax.bar(lags - width/2, acf_returns, width, label='Returns', color='steelblue', alpha=0.7)
    ax.bar(lags + width/2, acf_squared, width, label='Squared Returns', color='darkred', alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=1.96/np.sqrt(len(sample_returns)), color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(y=-1.96/np.sqrt(len(sample_returns)), color='gray', linestyle='--', linewidth=0.5)
    ax.set_title('(c) Autocorrelation: Returns vs Squared Returns', fontweight='bold')
    ax.set_xlabel('Lag (days)')
    ax.set_ylabel('ACF')
    ax.legend()
    ax.set_xlim(-1, 51)

    # (d) VIX
    ax = axes[1, 1]
    market['VIX'].plot(ax=ax, linewidth=0.8, color='purple', alpha=0.8)
    ax.fill_between(market.index, 0, market['VIX'], alpha=0.3, color='purple')
    ax.set_title('(d) VIX Index', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('VIX')
    ax.set_xlim(market.index.min(), market.index.max())

    # Add recession shading (approximate)
    for ax_i in [axes[0, 0], axes[0, 1], axes[1, 1]]:
        # 2001 recession
        ax_i.axvspan(pd.Timestamp('2001-03-01'), pd.Timestamp('2001-11-30'),
                     alpha=0.1, color='gray')
        # 2008-2009 recession
        ax_i.axvspan(pd.Timestamp('2007-12-01'), pd.Timestamp('2009-06-30'),
                     alpha=0.1, color='gray')
        # COVID
        ax_i.axvspan(pd.Timestamp('2020-02-01'), pd.Timestamp('2020-04-30'),
                     alpha=0.1, color='gray')

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.savefig(filepath.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filepath}")


def main():
    print("="*70)
    print("   MODULE 1: DATA DESCRIPTION")
    print("="*70)

    # Create output directories
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load data
    returns, prices, vol_proxy, market, macro = load_data()

    # Compute statistics
    stats_df = compute_summary_stats(returns)
    panel_stats = compute_panel_stats(returns, vol_proxy, market)
    sector_stats = compute_sector_breakdown(stats_df)

    # Print summary
    print("\n" + "="*70)
    print("   SUMMARY STATISTICS")
    print("="*70)

    print("\nPanel Statistics:")
    for category, values in panel_stats.items():
        print(f"\n{category}:")
        for key, val in values.items():
            print(f"  {key}: {val:.2f}")

    print("\nSector Breakdown:")
    print(sector_stats.to_string())

    # Export tables
    export_table1_latex(stats_df, panel_stats,
                        os.path.join(TABLES_DIR, "table1_summary_stats.tex"))
    export_table2_latex(sector_stats, macro,
                        os.path.join(TABLES_DIR, "table2_data_description.tex"))

    # Create figures
    create_figure1(returns, vol_proxy, market,
                   os.path.join(FIGURES_DIR, "fig1_data_overview.pdf"))

    # Save intermediate results
    stats_df.to_csv(os.path.join(RESULTS_DIR, "intermediate", "stock_stats.csv"))

    print("\n" + "="*70)
    print("   MODULE 1 COMPLETE")
    print("="*70)
    print("\nOutputs created:")
    print(f"  - {TABLES_DIR}/table1_summary_stats.tex")
    print(f"  - {TABLES_DIR}/table2_data_description.tex")
    print(f"  - {FIGURES_DIR}/fig1_data_overview.pdf")
    print(f"  - {FIGURES_DIR}/fig1_data_overview.png")


if __name__ == "__main__":
    main()
