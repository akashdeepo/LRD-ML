"""
Module 8: Robustness Checks
===========================

Tests robustness of main findings:
1. Subsample analysis (using Module 5 rolling forecasts)
2. Different forecast horizons (1-day, 5-day, 22-day)
3. Generates Table 7 (subsamples) and Table 8 (horizons) for paper
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = r"C:\Users\Akash\OneDrive\Desktop\LRD_Nicholas"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "results", "intermediate")
TABLE_DIR = os.path.join(BASE_DIR, "results", "tables")


def print_header(text):
    print("\n" + "=" * 70)
    print(f"   {text}")
    print("=" * 70)


if __name__ == "__main__":
    print_header("MODULE 8: ROBUSTNESS CHECKS")

    # Load hybrid forecasts from Module 5
    print("\nLoading Module 5 forecasts...")
    with open(os.path.join(INTERMEDIATE_DIR, 'hybrid_forecasts.pkl'), 'rb') as f:
        results = pickle.load(f)

    stocks = list(results['har_lrd'].keys())
    print(f"  Using forecasts from {len(stocks)} stocks")

    # =========================================================================
    # PART 1: SUBSAMPLE ANALYSIS
    # =========================================================================
    print_header("PART 1: SUBSAMPLE ANALYSIS")

    subperiods = {
        'Pre-Crisis (2005-2007)': ('2005-01-01', '2007-12-31'),
        'Financial Crisis (2008-2009)': ('2008-01-01', '2009-12-31'),
        'Recovery (2010-2015)': ('2010-01-01', '2015-12-31'),
        'Bull Market (2016-2019)': ('2016-01-01', '2019-12-31'),
        'COVID & After (2020-2024)': ('2020-01-01', '2024-12-31'),
    }

    print(f"\n{'Period':<30} {'HAR MSE':>10} {'LRD MSE':>10} {'Improv':>10} {'N':>6}")
    print("-" * 70)

    subsample_results = []

    for period_name, (start, end) in subperiods.items():
        period_har_mse = []
        period_lrd_mse = []

        for ticker in stocks:
            actual = results['actual'][ticker]
            har = results['har_rv'][ticker]
            lrd = results['har_lrd'][ticker]

            # Filter to period
            mask = (actual.index >= start) & (actual.index <= end)
            a = actual[mask]
            h = har.reindex(a.index)
            l = lrd.reindex(a.index)

            # Align
            valid = a.notna() & h.notna() & l.notna()
            if valid.sum() < 20:
                continue

            a_vals = a[valid].values
            h_vals = h[valid].values
            l_vals = l[valid].values

            period_har_mse.append(np.mean((a_vals - h_vals) ** 2))
            period_lrd_mse.append(np.mean((a_vals - l_vals) ** 2))

        if period_har_mse:
            avg_har = np.mean(period_har_mse) * 1e7
            avg_lrd = np.mean(period_lrd_mse) * 1e7
            impr = (np.mean(period_har_mse) - np.mean(period_lrd_mse)) / np.mean(period_har_mse) * 100
            n = len(period_har_mse)

            subsample_results.append({
                'Period': period_name,
                'MSE_HAR': avg_har,
                'MSE_LRD': avg_lrd,
                'Improvement': impr,
                'N': n
            })

            sign = '+' if impr > 0 else ''
            print(f"{period_name:<30} {avg_har:>10.2f} {avg_lrd:>10.2f} {sign}{impr:>9.1f}% {n:>6}")

    # Summary
    n_positive = sum(1 for r in subsample_results if r['Improvement'] > 0)
    print("-" * 70)
    print(f"LRD features improve in {n_positive}/{len(subsample_results)} subperiods")

    # =========================================================================
    # PART 2: MULTI-HORIZON ANALYSIS
    # =========================================================================
    print_header("PART 2: MULTI-HORIZON ANALYSIS")

    vol_proxy = pd.read_csv(os.path.join(PROCESSED_DIR, "volatility_proxy.csv"),
                           index_col=0, parse_dates=True)
    rolling_d = pd.read_csv(os.path.join(INTERMEDIATE_DIR, "rolling_d_hat.csv"),
                           index_col=0, parse_dates=True)
    cs_features = pd.read_csv(os.path.join(INTERMEDIATE_DIR, "cross_sectional_features.csv"),
                             index_col=0, parse_dates=True)

    horizons = [1, 5, 22]
    horizon_results = []

    print(f"\n{'Horizon':<15} {'HAR MSE':>10} {'LRD MSE':>10} {'Improv':>10}")
    print("-" * 50)

    for h in horizons:
        h_har_mse = []
        h_lrd_mse = []

        for ticker in stocks:
            rv = vol_proxy[ticker].dropna()
            d_hat = rolling_d[ticker] if ticker in rolling_d.columns else None
            cs_mean = cs_features['mean_d']

            if d_hat is None:
                continue

            # Target: h-day ahead average RV
            if h == 1:
                rv_target = rv.shift(-1)
            else:
                rv_target = rv.rolling(h).mean().shift(-h)

            # Features (lagged)
            rv_d = rv.shift(1)
            rv_w = rv.rolling(5).mean().shift(1)
            rv_m = rv.rolling(22).mean().shift(1)
            d_lag = d_hat.shift(1).reindex(rv.index)
            delta_d = d_hat.diff().shift(1).reindex(rv.index)
            cs_lag = cs_mean.shift(1).reindex(rv.index)

            # Build dataset
            df = pd.DataFrame({
                'target': rv_target,
                'rv_d': rv_d,
                'rv_w': rv_w,
                'rv_m': rv_m,
                'd_hat': d_lag,
                'delta_d': delta_d,
                'cs_mean': cs_lag
            }).dropna()

            if len(df) < 500:
                continue

            # Train/test split (80/20)
            train_size = int(len(df) * 0.8)
            train = df.iloc[:train_size]
            test = df.iloc[train_size:]

            if len(test) < 50:
                continue

            # HAR
            har_cols = ['rv_d', 'rv_w', 'rv_m']
            har_model = LinearRegression().fit(train[har_cols], train['target'])
            har_pred = har_model.predict(test[har_cols])

            # HAR-LRD
            lrd_cols = ['rv_d', 'rv_w', 'rv_m', 'd_hat', 'delta_d', 'cs_mean']
            lrd_model = LinearRegression().fit(train[lrd_cols], train['target'])
            lrd_pred = lrd_model.predict(test[lrd_cols])

            h_har_mse.append(np.mean((test['target'].values - har_pred) ** 2))
            h_lrd_mse.append(np.mean((test['target'].values - lrd_pred) ** 2))

        if h_har_mse:
            avg_har = np.mean(h_har_mse) * 1e7
            avg_lrd = np.mean(h_lrd_mse) * 1e7
            impr = (np.mean(h_har_mse) - np.mean(h_lrd_mse)) / np.mean(h_har_mse) * 100

            horizon_results.append({
                'Horizon': f'{h}-day',
                'MSE_HAR': avg_har,
                'MSE_LRD': avg_lrd,
                'Improvement': impr
            })

            sign = '+' if impr > 0 else ''
            print(f"{h}-day ahead{' ' * 4} {avg_har:>10.2f} {avg_lrd:>10.2f} {sign}{impr:>9.1f}%")

    # =========================================================================
    # EXPORT TABLES
    # =========================================================================
    print_header("EXPORTING TABLES")

    # Table 7: Subsample Analysis
    table7_path = os.path.join(TABLE_DIR, 'table7_subsamples.tex')
    with open(table7_path, 'w') as f:
        f.write("% Table 7: Subsample Analysis\n")
        f.write("% Generated by module8_robustness.py\n\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Forecast Performance Across Subsamples}\n")
        f.write("\\label{tab:subsamples}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Period & MSE (HAR) & MSE (LRD) & Improvement & N \\\\\n")
        f.write("\\midrule\n")

        for r in subsample_results:
            f.write(f"{r['Period']} & {r['MSE_HAR']:.2f} & {r['MSE_LRD']:.2f} & ")
            f.write(f"{r['Improvement']:.1f}\\% & {r['N']} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write("\\item Notes: MSE values multiplied by $10^7$. ")
        f.write("Rolling forecast methodology from Module 5.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{table}\n")

    print(f"  Saved {table7_path}")

    # Table 8: Multi-Horizon
    table8_path = os.path.join(TABLE_DIR, 'table8_horizons.tex')
    with open(table8_path, 'w') as f:
        f.write("% Table 8: Multi-Horizon Forecasts\n")
        f.write("% Generated by module8_robustness.py\n\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Forecast Performance Across Horizons}\n")
        f.write("\\label{tab:horizons}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Horizon & MSE (HAR) & MSE (LRD) & Improvement \\\\\n")
        f.write("\\midrule\n")

        for r in horizon_results:
            f.write(f"{r['Horizon']} & {r['MSE_HAR']:.2f} & {r['MSE_LRD']:.2f} & ")
            f.write(f"{r['Improvement']:.1f}\\% \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write("\\item Notes: MSE values multiplied by $10^7$. ")
        f.write("h-day horizon forecasts average volatility over next h days.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{table}\n")

    print(f"  Saved {table8_path}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("ROBUSTNESS CHECK SUMMARY")

    print("\nSubsample Results:")
    print("-" * 60)
    for r in subsample_results:
        status = "BETTER" if r['Improvement'] > 0 else "worse"
        print(f"  {r['Period']:<30} {r['Improvement']:>6.1f}%  [{status}]")

    print("\nHorizon Results:")
    print("-" * 60)
    for r in horizon_results:
        status = "BETTER" if r['Improvement'] > 0 else "worse"
        print(f"  {r['Horizon']:<30} {r['Improvement']:>6.1f}%  [{status}]")

    n_pos_sub = sum(1 for r in subsample_results if r['Improvement'] > 0)
    n_pos_hor = sum(1 for r in horizon_results if r['Improvement'] > 0)

    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    print(f"  - LRD improves in {n_pos_sub}/{len(subsample_results)} subsamples")
    print(f"  - LRD improves in {n_pos_hor}/{len(horizon_results)} horizons")
    print("=" * 70)
