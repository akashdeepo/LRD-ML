"""
Module 7: Feature Importance & Interpretation
==============================================

Analyzes which features contribute most to volatility forecasting:
- XGBoost feature importance (gain)
- Permutation importance
- Creates Figure 5 and Table 6 for the paper
"""

import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = r"C:\Users\Akash\OneDrive\Desktop\LRD_Nicholas"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "results", "intermediate")
FIGURE_DIR = os.path.join(BASE_DIR, "results", "figures")
TABLE_DIR = os.path.join(BASE_DIR, "results", "tables")

def print_header(text):
    print("\n" + "=" * 60)
    print(f"   {text}")
    print("=" * 60)


if __name__ == "__main__":
    print_header("MODULE 7: FEATURE IMPORTANCE & INTERPRETATION")

    # Load data
    print("\nLoading data...")
    vol_proxy = pd.read_csv(os.path.join(PROCESSED_DIR, "volatility_proxy.csv"),
                           index_col=0, parse_dates=True)
    rolling_d = pd.read_csv(os.path.join(INTERMEDIATE_DIR, "rolling_d_hat.csv"),
                           index_col=0, parse_dates=True)
    cs_features = pd.read_csv(os.path.join(INTERMEDIATE_DIR, "cross_sectional_features.csv"),
                             index_col=0, parse_dates=True)
    market = pd.read_csv(os.path.join(PROCESSED_DIR, "market.csv"),
                        index_col=0, parse_dates=True)

    print("Building pooled dataset...")

    # Build pooled feature matrix
    feature_data = []
    stocks = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ', 'GE', 'IBM', 'WMT', 'KO', 'PG']

    for ticker in stocks:
        rv = vol_proxy[ticker].dropna()
        d_hat = rolling_d[ticker]

        common_idx = (rv.index
                     .intersection(d_hat.dropna().index)
                     .intersection(cs_features.index)
                     .intersection(market.index))

        rv_aligned = rv.loc[common_idx]
        d_aligned = d_hat.loc[common_idx]
        cs_aligned = cs_features.loc[common_idx]
        vix = market['VIX'].loc[common_idx]

        df = pd.DataFrame({
            'rv_d': rv_aligned.shift(1),
            'rv_w': rv_aligned.rolling(5).mean().shift(1),
            'rv_m': rv_aligned.rolling(22).mean().shift(1),
            'd_hat': d_aligned.shift(1),
            'delta_d': d_aligned.diff().shift(1),
            'cs_mean_d': cs_aligned['mean_d'].shift(1),
            'cs_std_d': cs_aligned['std_d'].shift(1),
            'VIX': vix.shift(1),
            'target': rv_aligned
        }).dropna()

        feature_data.append(df)

    pooled = pd.concat(feature_data, ignore_index=True)
    print(f"  Pooled dataset: {pooled.shape}")

    # Feature names for display
    feature_names = ['RV_daily', 'RV_weekly', 'RV_monthly', 'd_hat',
                    'delta_d', 'CS_mean_d', 'CS_std_d', 'VIX']

    X = pooled[['rv_d', 'rv_w', 'rv_m', 'd_hat', 'delta_d',
               'cs_mean_d', 'cs_std_d', 'VIX']].values
    y = pooled['target'].values

    # Fit XGBoost
    print("\nFitting XGBoost model...")
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                        random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Feature importance (gain)
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    print("\n" + "-" * 50)
    print("FEATURE IMPORTANCE (XGBoost Gain)")
    print("-" * 50)
    for _, row in importance_df.iterrows():
        bar = '#' * int(row['Importance'] * 50)
        print(f"{row['Feature']:<15} {row['Importance']:.3f} {bar}")

    # Permutation importance
    print("\nComputing permutation importance...")
    perm_importance = permutation_importance(model, X, y, n_repeats=10,
                                            random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean
    }).sort_values('Importance', ascending=False)

    print("\n" + "-" * 50)
    print("PERMUTATION IMPORTANCE")
    print("-" * 50)
    for _, row in perm_df.iterrows():
        bar = '#' * int(row['Importance'] * 50 / perm_df['Importance'].max())
        print(f"{row['Feature']:<15} {row['Importance']:.2e} {bar}")

    # =========================================================================
    # Create Figure 5
    # =========================================================================
    print("\nCreating Figure 5...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Color scheme: green for LRD features, blue for HAR/other
    def get_color(feature):
        if 'd' in feature.lower() or 'cs' in feature.lower():
            return '#2ecc71'  # Green for LRD
        return '#3498db'  # Blue for HAR

    # Panel A: XGBoost Feature Importance
    ax1 = axes[0]
    colors = [get_color(f) for f in importance_df['Feature']]
    ax1.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    ax1.set_xlabel('Importance (Gain)', fontsize=11)
    ax1.set_title('(a) XGBoost Feature Importance', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='LRD Features'),
                      Patch(facecolor='#3498db', label='HAR Features')]
    ax1.legend(handles=legend_elements, loc='lower right')

    # Panel B: Permutation Importance
    ax2 = axes[1]
    colors = [get_color(f) for f in perm_df['Feature']]
    ax2.barh(perm_df['Feature'], perm_df['Importance'], color=colors)
    ax2.set_xlabel('Importance (MSE Increase)', fontsize=11)
    ax2.set_title('(b) Permutation Importance', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'fig5_interpretation.pdf'),
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURE_DIR, 'fig5_interpretation.png'),
               dpi=150, bbox_inches='tight')
    print(f"  Saved fig5_interpretation.pdf")

    # =========================================================================
    # Export Table 6
    # =========================================================================
    print("\nExporting Table 6...")
    table6_path = os.path.join(TABLE_DIR, 'table6_feature_importance.tex')

    with open(table6_path, 'w') as f:
        f.write("% Table 6: Feature Importance Analysis\n")
        f.write("% Generated by module7_interpretation.py\n\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Feature Importance for Volatility Forecasting}\n")
        f.write("\\label{tab:feature_importance}\n")
        f.write("\\begin{tabular}{llcc}\n")
        f.write("\\toprule\n")
        f.write("Rank & Feature & XGBoost Gain & Permutation Imp. \\\\\n")
        f.write("\\midrule\n")

        for i, (_, row) in enumerate(importance_df.iterrows()):
            perm_val = perm_df[perm_df['Feature'] == row['Feature']]['Importance'].values[0]
            # Mark LRD features
            is_lrd = 'd' in row['Feature'].lower() or 'cs' in row['Feature'].lower()
            marker = "$^\\dagger$" if is_lrd else ""
            f.write(f"{i+1} & {row['Feature']}{marker} & {row['Importance']:.3f} & {perm_val:.2e} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write("\\item Notes: $^\\dagger$ indicates LRD-based features. ")
        f.write("XGBoost Gain measures feature contribution to model splits. ")
        f.write("Permutation importance measures MSE increase when feature is shuffled.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{table}\n")

    print(f"  Saved {table6_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print_header("KEY FINDINGS")

    lrd_features = ['d_hat', 'delta_d', 'CS_mean_d', 'CS_std_d']
    lrd_importance = importance_df[importance_df['Feature'].isin(lrd_features)]['Importance'].sum()
    har_importance = importance_df[~importance_df['Feature'].isin(lrd_features)]['Importance'].sum()

    print(f"\nTotal importance share:")
    print(f"  HAR features (RV_d, RV_w, RV_m, VIX): {har_importance:.1%}")
    print(f"  LRD features (d_hat, delta_d, CS_*):  {lrd_importance:.1%}")

    print(f"\nTop 3 most important features:")
    for i, (_, row) in enumerate(importance_df.head(3).iterrows()):
        print(f"  {i+1}. {row['Feature']}: {row['Importance']:.3f}")

    print("\n" + "=" * 60)
    print("   MODULE 7 COMPLETE")
    print("=" * 60)
