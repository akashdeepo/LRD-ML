"""
MODULE 1: Data Description (v2 — new Bloomberg panel)
======================================================
Produces Table 1, Table 2, and Figure 1 for the paper.

Uses modules.io_v2.build_clean_panel() so that all downstream modules see the
same 115-stock panel (after 70% coverage filter), the same winsorized log
returns, and the same Parkinson range-based realized volatility.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from modules.io_v2 import build_clean_panel, sector_map

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
TABLES_DIR = BASE / "results" / "tables"
FIGURES_DIR = BASE / "results" / "figures"
INTERMEDIATE_DIR = BASE / "results" / "intermediate"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "font.family": "serif",
})


def per_stock_stats(returns: pd.DataFrame, sectors: dict) -> pd.DataFrame:
    rows = {}
    for col in returns.columns:
        r = returns[col].dropna()
        rows[col] = {
            "N": len(r),
            "Mean (%)": r.mean() * 252 * 100,
            "Std (%)": r.std() * np.sqrt(252) * 100,
            "Skewness": r.skew(),
            "Kurtosis": r.kurtosis(),
            "Min (%)": r.min() * 100,
            "Max (%)": r.max() * 100,
            "JB Stat": stats.jarque_bera(r)[0],
            "JB p-val": stats.jarque_bera(r)[1],
        }
    df = pd.DataFrame(rows).T
    df["Sector"] = df.index.map(lambda t: sectors.get(t, "Other"))
    return df


def panel_stats(returns: pd.DataFrame, rv_pk: pd.DataFrame, market: pd.DataFrame) -> dict:
    r = returns.values.flatten()
    r = r[~np.isnan(r)]
    v = rv_pk.values.flatten()
    v = v[~np.isnan(v) & (v > 0)]
    return {
        "Returns": {
            "Mean (% ann.)": r.mean() * 252 * 100,
            "Std (% ann.)": r.std() * np.sqrt(252) * 100,
            "Skewness": stats.skew(r),
            "Kurtosis": stats.kurtosis(r),
            "Min (%)": r.min() * 100,
            "Max (%)": r.max() * 100,
            "1st Percentile (%)": np.percentile(r, 1) * 100,
            "99th Percentile (%)": np.percentile(r, 99) * 100,
        },
        "Parkinson RV": {
            "Mean ann. vol (%)": np.sqrt(v.mean() * 252) * 100,
            "Median ann. vol (%)": np.sqrt(np.median(v) * 252) * 100,
            "Skewness": stats.skew(v),
            "Kurtosis": stats.kurtosis(v),
        },
        "VIX": {
            "Mean": market["VIX"].mean(),
            "Std": market["VIX"].std(),
            "Min": market["VIX"].min(),
            "Max": market["VIX"].max(),
        },
    }


def sector_breakdown(stats_df: pd.DataFrame) -> pd.DataFrame:
    sector_stats = stats_df.groupby("Sector").agg({
        "N": "count",
        "Mean (%)": "mean",
        "Std (%)": "mean",
        "Skewness": "mean",
        "Kurtosis": "mean",
    }).round(2)
    sector_stats.columns = ["Count", "Mean Return (%)", "Mean Vol (%)",
                            "Avg Skewness", "Avg Kurtosis"]
    return sector_stats.sort_values("Count", ascending=False)


def acf(x: pd.Series, nlags: int = 50) -> np.ndarray:
    x = x.dropna()
    n = len(x)
    xd = x - x.mean()
    out = np.correlate(xd, xd, mode="full")[n - 1:] / (x.var() * n)
    return out[: nlags + 1]


def export_table1(stats_df: pd.DataFrame, ps: dict, n_stocks: int, t0, t1, fp: Path) -> None:
    with open(fp, "w") as f:
        f.write("% Table 1: Summary Statistics for Stock Returns (new Bloomberg panel)\n\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Summary Statistics for Daily Stock Returns}\n")
        f.write("\\label{tab:summary_stats}\n\\small\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write(" & Mean & Std & Skewness & Kurtosis & Min & Max \\\\\n")
        f.write(" & (\\% ann.) & (\\% ann.) & & & (\\%) & (\\%) \\\\\n\\midrule\n")
        f.write("\\multicolumn{7}{l}{\\textbf{Panel A: Pooled Sample}} \\\\\n")
        r = ps["Returns"]
        f.write(f"All Stocks (N={n_stocks}) & {r['Mean (% ann.)']:.2f} & {r['Std (% ann.)']:.2f} & "
                f"{r['Skewness']:.2f} & {r['Kurtosis']:.2f} & "
                f"{r['Min (%)']:.2f} & {r['Max (%)']:.2f} \\\\\n\\midrule\n")
        f.write("\\multicolumn{7}{l}{\\textbf{Panel B: By GICS Sector}} \\\\\n")
        sec = stats_df.groupby("Sector").agg({
            "Mean (%)": "mean", "Std (%)": "mean", "Skewness": "mean",
            "Kurtosis": "mean", "Min (%)": "mean", "Max (%)": "mean"})
        for s in sec.index:
            row = sec.loc[s]
            n = (stats_df["Sector"] == s).sum()
            f.write(f"{s} (n={n}) & {row['Mean (%)']:.2f} & {row['Std (%)']:.2f} & "
                    f"{row['Skewness']:.2f} & {row['Kurtosis']:.2f} & "
                    f"{row['Min (%)']:.2f} & {row['Max (%)']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write(f"\\item Notes: Sample period {t0:%b %Y} -- {t1:%b %Y}, "
                f"{n_stocks} S\\&P 500 constituents that pass a 70\\% coverage filter. "
                "Mean and Std are annualized; Skewness and Kurtosis are excess values. "
                "Returns are log returns winsorized at the 0.1\\% and 99.9\\% percentiles.\n")
        f.write("\\end{tablenotes}\n\\end{table}\n")


def export_table2(sector_stats: pd.DataFrame, market: pd.DataFrame, t0, t1, fp: Path) -> None:
    with open(fp, "w") as f:
        f.write("% Table 2: Data Description\n\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Data Description}\n\\label{tab:data_description}\n\\small\n")
        f.write("\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("\\multicolumn{3}{l}{\\textbf{Panel A: Sample Composition by GICS Sector}} \\\\\n")
        f.write("Sector & N Stocks & \\% of Sample \\\\\n\\midrule\n")
        total = int(sector_stats["Count"].sum())
        for s in sector_stats.index:
            n = int(sector_stats.loc[s, "Count"])
            f.write(f"{s} & {n} & {100 * n / total:.1f}\\% \\\\\n")
        f.write(f"\\midrule\nTotal & {total} & 100.0\\% \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\n")
        f.write("\\vspace{0.5cm}\n\n")
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("\\multicolumn{6}{l}{\\textbf{Panel B: Market-Level Variables}} \\\\\n")
        f.write("Variable & Mean & Std & Min & Max & N \\\\\n\\midrule\n")
        for col in market.columns:
            m = market[col].dropna()
            if len(m) == 0:
                continue
            f.write(f"{col.replace('_', ' ')} & {m.mean():.2f} & {m.std():.2f} & "
                    f"{m.min():.2f} & {m.max():.2f} & {len(m)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write(f"\\item Notes: Sample period {t0:%b %Y} -- {t1:%b %Y}. "
                "Market-level variables sourced from Bloomberg.\n")
        f.write("\\end{tablenotes}\n\\end{table}\n")


def figure1(returns: pd.DataFrame, rv_pk: pd.DataFrame, market: pd.DataFrame,
            sample: str, fp: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if sample not in returns.columns:
        sample = returns.columns[0]
    r = returns[sample].dropna()
    v = rv_pk[sample].dropna()

    ax = axes[0, 0]
    r.plot(ax=ax, linewidth=0.5, alpha=0.8, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"(a) Daily Log Returns: {sample}", fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("Log Return")

    ax = axes[0, 1]
    v.plot(ax=ax, linewidth=0.5, alpha=0.8, color="darkred")
    ax.set_title(f"(b) Parkinson Realized Variance: {sample}", fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("RV (variance)")

    ax = axes[1, 0]
    r2 = (r ** 2).dropna()
    ar = acf(r, 50); arsq = acf(r2, 50); arv = acf(v, 50)
    lags = np.arange(len(ar))
    w = 0.27
    ax.bar(lags - w, ar, w, label="Returns", color="steelblue", alpha=0.75)
    ax.bar(lags, arsq, w, label="Squared Returns", color="darkred", alpha=0.75)
    ax.bar(lags + w, arv, w, label="Parkinson RV", color="forestgreen", alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.5)
    ci = 1.96 / np.sqrt(len(r))
    ax.axhline(ci, color="gray", linestyle="--", linewidth=0.5)
    ax.axhline(-ci, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title("(c) ACF Comparison", fontweight="bold")
    ax.set_xlabel("Lag (days)"); ax.set_ylabel("ACF")
    ax.legend(); ax.set_xlim(-1, 51)

    ax = axes[1, 1]
    market["VIX"].plot(ax=ax, linewidth=0.8, color="purple", alpha=0.85)
    ax.fill_between(market.index, 0, market["VIX"], alpha=0.25, color="purple")
    ax.set_title("(d) VIX Index", fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("VIX")

    for ax_i in [axes[0, 0], axes[0, 1], axes[1, 1]]:
        ax_i.axvspan(pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-30"),
                     alpha=0.1, color="gray")
        ax_i.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30"),
                     alpha=0.1, color="gray")

    plt.tight_layout()
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.savefig(str(fp).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("=" * 70)
    print("   MODULE 1: DATA DESCRIPTION (Bloomberg panel)")
    print("=" * 70)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_clean_panel()
    sectors = sector_map(panel.metadata)
    n = len(panel.kept)
    t0, t1 = panel.prices.index.min(), panel.prices.index.max()
    print(f"  N={n}  T={len(panel.prices)}  range {t0.date()} -> {t1.date()}")

    stats_df = per_stock_stats(panel.returns, sectors)
    ps = panel_stats(panel.returns, panel.rv_parkinson, panel.market)
    sec = sector_breakdown(stats_df)

    print("\nPooled returns:")
    for k, v in ps["Returns"].items():
        print(f"  {k}: {v:.2f}")
    print("\nParkinson RV:")
    for k, v in ps["Parkinson RV"].items():
        print(f"  {k}: {v:.2f}")
    print("\nVIX:")
    for k, v in ps["VIX"].items():
        print(f"  {k}: {v:.2f}")
    print("\nSector breakdown:")
    print(sec.to_string())

    export_table1(stats_df, ps, n, t0, t1, TABLES_DIR / "table1_summary_stats.tex")
    export_table2(sec, panel.market, t0, t1, TABLES_DIR / "table2_data_description.tex")
    figure1(panel.returns, panel.rv_parkinson, panel.market,
            sample="AAPL", fp=FIGURES_DIR / "fig1_data_overview.pdf")
    stats_df.to_csv(INTERMEDIATE_DIR / "stock_stats.csv")

    print("\n" + "=" * 70)
    print("Outputs:")
    print(f"  {TABLES_DIR/'table1_summary_stats.tex'}")
    print(f"  {TABLES_DIR/'table2_data_description.tex'}")
    print(f"  {FIGURES_DIR/'fig1_data_overview.pdf'} (+ .png)")


if __name__ == "__main__":
    main()
