"""
MODULE 1B: Preliminary Diagnostics and Model Illustration (v2)
==============================================================
Fits ARFIMA(1,d,1)-FIGARCH(1,d,1) on a sample of stocks; produces:
  - Figure: 4-panel preliminary diagnostics (return distribution, vol
    correlation heatmap, FIGARCH conditional vs realized, QQ-plot)
  - Table: FIGARCH parameter estimates across the sample.

Operates on the new Bloomberg panel via modules.io_v2.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from arch import arch_model
from scipy import stats

from modules.io_v2 import build_clean_panel

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
TABLES_DIR = BASE / "results" / "tables"
FIGURES_DIR = BASE / "results" / "figures"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (14, 12),
    "font.size": 12.5,
    "axes.labelsize": 13.5,
    "axes.titlesize": 14.5,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11.5,
    "font.family": "serif",
})

SAMPLE_STOCKS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "GE", "WMT", "KO", "PG", "BA"]


def fit_figarch_one(r: pd.Series, p: int = 1, q: int = 1, power: float = 2.0):
    y = (r.dropna() * 100)
    try:
        model = arch_model(y, mean="AR", lags=1, vol="FIGARCH", p=p, q=q,
                           power=power, dist="normal")
        res = model.fit(disp="off", show_warning=False)
        return res, res.conditional_volatility / 100, "FIGARCH"
    except Exception as exc:
        print(f"  FIGARCH failed ({exc}); falling back to GARCH(1,1)")
        model = arch_model(y, mean="AR", lags=1, vol="GARCH", p=1, q=1)
        res = model.fit(disp="off", show_warning=False)
        return res, res.conditional_volatility / 100, "GARCH"


def fit_panel(returns: pd.DataFrame, stocks: list[str]) -> dict:
    out = {}
    print(f"\nFitting FIGARCH to {len(stocks)} stocks...")
    for i, s in enumerate(stocks, 1):
        if s not in returns.columns:
            print(f"  [{i}/{len(stocks)}] {s}: not in panel — skipped")
            continue
        print(f"  [{i}/{len(stocks)}] {s}...", end=" ")
        res, cond_vol, kind = fit_figarch_one(returns[s])
        d_hat = res.params["d"] if "d" in res.params.index else None
        if d_hat is not None:
            print(f"d={d_hat:.3f}")
        else:
            print(f"({kind})")
        out[s] = {"model": res, "conditional_vol": cond_vol, "d_hat": d_hat, "kind": kind}
    return out


def figure_diagnostics(returns: pd.DataFrame, rv_pk: pd.DataFrame,
                       results: dict, fp: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    all_r = returns.values.flatten()
    all_r = all_r[~np.isnan(all_r)]
    mu, sig = stats.norm.fit(all_r)

    # (a) Distribution
    ax = axes[0, 0]
    ax.hist(all_r, bins=120, density=True, alpha=0.7, color="steelblue",
            edgecolor="white", linewidth=0.4, label="Empirical")
    x = np.linspace(all_r.min(), all_r.max(), 300)
    ax.plot(x, stats.norm.pdf(x, mu, sig), "r-", linewidth=2,
            label=f"Normal ($\\mu$={mu:.4f}, $\\sigma$={sig:.4f})")
    df, loc, scale = stats.t.fit(all_r)
    ax.plot(x, stats.t.pdf(x, df, loc, scale), "g--", linewidth=2,
            label=f"Student-t (df={df:.1f})")
    ax.set_xlabel("Log Return"); ax.set_ylabel("Density")
    ax.set_title("(a) Log-Return Distribution: Empirical vs Theoretical", fontweight="bold")
    ax.legend(loc="upper right"); ax.set_xlim(-0.15, 0.15)
    txt = (f"Skewness: {stats.skew(all_r):.3f}\n"
           f"Kurtosis: {stats.kurtosis(all_r):.2f}\n"
           f"JB Stat: {stats.jarque_bera(all_r)[0]:.0f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # (b) Vol correlation heatmap (Parkinson RV)
    ax = axes[0, 1]
    sub = [s for s in SAMPLE_STOCKS if s in rv_pk.columns]
    vol_subset = rv_pk[sub].dropna()
    corr = vol_subset.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                annot_kws={"size": 9})
    ax.set_title("(b) Cross-Stock Parkinson-RV Correlation", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # (c) FIGARCH cond vol vs Parkinson RV (sqrt -> vol scale)
    ax = axes[1, 0]
    sample = "AAPL" if "AAPL" in results else next(iter(results))
    cond = results[sample]["conditional_vol"]
    d_hat = results[sample]["d_hat"]
    realized_vol = np.sqrt(rv_pk[sample]).dropna()
    common = cond.index.intersection(realized_vol.index)
    rs = realized_vol.loc[common].rolling(22).mean()
    cs = cond.loc[common].rolling(22).mean()
    ax.plot(rs.index, rs, "steelblue", linewidth=0.8, alpha=0.85,
            label="Realized Vol ($\\sqrt{RV^{PK}}$, 22-day MA)")
    ax.plot(cs.index, cs, "darkred", linewidth=1.2, alpha=0.9,
            label="FIGARCH Conditional Vol")
    ax.axvspan(pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-30"), alpha=0.15, color="gray")
    ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-30"), alpha=0.15, color="gray")
    d_label = f" (d={d_hat:.3f})" if d_hat is not None else " (GARCH)"
    ax.set_title(f"(c) FIGARCH Fitted vs Realized Volatility: {sample}{d_label}", fontweight="bold")
    ax.set_ylabel("Volatility"); ax.set_xlabel("")
    ax.legend(loc="upper right")

    # (d) QQ
    ax = axes[1, 1]
    rng = np.random.default_rng(0)
    n_sample = min(10000, len(all_r))
    sample_r = rng.choice(all_r, n_sample, replace=False)
    sorted_r = np.sort(sample_r)
    n = len(sorted_r)
    th = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    ax.scatter(th, sorted_r, s=3, alpha=0.5, color="steelblue")
    slope, intercept = np.polyfit(th, sorted_r, 1)
    ax.plot(th, slope * th + intercept, "r-", linewidth=2, label="Reference line")
    xlim = ax.get_xlim()
    ax.plot([xlim[0], xlim[1]], [xlim[0] * sig + mu, xlim[1] * sig + mu],
            "g--", linewidth=1.5, alpha=0.7, label="Normal theoretical")
    ax.set_xlabel("Theoretical Quantiles (Normal)"); ax.set_ylabel("Sample Quantiles")
    ax.set_title("(d) QQ-Plot: Log Returns vs Normal Distribution", fontweight="bold")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.savefig(str(fp).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()


def export_figarch_table(results: dict, fp: Path) -> None:
    rows = []
    for stock, res in results.items():
        m = res["model"]
        params = m.params
        rows.append({
            "Stock": stock,
            "omega": params.get("omega", np.nan),
            "d": params.get("d", np.nan),
            "phi": params.get("phi", np.nan),
            "beta": params.get("beta[1]", params.get("beta", np.nan)),
            "AIC": m.aic, "BIC": m.bic, "LogLik": m.loglikelihood,
        })
    df = pd.DataFrame(rows)

    with open(fp, "w") as f:
        f.write("% Table: FIGARCH(1,d,1) Estimates\n\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{FIGARCH(1,d,1) Model Estimates}\n")
        f.write("\\label{tab:figarch_estimates}\n\\small\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("Stock & $\\omega$ & $d$ & $\\phi$ & $\\beta$ & AIC & BIC \\\\\n\\midrule\n")
        for _, row in df.iterrows():
            d_str = f"{row['d']:.3f}" if not np.isnan(row["d"]) else "--"
            phi_str = f"{row['phi']:.3f}" if not np.isnan(row["phi"]) else "--"
            beta_str = f"{row['beta']:.3f}" if not np.isnan(row["beta"]) else "--"
            f.write(f"{row['Stock']} & {row['omega']:.4f} & {d_str} & {phi_str} & "
                    f"{beta_str} & {row['AIC']:.1f} & {row['BIC']:.1f} \\\\\n")
        f.write("\\midrule\n")
        d_vals = df["d"].dropna()
        if len(d_vals):
            f.write(f"\\textbf{{Mean}} & -- & \\textbf{{{d_vals.mean():.3f}}} & -- & -- & -- & -- \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write("\\item Notes: FIGARCH(1,d,1) with AR(1) mean and normal innovations. "
                "$d \\in (0, 0.5)$ indicates stationary long memory in conditional variance.\n")
        f.write("\\end{tablenotes}\n\\end{table}\n")


def diagnostic_summary(returns: pd.DataFrame, results: dict) -> None:
    print("\n" + "=" * 70)
    print("   DIAGNOSTIC SUMMARY")
    print("=" * 70)
    r = returns.values.flatten()
    r = r[~np.isnan(r)]
    print("\n1. RETURN DISTRIBUTION")
    print(f"   N={len(r):,}  mean(ann)={r.mean()*252*100:.2f}%  std(ann)={r.std()*np.sqrt(252)*100:.2f}%")
    print(f"   Skewness={stats.skew(r):.3f}  Kurtosis={stats.kurtosis(r):.2f}  JB={stats.jarque_bera(r)[0]:.0f}")
    print("\n2. FIGARCH d ESTIMATES")
    ds = [v["d_hat"] for v in results.values() if v["d_hat"] is not None]
    if ds:
        print(f"   N={len(ds)}  mean d={np.mean(ds):.3f}  std={np.std(ds):.3f}  range=[{min(ds):.3f}, {max(ds):.3f}]")
    else:
        print("   None — all fits fell back to GARCH")


def main() -> None:
    print("=" * 70)
    print("   MODULE 1B: PRELIMINARY DIAGNOSTICS")
    print("=" * 70)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_clean_panel()
    print(f"  N={len(panel.kept)}  T={len(panel.prices)}")

    results = fit_panel(panel.returns, SAMPLE_STOCKS)

    figure_diagnostics(panel.returns, panel.rv_parkinson, results,
                       FIGURES_DIR / "fig_preliminary_diagnostics.pdf")
    export_figarch_table(results, TABLES_DIR / "table_figarch_estimates.tex")
    diagnostic_summary(panel.returns, results)

    print("\n" + "=" * 70)
    print("Outputs:")
    print(f"  {FIGURES_DIR / 'fig_preliminary_diagnostics.pdf'} (+ .png)")
    print(f"  {TABLES_DIR / 'table_figarch_estimates.tex'}")


if __name__ == "__main__":
    main()
