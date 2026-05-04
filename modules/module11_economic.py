"""
MODULE 11: Economic Significance — Volatility-Managed Portfolios
================================================================
Moreira-Muir (2017) volatility-managed portfolio test.

Setup (h=5, weekly rebalancing on the same sample-date stride as forecasts):
    sigma2_hat_{m,i,t} = exp( yhat_{m,i,t} )
        = model-m forecast of mean RV^{PK} over [t+1, t+5]
    r_{i,t->t+5}        = sum of daily log returns over the next 5 trading days
    f_{m,i,t}           = (c_{m,i} / sigma2_hat_{m,i,t}) * r_{i,t->t+5}
    c_{m,i}             chosen per-stock so that var(f_{m,i}) = var(r_{i})
                        (Moreira-Muir normalization: managed and unmanaged
                        have the same unconditional variance per stock).
    portfolio:          equal-weight average across the 115 stocks each week.

We compare:
    * unmanaged buy-and-hold equal-weight portfolio
    * Model-A-managed portfolio (HAR baseline)
    * Model-A1-managed portfolio (HAR-X)
    * Model-C-managed portfolio (full structural)

Metrics: annualized return, annualized volatility, Sharpe, max drawdown,
CER for a mean-variance investor with risk aversion gamma = 5. Also reports
the same metrics within calm / high-VIX / COVID regimes.

Outputs:
    results/tables/table10_volmanaged.tex
    results/intermediate/table10_raw.csv
    results/figures/fig9_volmanaged_cumulative.pdf
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.forecast_io import load_bundle

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 12.5,
    "axes.labelsize": 13.5,
    "axes.titlesize": 14.5,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11.5,
    "font.family": "serif",
})

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
FCST = BASE / "results" / "intermediate" / "forecasts"
INTERM = BASE / "results" / "intermediate"
TABLES_DIR = BASE / "results" / "tables"
FIG_DIR = BASE / "results" / "figures"

H = 5                # forecast / rebalance horizon
GAMMA = 5.0          # risk aversion for CER
ANN_PER_PERIOD = 252 / H  # annualization factor for 5-day returns


def _five_day_log_returns(bundle, sample_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """For each sample date t, compute sum of next-H daily log returns per stock."""
    ret_full = bundle.returns
    full_dates = ret_full.index
    pos = full_dates.get_indexer(sample_dates)
    out = np.full((len(sample_dates), ret_full.shape[1]), np.nan)
    arr = ret_full.values
    for k, p in enumerate(pos):
        if p < 0 or p + H >= len(full_dates):
            continue
        out[k, :] = np.nansum(arr[p + 1: p + 1 + H, :], axis=0)
    return pd.DataFrame(out, index=sample_dates, columns=ret_full.columns)


def _stats(r: pd.Series) -> dict[str, float]:
    r = r.dropna()
    if len(r) < 5:
        return {"mean_ann": np.nan, "vol_ann": np.nan,
                "sharpe": np.nan, "max_dd": np.nan, "cer": np.nan,
                "n_periods": int(len(r))}
    mean_ann = float(r.mean() * ANN_PER_PERIOD)
    vol_ann = float(r.std() * np.sqrt(ANN_PER_PERIOD))
    sharpe = mean_ann / vol_ann if vol_ann > 0 else np.nan
    cum = np.exp(r.cumsum())
    peak = np.maximum.accumulate(cum.values)
    dd = float((cum.values / peak - 1).min())
    cer = mean_ann - 0.5 * GAMMA * vol_ann ** 2
    return {"mean_ann": mean_ann, "vol_ann": vol_ann,
            "sharpe": float(sharpe), "max_dd": dd, "cer": cer,
            "n_periods": int(len(r))}


def _vol_managed_returns(yhat: pd.DataFrame, ret_h: pd.DataFrame
                         ) -> tuple[pd.Series, pd.DataFrame]:
    """Returns (portfolio_managed, per_stock_managed_returns).

    Per-stock normalization: c_i = std(r_i) / std(r_i / sigma2_hat_i).
    Then portfolio = equal-weight mean across stocks.
    """
    common_idx = yhat.index.intersection(ret_h.index)
    common_cols = sorted(set(yhat.columns) & set(ret_h.columns))
    yh = yhat.loc[common_idx, common_cols]
    rh = ret_h.loc[common_idx, common_cols]
    sigma2 = np.exp(yh)
    raw = rh / sigma2
    c = rh.std(axis=0, skipna=True) / raw.std(axis=0, skipna=True)
    managed = raw.mul(c, axis=1)
    port = managed.mean(axis=1, skipna=True)
    return port, managed


def regime_masks_5d(idx: pd.DatetimeIndex,
                    market: pd.DataFrame) -> dict[str, np.ndarray]:
    vix = market["VIX"].reindex(idx).ffill()
    q1, q3 = vix.quantile(0.25), vix.quantile(0.75)
    return {
        "Full sample": np.ones(len(idx), dtype=bool),
        "Low VIX (Q1)": (vix <= q1).values,
        "High VIX (Q4)": (vix >= q3).values,
        "COVID 2020": ((idx >= "2020-03-01") & (idx <= "2020-12-31")),
    }


def _load_model_yhat(model: str) -> pd.DataFrame | None:
    fp = FCST / f"{model}_h{H:02d}_yhat.csv"
    if not fp.exists():
        return None
    return pd.read_csv(fp, index_col=0, parse_dates=True)


def main() -> None:
    print("=" * 70)
    print("   MODULE 11: ECONOMIC SIGNIFICANCE — VOL-MANAGED PORTFOLIO")
    print("=" * 70)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle()
    sd = bundle.sample_dates

    # 5-day forward log returns aligned to each sample date
    ret_h = _five_day_log_returns(bundle, sd)

    # unmanaged equal-weight portfolio
    port_unmanaged = ret_h.mean(axis=1, skipna=True)

    portfolios = {"Unmanaged (buy-and-hold)": port_unmanaged}
    for model in ["A", "A1", "C"]:
        yhat = _load_model_yhat(model)
        if yhat is None:
            print(f"  [WARN] no yhat for {model}; skipping")
            continue
        # restrict to OOS dates where forecast is non-null
        common_idx = yhat.index.intersection(ret_h.index)
        # only include rows where yhat has at least one stock
        keep = yhat.loc[common_idx].notna().any(axis=1)
        common_idx = common_idx[keep]
        yhat = yhat.loc[common_idx]
        port_m, _ = _vol_managed_returns(yhat, ret_h.loc[common_idx])
        portfolios[f"{model}-managed"] = port_m

    # Align all portfolios on a common OOS index — the intersection of
    # non-null dates across managed portfolios.
    managed_keys = [k for k in portfolios if "managed" in k]
    if managed_keys:
        common = portfolios[managed_keys[0]].dropna().index
        for k in managed_keys[1:]:
            common = common.intersection(portfolios[k].dropna().index)
        portfolios = {k: v.reindex(common) for k, v in portfolios.items()}

    # build market-state masks once on common index
    masks = regime_masks_5d(common, bundle.market)

    rows = []
    for regime, mask in masks.items():
        for label, p in portfolios.items():
            sub = p[mask]
            s = _stats(sub)
            rows.append({"regime": regime, "portfolio": label, **s})
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "table10_raw.csv", index=False)

    print()
    pivot_sharpe = df.pivot_table(index="portfolio", columns="regime",
                                  values="sharpe").round(3)
    pivot_cer = df.pivot_table(index="portfolio", columns="regime",
                               values="cer").round(4)
    print("Annualized Sharpe ratios:")
    print(pivot_sharpe.to_string())
    print("\nAnnualized CER (gamma=5):")
    print(pivot_cer.to_string())

    # --- LaTeX export ---
    fp = TABLES_DIR / "table10_volmanaged.tex"
    portfolios_order = ["Unmanaged (buy-and-hold)", "A-managed",
                        "A1-managed", "C-managed"]
    portfolios_order = [p for p in portfolios_order if p in portfolios]
    cols_metric = ["mean_ann", "vol_ann", "sharpe", "max_dd", "cer"]
    metric_label = {"mean_ann": "Ann.\\ Ret.", "vol_ann": "Ann.\\ Vol.",
                    "sharpe": "Sharpe", "max_dd": "Max DD", "cer": "CER"}
    with open(fp, "w") as f:
        f.write("% Table 10: Vol-managed portfolio metrics by regime\n\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Volatility-Managed Portfolios: "
                "Risk-Adjusted Performance (Moreira-Muir 2017 Construction)}\n")
        f.write("\\label{tab:volmanaged}\n\\small\n")
        f.write("\\begin{tabular}{ll" + "c" * len(cols_metric) + "}\n\\toprule\n")
        f.write("Regime & Portfolio")
        for c in cols_metric:
            f.write(f" & {metric_label[c]}")
        f.write(" \\\\\n\\midrule\n")
        prev = None
        for regime in masks:
            for p in portfolios_order:
                row = df[(df["regime"] == regime) & (df["portfolio"] == p)]
                if row.empty:
                    continue
                r = row.iloc[0]
                reg_disp = "" if regime == prev else regime
                f.write(f"{reg_disp} & {p}")
                for c in cols_metric:
                    v = r[c]
                    if pd.isna(v):
                        f.write(" & --")
                    elif c in ("mean_ann", "vol_ann", "max_dd"):
                        f.write(f" & {v*100:+.2f}\\%")
                    elif c == "cer":
                        f.write(f" & {v*100:+.2f}\\%")
                    else:
                        f.write(f" & {v:+.2f}")
                f.write(" \\\\\n")
                prev = regime
            f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write(
            "\\item Notes: Equal-weight portfolio of 115 stocks, weekly "
            "($h=5$ trading days) rebalancing aligned to the forecast sample "
            "stride. The volatility-managed portfolio uses position weight "
            "$w_{m,i,t} = c_{m,i} / \\hat\\sigma^2_{m,i,t}$ where "
            "$\\hat\\sigma^2_{m,i,t} = \\exp(\\hat y_{m,i,t})$ is the "
            "model-$m$ forecast of mean $RV^{PK}$ over $[t+1, t+5]$. The "
            "constant $c_{m,i}$ is set so that $\\mathrm{Var}(f_{m,i}) "
            "= \\mathrm{Var}(r_i)$ at the stock level. Annualized assuming "
            "$252/5$ five-day periods per year. CER computed for a "
            "mean-variance investor with risk aversion $\\gamma = 5$.\n"
        )
        f.write("\\end{tablenotes}\n\\end{table}\n")
    print(f"\nSaved {fp}")

    # --- cumulative-wealth figure ---
    fig, ax = plt.subplots(figsize=(11, 5.2))
    color_map = {"Unmanaged (buy-and-hold)": "#666666",
                 "A-managed": "#2E86AB",
                 "A1-managed": "#3CB371",
                 "C-managed": "#C0392B"}
    for label in portfolios_order:
        p = portfolios[label].dropna()
        cum = np.exp(p.cumsum())
        ax.plot(cum.index, cum.values, lw=1.6, label=label,
                color=color_map.get(label, "black"))
    for start, end, lab, c in [
        ("2020-03-01", "2020-12-31", "COVID", "#E74C3C"),
    ]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15,
                   color=c, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative wealth (log scale, $1 invested at OOS start)")
    ax.set_title("Volatility-managed portfolios vs buy-and-hold (115 stocks, EW)")
    ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    fig_fp = FIG_DIR / "fig9_volmanaged_cumulative.pdf"
    fig.savefig(fig_fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_fp}")


if __name__ == "__main__":
    main()
