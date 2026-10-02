"""
MODULE 11: Economic Significance — Volatility-Managed Portfolios
================================================================
Moreira-Muir (2017)-style volatility management applied stock by stock, on
the weekly forecast stride (h = 5). Rewritten 2026-10-02 after the audit
(issues I2, I5, I6, I26 in the audit doc; docs/EXPERIMENTS.md third block).

Construction, for stock i and weekly origin t:
    R_{i,t}      = simple return over trading days t+1..t+5 (price returns:
                   closes are split- but not dividend-adjusted)
    rf_t         = 3-month T-bill (USGG3M) compounded over 5 trading days
    x_{i,t}      = R_{i,t} - rf_t                      (excess return)
    s2_{m,i,t}   = exp(yhat_{m,i,t}), model m's forecast of mean daily
                   Parkinson variance over t+1..t+5
    w_{m,i,t}    = c_{m,i,t} / s2_{m,i,t}
    c_{m,i,t}    = sd(x_i) / sd(x_i / s2_{m,i}) over weeks strictly before t
                   (expanding, 52-week warm-up): real-time normalisation
    portfolio m  = (1/N_t) sum_i w_{m,i,t} x_{i,t}     (excess return; the
                   unlevered remainder earns the bill rate)

Portfolios: equal-weight (weekly rebalanced, w = 1), HAR-, HAR-X- and
C-managed, a model-free benchmark whose variance forecast is the trailing
22-day mean Parkinson variance through day t, and C-managed with the
full-sample Moreira-Muir constant (robustness). All portfolios use the same
(date, stock) cells. Sharpe ratios, CER and drawdowns are on excess returns
(drawdown on total-return wealth); turnover counts drift between rebalances;
net Sharpe ratios are reported at 0, 5 and 10 bp per unit of notional traded.

Outputs:
    results/tables/table10_volmanaged.tex, results/tables/table10b_costs.tex
    results/intermediate/table10_raw.csv, table10b_costs.csv, portfolio_weekly.csv
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
    "figure.dpi": 110, "font.size": 12.5, "axes.labelsize": 13.5,
    "axes.titlesize": 14.5, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 11.5, "font.family": "serif",
})
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
FCST = BASE / "results" / "intermediate" / "forecasts"
INTERM = BASE / "results" / "intermediate"
TABLES_DIR = BASE / "results" / "tables"
FIG_DIR = BASE / "results" / "figures"

H = 5                        # rebalance horizon (trading days)
GAMMA = 5.0                  # risk aversion for CER
ANN_PER_PERIOD = 252 / H     # five-day periods per year
C_WARMUP_WEEKS = 52
COST_BPS = (0, 5, 10)

MANAGED = {"A": "HAR-managed", "A1": "HAR-X-managed", "C": "C-managed"}
UNMANAGED = "Equal-weight (unmanaged)"
NAIVE = "Trailing-22d-RV-managed"
C_FULL = "C-managed (full-sample c)"
ORDER = [UNMANAGED, "HAR-managed", "HAR-X-managed", NAIVE, "C-managed", C_FULL]


# ------------------------------------------------------------------ inputs
def five_day_simple_returns(bundle, sample_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Simple return of each stock over trading days t+1..t+5, from raw
    (unwinsorised) log returns. A window with no observed return is NaN
    (the old code summed it to zero)."""
    lr = bundle.returns_raw
    pos = lr.index.get_indexer(sample_dates)
    arr = lr.values
    out = np.full((len(sample_dates), lr.shape[1]), np.nan)
    for k, p in enumerate(pos):
        if p < 0 or p + H >= len(lr.index):
            continue
        block = arr[p + 1: p + 1 + H, :]
        ok = ~np.isnan(block).all(axis=0)
        out[k, ok] = np.expm1(np.nansum(block[:, ok], axis=0))
    return pd.DataFrame(out, index=sample_dates, columns=lr.columns)


def weekly_riskfree(bundle, sample_dates: pd.DatetimeIndex) -> pd.Series:
    """3-month T-bill (percent, annual) compounded over five trading days,
    observed at the origin."""
    y = bundle.panel.market["USGG3M"].ffill().reindex(sample_dates).ffill()
    return (1.0 + y.clip(lower=0) / 100.0) ** (H / 252.0) - 1.0


def load_yhat(model: str) -> pd.DataFrame | None:
    fp = FCST / f"{model}_h{H:02d}_yhat.csv"
    return pd.read_csv(fp, index_col=0, parse_dates=True) if fp.exists() else None


def realtime_constant(x: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """c_{i,t} = sd(x_i)/sd(raw_i) over the weeks strictly before t."""
    sd_x = x.expanding(min_periods=C_WARMUP_WEEKS).std().shift(1)
    sd_r = raw.expanding(min_periods=C_WARMUP_WEEKS).std().shift(1)
    return sd_x / sd_r


# ------------------------------------------------------------------ build
def build_portfolios(bundle) -> tuple[pd.DataFrame, dict]:
    """Weekly EXCESS returns of every portfolio on common cells, plus
    diagnostics (weights, turnover, total returns)."""
    sd = bundle.sample_dates
    R = five_day_simple_returns(bundle, sd)
    rf = weekly_riskfree(bundle, sd)
    X = R.sub(rf, axis=0)

    s2 = {}
    for m in MANAGED:
        yh = load_yhat(m)
        s2[m] = np.exp(yh.reindex(index=sd, columns=X.columns))
    naive = bundle.feat["har_m"].reindex(index=sd, columns=X.columns)
    s2["naive"] = naive.where(naive > 0)

    # common universe: every forecast, the benchmark and the return exist
    cell = X.notna()
    for v in s2.values():
        cell &= v.notna()
    X = X.where(cell)

    weights = {}
    for key, v in s2.items():
        raw = X / v
        c = realtime_constant(X, raw)
        weights[key] = (c / v).where(cell)
    yhC = s2["C"]
    raw = X / yhC
    c_full = X.std() / raw.std()
    weights["C_full"] = (1.0 / yhC).mul(c_full, axis=1).where(cell)

    # evaluation weeks: every managed weight defined for at least one stock
    live = pd.Series(True, index=sd)
    for w in weights.values():
        live &= w.notna().any(axis=1)
    live &= cell.any(axis=1)

    labels = {"A": "HAR-managed", "A1": "HAR-X-managed", "C": "C-managed",
              "naive": NAIVE, "C_full": C_FULL}
    out = {UNMANAGED: X.where(weights["C"].notna()).mean(axis=1)}
    W = {UNMANAGED: pd.DataFrame(1.0, index=sd, columns=X.columns).where(weights["C"].notna())}
    for key, w in weights.items():
        w = w.where(weights["C"].notna())
        out[labels[key]] = (w * X).sum(axis=1, min_count=1) / w.notna().sum(axis=1)
        W[labels[key]] = w
    P = pd.DataFrame(out).loc[live]
    diag = {"weights": {k: v.loc[live] for k, v in W.items()},
            "R": R.loc[live], "rf": rf.loc[live]}
    return P[ORDER], diag


def turnover(w: pd.DataFrame, R: pd.DataFrame, rf: pd.Series) -> pd.Series:
    """Notional traded per unit of capital each week, including drift.
    Position fraction of stock i = w_i / N; the remainder sits in bills."""
    n = w.notna().sum(axis=1).replace(0, np.nan)
    frac = w.div(n, axis=0).fillna(0.0)
    port_tot = (frac * R.fillna(0.0)).sum(axis=1) + (1 - frac.sum(axis=1)) * rf
    drifted = frac.mul(1 + R.fillna(0.0)).div(1 + port_tot, axis=0)
    return (frac - drifted.shift(1)).abs().sum(axis=1).iloc[1:]


def stats(x: pd.Series, rf: pd.Series) -> dict[str, float]:
    x = x.dropna()
    if len(x) < 5:
        return {k: np.nan for k in ("mean_ann", "vol_ann", "sharpe", "max_dd", "cer")} | {"n_periods": len(x)}
    mean_ann = float(x.mean() * ANN_PER_PERIOD)
    vol_ann = float(x.std(ddof=1) * np.sqrt(ANN_PER_PERIOD))
    wealth = np.cumprod(1.0 + x.values + rf.reindex(x.index).values)
    dd = float((wealth / np.maximum.accumulate(wealth) - 1.0).min())
    return {"mean_ann": mean_ann, "vol_ann": vol_ann,
            "sharpe": mean_ann / vol_ann if vol_ann > 0 else np.nan,
            "max_dd": dd, "cer": mean_ann - 0.5 * GAMMA * vol_ann ** 2,
            "n_periods": int(len(x))}


def regime_masks_weekly(idx: pd.DatetimeIndex, bundle) -> dict[str, np.ndarray]:
    """Same regime definition as every forecast table: VIX quartiles over the
    out-of-sample forecast dates (audit I10)."""
    from modules.module6_forecast_eval import regime_masks
    ref = load_yhat("A")
    ref_idx = ref.index[ref.notna().any(axis=1)]
    return {"Full sample": np.ones(len(idx), dtype=bool)} | regime_masks(idx, bundle.market, ref_idx)


# ------------------------------------------------------------------ main
def main() -> None:
    print("=" * 70)
    print("   MODULE 11: VOLATILITY-MANAGED PORTFOLIOS (excess returns)")
    print("=" * 70)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    P, diag = build_portfolios(bundle)
    rf = diag["rf"]
    print(f"  evaluation weeks: {len(P)}  ({P.index[0].date()} to {P.index[-1].date()})")
    P.assign(rf=rf).to_csv(INTERM / "portfolio_weekly.csv")

    masks = regime_masks_weekly(P.index, bundle)
    rows = [{"regime": reg, "portfolio": lab, **stats(P[lab][m], rf)}
            for reg, m in masks.items() for lab in ORDER]
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "table10_raw.csv", index=False)
    print(df.pivot_table(index="portfolio", columns="regime", values="sharpe").round(3).to_string())

    # costs, turnover and leverage
    crow = []
    for lab in ORDER:
        w = diag["weights"][lab]
        to = turnover(w, diag["R"], rf).reindex(P.index).fillna(0.0)
        gross = w.sum(axis=1) / w.notna().sum(axis=1)
        wv = w.stack()
        row = {"portfolio": lab, "turnover_ann": float(to.mean() * ANN_PER_PERIOD),
               "w_median": float(wv.median()), "w_p99": float(wv.quantile(0.99)),
               "w_max": float(wv.max()), "gross_mean": float(gross.mean()),
               "gross_min": float(gross.min()), "gross_max": float(gross.max())}
        for bp in COST_BPS:
            row[f"sharpe_{bp}bp"] = stats(P[lab] - bp / 1e4 * to, rf)["sharpe"]
        crow.append(row)
    cdf = pd.DataFrame(crow)
    cdf.to_csv(INTERM / "table10b_costs.csv", index=False)
    print(cdf.round(3).to_string(index=False))

    # ---- Table 10
    with open(TABLES_DIR / "table10_volmanaged.tex", "w", encoding="utf-8") as f:
        f.write("% Table 10: volatility-managed portfolios on excess returns (audit rewrite 2026-10-02)\n\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Volatility-Managed Portfolios: Excess-Return Performance}\n")
        f.write("\\label{tab:volmanaged}\n\\small\n\\begin{tabular}{llccccc}\n\\toprule\n")
        f.write("Regime & Portfolio & Ann.\\ excess ret. & Ann.\\ vol. & Sharpe & Max DD & CER \\\\\n\\midrule\n")
        for reg in masks:
            for i, lab in enumerate(ORDER):
                r = df[(df.regime == reg) & (df.portfolio == lab)].iloc[0]
                f.write(f"{reg if i == 0 else ''} & {lab} & {r.mean_ann*100:+.2f}\\% & {r.vol_ann*100:.2f}\\% & "
                        f"{r.sharpe:+.2f} & {r.max_dd*100:.1f}\\% & {r.cer*100:+.2f}\\% \\\\\n")
            f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\begin{tablenotes}\\small\n")
        f.write("\\item Notes: Weekly returns over trading days $t+1$ to $t+5$ in excess of the three-month "
                "Treasury bill. Managed positions are $w_{m,i,t} = c_{m,i,t}/\\hat\\sigma^2_{m,i,t}$ per stock, "
                "with $c_{m,i,t}$ equating managed and unmanaged excess-return variance over the weeks before $t$ "
                "(52-week warm-up); the unlevered remainder earns the bill rate. The trailing-22d-RV benchmark "
                "uses the mean Parkinson variance over the last 22 trading days as its variance forecast and no "
                "model. All portfolios use the same (date, stock) cells. Maximum drawdown is on total-return "
                "wealth; CER is for a mean--variance investor with $\\gamma=5$. Prices are split- but not "
                "dividend-adjusted. VIX quartiles over the out-of-sample forecast dates.\n")
        f.write("\\end{tablenotes}\n\\end{table}\n")

    with open(TABLES_DIR / "table10b_costs.tex", "w", encoding="utf-8") as f:
        f.write("% Table 10b: turnover, leverage and net Sharpe ratios\n\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Turnover, Leverage and Sharpe Ratios Net of Trading Costs}\n")
        f.write("\\label{tab:costs}\n\\small\n\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("Portfolio & Turnover (x/yr) & Weight p99 & Weight max & Sharpe, 0 bp & 5 bp & 10 bp \\\\\n\\midrule\n")
        for r in cdf.itertuples():
            f.write(f"{r.portfolio} & {r.turnover_ann:.1f} & {r.w_p99:.2f} & {r.w_max:.2f} & "
                    f"{r.sharpe_0bp:+.2f} & {r.sharpe_5bp:+.2f} & {r.sharpe_10bp:+.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\begin{tablenotes}\\small\n")
        f.write("\\item Notes: Turnover is notional traded per unit of capital per year, including drift "
                "between weekly rebalances. Weights are per-stock position multipliers $w_{m,i,t}$. Net "
                "Sharpe ratios deduct the stated cost per unit of notional traded.\n")
        f.write("\\end{tablenotes}\n\\end{table}\n")
    print(f"Saved Tables 10 and 10b")

    # ---- Figure 9: total-return wealth
    fig, ax = plt.subplots(figsize=(11, 5.2))
    colors = {UNMANAGED: "#666666", "HAR-managed": "#2E86AB", "HAR-X-managed": "#3CB371",
              NAIVE: "#B8860B", "C-managed": "#C0392B", C_FULL: "#000000"}
    for lab in ORDER:
        x = P[lab].dropna()
        ax.plot(x.index, np.cumprod(1 + x.values + rf.reindex(x.index).values),
                lw=1.5, label=lab, color=colors[lab])
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-12-31"), alpha=0.15,
               color="#E74C3C", label="COVID")
    ax.set_yscale("log")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth, total return (log scale)")
    ax.set_title("Volatility-managed portfolios vs equal weight (115 stocks)")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_volmanaged_cumulative.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 9")


if __name__ == "__main__":
    main()
