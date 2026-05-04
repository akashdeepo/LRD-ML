"""
MODULE 9: Robustness Summary (Table 9)
======================================
For each robustness perturbation, refits / re-evaluates Model C at h=5 (and the
HAR baseline A) and reports MSE, %-improvement vs A, and HLN-corrected
Diebold-Mariano statistic.

Variants:
    headline          baseline (d_GPH window=750, target=log mean Parkinson RV)
    estimator_LW      replace d_GPH with d_LW in Model C
    window_500        d_GPH refit with rolling window 500 (and derived features)
    window_1000       d_GPH refit with rolling window 1000
    target_sqret      replace target with log mean future squared returns
    liquidity_low     restrict evaluation to low-illiquidity half of stocks
    liquidity_high    restrict to high-illiquidity half
    regime_high_vix   pre-existing high-VIX-quartile result (from Table 7)
    regime_low_vix    pre-existing low-VIX-quartile result
    regime_covid      pre-existing COVID 2020 result
    inference_plain   plain (iid pooled-cell) DM stat for headline (illustrative)

Output:
    results/tables/table9_robustness.tex
    results/intermediate/table9_raw.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from modules.forecast_io import (
    HORIZONS, MODEL_FEATURES, build_targets, load_bundle,
    aligned_xy, stock_matrix,
)
from modules.module2_lrd_estimation import (
    gph, local_whittle, rolling_panel,
)
from modules.module3_feature_engineering import (
    cross_sectional_features, memory_dynamics, sector_mean_panel,
)
from modules.module4_benchmarks import (
    INIT_TRAIN_FRAC, expanding_forecast,
)
from modules.module6_forecast_eval import (
    diebold_mariano, qlike_loss, squared_loss,
)

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
FCST = BASE / "results" / "intermediate" / "forecasts"
INTERM = BASE / "results" / "intermediate"
FEAT_DIR = INTERM / "features"
TABLES_DIR = BASE / "results" / "tables"

H = 5  # headline horizon for Table 9
STRIDE = 5


# ----------------------------------------------------------------------- helpers
def _evaluate(yhat: pd.DataFrame, y: pd.DataFrame,
              base_yhat: pd.DataFrame, base_y: pd.DataFrame,
              h: int, stocks: list[str] | None = None
              ) -> dict[str, float]:
    """Compute pooled MSE and HLN-DM for (yhat,y) vs the baseline (base_yhat,base_y).

    Restricts to a stock subset if `stocks` is given.
    """
    if stocks is not None:
        cols = [c for c in stocks if c in yhat.columns and c in base_yhat.columns]
        yhat = yhat[cols]; y = y[cols]
        base_yhat = base_yhat[cols]; base_y = base_y[cols]
    L = squared_loss(yhat, y)
    L_base = squared_loss(base_yhat, base_y)
    mse = float(np.nanmean(L.values))
    mse_base = float(np.nanmean(L_base.values))
    imp_pct = 100 * (1 - mse / mse_base)
    _, t_hln, p, T = diebold_mariano(L_base, L, h=h)
    return {"MSE": mse, "MSE_baseline_A": mse_base,
            "imp_pct": imp_pct, "HLN_DM_t": t_hln, "p": p, "T_dates": T}


def _refit_model_C(bundle, h: int = H,
                   feat_overrides: dict[str, pd.DataFrame] | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit Model C with optional feature-panel overrides applied to bundle.feat
    in place. Returns (yhat_panel, y_panel)."""
    if feat_overrides:
        for k, v in feat_overrides.items():
            bundle.feat[k] = v
    init_n = int(len(bundle.sample_dates) * INIT_TRAIN_FRAC)
    targets = build_targets(bundle)
    yhat_panel = pd.DataFrame(index=bundle.sample_dates,
                              columns=bundle.panel.kept, dtype=float)
    y_panel = pd.DataFrame(index=bundle.sample_dates,
                           columns=bundle.panel.kept, dtype=float)
    for tkr in bundle.panel.kept:
        sm = stock_matrix(bundle, tkr, "C", targets)
        X, y = aligned_xy(sm, h)
        if len(X) < init_n + 5:
            continue
        yhat = expanding_forecast(X, y, init_n)
        yhat_panel.loc[yhat.index, tkr] = yhat.values
        y_panel.loc[y.index, tkr] = y.values
    return yhat_panel, y_panel


def _build_derived_from_d(d_panel: pd.DataFrame, sectors: dict
                          ) -> dict[str, pd.DataFrame]:
    """Reproduce the d-derived feature panels used by Model C from a candidate
    d panel. The panel must already be on the sample-date index."""
    dyn = memory_dynamics(d_panel)
    cs = cross_sectional_features(d_panel)
    sec = sector_mean_panel(d_panel, sectors)
    return {
        "d_gph": d_panel,
        "delta_d_gph": dyn["delta"],
        "vol_d_gph": dyn["vol"],
        "trend_d_gph": dyn["trend"],
        # cs features stored as a single multi-column frame in bundle.cs;
        # for substitution we keep mean and std only since C uses those.
        "_cs_mean_d": cs["cs_mean_d"],
        "_cs_std_d": cs["cs_std_d"],
        "sector_mean_d": sec,
    }


def _apply_d_overrides(bundle, overrides: dict[str, pd.DataFrame]) -> None:
    """Substitute d-derived feature panels and rebuild d*VIX, d*MOVE."""
    bundle.feat["d_gph"] = overrides["d_gph"]
    bundle.feat["delta_d_gph"] = overrides["delta_d_gph"]
    bundle.feat["vol_d_gph"] = overrides["vol_d_gph"]
    bundle.feat["trend_d_gph"] = overrides["trend_d_gph"]
    bundle.feat["sector_mean_d"] = overrides["sector_mean_d"]
    # cs frame
    bundle.cs = bundle.cs.copy()
    bundle.cs["cs_mean_d"] = overrides["_cs_mean_d"].reindex(bundle.cs.index)
    bundle.cs["cs_std_d"] = overrides["_cs_std_d"].reindex(bundle.cs.index)
    # interactions
    vix = bundle.market["VIX"]
    move = bundle.market["MOVE"]
    bundle.feat["d_x_vix"] = overrides["d_gph"].mul(vix, axis=0)
    bundle.feat["d_x_move"] = overrides["d_gph"].mul(move, axis=0)


def _rolling_d_at_window(bundle, window: int, estimator) -> pd.DataFrame:
    """Recompute rolling d on log Parkinson RV at a custom window. Returns a
    panel reindexed to bundle.sample_dates (forward-filled to the existing
    weekly-sample grid)."""
    log_rv = bundle.log_rv
    panel = rolling_panel(log_rv, estimator, window=window, stride=STRIDE,
                          label=f"rolling-d (window={window})")
    return panel.reindex(bundle.sample_dates).ffill()


# ---------------------------------------------------------------- variant runners
def variant_headline(bundle, base_yhat, base_y) -> dict:
    yhat = pd.read_csv(FCST / f"C_h{H:02d}_yhat.csv",
                       index_col=0, parse_dates=True)
    y = pd.read_csv(FCST / f"C_h{H:02d}_y.csv",
                    index_col=0, parse_dates=True)
    return _evaluate(yhat, y, base_yhat, base_y, H)


def variant_estimator_LW(bundle, base_yhat, base_y) -> dict:
    """Replace d_GPH with d_LW everywhere. Note: feat_d_lw lives at the same
    sample-date grid as feat_d_gph."""
    d_lw = bundle.feat["d_lw"]
    derived = _build_derived_from_d(d_lw, bundle.sectors)
    _apply_d_overrides(bundle, derived)
    yhat, y = _refit_model_C(bundle)
    return _evaluate(yhat, y, base_yhat, base_y, H)


def variant_window(bundle, window: int, base_yhat, base_y) -> dict:
    new_d = _rolling_d_at_window(bundle, window=window, estimator=gph)
    derived = _build_derived_from_d(new_d, bundle.sectors)
    _apply_d_overrides(bundle, derived)
    yhat, y = _refit_model_C(bundle)
    return _evaluate(yhat, y, base_yhat, base_y, H)


def variant_target_sqret(bundle, base_yhat, base_y) -> dict:
    """Replace forecasting target with log mean future squared returns. We
    refit Model C *and* Model A with the new target (so improvement is
    apples-to-apples)."""
    sq = bundle.returns ** 2
    full_dates = sq.index
    sd = bundle.sample_dates
    pos = full_dates.get_indexer(sd)
    new_target = np.full((len(sd), sq.shape[1]), np.nan)
    for k, p in enumerate(pos):
        if p < 0 or p + H >= len(full_dates):
            continue
        block = sq.values[p + 1: p + 1 + H, :]
        mean = np.nanmean(block, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            new_target[k, :] = np.where(mean > 0, np.log(mean), np.nan)
    new_target_df = pd.DataFrame(new_target, index=sd, columns=sq.columns)

    init_n = int(len(sd) * INIT_TRAIN_FRAC)

    def _fit(model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        targets = {h: new_target_df for h in HORIZONS}
        yh = pd.DataFrame(index=sd, columns=bundle.panel.kept, dtype=float)
        yy = pd.DataFrame(index=sd, columns=bundle.panel.kept, dtype=float)
        for tkr in bundle.panel.kept:
            sm = stock_matrix(bundle, tkr, model, targets)
            X, y = aligned_xy(sm, H)
            if len(X) < init_n + 5:
                continue
            yh.loc[X.index, tkr] = expanding_forecast(X, y, init_n).values
            yy.loc[y.index, tkr] = y.values
        return yh, yy

    yh_C, yy_C = _fit("C")
    yh_A, yy_A = _fit("A")
    return _evaluate(yh_C, yy_C, yh_A, yy_A, H)


def variant_liquidity(bundle, base_yhat, base_y, half: str = "low") -> dict:
    """Split stocks by static-median illiquidity (mean inverse dollar volume
    across the full panel). Low-illiquidity = high-liquidity, large-cap stocks;
    high-illiquidity = small/mid-cap stocks where the persistence signal might
    be noisier."""
    BB = BASE / "bloomberg_pull" / "processed"
    prc = pd.read_csv(BB / "prices_close.csv", index_col=0, parse_dates=True)
    vol = pd.read_csv(BB / "volume.csv", index_col=0, parse_dates=True)
    common = sorted(set(prc.columns) & set(vol.columns) & set(bundle.panel.kept))
    prc = prc[common]; vol = vol[common]
    dv = (prc * vol).rolling(22).mean()  # 22-day mean dollar volume per stock
    illiq_score = (1.0 / dv).mean(axis=0)
    illiq_score = illiq_score.dropna()
    median = illiq_score.median()
    low_liq = illiq_score[illiq_score >= median].index.tolist()  # high illiq
    high_liq = illiq_score[illiq_score < median].index.tolist()  # low illiq
    yhat = pd.read_csv(FCST / f"C_h{H:02d}_yhat.csv",
                       index_col=0, parse_dates=True)
    y = pd.read_csv(FCST / f"C_h{H:02d}_y.csv",
                    index_col=0, parse_dates=True)
    stocks = low_liq if half == "low" else high_liq
    out = _evaluate(yhat, y, base_yhat, base_y, H, stocks=stocks)
    out["n_stocks"] = len(stocks)
    return out


def variant_garch(bundle, base_yhat, base_y) -> dict:
    """GARCH(1,1) on log returns, fitted by Module 4b. Compares its h=5 log
    mean variance forecast against the same HAR baseline."""
    fp_yhat = FCST / f"G_h{H:02d}_yhat.csv"
    fp_y = FCST / f"G_h{H:02d}_y.csv"
    if not fp_yhat.exists() or not fp_y.exists():
        return {"MSE": np.nan, "imp_pct": np.nan, "HLN_DM_t": np.nan}
    yhat = pd.read_csv(fp_yhat, index_col=0, parse_dates=True)
    y = pd.read_csv(fp_y, index_col=0, parse_dates=True)
    return _evaluate(yhat, y, base_yhat, base_y, H)


def variant_inference_plain(bundle, base_yhat, base_y) -> dict:
    """Same headline pooled MSE, but the 'plain' iid (cell-level) DM stat for
    illustrative comparison with HLN-DM."""
    yhat = pd.read_csv(FCST / f"C_h{H:02d}_yhat.csv",
                       index_col=0, parse_dates=True)
    y = pd.read_csv(FCST / f"C_h{H:02d}_y.csv",
                    index_col=0, parse_dates=True)
    L = squared_loss(yhat, y).values.ravel()
    L_base = squared_loss(base_yhat, base_y).values.ravel()
    d = (L_base - L)
    d = d[~np.isnan(d)]
    plain_t = float(d.mean() / (d.std() / np.sqrt(len(d))))
    out = _evaluate(yhat, y, base_yhat, base_y, H)
    out["plain_DM_t"] = plain_t
    return out


# ------------------------------------------------------------------- regime rows
def regime_rows() -> list[dict]:
    raw = pd.read_csv(INTERM / "table7_raw.csv")
    out = []
    for label, regime in [("regime_high_vix", "High VIX (Q4)"),
                          ("regime_low_vix", "Low VIX (Q1)"),
                          ("regime_covid", "COVID (2020)"),
                          ("regime_gfc", "GFC (2008-Q3 to 2009-Q4)")]:
        sub = raw[(raw["regime"] == regime) & (raw["model"] == "C") & (raw["h"] == H)]
        if sub.empty:
            continue
        out.append({"variant": label,
                    "imp_pct": float(sub["imp_vs_A_pct"].iloc[0])})
    return out


# ----------------------------------------------------------------- main
def main() -> None:
    print("=" * 70)
    print("   MODULE 9: ROBUSTNESS SUMMARY (Table 9)")
    print("=" * 70)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # baseline A forecasts at h=5 (used as DM denominator in every variant)
    base_yhat = pd.read_csv(FCST / f"A_h{H:02d}_yhat.csv",
                            index_col=0, parse_dates=True)
    base_y = pd.read_csv(FCST / f"A_h{H:02d}_y.csv",
                         index_col=0, parse_dates=True)

    rows = []

    print("\n[1] headline (existing C h=5)")
    rows.append({"variant": "headline",
                 **variant_headline(None, base_yhat, base_y)})

    print("\n[2] inference: plain DM vs HLN-DM")
    rows.append({"variant": "inference_plain",
                 **variant_inference_plain(None, base_yhat, base_y)})

    print("\n[2b] benchmark: GARCH(1,1) on returns")
    rows.append({"variant": "benchmark_garch11",
                 **variant_garch(None, base_yhat, base_y)})

    print("\n[3] liquidity: high-illiq half (low-liquidity stocks)")
    rows.append({"variant": "liquidity_high_illiq",
                 **variant_liquidity(load_bundle(), base_yhat, base_y, half="low")})

    print("\n[4] liquidity: low-illiq half (high-liquidity stocks)")
    rows.append({"variant": "liquidity_low_illiq",
                 **variant_liquidity(load_bundle(), base_yhat, base_y, half="high")})

    print("\n[5] alternative estimator: LW d in place of GPH d")
    rows.append({"variant": "estimator_LW",
                 **variant_estimator_LW(load_bundle(), base_yhat, base_y)})

    print("\n[6] alternative window: 500 days")
    rows.append({"variant": "window_500",
                 **variant_window(load_bundle(), 500, base_yhat, base_y)})

    print("\n[7] alternative window: 1000 days")
    rows.append({"variant": "window_1000",
                 **variant_window(load_bundle(), 1000, base_yhat, base_y)})

    print("\n[8] alternative target: log mean future squared returns")
    rows.append({"variant": "target_sqret",
                 **variant_target_sqret(load_bundle(), base_yhat, base_y)})

    print("\n[9] regime rows (from Table 7)")
    for r in regime_rows():
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "table9_raw.csv", index=False)
    print("\n" + "=" * 70)
    print(df.round(3).to_string(index=False))

    # LaTeX export
    fp = TABLES_DIR / "table9_robustness.tex"
    with open(fp, "w") as f:
        f.write("% Table 9: Robustness summary -- Model C vs Model A at h=5\n\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Robustness Summary: Model C vs Model A, $h=5$}\n")
        f.write("\\label{tab:robustness}\n\\small\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Variant & MSE & \\%$\\Delta$ vs A & HLN DM-$t$ \\\\\n")
        f.write("\\midrule\n")
        labels = {
            "headline": "Headline (Model C, full panel)",
            "inference_plain": "Inference: plain DM (illustrative)",
            "benchmark_garch11": "Benchmark: GARCH(1,1) on returns",
            "liquidity_high_illiq": "Liquidity: low-liquidity half",
            "liquidity_low_illiq": "Liquidity: high-liquidity half",
            "estimator_LW": "Estimator: $\\hat d_{LW}$ replaces $\\hat d_{GPH}$",
            "window_500": "Rolling window: 500 days",
            "window_1000": "Rolling window: 1000 days",
            "target_sqret": "Target: log mean future squared returns",
            "regime_high_vix": "Regime: high VIX (Q4)",
            "regime_low_vix": "Regime: low VIX (Q1)",
            "regime_covid": "Regime: COVID 2020",
            "regime_gfc": "Regime: GFC 2008--2009",
        }
        for _, r in df.iterrows():
            label = labels.get(r["variant"], r["variant"])
            mse_val = r.get("MSE", np.nan)
            mse = "--" if pd.isna(mse_val) else f"{mse_val:.4f}"
            imp_pct_val = r.get("imp_pct", np.nan)
            imp = "--" if pd.isna(imp_pct_val) else f"{imp_pct_val:+.2f}\\%"
            t_val = r.get("HLN_DM_t", np.nan)
            t = "--" if pd.isna(t_val) else f"${t_val:+.2f}$"
            if r["variant"] == "inference_plain":
                t = (f"${r.get('HLN_DM_t', np.nan):+.2f}$ "
                     f"(plain $+{r.get('plain_DM_t', np.nan):.2f}$)")
            f.write(f"{label} & {mse} & {imp} & {t} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}%\n}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write(
            "\\item Notes: Robustness checks for Model C against Model A at "
            "$h=5$ on $\\log RV^{PK}$. Liquidity halves are formed from the "
            "static median of mean inverse dollar volume across the sample. "
            "$\\hat d_{LW}$ replaces $\\hat d_{GPH}$ in the entire feature "
            "block (including memory dynamics and interactions). Window "
            "variants refit the rolling LRD estimation at the alternative "
            "window before regenerating the full feature stack. The target "
            "variant replaces $\\log RV^{PK}$ with log mean future squared "
            "returns (Models A and C both refit). The GARCH(1,1) row fits a "
            "constant-mean GARCH(1,1) on log returns with refit every 20 "
            "sample steps and reports the resulting log mean conditional "
            "variance forecast against the same Parkinson target; its "
            "forecast scale differs from the Parkinson RV target (returns "
            "variance vs.\\ range-based variance proxy), so the negative "
            "$\\%\\Delta$ vs A reflects both this level mismatch and the "
            "well-known limitation of returns-only GARCH for forecasting "
            "realised variance. Regime rows come from the Table 7 split at "
            "$h=5$. The plain-DM stat is the unscaled iid pooled-cell "
            "statistic; the HLN stat is the panel-aware "
            "Harvey-Leybourne-Newbold finite-sample-corrected version.\n"
        )
        f.write("\\end{tablenotes}\n\\end{table}\n")
    print(f"\nSaved {fp}")


if __name__ == "__main__":
    main()
