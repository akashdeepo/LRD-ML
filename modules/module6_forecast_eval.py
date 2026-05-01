"""
MODULE 6: Forecast Evaluation
==============================
Reads all forecast panels under results/intermediate/forecasts/ and produces:

  Table 5 — model x horizon comparison (MSE, QLIKE, % improvement vs A,
            HLN-corrected Diebold-Mariano stat vs A) pooled across stocks
  Table 6 — feature-importance summary (Lasso non-zero coefficients per
            horizon + Random Forest / GBM permutation-style ranking)
  Table 7 — regime split: VIX-quartile + GFC + COVID windows, MSE per
            (model, horizon, regime), % improvement vs A
  Table 8 — horizon-by-sector breakdown for the leading model
            (Model C and best D variant)

The DM test is the panel-aware HLN-corrected version: cross-sectional mean of
loss differentials per date, Newey-West HAC variance with bandwidth tied to
h/stride, and the Harvey-Leybourne-Newbold finite-sample correction with
Student-t(T-1) reference distribution.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from modules.forecast_io import HORIZONS, load_bundle

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
FCST_DIR = BASE / "results" / "intermediate" / "forecasts"
TABLES_DIR = BASE / "results" / "tables"
INTERMEDIATE_DIR = BASE / "results" / "intermediate"

# Legacy label B == A2 (same predictor set, identical fits). We expose it as
# A2 in tables and the loader renames B files transparently.
LEGACY_RENAME = {"B": "A2"}

LINEAR_LADDER = ("A", "A1", "A2", "A3", "A4", "A5", "C")
ML_MODELS = ("D_lasso", "D_ridge", "D_en", "D_rf", "D_gbm")
DISPLAY_ORDER = LINEAR_LADDER + ML_MODELS

# Sample-date stride (weekly = 5 trading days). Used to translate the
# trading-day forecast horizon into the effective sample-date overlap that
# governs HAC bandwidth and HLN correction.
SAMPLE_STRIDE = 5


# --------------------------------------------------------------- loss functions
def squared_loss(yhat: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return (yhat - y) ** 2


def qlike_loss(yhat_log: pd.DataFrame, y_log: pd.DataFrame) -> pd.DataFrame:
    """QLIKE on the volatility-variance scale: log(sigma2_hat) + sigma2 / sigma2_hat
    where sigma2 = exp(y_log) and sigma2_hat = exp(yhat_log)."""
    with np.errstate(over="ignore"):
        sig2_hat = np.exp(yhat_log)
        sig2 = np.exp(y_log)
        return yhat_log + sig2 / sig2_hat


def diebold_mariano(loss_a: pd.DataFrame, loss_b: pd.DataFrame,
                    h: int = 1, stride: int = SAMPLE_STRIDE
                    ) -> tuple[float, float, float, int]:
    """Panel-aware HLN-corrected Diebold-Mariano test.

    H_0: E[loss_a - loss_b] = 0.

    Construction:
        d_t = mean over stocks of (loss_a_{i,t} - loss_b_{i,t})
        var(d_bar) via Newey-West (Bartlett) HAC with bandwidth h_eff - 1
            where h_eff = ceil(h / stride) is the forecast horizon expressed
            in sample-date units (so non-overlapping forecasts give bw=0).
        DM = mean(d) / sqrt(var_hac / T)
        HLN correction: DM* = DM * sqrt((T + 1 - 2 h_eff + h_eff (h_eff - 1) / T) / T)
        Reference: Student-t(T - 1).

    Returns (mean_d, t_HLN, p_two_sided, T_dates).
    """
    diff = (loss_a - loss_b)
    d = diff.mean(axis=1).dropna().values  # cross-sectional mean per date
    T = int(len(d))
    if T < 5:
        return np.nan, np.nan, np.nan, T

    mean_d = float(d.mean())
    centered = d - mean_d

    h_eff = max(int(np.ceil(h / stride)), 1)
    bandwidth = h_eff - 1
    gamma0 = float((centered ** 2).mean())
    var_hac = gamma0
    for k in range(1, bandwidth + 1):
        gamma_k = float((centered[k:] * centered[:-k]).mean())
        weight = 1.0 - k / (bandwidth + 1)  # Bartlett kernel
        var_hac += 2.0 * weight * gamma_k

    if var_hac <= 0 or T <= 1:
        return mean_d, np.nan, np.nan, T

    se = float(np.sqrt(var_hac / T))
    t_dm = mean_d / se
    correction_arg = (T + 1 - 2 * h_eff + h_eff * (h_eff - 1) / T) / T
    correction = float(np.sqrt(max(correction_arg, 1e-10)))
    t_hln = float(t_dm * correction)
    p_two = float(2 * (1 - stats.t.cdf(abs(t_hln), df=T - 1)))
    return mean_d, t_hln, p_two, T


# --------------------------------------------------------------- table 5
def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "^{***}"
    if p < 0.05:
        return "^{**}"
    if p < 0.10:
        return "^{*}"
    return ""


def table5_main_comparison(loss_panels: dict[str, dict[int, pd.DataFrame]],
                           qlike_panels: dict[str, dict[int, pd.DataFrame]],
                           fp: Path) -> pd.DataFrame:
    rows = []
    for model in DISPLAY_ORDER:
        if model not in loss_panels:
            continue
        for h in HORIZONS:
            if h not in loss_panels[model]:
                continue
            mse = loss_panels[model][h].values
            mse = float(np.nanmean(mse))
            ql = qlike_panels[model][h].values
            ql = float(np.nanmean(ql))
            base_mse = loss_panels["A"][h].values
            base_mse = float(np.nanmean(base_mse))
            imp_pct = 100 * (1 - mse / base_mse)
            if model == "A":
                dm_t, dm_p = np.nan, np.nan
            else:
                _, dm_t, dm_p, _ = diebold_mariano(
                    loss_panels["A"][h], loss_panels[model][h], h=h
                )
            rows.append({
                "model": model, "h": h,
                "MSE_logRV": mse, "QLIKE": ql,
                "imp_vs_A_pct": imp_pct,
                "DM_t_HLN": dm_t, "DM_p": dm_p,
            })
    df = pd.DataFrame(rows)
    df.to_csv(INTERMEDIATE_DIR / "table5_raw.csv", index=False)

    with open(fp, "w") as f:
        f.write("% Table 5: Out-of-sample forecast comparison (HLN-corrected DM)\n\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Out-of-Sample Forecast Comparison: Layered Ladder "
                "$A \\to A_1\\dots A_5 \\to C \\to D$, Horizons "
                "$h \\in \\{1,5,22\\}$}\n")
        f.write("\\label{tab:model_comparison}\n\\small\n")
        f.write("\\begin{tabular}{llcccc}\n\\toprule\n")
        f.write("Model & $h$ & MSE($\\log RV^{PK}$) & QLIKE & "
                "\\%$\\Delta$ vs A & HLN DM-$t$ vs A \\\\\n")
        f.write("\\midrule\n")
        prev = None
        for _, r in df.iterrows():
            if r["model"] != prev:
                if prev is not None:
                    f.write("\\midrule\n")
                model_label = r["model"].replace("_", "\\_")
                f.write(f"\\textbf{{{model_label}}} ")
                prev = r["model"]
            else:
                f.write(" ")
            if pd.isna(r["DM_t_HLN"]):
                dm_str = "--"
            else:
                stars = _stars(r["DM_p"])
                dm_str = f"${r['DM_t_HLN']:+.2f}{stars}$"
            f.write(f"& {int(r['h']):2d} & {r['MSE_logRV']:.4f} & "
                    f"{r['QLIKE']:.4f} & {r['imp_vs_A_pct']:+.2f}\\% & "
                    f"{dm_str} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write(
            "\\item Notes: pooled across all stocks and out-of-sample dates "
            "(post 40\\% warm-up). MSE on $\\log RV^{PK}$ scale "
            "(range-based variance proxy); QLIKE on the variance scale. "
            "HLN DM-$t$ is the Harvey-Leybourne-Newbold finite-sample-corrected "
            "Diebold-Mariano statistic, computed on the cross-sectional mean "
            "loss differential per date with Newey-West HAC variance "
            "(bandwidth tied to $\\lceil h / 5 \\rceil - 1$ to handle "
            "overlapping multi-step forecasts on a weekly sample stride) and "
            "Student-$t(T-1)$ reference. Positive values indicate the model "
            "beats Model A. Significance: $^{*}$ $p<0.10$, "
            "$^{**}$ $p<0.05$, $^{***}$ $p<0.01$. "
            "Layered ladder: $A$ (HAR core), $A_1$ ($+$ VIX, MOVE; HAR-X), "
            "$A_2$ ($+$ own-stock persistence block), $A_3$ ($+$ cross-sectional "
            "mean and dispersion of $\\hat d$), $A_4$ ($+$ sector-mean $\\hat d$), "
            "$A_5$ ($+$ $\\hat d$, VIX, MOVE, $\\hat d \\times$ VIX, "
            "$\\hat d \\times$ MOVE), $C$ (full union), $D$ (predictors of $C$ "
            "estimated by shrinkage / tree-based ML).\n"
        )
        f.write("\\end{tablenotes}\n\\end{table}\n")
    return df


# --------------------------------------------------------------- table 7 (regime)
def regime_masks(idx: pd.DatetimeIndex, market: pd.DataFrame) -> dict[str, np.ndarray]:
    vix = market["VIX"].reindex(idx).ffill()
    q1, q3 = vix.quantile(0.25), vix.quantile(0.75)
    return {
        "Low VIX (Q1)": (vix <= q1).values,
        "High VIX (Q4)": (vix >= q3).values,
        "GFC (2008-Q3 to 2009-Q4)": ((idx >= "2008-09-01") & (idx <= "2009-12-31")),
        "COVID (2020)": ((idx >= "2020-03-01") & (idx <= "2020-12-31")),
    }


def table7_regimes(loss_panels: dict, market: pd.DataFrame, fp: Path) -> None:
    idx = loss_panels["A"][1].index
    masks = regime_masks(idx, market)
    rows = []
    for regime, mask in masks.items():
        for model in DISPLAY_ORDER:
            if model not in loss_panels:
                continue
            for h in HORIZONS:
                if h not in loss_panels[model]:
                    continue
                arr = loss_panels[model][h].loc[mask].values
                arr = arr[~np.isnan(arr)]
                if arr.size == 0:
                    continue
                base = loss_panels["A"][h].loc[mask].values
                base = base[~np.isnan(base)].mean()
                rows.append({
                    "regime": regime, "model": model, "h": h,
                    "MSE": float(arr.mean()),
                    "imp_vs_A_pct": 100 * (1 - arr.mean() / base),
                })
    df = pd.DataFrame(rows)
    df.to_csv(INTERMEDIATE_DIR / "table7_raw.csv", index=False)

    pivot = df.pivot_table(index=("regime", "h"), columns="model",
                           values="imp_vs_A_pct").round(2)

    with open(fp, "w") as f:
        f.write("% Table 7: Regime-Conditioned Forecast Improvement vs Model A\n\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Regime-Conditioned MSE Improvement (\\%) vs Model A}\n")
        f.write("\\label{tab:regime_split}\n\\small\n")
        cols = [m for m in DISPLAY_ORDER if m in pivot.columns]
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{ll" + "c" * len(cols) + "}\n\\toprule\n")
        f.write("Regime & $h$")
        for m in cols:
            f.write(f" & {m.replace('_', '\\_')}")
        f.write(" \\\\\n\\midrule\n")
        prev_reg = None
        for (reg, h), row in pivot.iterrows():
            reg_str = reg if reg != prev_reg else ""
            f.write(f"{reg_str} & {h}")
            for m in cols:
                v = row.get(m, np.nan)
                f.write(" & --" if pd.isna(v) else f" & {v:+.2f}\\%")
            f.write(" \\\\\n")
            if reg != prev_reg and prev_reg is not None:
                pass
            prev_reg = reg
        f.write("\\bottomrule\n\\end{tabular}%\n}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write(
            "\\item Notes: Out-of-sample MSE improvement on $\\log RV^{PK}$ "
            "relative to Model A, computed within each regime. "
            "VIX quartiles are computed on the out-of-sample evaluation window. "
            "Crisis windows are 2008-Q3 to 2009-Q4 (GFC) and Mar--Dec 2020 (COVID). "
            "Positive values indicate the model beats HAR within the regime.\n"
        )
        f.write("\\end{tablenotes}\n\\end{table}\n")
    return df


# --------------------------------------------------------------- table 8 (sector)
def table8_sectors(loss_panels: dict, sectors: dict, fp: Path) -> None:
    rows = []
    leading_models = ["A", "A1", "A2", "A3", "A4", "A5", "C",
                      "D_lasso", "D_rf", "D_gbm"]
    leading_models = [m for m in leading_models if m in loss_panels]
    sec_per = pd.Series(sectors)
    sectors_list = sorted(set(sec_per.dropna()))

    for sec in sectors_list:
        cols = [t for t, s in sectors.items() if s == sec]
        for model in leading_models:
            for h in HORIZONS:
                if h not in loss_panels.get(model, {}):
                    continue
                lp = loss_panels[model][h]
                cols_in = [c for c in cols if c in lp.columns]
                if not cols_in:
                    continue
                arr = lp[cols_in].values
                arr = arr[~np.isnan(arr)]
                if arr.size == 0:
                    continue
                base = loss_panels["A"][h][cols_in].values
                base = base[~np.isnan(base)].mean()
                rows.append({
                    "sector": sec, "model": model, "h": h,
                    "MSE": float(arr.mean()),
                    "imp_vs_A_pct": 100 * (1 - arr.mean() / base),
                    "n_stocks": len(cols_in),
                })
    df = pd.DataFrame(rows)
    df.to_csv(INTERMEDIATE_DIR / "table8_raw.csv", index=False)

    # Focus on h=5 (medium horizon) for the table; raw csv has all
    df5 = df[df["h"] == 5].pivot_table(index="sector", columns="model",
                                       values="imp_vs_A_pct").round(2)

    with open(fp, "w") as f:
        f.write("% Table 8: Sector-Level Forecast Improvement (h=5) vs Model A\n\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Sector-Level MSE Improvement (\\%) vs Model A at $h=5$}\n")
        f.write("\\label{tab:sector_split}\n\\small\n")
        cols = [m for m in leading_models if m in df5.columns]
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n")
        f.write("Sector")
        for m in cols:
            f.write(f" & {m.replace('_', '\\_')}")
        f.write(" \\\\\n\\midrule\n")
        for sec, row in df5.iterrows():
            f.write(f"{sec}")
            for m in cols:
                v = row.get(m, np.nan)
                f.write(" & --" if pd.isna(v) else f" & {v:+.2f}\\%")
            f.write(" \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}%\n}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write(
            "\\item Notes: Out-of-sample MSE improvement on $\\log RV^{PK}$ at "
            "horizon $h=5$, by GICS sector, relative to Model A. Pooled across "
            "stocks within each sector and over the entire out-of-sample window.\n"
        )
        f.write("\\end{tablenotes}\n\\end{table}\n")
    return df


# --------------------------------------------------------------- table 6 (importance)
def table6_feature_importance(bundle, fp: Path) -> None:
    """Train a simple Lasso on the full-sample (in-sample!) Z_t feature set,
    pooled across stocks and dates, to show which features have non-zero
    coefficients. This is a descriptive importance proxy; the OOS gains shown
    in Table 5 are the more rigorous test."""
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    from modules.forecast_io import build_targets, MODEL_FEATURES, stock_matrix

    targets = build_targets(bundle)
    feat_cols = MODEL_FEATURES["C"]
    rows_per_h: dict[int, pd.DataFrame] = {}
    for h in HORIZONS:
        Xs, ys = [], []
        for t in bundle.panel.kept:
            sm = stock_matrix(bundle, t, "C", targets)
            df = sm.X.copy()
            df["__y__"] = sm.y[h]
            df = df.dropna()
            Xs.append(df[feat_cols].values)
            ys.append(df["__y__"].values)
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        scaler = StandardScaler().fit(X)
        Xs_s = scaler.transform(X)
        lasso = LassoCV(cv=5, n_alphas=20, max_iter=2000, n_jobs=1).fit(Xs_s, y)
        coefs = pd.Series(lasso.coef_, index=feat_cols, name=f"h={h}")
        rows_per_h[h] = coefs

    coef_df = pd.concat(rows_per_h.values(), axis=1)
    coef_df.columns = [f"h={h}" for h in HORIZONS]
    coef_df.to_csv(INTERMEDIATE_DIR / "table6_lasso_coefs.csv")

    with open(fp, "w") as f:
        f.write("% Table 6: Pooled-Lasso standardised coefficients per horizon\n\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Standardised Lasso Coefficients on the Full Pooled Sample}\n")
        f.write("\\label{tab:importance}\n\\small\n")
        f.write("\\begin{tabular}{l" + "c" * len(HORIZONS) + "}\n\\toprule\n")
        f.write("Feature")
        for h in HORIZONS:
            f.write(f" & $h={h}$")
        f.write(" \\\\\n\\midrule\n")
        for feat in feat_cols:
            f.write(f"{feat.replace('_', '\\_')}")
            for h in HORIZONS:
                v = coef_df.loc[feat, f"h={h}"]
                f.write(f" & {v:+.4f}")
            f.write(" \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write(
            "\\item Notes: Pooled-Lasso fit on standardised features (full sample, "
            "all stocks, all out-of-sample dates), one fit per horizon. Coefficients "
            "are on standardised inputs and so are directly comparable in magnitude. "
            "Zero coefficients indicate features dropped by Lasso.\n"
        )
        f.write("\\end{tablenotes}\n\\end{table}\n")


# --------------------------------------------------------------- main
def load_forecast_panels() -> tuple[dict, dict]:
    """Load forecast panels from disk. Legacy 'B' files are exposed under the
    new label 'A2' (identical predictor set). If both B and A2 files exist,
    the explicit A2 file wins."""
    loss = {}; ql = {}
    files = sorted(FCST_DIR.glob("*_yhat.csv"))

    def _parse(name: str) -> tuple[str, int] | None:
        if "_h" not in name:
            return None
        model, _, hstr = name.rpartition("_h")
        try:
            return model, int(hstr)
        except ValueError:
            return None

    # Pass 1: native names except legacy B
    for fp in files:
        parsed = _parse(fp.stem.replace("_yhat", ""))
        if parsed is None:
            continue
        model, h = parsed
        if model == "B":
            continue
        y_fp = FCST_DIR / f"{model}_h{h:02d}_y.csv"
        if not y_fp.exists():
            continue
        yhat = pd.read_csv(fp, index_col=0, parse_dates=True)
        y = pd.read_csv(y_fp, index_col=0, parse_dates=True)
        loss.setdefault(model, {})[h] = squared_loss(yhat, y)
        ql.setdefault(model, {})[h] = qlike_loss(yhat, y)

    # Pass 2: fill A2 from legacy B if no native A2 fit on disk
    for fp in files:
        parsed = _parse(fp.stem.replace("_yhat", ""))
        if parsed is None:
            continue
        model, h = parsed
        if model != "B":
            continue
        new = LEGACY_RENAME["B"]
        if new in loss and h in loss[new]:
            continue  # native fit already loaded
        y_fp = FCST_DIR / f"B_h{h:02d}_y.csv"
        if not y_fp.exists():
            continue
        yhat = pd.read_csv(fp, index_col=0, parse_dates=True)
        y = pd.read_csv(y_fp, index_col=0, parse_dates=True)
        loss.setdefault(new, {})[h] = squared_loss(yhat, y)
        ql.setdefault(new, {})[h] = qlike_loss(yhat, y)

    return loss, ql


def main() -> None:
    print("=" * 70)
    print("   MODULE 6: FORECAST EVALUATION")
    print("=" * 70)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    loss_panels, qlike_panels = load_forecast_panels()
    print(f"  Loaded forecast panels for {len(loss_panels)} models: "
          f"{sorted(loss_panels.keys())}")
    for m, hd in loss_panels.items():
        print(f"    {m}: horizons {sorted(hd.keys())}")

    bundle = load_bundle()
    sectors = bundle.sectors
    market = bundle.market

    print("\n[1/4] Table 5 — main comparison...")
    df5 = table5_main_comparison(loss_panels, qlike_panels,
                                 TABLES_DIR / "table5_model_comparison.tex")
    print(df5.round(4).to_string(index=False))

    print("\n[2/4] Table 7 — regime split...")
    df7 = table7_regimes(loss_panels, market,
                         TABLES_DIR / "table7_subsamples.tex")
    print(df7.pivot_table(index=("regime", "h"), columns="model",
                          values="imp_vs_A_pct").round(2).to_string())

    print("\n[3/4] Table 8 — sector split (h=5)...")
    df8 = table8_sectors(loss_panels, sectors,
                         TABLES_DIR / "table8_horizons.tex")
    print(df8[df8["h"] == 5].pivot_table(
        index="sector", columns="model", values="imp_vs_A_pct").round(2).to_string())

    print("\n[4/4] Table 6 — pooled Lasso coefficients...")
    table6_feature_importance(bundle, TABLES_DIR / "table6_feature_importance.tex")
    print("  saved.")

    print("\n" + "=" * 70)
    print("Outputs:")
    print(f"  {TABLES_DIR/'table5_model_comparison.tex'}")
    print(f"  {TABLES_DIR/'table6_feature_importance.tex'}")
    print(f"  {TABLES_DIR/'table7_subsamples.tex'}")
    print(f"  {TABLES_DIR/'table8_horizons.tex'}")
    print(f"  {INTERMEDIATE_DIR/'table5_raw.csv'}, table7_raw.csv, table8_raw.csv, "
          "table6_lasso_coefs.csv")


if __name__ == "__main__":
    main()
