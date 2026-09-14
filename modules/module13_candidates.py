"""
MODULE 13: Pre-registered candidate designs (docs/EXPERIMENTS.md runs 05-07)
============================================================================
All three ask the same question from a different angle: the persistence
regressors have non-zero population coefficients (Clark-West rejects, finding
#7) but per-stock OLS cannot turn that into out-of-sample accuracy. Each design
attacks the estimation-noise explanation or tests the "duration" interpretation
directly. Every forecast is point-in-time (embargo from forecast_io).

run-05  pooled panel with stock fixed effects  -> forecasts P{model}_h{h}
run-06  duration target D_t = y_{t,22} - y_{t,5} -> forecasts DUR_{model}_h22
run-07  market-level target (cross-sectional mean of the stock targets)
        with market HAR + VIX/MOVE + persistence state -> forecasts M{model}_h{h}

Outputs: results/tables/table13_candidates.tex, results/intermediate/candidates_*.csv
Run from the repo root:  python -m modules.module13_candidates [--only 05 06 07]
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from modules.forecast_io import (
    HORIZONS, MODEL_FEATURES, aligned_xy, build_targets, load_bundle,
    n_train_rows, row_positions, stock_matrix,
)
from modules.module4_benchmarks import INIT_TRAIN_FRAC, _ols, expanding_forecast
from modules.module6_forecast_eval import diebold_mariano, squared_loss
from modules.module12_incremental_tests import _hac_mean_test, clark_west, holm, _stars

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
FCST = BASE / "results" / "intermediate" / "forecasts"
TABLES = BASE / "results" / "tables"
INTERM = BASE / "results" / "intermediate"


# ------------------------------------------------------------------ helpers
def _save(tag: str, h: int, yhat: pd.DataFrame, y: pd.DataFrame) -> None:
    yhat.to_csv(FCST / f"{tag}_h{h:02d}_yhat.csv")
    y.to_csv(FCST / f"{tag}_h{h:02d}_y.csv")


def _compare(label: str, yh_base, y_base, yh_new, y_new, h: int) -> dict:
    Lb, Ln = squared_loss(yh_base, y_base), squared_loss(yh_new, y_new)
    mse_b, mse_n = float(np.nanmean(Lb.values)), float(np.nanmean(Ln.values))
    _, t_dm, p_dm, T = diebold_mariano(Lb, Ln, h=h)
    cw = clark_west(yh_base, yh_new, y_new, h)
    return {"comparison": label, "h": h, "T_dates": T, "mse_base": mse_b, "mse_new": mse_n,
            "gain_pct": 100 * (1 - mse_n / mse_b), "dm_t": t_dm, "dm_p": p_dm,
            "cw_t": cw["cw_t"], "cw_p": cw["cw_p_one_sided"]}


# ------------------------------------------------------------------ run-05: pooled
def _stack(bundle, targets, model: str, h: int, init_n: int):
    frames = []
    for tkr in bundle.panel.kept:
        sm = stock_matrix(bundle, tkr, model, targets)
        X, y = aligned_xy(sm, h)
        if len(X) < init_n + 5:
            continue
        df = X.copy()
        df["__y__"] = y.values
        df["__pos__"] = row_positions(X.index, bundle.rv.index)
        df["__tkr__"] = tkr
        df["__date__"] = X.index
        frames.append(df)
    P = pd.concat(frames, ignore_index=True).sort_values(["__pos__", "__tkr__"], kind="stable")
    return P.reset_index(drop=True)


def _fe_fit(X: np.ndarray, y: np.ndarray, codes: np.ndarray, K: int
            ) -> tuple[np.ndarray, np.ndarray]:
    """Pooled OLS with group (stock) fixed effects via the within
    transformation. Returns (beta, alpha) where alpha[k] is group k's
    intercept; groups absent from the sample get alpha = nan. Equivalent to
    OLS on [X, group dummies] (tests/test_pooled.py)."""
    cnt = np.bincount(codes, minlength=K).astype(float)
    cnt[cnt == 0] = np.nan
    xm = np.vstack([np.bincount(codes, weights=X[:, j], minlength=K)
                    for j in range(X.shape[1])]).T / cnt[:, None]
    ym = np.bincount(codes, weights=y, minlength=K) / cnt
    beta, *_ = np.linalg.lstsq(X - xm[codes], y - ym[codes], rcond=None)
    alpha = ym - xm @ beta
    return beta, alpha


def pooled_forecast(bundle, targets, model: str, h: int, init_n: int
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One regression per origin on the stacked panel of all stocks, stock
    fixed effects via the within transformation computed on training rows
    only, expanding window with the point-in-time embargo."""
    feats = MODEL_FEATURES[model]
    P = _stack(bundle, targets, model, h, init_n)
    Xall = P[feats].values
    yall = P["__y__"].values
    pos_all = P["__pos__"].values
    tk_all = P["__tkr__"].values
    origins = bundle.sample_dates[init_n:]
    origin_pos = bundle.rv.index.get_indexer(origins)
    tickers = np.array(bundle.panel.kept)
    tk_codes = pd.Categorical(tk_all, categories=tickers).codes
    yhat = pd.DataFrame(index=bundle.sample_dates, columns=tickers, dtype=float)
    ytrue = pd.DataFrame(index=bundle.sample_dates, columns=tickers, dtype=float)
    date_all = P["__date__"].values
    for od, op in zip(origins, origin_pos):
        n = int(np.searchsorted(pos_all, op - h, side="right"))     # rows with pos + h <= op
        if n < 50:
            continue
        Xtr, ytr, ctr = Xall[:n], yall[:n], tk_codes[:n]
        beta, alpha = _fe_fit(Xtr, ytr, ctr, len(tickers))       # within-FE OLS, training rows only
        test = np.where(date_all == np.datetime64(od))[0]
        for i in test:
            c = tk_codes[i]
            if not np.isfinite(alpha[c]):
                continue
            yhat.iloc[yhat.index.get_loc(od), c] = alpha[c] + Xall[i] @ beta
            ytrue.iloc[ytrue.index.get_loc(od), c] = yall[i]
    return yhat, ytrue


def run05(bundle, targets, init_n) -> pd.DataFrame:
    print("\n[run-05] pooled panel with stock fixed effects", flush=True)
    rows = []
    for h in HORIZONS:
        got = {}
        for model in ["A", "A1", "C", "A1cs"]:
            yh, y = pooled_forecast(bundle, targets, model, h, init_n)
            _save(f"P{model}", h, yh, y)
            got[model] = (yh, y)
            print(f"   P{model:5s} h={h:2d} MSE={float(np.nanmean(((yh - y)**2).values)):.4f}", flush=True)
        a1 = pd.read_csv(FCST / f"A1_h{h:02d}_yhat.csv", index_col=0, parse_dates=True)
        a1y = pd.read_csv(FCST / f"A1_h{h:02d}_y.csv", index_col=0, parse_dates=True)
        rows.append(_compare("P-C vs P-A1", *got["A1"], *got["C"], h))
        rows.append(_compare("P-A1cs vs P-A1", *got["A1"], *got["A1cs"], h))
        rows.append(_compare("P-A1 vs per-stock A1", a1, a1y, *got["A1"], h))
        rows.append(_compare("P-A1 vs P-A", *got["A"], *got["A1"], h))
    df = pd.DataFrame(rows)
    pre = df["comparison"].isin(["P-C vs P-A1", "P-A1cs vs P-A1"])
    df["cw_p_holm"] = np.nan
    df.loc[pre, "cw_p_holm"] = holm(df.loc[pre, "cw_p"].tolist())
    df["passes_rule"] = (df["cw_p_holm"] < 0.05) & (df["gain_pct"] > 0)
    df.to_csv(INTERM / "candidates_run05_pooled.csv", index=False)
    return df


# ------------------------------------------------------------------ run-06: duration
def run06(bundle, targets, init_n) -> pd.DataFrame:
    print("\n[run-06] duration target D = y22 - y5", flush=True)
    D = targets[22] - targets[5]
    dtargets = {h: D for h in HORIZONS}          # stock_matrix expects all horizons
    got = {}
    for model in ["A", "A1", "A1cs", "C"]:
        yhat = pd.DataFrame(index=bundle.sample_dates, columns=bundle.panel.kept, dtype=float)
        ytrue = yhat.copy()
        for tkr in bundle.panel.kept:
            sm = stock_matrix(bundle, tkr, model, dtargets)
            X, y = aligned_xy(sm, 22)
            if len(X) < init_n + 5:
                continue
            yh = expanding_forecast(X, y, init_n, 22, bundle.rv.index)
            yhat.loc[yh.index, tkr] = yh.values
            ytrue.loc[y.index, tkr] = y.values
        _save(f"DUR_{model}", 22, yhat, ytrue)
        got[model] = (yhat, ytrue)
        print(f"   DUR_{model:5s} MSE={float(np.nanmean(((yhat - ytrue)**2).values)):.4f}", flush=True)
    # unconditional benchmark: expanding mean of D per stock (no regressors)
    rows = [_compare("A1 vs A (duration target)", *got["A"], *got["A1"], 22),
            _compare("A1cs vs A1 (duration target)", *got["A1"], *got["A1cs"], 22),
            _compare("C vs A1 (duration target)", *got["A1"], *got["C"], 22)]
    df = pd.DataFrame(rows)
    pre = df["comparison"].str.startswith(("A1cs", "C vs"))
    df["cw_p_holm"] = np.nan
    df.loc[pre, "cw_p_holm"] = holm(df.loc[pre, "cw_p"].tolist())
    df["passes_rule"] = (df["cw_p_holm"] < 0.05) & (df["gain_pct"] > 0)
    df.to_csv(INTERM / "candidates_run06_duration.csv", index=False)
    return df


# ------------------------------------------------------------------ run-07: market level
M_FEATURES = {
    "MA": ["m_har_d", "m_har_w", "m_har_m"],
    "MA1": ["m_har_d", "m_har_w", "m_har_m", "vix", "move"],
    "MC": ["m_har_d", "m_har_w", "m_har_m", "vix", "move",
           "cs_mean_d", "cs_std_d", "csd_x_vix", "csd_x_move"],
}


def run07(bundle, targets, init_n) -> pd.DataFrame:
    print("\n[run-07] market-level target (cross-sectional mean of stock targets)", flush=True)
    sd = bundle.sample_dates
    m_daily = bundle.log_rv.mean(axis=1)                       # cross-sectional mean daily log RV
    feats = pd.DataFrame(index=sd)
    feats["m_har_d"] = m_daily.shift(1).reindex(sd)
    feats["m_har_w"] = m_daily.rolling(5).mean().shift(1).reindex(sd)
    feats["m_har_m"] = m_daily.rolling(22).mean().shift(1).reindex(sd)
    feats["vix"] = bundle.market["VIX"].reindex(sd)
    feats["move"] = bundle.market["MOVE"].reindex(sd)
    feats["cs_mean_d"] = bundle.cs["cs_mean_d"].reindex(sd)
    feats["cs_std_d"] = bundle.cs["cs_std_d"].reindex(sd)
    feats["csd_x_vix"] = feats["cs_mean_d"] * feats["vix"]
    feats["csd_x_move"] = feats["cs_mean_d"] * feats["move"]
    rows = []
    for h in HORIZONS:
        yM = targets[h].mean(axis=1).reindex(sd)
        got = {}
        for model, cols in M_FEATURES.items():
            df = feats[cols].copy(); df["__y__"] = yM
            df = df.dropna()
            X, y = df[cols], df["__y__"]
            yh = expanding_forecast(X, y, init_n, h, bundle.rv.index)
            yhat = yh.to_frame("MKT"); ytrue = y.to_frame("MKT")
            _save(model, h, yhat, ytrue)
            got[model] = (yhat, ytrue)
            print(f"   {model:4s} h={h:2d} MSE={float(np.nanmean(((yhat - ytrue)**2).values)):.5f}", flush=True)
        rows.append(_compare("M-A1 vs M-A", *got["MA"], *got["MA1"], h))
        rows.append(_compare("M-C vs M-A1", *got["MA1"], *got["MC"], h))
    df = pd.DataFrame(rows)
    pre = df["comparison"] == "M-C vs M-A1"
    df["cw_p_holm"] = np.nan
    df.loc[pre, "cw_p_holm"] = holm(df.loc[pre, "cw_p"].tolist())
    df["passes_rule"] = (df["cw_p_holm"] < 0.05) & (df["gain_pct"] > 0)
    df.to_csv(INTERM / "candidates_run07_market.csv", index=False)
    return df


# ------------------------------------------------------------------ table
def write_table13(res: dict[str, pd.DataFrame]) -> None:
    titles = {"05": "Panel A: pooled estimation with stock fixed effects (level target)",
              "06": "Panel B: duration target $D_t=y_{t,22}-y_{t,5}$, per-stock OLS",
              "07": "Panel C: market-level target (cross-sectional mean), single series"}
    with open(TABLES / "table13_candidates.tex", "w", encoding="utf-8") as f:
        f.write("% Table 13: pre-registered candidate designs (docs/EXPERIMENTS.md runs 05-07)\n\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Pre-Registered Follow-Up Designs: Does the Persistence Information Become Usable?}\n")
        f.write("\\label{tab:candidates}\n\\small\n\\begin{tabular}{llcccc}\n\\toprule\n")
        f.write("Comparison & $h$ & \\%$\\Delta$MSE & HLN DM-$t$ & CW-$t$ & CW $p$ (Holm) \\\\\n")
        for key in ["05", "06", "07"]:
            if key not in res:
                continue
            f.write("\\midrule\n\\multicolumn{6}{l}{\\textit{" + titles[key] + "}} \\\\\n")
            for r in res[key].itertuples():
                hp = f"{r.cw_p_holm:.3f}" if np.isfinite(r.cw_p_holm) else "--"
                f.write(f"{r.comparison} & {r.h} & {r.gain_pct:+.2f}\\% & ${r.dm_t:+.2f}${_stars(r.dm_p)} & "
                        f"${r.cw_t:+.2f}${_stars(r.cw_p)} & {hp} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\begin{tablenotes}\\small\n")
        f.write("\\item Notes: All forecasts are point-in-time expanding-window OLS. \\%$\\Delta$MSE is the "
                "change in pooled MSE of the second-named model relative to the first (positive favours the "
                "second). HLN DM-$t$ (two-sided) and Clark--West CW-$t$ (one-sided, nested) as in "
                "Table~\\ref{tab:harx_tests}. Holm adjustment is applied within each panel to the "
                "pre-registered comparisons; other rows are unadjusted (--). Panel A: one regression per "
                "origin on the stacked panel of all stocks with stock fixed effects (within transformation "
                "on training rows). Panel B: the target is the log ratio of monthly to weekly future mean "
                "variance. Panel C: a single market series with HAR components of the cross-sectional mean "
                "log variance; $T\\approx645$ evaluation dates. $^{*}$ $p<0.10$, $^{**}$ $p<0.05$, "
                "$^{***}$ $p<0.01$.\n")
        f.write("\\end{tablenotes}\n\\end{table}\n")
    print(f"Saved {TABLES / 'table13_candidates.tex'}")


def main(only: tuple[str, ...] | None = None) -> None:
    print("=" * 70)
    print("   MODULE 13: PRE-REGISTERED CANDIDATE DESIGNS (runs 05-07)")
    print("=" * 70)
    bundle = load_bundle()
    targets = build_targets(bundle)
    init_n = int(len(bundle.sample_dates) * INIT_TRAIN_FRAC)
    res = {}
    todo = only or ("05", "06", "07")
    if "05" in todo:
        res["05"] = run05(bundle, targets, init_n)
    if "06" in todo:
        res["06"] = run06(bundle, targets, init_n)
    if "07" in todo:
        res["07"] = run07(bundle, targets, init_n)
    # merge with previously saved results for the table
    for key, fn in [("05", "candidates_run05_pooled.csv"), ("06", "candidates_run06_duration.csv"),
                    ("07", "candidates_run07_market.csv")]:
        if key not in res and (INTERM / fn).exists():
            res[key] = pd.read_csv(INTERM / fn)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        for key, df in res.items():
            print(f"\nrun-{key}:\n" + df.to_string(index=False))
    write_table13(res)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", default=None, help="Subset of runs: 05 06 07")
    a = p.parse_args()
    main(only=tuple(a.only) if a.only else None)
