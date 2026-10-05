"""
MODULE 18: Conditional Predictive Ability with Bounded Estimation Memory (run-19)
================================================================================
The Giacomini-White (2006) test assumes that parameter-estimation error does
not vanish, which holds for a fixed-length rolling estimation window but not
for the expanding windows of module 4. This module re-estimates HAR-X and the
persistence specifications with a rolling window of ROLL_MAIN weekly rows (the
length of the initial training sample, so the evaluation period is unchanged)
and ROLL_ALT rows as a sensitivity check, applies the same point-in-time
embargo, and reports:

  * relative MSE of each specification vs HAR-X on the rolling forecasts;
  * the GW statistic with instruments (1, d_{t-k}), k = ceil(h/5), and the
    coefficients of the conditional-moment regression d_t on (1, d_{t-k}) with
    HAC t-statistics, so the direction of any rejection is visible;
  * a check that the instrument's target window has closed by the origin.

Outputs: results/intermediate/forecasts/R{W}_{spec}_h{h}_yhat.csv,
         results/intermediate/rolling_gw.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from modules.forecast_io import (
    HORIZONS, aligned_xy, build_targets, first_eval_row, load_bundle, n_train_rows,
    row_positions, stock_matrix,
)
from modules.module4_benchmarks import INIT_TRAIN_FRAC, _ols
from modules.module6_forecast_eval import SAMPLE_STRIDE, diebold_mariano, hac_bandwidth, squared_loss
from modules.module12_incremental_tests import _hac_mean_test, _wald_hac

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
FCST = BASE / "results" / "intermediate" / "forecasts"
INTERM = BASE / "results" / "intermediate"

ROLL_MAIN, ROLL_ALT = 431, 260
SPECS = ["A1", "A1cs", "A1sec", "A1mod", "C"]


def rolling_forecast(X: pd.DataFrame, y: pd.Series, init_date, h: int,
                     full_index: pd.DatetimeIndex, window: int) -> pd.Series:
    """Fixed-length rolling OLS with the module-4 embargo: at origin t the fit
    uses the last `window` embargoed rows."""
    Xa, ya = X.values, y.values
    pos = row_positions(X.index, full_index)
    yhat = np.full(len(ya), np.nan)
    for t in range(first_eval_row(X.index, init_date), len(ya)):
        n = n_train_rows(pos, t, h)
        lo = max(0, n - window)
        if n - lo < 30:
            continue
        beta = _ols(Xa[lo:n], ya[lo:n])
        yhat[t] = beta[0] + Xa[t] @ beta[1:]
    return pd.Series(yhat, index=y.index)


def run_spec(bundle, targets, spec: str, h: int, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    init_date = bundle.sample_dates[int(len(bundle.sample_dates) * INIT_TRAIN_FRAC)]
    jobs = []
    for t in bundle.panel.kept:
        X, y = aligned_xy(stock_matrix(bundle, t, spec, targets), h)
        if len(X) >= 60:
            jobs.append((t, X, y))
    fits = Parallel(n_jobs=12)(delayed(rolling_forecast)(X, y, init_date, h, bundle.rv.index, window)
                               for _, X, y in jobs)
    yhat = pd.DataFrame(index=bundle.sample_dates, columns=bundle.panel.kept, dtype=float)
    yy = yhat.copy()
    for (t, X, y), f in zip(jobs, fits):
        yhat.loc[f.index, t] = f.values
        yy.loc[y.index, t] = y.values
    yhat.to_csv(FCST / f"R{window}_{spec}_h{h:02d}_yhat.csv")
    yy.to_csv(FCST / f"R{window}_{spec}_h{h:02d}_y.csv")
    return yhat, yy


def gw_with_direction(Lb: pd.DataFrame, Ls: pd.DataFrame, h: int) -> dict:
    """GW statistic plus the conditional-moment regression d_t = a + b d_{t-k}."""
    d = (Lb - Ls).mean(axis=1).dropna().values          # >0 favours the spec
    k = max(int(np.ceil(h / SAMPLE_STRIDE)), 1)
    Z = np.column_stack([d[k:], d[k:] * d[:-k]])
    stat, T = _wald_hac(Z, h)
    X = np.column_stack([np.ones(len(d) - k), d[:-k]])
    beta = np.linalg.lstsq(X, d[k:], rcond=None)[0]
    resid = d[k:] - X @ beta
    # HAC t-statistics for (a, b)
    bw = hac_bandwidth(T, h)
    S = np.zeros((2, 2)); U = X * resid[:, None]
    S += U.T @ U / T
    for j in range(1, bw + 1):
        G = U[j:].T @ U[:-j] / T
        S += (1 - j / (bw + 1)) * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X / T)
    V = XtX_inv @ S @ XtX_inv / T
    se = np.sqrt(np.diag(V))
    return {"gw_stat": stat, "gw_p": float(1 - stats.chi2.cdf(stat, 2)), "T": T,
            "cm_intercept": beta[0], "cm_intercept_t": beta[0] / se[0],
            "cm_slope": beta[1], "cm_slope_t": beta[1] / se[1],
            "instrument_lag_days": k * SAMPLE_STRIDE, "instrument_target_closed": k * SAMPLE_STRIDE >= h}


def main() -> None:
    pd.set_option("display.width", 220)
    print("=" * 70); print("   MODULE 18: ROLLING-WINDOW GIACOMINI-WHITE"); print("=" * 70)
    bundle = load_bundle()
    targets = build_targets(bundle)
    rows = []
    for window in (ROLL_MAIN, ROLL_ALT):
        for h in HORIZONS:
            print(f"  window {window}, h={h}: ", end="", flush=True)
            panels = {}
            for spec in SPECS:
                print(spec, end=" ", flush=True)
                panels[spec] = run_spec(bundle, targets, spec, h, window)
            print()
            yb, y = panels["A1"]
            Lb_full = squared_loss(yb, y)
            for spec in SPECS[1:]:
                ys, _ = panels[spec]
                Ls_full = squared_loss(ys, y)
                cm = Lb_full.notna() & Ls_full.notna()
                Lb, Ls = Lb_full.where(cm), Ls_full.where(cm)
                _, t_dm, p_dm, T = diebold_mariano(Lb, Ls, h=h)
                r = {"window": window, "spec": spec, "h": h,
                     "mse_harx": float(np.nanmean(Lb.values)), "mse_spec": float(np.nanmean(Ls.values))}
                r["gain_vs_harx_pct"] = 100 * (1 - r["mse_spec"] / r["mse_harx"])
                r["dm_t"] = t_dm; r["dm_p"] = p_dm
                r.update(gw_with_direction(Lb, Ls, h))
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "rolling_gw.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
