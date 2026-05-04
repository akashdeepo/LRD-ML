"""
MODULE 4B: GARCH(1,1) Benchmark Forecasts
=========================================
Per-stock GARCH(1,1) on log returns, walk-forward over the same sample-date
grid as Modules 4 and 5. For each (stock, sample date t, horizon h) we
produce a forecast of

    y_{t,h} = log( (1/h) * sum_{j=1..h} sigma^2_{t+j|t} )

where sigma^2_{t+j|t} is the GARCH(1,1) j-step-ahead conditional variance
forecast on returns. This matches the target-scale of Modules 4 and 5 (log
mean future Parkinson range-based variance proxy) up to the standard
GARCH-on-returns vs RV proxy mapping.

Refit cadence: every 20 sample steps (matches Module 5's ML pipeline). The
manuscript already notes results are insensitive to this stride.

Outputs (results/intermediate/forecasts/):
    G_h{01,05,22}_yhat.csv   T_eval x N forecasts on log mean variance scale
    G_h{01,05,22}_y.csv      T_eval x N realised log mean future RV^PK
                             (same target as the rest of the ladder)
    coverage_summary appended in coverage_summary.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

from modules.forecast_io import HORIZONS, build_targets, load_bundle
from modules.module4_benchmarks import INIT_TRAIN_FRAC

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
FCST_DIR = BASE / "results" / "intermediate" / "forecasts"

REFIT_STRIDE = 20    # refit every 20 sample steps; reuse params in between
RETURN_SCALE = 100   # arch convention: scale returns to percent for stability
MIN_OBS_FIT = 252    # need at least one year of returns to fit
MAX_TRAIN_LEN = 5000 # cap training window to keep fits fast (~20 years)


def _fit_garch11(r: np.ndarray):
    """Fit GARCH(1,1) with constant mean to a numpy array of returns (already
    on the percent scale). Returns (omega, alpha, beta, last_var, last_resid)
    needed to roll forecasts forward."""
    model = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal",
                       rescale=False)
    res = model.fit(disp="off", show_warning=False)
    p = res.params
    omega = float(p["omega"])
    alpha = float(p["alpha[1]"])
    beta = float(p["beta[1]"])
    cond_var = np.asarray(res.conditional_volatility) ** 2
    mu = float(p["mu"])
    last_var = float(cond_var[-1])
    last_resid = float(r[-1] - mu)
    return omega, alpha, beta, mu, last_var, last_resid


def _multi_step_var(omega: float, alpha: float, beta: float,
                    var_t: float, resid_t: float, h: int) -> np.ndarray:
    """Iterate the GARCH(1,1) recursion forward for h steps starting from
    (var_t, resid_t).  Returns array of length h with sigma^2_{t+j} on the
    % scale."""
    out = np.empty(h, dtype=float)
    # one-step ahead given current squared residual:
    var_next = omega + alpha * resid_t * resid_t + beta * var_t
    out[0] = var_next
    # subsequent steps use unconditional expectation: E[r_{t+j}^2] = var_{t+j}
    persistence = alpha + beta
    long_run = omega / max(1.0 - persistence, 1e-8)
    for j in range(1, h):
        out[j] = long_run + persistence * (out[j - 1] - long_run)
    return out


def _forecast_one_stock(returns: pd.Series, log_rv: pd.Series,
                        sample_dates: pd.DatetimeIndex,
                        init_n: int) -> dict[int, pd.Series]:
    """For one stock, produce log(mean future variance) forecasts at each
    sample date for h in {1,5,22}. Refit GARCH every REFIT_STRIDE sample
    steps and advance the variance recursion forward between refits using
    the actual returns and fixed parameters."""
    out = {h: pd.Series(np.nan, index=sample_dates, dtype=float)
           for h in HORIZONS}

    ret = returns.dropna()
    full_dates = ret.index
    pos = full_dates.get_indexer(sample_dates)

    params = None        # (omega, alpha, beta, mu)
    bias = 0.0           # in-sample mean bias of log GARCH var vs log Parkinson RV
    var_t = None         # conditional variance at the most recent sample date
    last_fit_idx = -10**9
    last_pos = None      # trading-day position corresponding to var_t

    log_rv_full = log_rv.reindex(returns.index).values

    for k in range(init_n, len(sample_dates)):
        p = pos[k]
        if p < 0:
            continue
        end = p + 1
        start = max(0, end - MAX_TRAIN_LEN)
        train = ret.values[start:end] * RETURN_SCALE
        if len(train) < MIN_OBS_FIT:
            continue

        if (k - last_fit_idx) >= REFIT_STRIDE or params is None:
            try:
                omega, alpha, beta, mu, var_t, _resid = _fit_garch11(train)
                params = (omega, alpha, beta, mu)
                last_fit_idx = k
                last_pos = p
            except Exception:
                continue
        else:
            # Advance recursion forward from last_pos+1 to p using the fixed
            # parameters and the actual returns observed in between.
            omega, alpha, beta, mu = params
            for j in range(last_pos + 1, p + 1):
                eps = ret.values[j] * RETURN_SCALE - mu
                var_t = omega + alpha * eps * eps + beta * var_t
            last_pos = p

        # Multi-step variance forecast given (var_t, last_residual at p)
        omega, alpha, beta, mu = params
        resid_t = ret.values[p] * RETURN_SCALE - mu
        # var_{t+1} uses resid_t (already known); deeper steps use the long
        # run convergence form.
        var_next = omega + alpha * resid_t * resid_t + beta * var_t
        persistence = alpha + beta
        long_run = omega / max(1.0 - persistence, 1e-8)

        for h in HORIZONS:
            path = np.empty(h, dtype=float)
            path[0] = var_next
            for j in range(1, h):
                path[j] = long_run + persistence * (path[j - 1] - long_run)
            mean_var = path.mean() / (RETURN_SCALE * RETURN_SCALE)
            if mean_var > 0 and np.isfinite(mean_var):
                out[h].iat[k] = float(np.log(mean_var))
    return out


def main() -> None:
    print("=" * 70)
    print("   MODULE 4B: GARCH(1,1) BENCHMARK FORECASTS")
    print("=" * 70)
    FCST_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle()
    sd = bundle.sample_dates
    init_n = int(len(sd) * INIT_TRAIN_FRAC)
    print(f"  N={len(bundle.panel.kept)}  T_sample={len(sd)}  "
          f"init_train={init_n}  refit_stride={REFIT_STRIDE}")

    targets = build_targets(bundle)

    yhat = {h: pd.DataFrame(np.nan, index=sd,
                            columns=bundle.panel.kept, dtype=float)
            for h in HORIZONS}
    yreal = {h: pd.DataFrame(np.nan, index=sd,
                             columns=bundle.panel.kept, dtype=float)
             for h in HORIZONS}

    for i, t in enumerate(bundle.panel.kept):
        if i % 10 == 0:
            print(f"  [{i+1:3d}/{len(bundle.panel.kept)}] {t} ...")
        per_h = _forecast_one_stock(bundle.returns[t], bundle.log_rv[t],
                                    sd, init_n)
        for h in HORIZONS:
            yhat[h][t] = per_h[h].values
            yreal[h][t] = targets[h][t].values

    cov_rows = []
    for h in HORIZONS:
        yh_fp = FCST_DIR / f"G_h{h:02d}_yhat.csv"
        y_fp = FCST_DIR / f"G_h{h:02d}_y.csv"
        yhat[h].to_csv(yh_fp)
        yreal[h].to_csv(y_fp)
        cov = int(yhat[h].notna().sum().sum())
        share = cov / yhat[h].size * 100
        cov_rows.append({"model": "G", "horizon": h, "n_features": 0,
                         "n_forecasts": cov,
                         "share_non_null_pct": round(share, 2)})
        print(f"   saved {yh_fp.name}  cov: {cov} ({share:.1f}%)")

    cov_fp = FCST_DIR / "coverage_summary.csv"
    new_cov = pd.DataFrame(cov_rows)
    if cov_fp.exists():
        old = pd.read_csv(cov_fp)
        old = old[~((old["model"] == "G"))]
        merged = pd.concat([old, new_cov], ignore_index=True)
        merged.sort_values(["model", "horizon"]).to_csv(cov_fp, index=False)
    else:
        new_cov.to_csv(cov_fp, index=False)
    print("\n" + "=" * 70)
    print(f"GARCH(1,1) forecasts saved under {FCST_DIR}")


if __name__ == "__main__":
    main()
