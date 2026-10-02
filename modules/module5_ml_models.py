"""
MODULE 5: Machine-Learning Forecasts (Model D — non-linear)
============================================================
Same predictor set as Model C (per Rachev's nested design), but estimated
with shrinkage and tree-based learners:
    D_lasso  — LassoCV
    D_ridge  — RidgeCV
    D_en     — ElasticNetCV
    D_rf     — RandomForestRegressor
    D_gbm    — LightGBM regressor

Walk-forward fitting cadence is coarser than Module 4 (refit every K sample
steps) because tree models are much heavier than OLS. Default K=20 ≈ refit
every 20 weekly steps ≈ 5 months. Hyperparameters are chosen by time-series
CV inside each training window for the regularised linear models and for
gradient boosting (audit I32); the random forest uses fixed defaults
(200 trees, min leaf 20), to which it is insensitive.

Outputs (results/intermediate/forecasts/):
    D_{lasso,ridge,en,rf,gbm}_h{1,5,22}_yhat.csv
    D_{...}_y.csv              (same realised log-mean-future-RV)
    coverage_summary_D.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import TimeSeriesSplit
from joblib import Parallel, delayed

from modules.forecast_io import first_eval_row, n_train_rows, row_positions
import lightgbm as lgb

from modules.forecast_io import (
    HORIZONS, MODEL_FEATURES,
    aligned_xy, build_targets, load_bundle, stock_matrix,
)

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
FCST_DIR = BASE / "results" / "intermediate" / "forecasts"

INIT_TRAIN_FRAC = 0.40
REFIT_STRIDE = 20             # refit ML model every 20 sample steps
RF_TREES = 200
GBM_BASE = dict(
    objective="regression", learning_rate=0.05, max_depth=-1,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    verbose=-1, n_jobs=1,
)
# Searched by time-series CV at every refit (audit I32; the old fixed
# 400 trees x 31 leaves overfit the 430-1,100 rows available per stock).
GBM_GRID = [dict(num_leaves=nl, min_data_in_leaf=ml) for nl in (2, 4, 15) for ml in (20, 50)]
GBM_TREES = (25, 50, 100, 200, 400, 800)
STOCK_JOBS = 12               # stocks fitted in parallel for gradient boosting


# ------------------------------------------------------------------ estimators
# Hyperparameters of the shrinkage estimators are chosen by time-series
# cross-validation: expanding folds that validate on later data only, with a
# gap of ceil(h/5) rows so overlapping targets do not straddle a fold boundary
# (audit I13; the old cv=5 was unshuffled K-fold, validating on the past).
def _tscv(h: int) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=5, gap=int(np.ceil(h / 5)))


def _lasso(h: int, seed: int = 0):
    return LassoCV(cv=_tscv(h), n_alphas=20, max_iter=2000, n_jobs=1, random_state=seed)


def _ridge(h: int):
    return RidgeCV(alphas=np.logspace(-3, 3, 25), cv=_tscv(h))


def _en(h: int, seed: int = 0):
    return ElasticNetCV(cv=_tscv(h), l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                        n_alphas=15, max_iter=2000, n_jobs=1, random_state=seed)


def _rf(h: int, seed: int = 0):
    return RandomForestRegressor(n_estimators=RF_TREES, max_depth=None,
                                 min_samples_leaf=20, n_jobs=-1, random_state=seed)


class TunedLGBM:
    """LightGBM whose leaves, minimum leaf size and number of trees are chosen
    by expanding time-series CV on the training rows (MSE summed over folds),
    then refit on all of them. Tree counts are scored from one fit per grid
    point via staged prediction."""

    def __init__(self, h: int, seed: int = 0):
        self.h, self.seed = h, seed

    def fit(self, X, y):
        best = (np.inf, None, None)
        for p in GBM_GRID:
            err = np.zeros(len(GBM_TREES))
            for tr, va in TimeSeriesSplit(n_splits=3, gap=int(np.ceil(self.h / 5))).split(X):
                m = lgb.LGBMRegressor(**GBM_BASE, **p, n_estimators=max(GBM_TREES),
                                      random_state=self.seed).fit(X[tr], y[tr])
                for k, n in enumerate(GBM_TREES):
                    err[k] += np.mean((m.predict(X[va], num_iteration=n) - y[va]) ** 2)
            k = int(err.argmin())
            if err[k] < best[0]:
                best = (err[k], p, GBM_TREES[k])
        self.params_ = {**best[1], "n_estimators": best[2]}
        self.model_ = lgb.LGBMRegressor(**GBM_BASE, **self.params_,
                                        random_state=self.seed).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def _gbm(h: int):
    return TunedLGBM(h)


ESTIMATORS = {
    "lasso": _lasso,
    "ridge": _ridge,
    "en": _en,
    "rf": _rf,
    "gbm": _gbm,
}
NEEDS_SCALING = {"lasso", "ridge", "en"}
PARALLEL_STOCKS = {"gbm"}     # single-threaded fits; the others parallelise internally or are fast


# ------------------------------------------------------------------ walk-forward
def walk_forward(X: pd.DataFrame, y: pd.Series, init_date,
                 estimator_name: str, h: int, full_index: pd.DatetimeIndex,
                 refit_stride: int = REFIT_STRIDE) -> pd.Series:
    """Walk-forward ML forecast, refit every `refit_stride` origins, with the
    same point-in-time embargo as module 4: a refit at origin t uses only rows
    whose target window ends on or before pos_t (docs/FINDINGS.md #1)."""
    warnings.filterwarnings("ignore")     # joblib workers do not inherit the module-level filter
    Xa = X.values
    ya = y.values
    pos = row_positions(X.index, full_index)
    yhat = np.full(len(ya), np.nan)

    last_fit_t = -10**9
    model = None
    scaler = None
    needs_scale = estimator_name in NEEDS_SCALING

    for t in range(first_eval_row(X.index, init_date), len(ya)):
        if (t - last_fit_t) >= refit_stride or model is None:
            n = n_train_rows(pos, t, h)
            X_train, y_train = Xa[:n], ya[:n]
            if needs_scale:
                scaler = StandardScaler().fit(X_train)
                X_train_s = scaler.transform(X_train)
            else:
                X_train_s = X_train
            model = ESTIMATORS[estimator_name](h)
            model.fit(X_train_s, y_train)
            last_fit_t = t

        x_now = Xa[t:t + 1]
        if needs_scale:
            x_now = scaler.transform(x_now)
        yhat[t] = model.predict(x_now)[0]

    return pd.Series(yhat, index=y.index)


def run_estimator_horizon(bundle, est_name: str, h: int, init_n: int,
                          refit_stride: int = REFIT_STRIDE,
                          verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = build_targets(bundle)
    yhat_panel = pd.DataFrame(index=bundle.sample_dates,
                              columns=bundle.panel.kept, dtype=float)
    y_panel = pd.DataFrame(index=bundle.sample_dates,
                           columns=bundle.panel.kept, dtype=float)
    jobs = []
    for t in bundle.panel.kept:
        sm = stock_matrix(bundle, t, "D", targets)
        X, y = aligned_xy(sm, h)
        if len(X) >= 60:
            jobs.append((t, X, y))
    n_jobs = STOCK_JOBS if est_name in PARALLEL_STOCKS else 1
    fits = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
        delayed(walk_forward)(X, y, bundle.sample_dates[init_n], est_name, h,
                              bundle.rv.index, refit_stride)
        for _, X, y in jobs)
    for (t, X, y), yhat in zip(jobs, fits):
        yhat_panel.loc[yhat.index, t] = yhat.values
        y_panel.loc[y.index, t] = y.values
    return yhat_panel, y_panel


def main(only: tuple[str, ...] | None = None,
         horizons: tuple[int, ...] | None = None) -> None:
    print("=" * 70)
    print("   MODULE 5: ML FORECASTS (Model D - non-linear)")
    print("=" * 70)
    FCST_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle()
    init_n = int(len(bundle.sample_dates) * INIT_TRAIN_FRAC)
    print(f"  N={len(bundle.panel.kept)}  T_sample={len(bundle.sample_dates)}  "
          f"init_train={init_n} ({100 * INIT_TRAIN_FRAC:.0f}% of sample)")
    print(f"  Refit stride: every {REFIT_STRIDE} sample steps")
    print(f"  Predictor count (Model D == C): {len(MODEL_FEATURES['D'])}")

    coverage = []
    for est_name in (only or tuple(ESTIMATORS.keys())):
        for h in (horizons or HORIZONS):
            print(f"\n[D_{est_name}, h={h:2d}] walk-forward...", flush=True)
            yhat_panel, y_panel = run_estimator_horizon(bundle, est_name, h, init_n)
            yhat_fp = FCST_DIR / f"D_{est_name}_h{h:02d}_yhat.csv"
            y_fp = FCST_DIR / f"D_{est_name}_h{h:02d}_y.csv"
            yhat_panel.to_csv(yhat_fp)
            y_panel.to_csv(y_fp)
            cov = int(yhat_panel.notna().sum().sum())
            print(f"   saved {yhat_fp.name} (cov: {cov} non-null cells, "
                  f"{cov / yhat_panel.size * 100:.1f}%)")
            coverage.append({
                "estimator": est_name, "horizon": h,
                "n_features": len(MODEL_FEATURES["D"]),
                "n_forecasts": cov,
                "share_non_null_pct": round(cov / yhat_panel.size * 100, 2),
            })

    cov_fp = FCST_DIR / "coverage_summary_D.csv"
    new_cov = pd.DataFrame(coverage)
    if cov_fp.exists() and len(new_cov):
        old_cov = pd.read_csv(cov_fp)
        key = ["estimator", "horizon"]
        mask = ~old_cov.set_index(key).index.isin(new_cov.set_index(key).index)
        new_cov = pd.concat([old_cov[mask], new_cov], ignore_index=True).sort_values(key)
    new_cov.to_csv(cov_fp, index=False)
    print("\n" + "=" * 70)
    print(f"All Model D forecasts saved under {FCST_DIR}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", default=None, help="Subset of estimators")
    p.add_argument("--horizons", nargs="+", type=int, default=None, help="Subset of horizons")
    a = p.parse_args()
    main(only=tuple(a.only) if a.only else None,
         horizons=tuple(a.horizons) if a.horizons else None)
