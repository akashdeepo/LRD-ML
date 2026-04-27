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
every 20 weekly steps ≈ 5 months. Hyperparameters are CV'd inside each
training window for the regularised linear models; tree models use fixed
sensible defaults (and are less sensitive to small parameter changes).

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
GBM_PARAMS = dict(
    objective="regression",
    n_estimators=400, learning_rate=0.05,
    num_leaves=31, min_data_in_leaf=20, max_depth=-1,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    verbose=-1,
)


# ------------------------------------------------------------------ estimators
def _lasso(seed: int = 0):
    return LassoCV(cv=5, n_alphas=20, max_iter=2000, n_jobs=1, random_state=seed)


def _ridge():
    return RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5)


def _en(seed: int = 0):
    return ElasticNetCV(cv=5, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                        n_alphas=15, max_iter=2000, n_jobs=1, random_state=seed)


def _rf(seed: int = 0):
    return RandomForestRegressor(n_estimators=RF_TREES, max_depth=None,
                                 min_samples_leaf=20, n_jobs=-1, random_state=seed)


def _gbm():
    return lgb.LGBMRegressor(**GBM_PARAMS, random_state=0)


ESTIMATORS = {
    "lasso": _lasso,
    "ridge": _ridge,
    "en": _en,
    "rf": _rf,
    "gbm": _gbm,
}
NEEDS_SCALING = {"lasso", "ridge", "en"}


# ------------------------------------------------------------------ walk-forward
def walk_forward(X: pd.DataFrame, y: pd.Series, init_n: int,
                 estimator_name: str, refit_stride: int = REFIT_STRIDE) -> pd.Series:
    Xa = X.values
    ya = y.values
    yhat = np.full(len(ya), np.nan)

    last_fit_t = -10**9
    model = None
    scaler = None
    needs_scale = estimator_name in NEEDS_SCALING

    for t in range(init_n, len(ya)):
        if (t - last_fit_t) >= refit_stride or model is None:
            X_train, y_train = Xa[:t], ya[:t]
            if needs_scale:
                scaler = StandardScaler().fit(X_train)
                X_train_s = scaler.transform(X_train)
            else:
                X_train_s = X_train
            model = ESTIMATORS[estimator_name]()
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
    n = len(bundle.panel.kept)
    for i, t in enumerate(bundle.panel.kept):
        sm = stock_matrix(bundle, t, "D", targets)
        X, y = aligned_xy(sm, h)
        if len(X) < init_n + 5:
            continue
        yhat = walk_forward(X, y, init_n, est_name, refit_stride)
        yhat_panel.loc[yhat.index, t] = yhat.values
        y_panel.loc[y.index, t] = y.values
        if verbose and (i + 1) % 25 == 0:
            print(f"    [{i + 1}/{n}] {t}")
    return yhat_panel, y_panel


def main() -> None:
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
    for est_name in ESTIMATORS.keys():
        for h in HORIZONS:
            print(f"\n[D_{est_name}, h={h:2d}] walk-forward...")
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

    pd.DataFrame(coverage).to_csv(FCST_DIR / "coverage_summary_D.csv", index=False)
    print("\n" + "=" * 70)
    print(f"All Model D forecasts saved under {FCST_DIR}")


if __name__ == "__main__":
    main()
