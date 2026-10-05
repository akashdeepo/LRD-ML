"""
MODULE 23: Machine-Learning Feature Ablations (run-30; overnight)
=================================================================
Each learner is trained twice under identical tuning, splits, refit cadence
and evaluation cells: on HAR-X's 7 predictors and on model C's 18. The
difference isolates the value of the persistence features for that learner,
which the module-5 comparison (learner on 18 predictors vs linear HAR-X) did
not. Changes from module 5 requested in the review:

  * feature scaling and hyperparameter search inside each cross-validation
    fold (sklearn Pipeline inside GridSearchCV with TimeSeriesSplit and a gap);
  * the random forest is tuned like the other learners (LightGBM random-forest
    mode: leaves, minimum leaf size and feature fraction by time-series CV).

Run from the repo root (about 6 hours on 12 cores):
    python -m modules.module23_ml_ablation
Progress is appended to results/intermediate/ml_ablation_progress.txt; finished
(learner, feature set, horizon) combinations are skipped on restart.

Outputs: results/intermediate/forecasts/ablation/{learner}_{p7|p18}_h{h}_{yhat,y}.csv
         results/intermediate/ml_ablation.csv
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modules import module5_ml_models as m5
from modules.forecast_io import HORIZONS, MODEL_FEATURES, load_bundle
from modules.module6_forecast_eval import diebold_mariano, squared_loss

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "intermediate" / "forecasts" / "ablation"
INTERM = BASE / "results" / "intermediate"
PROGRESS = INTERM / "ml_ablation_progress.txt"

FEATURE_SETS = {"p7": MODEL_FEATURES["A1"], "p18": MODEL_FEATURES["C"]}
LEARNERS = ["lasso", "ridge", "en", "rf", "gbm"]


def _tscv(h):
    return TimeSeriesSplit(n_splits=5, gap=int(np.ceil(h / 5)))


def _grid(model, grid, h):
    return GridSearchCV(Pipeline([("scale", StandardScaler()), ("model", model)]),
                        grid, cv=_tscv(h), scoring="neg_mean_squared_error", n_jobs=1)


def lasso_p(h):
    return _grid(Lasso(max_iter=5000), {"model__alpha": np.logspace(-4, 0, 20)}, h)


def ridge_p(h):
    return _grid(Ridge(), {"model__alpha": np.logspace(-3, 3, 25)}, h)


def en_p(h):
    return _grid(ElasticNet(max_iter=5000),
                 {"model__alpha": np.logspace(-4, 0, 12), "model__l1_ratio": [0.1, 0.5, 0.9]}, h)


RF_BASE = dict(boosting_type="rf", n_estimators=200, bagging_fraction=0.8, bagging_freq=1,
               learning_rate=1.0, verbose=-1, n_jobs=1)
RF_GRID = [dict(num_leaves=nl, min_data_in_leaf=ml, feature_fraction=ff)
           for nl in (15, 63) for ml in (20, 50) for ff in (0.33, 0.67)]


class TunedRF:
    def __init__(self, h, seed=0):
        self.h, self.seed = h, seed

    def fit(self, X, y):
        best = (np.inf, None)
        for p in RF_GRID:
            err = 0.0
            for tr, va in TimeSeriesSplit(n_splits=3, gap=int(np.ceil(self.h / 5))).split(X):
                m = lgb.LGBMRegressor(**RF_BASE, **p, random_state=self.seed).fit(X[tr], y[tr])
                err += np.mean((m.predict(X[va]) - y[va]) ** 2)
            if err < best[0]:
                best = (err, p)
        self.params_ = best[1]
        self.model_ = lgb.LGBMRegressor(**RF_BASE, **self.params_, random_state=self.seed).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


# register with module 5 so its walk_forward / run_estimator_horizon can be reused
m5.ESTIMATORS.update({"lasso_p": lasso_p, "ridge_p": ridge_p, "en_p": en_p, "rf_lgb": TunedRF})
m5.PARALLEL_STOCKS.update({"lasso_p", "ridge_p", "en_p", "rf_lgb", "gbm"})
NAME = {"lasso": "lasso_p", "ridge": "ridge_p", "en": "en_p", "rf": "rf_lgb", "gbm": "gbm"}


def _done() -> set[str]:
    return set(PROGRESS.read_text().split()) if PROGRESS.exists() else set()


def run_all() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    init_n = int(len(bundle.sample_dates) * m5.INIT_TRAIN_FRAC)
    done = _done()
    for learner in LEARNERS:
        for fs, feats in FEATURE_SETS.items():
            MODEL_FEATURES["D"] = feats
            for h in HORIZONS:
                key = f"{learner}_{fs}_h{h:02d}"
                if key in done:
                    continue
                t0 = time.time()
                print(f"[{time.strftime('%H:%M')}] {key} ...", flush=True)
                yhat, y = m5.run_estimator_horizon(bundle, NAME[learner], h, init_n, verbose=False)
                yhat.to_csv(OUT / f"{key}_yhat.csv"); y.to_csv(OUT / f"{key}_y.csv")
                with open(PROGRESS, "a") as f:
                    f.write(key + "\n")
                print(f"    done in {(time.time() - t0) / 60:.1f} min", flush=True)


def evaluate() -> pd.DataFrame:
    F = BASE / "results" / "intermediate" / "forecasts"
    rows = []
    for learner in LEARNERS:
        for h in HORIZONS:
            p7 = OUT / f"{learner}_p7_h{h:02d}_yhat.csv"; p18 = OUT / f"{learner}_p18_h{h:02d}_yhat.csv"
            if not (p7.exists() and p18.exists()):
                continue
            rd = lambda p: pd.read_csv(p, index_col=0, parse_dates=True)
            y = rd(OUT / f"{learner}_p18_h{h:02d}_y.csv")
            L7, L18 = squared_loss(rd(p7), y), squared_loss(rd(p18), y)
            Lb = squared_loss(rd(F / f"A1_h{h:02d}_yhat.csv"), rd(F / f"A1_h{h:02d}_y.csv"))
            cm = L7.notna() & L18.notna() & Lb.reindex_like(L7).notna()
            L7, L18, Lb = L7.where(cm), L18.where(cm), Lb.reindex_like(L7).where(cm)
            m7, m18, mb = (float(np.nanmean(x.values)) for x in (L7, L18, Lb))
            _, t_ab, p_ab, T = diebold_mariano(L7, L18, h=h)
            _, t_b7, _, _ = diebold_mariano(Lb, L7, h=h)
            _, t_b18, _, _ = diebold_mariano(Lb, L18, h=h)
            rows.append({"learner": learner, "h": h, "T": T, "mse_p7": m7, "mse_p18": m18, "mse_linear_harx": mb,
                         "gain_p18_vs_p7_pct": 100 * (1 - m18 / m7), "dm_t_p18_vs_p7": t_ab, "dm_p": p_ab,
                         "gain_p7_vs_linear_harx_pct": 100 * (1 - m7 / mb), "dm_t_p7_vs_harx": t_b7,
                         "gain_p18_vs_linear_harx_pct": 100 * (1 - m18 / mb), "dm_t_p18_vs_harx": t_b18})
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "ml_ablation.csv", index=False)
    return df


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 220)
    if "--eval-only" not in sys.argv:
        run_all()
    print(evaluate().round(3).to_string(index=False))
