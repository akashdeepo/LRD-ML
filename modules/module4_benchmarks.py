"""
MODULE 4: Linear Benchmark Forecasts (Models A, B, C)
======================================================
For each model in {A, B, C}, each horizon h in {1, 5, 22}, and each stock,
produces an out-of-sample forecast series of log mean future Parkinson RV via
expanding-window OLS.

Models (predictor sets defined in modules.forecast_io.MODEL_FEATURES):
    A — HAR core              (5 predictors)
    B — A + own persistence   (11 predictors)
    C — B + cross-sectional + market + interactions  (18 predictors)

Outputs (results/intermediate/forecasts/):
    {A,B,C}_h{1,5,22}_yhat.csv   T_eval x N out-of-sample forecasts
    {A,B,C}_h{1,5,22}_y.csv      T_eval x N realised log mean future RV
    coverage_summary.csv          per-(model,h) non-null share
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from modules.forecast_io import (
    HORIZONS, MODEL_FEATURES,
    aligned_xy, build_targets, load_bundle, stock_matrix,
)

BASE = Path(__file__).resolve().parent.parent
FCST_DIR = BASE / "results" / "intermediate" / "forecasts"

INIT_TRAIN_FRAC = 0.40   # use first 40% of sample dates as training warm-up


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return beta


def expanding_forecast(X: pd.DataFrame, y: pd.Series,
                       init_n: int) -> pd.Series:
    """Expanding-window OLS forecast: at each step t >= init_n, fit on
    rows [0, t) and predict row t. Predictors at row t are X.iloc[t]."""
    Xa = X.values
    ya = y.values
    yhat = np.full(len(ya), np.nan)
    for t in range(init_n, len(ya)):
        beta = _ols(Xa[:t], ya[:t])
        yhat[t] = beta[0] + Xa[t] @ beta[1:]
    return pd.Series(yhat, index=y.index)


def run_model_horizon(bundle, model: str, h: int, init_n: int,
                      verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = build_targets(bundle)
    yhat_panel = pd.DataFrame(index=bundle.sample_dates,
                              columns=bundle.panel.kept, dtype=float)
    y_panel = pd.DataFrame(index=bundle.sample_dates,
                           columns=bundle.panel.kept, dtype=float)
    for i, t in enumerate(bundle.panel.kept):
        sm = stock_matrix(bundle, t, model, targets)
        X, y = aligned_xy(sm, h)
        if len(X) < init_n + 5:
            continue
        yhat = expanding_forecast(X, y, init_n)
        yhat_panel.loc[yhat.index, t] = yhat.values
        y_panel.loc[y.index, t] = y.values
        if verbose and i % 25 == 0:
            print(f"    [{i+1}/{len(bundle.panel.kept)}] {t}")
    return yhat_panel, y_panel


def main() -> None:
    print("=" * 70)
    print("   MODULE 4: LINEAR BENCHMARK FORECASTS (Models A, B, C)")
    print("=" * 70)
    FCST_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle()
    init_n = int(len(bundle.sample_dates) * INIT_TRAIN_FRAC)
    print(f"  N={len(bundle.panel.kept)}  T_sample={len(bundle.sample_dates)}  "
          f"init_train={init_n} ({100*INIT_TRAIN_FRAC:.0f}% of sample)")
    print(f"  Forecast eval starts {bundle.sample_dates[init_n].date()}")

    coverage = []
    for model in ("A", "B", "C"):
        nfeat = len(MODEL_FEATURES[model])
        for h in HORIZONS:
            print(f"\n[Model {model}, h={h:2d}, {nfeat} features] expanding-OLS, "
                  f"per-stock walk-forward...")
            yhat_panel, y_panel = run_model_horizon(bundle, model, h, init_n)
            yhat_fp = FCST_DIR / f"{model}_h{h:02d}_yhat.csv"
            y_fp = FCST_DIR / f"{model}_h{h:02d}_y.csv"
            yhat_panel.to_csv(yhat_fp)
            y_panel.to_csv(y_fp)
            cov = yhat_panel.notna().sum().sum()
            print(f"   saved {yhat_fp.name} (cov: {cov} non-null cells, "
                  f"{cov / yhat_panel.size * 100:.1f}%)")
            coverage.append({
                "model": model, "horizon": h,
                "n_features": nfeat,
                "n_forecasts": int(cov),
                "share_non_null_pct": round(cov / yhat_panel.size * 100, 2),
            })

    pd.DataFrame(coverage).to_csv(FCST_DIR / "coverage_summary.csv", index=False)
    print("\n" + "=" * 70)
    print(f"All forecast panels saved under {FCST_DIR}")


if __name__ == "__main__":
    main()
