"""
MODULE 4: Linear Benchmark Forecasts (Models A, A1..A5, C)
==========================================================
For each model in the layered ladder, each horizon h in {1, 5, 22}, and each
stock, produces an out-of-sample forecast series of log mean future
range-based variance proxy via expanding-window OLS.

Layered ladder (predictor sets defined in modules.forecast_io.MODEL_FEATURES):
    A   - HAR core                                              (5)
    A1  - HAR-X: HAR + VIX, MOVE                                (7)
    A2  - HAR + own-stock persistence block (== legacy B)       (11)
    A3  - HAR + cross-sectional mean/std of d                   (7)
    A4  - HAR + sector mean of d                                (6)
    A5  - HAR + d + VIX, MOVE + d*VIX, d*MOVE (with main eff.)  (10)
    C   - full structural (union)                               (18)

A2 is numerically identical to legacy B; we run it under the new label so the
forecast cache is self-contained.

Outputs (results/intermediate/forecasts/):
    {model}_h{1,5,22}_yhat.csv   T_eval x N out-of-sample forecasts
    {model}_h{1,5,22}_y.csv      T_eval x N realised log mean future RV
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

LINEAR_LADDER = ("A", "A1", "A2", "A3", "A4", "A5", "C")


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


def main(only: tuple[str, ...] | None = None,
         skip_existing: bool = False) -> None:
    """Fit the linear ladder.

    Parameters
    ----------
    only : tuple of model names, optional
        If given, fit only these models. Defaults to the full LINEAR_LADDER.
    skip_existing : bool
        If True, skip (model, h) combinations whose yhat file already exists.
        Useful for incremental runs after adding new layers.
    """
    models = only if only is not None else LINEAR_LADDER

    print("=" * 70)
    print(f"   MODULE 4: LINEAR LADDER FORECASTS  models={list(models)}")
    print("=" * 70)
    FCST_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle()
    init_n = int(len(bundle.sample_dates) * INIT_TRAIN_FRAC)
    print(f"  N={len(bundle.panel.kept)}  T_sample={len(bundle.sample_dates)}  "
          f"init_train={init_n} ({100*INIT_TRAIN_FRAC:.0f}% of sample)")
    print(f"  Forecast eval starts {bundle.sample_dates[init_n].date()}")

    coverage = []
    for model in models:
        if model not in MODEL_FEATURES:
            raise KeyError(f"Unknown model {model!r}; "
                           f"known: {sorted(MODEL_FEATURES)}")
        nfeat = len(MODEL_FEATURES[model])
        for h in HORIZONS:
            yhat_fp = FCST_DIR / f"{model}_h{h:02d}_yhat.csv"
            y_fp = FCST_DIR / f"{model}_h{h:02d}_y.csv"
            if skip_existing and yhat_fp.exists() and y_fp.exists():
                print(f"\n[Model {model}, h={h:2d}] cached -> skip")
                continue
            print(f"\n[Model {model}, h={h:2d}, {nfeat} features] expanding-OLS, "
                  f"per-stock walk-forward...")
            yhat_panel, y_panel = run_model_horizon(bundle, model, h, init_n)
            yhat_panel.to_csv(yhat_fp)
            y_panel.to_csv(y_fp)
            cov = int(yhat_panel.notna().sum().sum())
            print(f"   saved {yhat_fp.name} (cov: {cov} non-null cells, "
                  f"{cov / yhat_panel.size * 100:.1f}%)")
            coverage.append({
                "model": model, "horizon": h,
                "n_features": nfeat,
                "n_forecasts": cov,
                "share_non_null_pct": round(cov / yhat_panel.size * 100, 2),
            })

    cov_fp = FCST_DIR / "coverage_summary.csv"
    if coverage:
        new_cov = pd.DataFrame(coverage)
        if cov_fp.exists():
            old_cov = pd.read_csv(cov_fp)
            mask = ~old_cov.set_index(["model", "horizon"]).index.isin(
                new_cov.set_index(["model", "horizon"]).index
            )
            merged = pd.concat([old_cov[mask], new_cov], ignore_index=True)
            merged.sort_values(["model", "horizon"]).to_csv(cov_fp, index=False)
        else:
            new_cov.to_csv(cov_fp, index=False)
    print("\n" + "=" * 70)
    print(f"Linear-ladder forecasts under {FCST_DIR}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", default=None,
                   help="Subset of models to fit (default: full ladder)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip (model, h) combos whose forecast file exists")
    a = p.parse_args()
    main(only=tuple(a.only) if a.only else None,
         skip_existing=a.skip_existing)
