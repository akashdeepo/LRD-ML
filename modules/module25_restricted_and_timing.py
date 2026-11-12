"""
MODULE 25: Lagged Indices, Restricted Persistence, Trailing Maxima and
Combinations (runs 37, 38; JRFM round-1 revision)
=====================================================================
Registered in docs/EXPERIMENTS.md (sixth block, commit 4c4d1a4).

  run-37  HAR-X, A1cs, A1sec, A1mod and C refit with the VIX and MOVE lagged
          one trading day; the interactions d x VIX and d x MOVE rebuilt on
          the lagged levels. Each persistence specification is compared with
          lagged HAR-X on common cells.
  run-38  Against (aligned) HAR-X:
            A1d     HAR-X + own d-hat (economically restricted)
            A1pc    HAR-X + first principal component of C's eleven
                    persistence predictors; standardisation and loadings
                    from each training window only
            A1vmax  HAR-X + log of the trailing 750-day maximum VIX
            A1rvmax HAR-X + log of the trailing 750-day maximum of the
                    stock's 22-day mean Parkinson variance
            A1csmax HAR-X + cross-sectional state + log VIX maximum
            comb50  0.5 C + 0.5 HAR-X
            combrt  recursive MSE-optimal weight on C, pooled over stocks,
                    from errors whose targets were observed by the origin
                    (26-date warm-up, weight clipped to [0, 1])

Forecasts are cached in results/intermediate/forecasts/rev1/ and skipped if
present, so the module can be run in pieces:
    python -m modules.module25_restricted_and_timing --run 37
    python -m modules.module25_restricted_and_timing --run 38
    python -m modules.module25_restricted_and_timing --eval
Outputs: results/intermediate/lagged_indices.csv, restricted_persistence.csv
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from modules.forecast_io import (
    HORIZONS, MODEL_FEATURES, aligned_xy, build_targets, first_eval_row, load_bundle,
    n_train_rows, row_positions, stock_matrix,
)
from modules.module4_benchmarks import INIT_TRAIN_FRAC, _ols, expanding_forecast
from modules.module19_matched_robustness import compare

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
FCST = BASE / "results" / "intermediate" / "forecasts"
REV = FCST / "rev1"
INTERM = BASE / "results" / "intermediate"

W_MAX = 750                 # trailing window of the maxima = the state's window
MIN_OBS = 250               # as for the rolling d-hat
PERSISTENCE = [f for f in MODEL_FEATURES["C"] if f not in MODEL_FEATURES["A1"]]
COMB_WARMUP = 26
JOBS = 12


# ------------------------------------------------------------------ inputs
def _broadcast(s: pd.Series, dates, cols) -> pd.DataFrame:
    s = s.reindex(dates)
    return pd.DataFrame({c: s for c in cols})


def lagged_overrides(bundle) -> dict[str, pd.DataFrame]:
    daily = bundle.panel.market[["VIX", "MOVE"]].ffill().shift(1)
    sd, kept = bundle.sample_dates, bundle.panel.kept
    vix, move = _broadcast(daily["VIX"], sd, kept), _broadcast(daily["MOVE"], sd, kept)
    d = bundle.feat["d_gph"].reindex(index=sd, columns=kept)
    return {"vix": vix, "move": move, "d_x_vix": d * vix, "d_x_move": d * move}


def maxima(bundle) -> dict[str, pd.DataFrame]:
    sd, kept = bundle.sample_dates, bundle.panel.kept
    vix = bundle.panel.market["VIX"].ffill()
    vmax = np.log(vix.rolling(W_MAX, min_periods=MIN_OBS).max())
    rv22 = bundle.rv[kept].rolling(22, min_periods=15).mean()
    rvmax = np.log(rv22.rolling(W_MAX, min_periods=MIN_OBS).max())
    return {"vix_max": _broadcast(vmax, sd, kept), "rv_max": rvmax.reindex(sd)}


# ------------------------------------------------------------------ fitting
def _stock_xy(bundle, targets, tkr, model, h, overrides, extra):
    sm = stock_matrix(bundle, tkr, model, targets)
    for c, panel in overrides.items():
        if c in sm.X.columns:                 # only the model's own columns
            sm.X[c] = panel[tkr]
    for c, panel in extra.items():
        sm.X[c] = panel[tkr]
    return aligned_xy(sm, h)


def pc_forecast(X: pd.DataFrame, y: pd.Series, init_date, h: int, full_index,
                base_cols: list[str], pers_cols: list[str]) -> pd.Series:
    """Expanding OLS of y on HAR-X columns and the first principal component
    of the persistence block, the component re-estimated on each training
    window (point in time)."""
    B, P, ya = X[base_cols].values, X[pers_cols].values, y.values
    pos = row_positions(X.index, full_index)
    out = np.full(len(ya), np.nan)
    for t in range(first_eval_row(X.index, init_date), len(ya)):
        n = n_train_rows(pos, t, h)
        mu, sd = P[:n].mean(axis=0), P[:n].std(axis=0)
        sd[sd == 0] = 1.0
        Z = (P[:n] - mu) / sd
        w = np.linalg.svd(Z, full_matrices=False)[2][0]
        pc = Z @ w
        beta = _ols(np.column_stack([B[:n], pc]), ya[:n])
        out[t] = beta[0] + np.concatenate([B[t], [((P[t] - mu) / sd) @ w]]) @ beta[1:]
    return pd.Series(out, index=y.index)


def fit_panel(bundle, targets, name: str, model: str, h: int, overrides=None, extra=None,
              pca: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit `model` (plus `extra` columns, with `overrides` replacing the
    model's own columns) for every stock; cache to rev1/{name}_h{h}."""
    fy, fyy = REV / f"{name}_h{h:02d}_yhat.csv", REV / f"{name}_h{h:02d}_y.csv"
    if fy.exists():
        return (pd.read_csv(fy, index_col=0, parse_dates=True), pd.read_csv(fyy, index_col=0, parse_dates=True))
    overrides, extra = overrides or {}, extra or {}
    sd = bundle.sample_dates
    init_date = sd[int(len(sd) * INIT_TRAIN_FRAC)]
    jobs = []
    for tkr in bundle.panel.kept:
        X, y = _stock_xy(bundle, targets, tkr, model, h, overrides, extra)
        if len(X) >= 60:
            jobs.append((tkr, X, y))
    if pca:
        base_cols = MODEL_FEATURES["A1"]
        fn = lambda X, y: pc_forecast(X, y, init_date, h, bundle.rv.index, base_cols, PERSISTENCE)
    else:
        fn = lambda X, y: expanding_forecast(X, y, init_date, h, bundle.rv.index)
    fits = Parallel(n_jobs=JOBS)(delayed(fn)(X, y) for _, X, y in jobs)
    yh = pd.DataFrame(index=sd, columns=bundle.panel.kept, dtype=float); yy = yh.copy()
    for (tkr, X, y), f in zip(jobs, fits):
        yh.loc[f.index, tkr] = f.values
        yy.loc[y.index, tkr] = y.values
    REV.mkdir(parents=True, exist_ok=True)
    yh.to_csv(fy); yy.to_csv(fyy)
    return yh, yy


def _saved(model: str, h: int):
    return (pd.read_csv(FCST / f"{model}_h{h:02d}_yhat.csv", index_col=0, parse_dates=True),
            pd.read_csv(FCST / f"{model}_h{h:02d}_y.csv", index_col=0, parse_dates=True))


def recursive_combination(yb: pd.DataFrame, yc: pd.DataFrame, y: pd.DataFrame, h: int) -> pd.DataFrame:
    """Weight on C at origin t: pooled least-squares weight from the errors of
    origins whose targets were observed by t (lag ceil(h/5) rows), clipped to
    [0, 1]; NaN before COMB_WARMUP usable dates."""
    lag = int(np.ceil(h / 5))
    cm = yb.notna() & yc.notna() & y.notna()
    eb, dc = (y - yb).where(cm), (yc - yb).where(cm)
    num = (eb * dc).sum(axis=1, min_count=1)          # e_b (yc - yb) per date
    den = (dc ** 2).sum(axis=1, min_count=1)
    used = num.notna().cumsum()
    lam = (num.fillna(0).cumsum() / den.fillna(0).cumsum()).shift(lag).clip(0, 1)
    lam[used.shift(lag) < COMB_WARMUP] = np.nan
    return yb.add(dc.mul(lam, axis=0))


# ------------------------------------------------------------------ runs
def run37() -> None:
    bundle = load_bundle(); targets = build_targets(bundle)
    lag = lagged_overrides(bundle)
    for model in ("A1", "A1cs", "A1sec", "A1mod", "C"):
        for h in HORIZONS:
            fit_panel(bundle, targets, f"{model}_lagidx", model, h, overrides=lag)
            print(f"  run-37 {model} h={h} done", flush=True)


def run38() -> None:
    bundle = load_bundle(); targets = build_targets(bundle)
    mx = maxima(bundle)
    d = bundle.feat["d_gph"].reindex(index=bundle.sample_dates, columns=bundle.panel.kept)
    specs = {"A1d": ("A1", {"d_own": d}, False),
             "A1pc": ("C", {}, True),
             "A1vmax": ("A1", {"vix_max": mx["vix_max"]}, False),
             "A1rvmax": ("A1", {"rv_max": mx["rv_max"]}, False),
             "A1csmax": ("A1cs", {"vix_max": mx["vix_max"]}, False)}
    for name, (model, extra, pca) in specs.items():
        for h in HORIZONS:
            fit_panel(bundle, targets, name, model, h, extra=extra, pca=pca)
            print(f"  run-38 {name} h={h} done", flush=True)


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(3)
    rows37 = []
    for model in ("A1cs", "A1sec", "A1mod", "C"):
        for h in HORIZONS:
            yb_l, y_l = _cached("A1_lagidx", h)
            ys_l, ysy_l = _cached(f"{model}_lagidx", h)
            yb_a, y_a = _saved("A1", h); ys_a, ysy_a = _saved(model, h)
            lagged = compare(ys_l, ysy_l, yb_l, y_l, h, rng=rng)
            aligned = compare(ys_a, ysy_a, yb_a, y_a, h)
            rows37.append({"spec": model, "h": h, **{f"{k}_lagged": v for k, v in lagged.items()},
                           "gain_pct_aligned": aligned["gain_pct"], "dm_t_aligned": aligned["dm_t"],
                           "change_pp": lagged["gain_pct"] - aligned["gain_pct"]})
    rows38 = []
    for h in HORIZONS:
        yb, y = _saved("A1", h)
        yc, _ = _saved("C", h)
        for name in ("A1d", "A1pc", "A1vmax", "A1rvmax", "A1csmax"):
            ys, yy = _cached(name, h)
            rows38.append({"spec": name, "h": h, **compare(ys, yy, yb, y, h, rng=rng)})
        cm = yb.notna() & yc.notna() & y.notna()
        rows38.append({"spec": "comb50", "h": h, **compare((0.5 * yb + 0.5 * yc).where(cm), y, yb, y, h, rng=rng)})
        rt = recursive_combination(yb, yc, y, h)
        rows38.append({"spec": "combrt", "h": h, **compare(rt, y, yb.where(rt.notna()), y, h, rng=rng)})
        ycs, _ = _saved("A1cs", h)
        rows38.append({"spec": "A1cs (registered)", "h": h, **compare(ycs, y, yb, y, h, rng=rng)})
    d37, d38 = pd.DataFrame(rows37), pd.DataFrame(rows38)
    d37.to_csv(INTERM / "lagged_indices.csv", index=False)
    d38.to_csv(INTERM / "restricted_persistence.csv", index=False)
    return d37, d38


def _cached(name: str, h: int):
    return (pd.read_csv(REV / f"{name}_h{h:02d}_yhat.csv", index_col=0, parse_dates=True),
            pd.read_csv(REV / f"{name}_h{h:02d}_y.csv", index_col=0, parse_dates=True))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=["37", "38"], default=None)
    p.add_argument("--eval", action="store_true")
    a = p.parse_args()
    pd.set_option("display.width", 220)
    if a.run == "37":
        run37()
    elif a.run == "38":
        run38()
    if a.eval or a.run is None:
        d37, d38 = evaluate()
        print("\n[run-37] lagged VIX and MOVE: each spec vs lagged HAR-X")
        print(d37[["spec", "h", "gain_pct_lagged", "dm_t_lagged", "ci_lo_lagged", "ci_hi_lagged",
                   "gain_pct_aligned", "change_pp"]].round(3).to_string(index=False))
        print("\n[run-38] restricted persistence, maxima, combinations vs HAR-X")
        print(d38[["spec", "h", "gain_pct", "dm_t", "dm_p", "ci_lo", "ci_hi", "n_stocks"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
