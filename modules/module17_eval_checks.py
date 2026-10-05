"""
MODULE 17: Evaluation Checks (ledger runs 18, 20, 21)
=====================================================
Three checks on the saved forecasts, requested in the co-author review:

  run-18  Calendar report by horizon: last predictor date, last fully observed
          target date, candidate origins, exclusions, common cells, and the
          embargoed training count at the first and last evaluation origin.
  run-20  Clark-West decomposition: adjusted differential = raw MSE
          differential + adjustment term, each with its HAC t; and a Monte
          Carlo of the test's size for nested per-stock OLS with a common
          error factor.
  run-21  Circular block bootstrap (12-week blocks, 5000 draws) percentile
          intervals for every relative MSE reduction vs HAR and vs HAR-X.

Outputs (results/intermediate): calendar_report.csv, cw_decomposition.csv,
cw_size_mc.csv, bootstrap_intervals.csv.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from modules.forecast_io import (
    HORIZONS, MIN_TRAIN_ROWS, aligned_xy, build_targets, first_eval_row, load_bundle,
    n_train_rows, row_positions, stock_matrix,
)
from modules.module4_benchmarks import INIT_TRAIN_FRAC
from modules.module6_forecast_eval import hac_bandwidth, squared_loss
from modules.module12_incremental_tests import NESTED_OLS, _hac_mean_test

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
FCST = BASE / "results" / "intermediate" / "forecasts"
INTERM = BASE / "results" / "intermediate"

BLOCK, N_BOOT, SEED = 12, 5000, 0
SPECS_VS_HAR = ["A1", "A2", "A3", "A4", "A5", "C", "D_lasso", "D_ridge", "D_en", "D_rf", "D_gbm"]
SPECS_VS_HARX = ["A1cs", "A1sec", "A1mod", "C", "A5", "D_lasso", "D_ridge", "D_en", "D_rf", "D_gbm"]


def _load(model: str, h: int):
    p = FCST / f"{model}_h{h:02d}_yhat.csv"
    if not p.exists():
        return None
    return (pd.read_csv(p, index_col=0, parse_dates=True),
            pd.read_csv(FCST / f"{model}_h{h:02d}_y.csv", index_col=0, parse_dates=True))


# ------------------------------------------------------------------ run-18
def calendar_report(bundle) -> pd.DataFrame:
    full = bundle.rv.index
    sd = bundle.sample_dates
    init_n = int(len(sd) * INIT_TRAIN_FRAC)
    rows = []
    targets = build_targets(bundle)
    for h in HORIZONS:
        yh, y = _load("A", h)
        has_target = targets[h].notna().any(axis=1)
        ok = (yh.notna() & y.notna()).any(axis=1)
        ev = sd[ok]
        # training counts for a representative full-history stock
        sm = stock_matrix(bundle, "JPM", "A", targets)
        X, yy = aligned_xy(sm, h)
        pos = row_positions(X.index, full)
        t0 = first_eval_row(X.index, sd[init_n])
        rows.append({
            "h": h,
            "trading_days": len(full), "last_trading_day": full[-1].date(),
            "weekly_origins": len(sd), "first_origin": sd[0].date(), "last_origin": sd[-1].date(),
            "origins_with_observed_target": int(has_target.sum()),
            "last_origin_with_target": sd[has_target][-1].date(),
            "initial_training_origins": init_n, "last_training_origin": sd[init_n - 1].date(),
            "first_eval_origin": ev[0].date(), "last_eval_origin": ev[-1].date(),
            "eval_origins": int(ok.sum()),
            "origins_excluded_no_target": int((~has_target).sum()),
            "common_cells_HAR_HARX": int((yh.notna() & _load("A1", h)[0].notna()).sum().sum()),
            "train_rows_first_eval_JPM": n_train_rows(pos, t0, h),
            "train_rows_last_eval_JPM": n_train_rows(pos, len(X) - 1, h),
            "embargo_rows_dropped_JPM": (len(X) - 1) - n_train_rows(pos, len(X) - 1, h),
            "min_train_rows": MIN_TRAIN_ROWS,
        })
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "calendar_report.csv", index=False)
    return df


# ------------------------------------------------------------------ run-20
def cw_decomposition() -> pd.DataFrame:
    rows = []
    for spec in ["A5", "A1cs", "A1sec", "A1mod", "C"]:
        for h in HORIZONS:
            yb, y = _load("A1", h)
            ys, _ = _load(spec, h)
            cm = yb.notna() & ys.notna() & y.notna()
            eb, es = (y - yb).where(cm), (y - ys).where(cm)
            raw = (eb ** 2 - es ** 2).mean(axis=1).dropna().values          # >0 favours spec
            adj = ((yb - ys) ** 2).where(cm).mean(axis=1).dropna().values
            cw = raw + adj
            T = len(cw)
            r = {"spec": spec, "h": h, "T": T}
            for name, d in (("raw", raw), ("adjustment", adj), ("cw_adjusted", cw)):
                m, se, t, _ = _hac_mean_test(d, h)
                r[f"{name}_mean"] = m; r[f"{name}_t"] = t
            r["adjustment_share"] = r["adjustment_mean"] / r["cw_adjusted_mean"] if r["cw_adjusted_mean"] else np.nan
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "cw_decomposition.csv", index=False)
    return df


def cw_size_mc(n_rep: int = 500, N: int = 115, T_eval: int = 646, T_train0: int = 431,
               k_extra: int = 11, rho_cs: float | None = None, seed: int = SEED) -> pd.DataFrame:
    """Size of the panel Clark-West test under the null that the k_extra added
    regressors have zero coefficients, for per-stock expanding OLS with a
    common error factor. rho_cs: cross-sectional error correlation (from the
    data if None)."""
    rng = np.random.default_rng(seed)
    if rho_cs is None:
        yb, y = _load("A1", 1)
        e = (y - yb)
        c = e.corr().values
        rho_cs = float(np.nanmean(c[np.triu_indices_from(c, 1)]))
    T = T_train0 + T_eval
    out = {h: [] for h in (1, 5)}
    rej = {h: 0 for h in (1, 5)}
    for r in range(n_rep):
        f = rng.standard_normal((T, 1))                     # common factor
        eps = np.sqrt(rho_cs) * f + np.sqrt(1 - rho_cs) * rng.standard_normal((T, N))
        xb = rng.standard_normal((T, 2))                    # small model regressors
        xe = rng.standard_normal((T, k_extra))              # extra regressors, true coef 0
        Y = xb @ np.array([0.5, 0.3]) [:, None] + eps
        Xs = np.column_stack([np.ones(T), xb]); Xl = np.column_stack([Xs, xe])
        cw_h = np.zeros(T_eval)
        # refit every 20 origins to keep the MC affordable (forecasts are OLS)
        for i, t in enumerate(range(T_train0, T)):
            if i % 20 == 0:
                bs = np.linalg.lstsq(Xs[:t], Y[:t], rcond=None)[0]
                bl = np.linalg.lstsq(Xl[:t], Y[:t], rcond=None)[0]
            ps, pl = Xs[t] @ bs, Xl[t] @ bl
            cw_h[i] = np.mean((Y[t] - ps) ** 2 - (Y[t] - pl) ** 2 + (ps - pl) ** 2)
        for h in (1, 5):
            _, _, tstat, _ = _hac_mean_test(cw_h, h)
            out[h].append(tstat)
            rej[h] += tstat > 1.645
    rows = [{"h": h, "n_rep": n_rep, "rho_cs": rho_cs, "size_5pct_one_sided": rej[h] / n_rep,
             "mean_t": float(np.mean(out[h])), "sd_t": float(np.std(out[h], ddof=1))} for h in (1, 5)]
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "cw_size_mc.csv", index=False)
    return df


# ------------------------------------------------------------------ run-21
def _block_boot_ratio(La: np.ndarray, Lb: np.ndarray, rng, block=BLOCK, n_boot=N_BOOT):
    """Percentile CI for 100*(1 - mean(Lb)/mean(La)) with a circular block
    bootstrap over dates (La, Lb are date-level mean losses)."""
    T = len(La)
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :T] % T
    ra = La[idx].mean(axis=1); rb = Lb[idx].mean(axis=1)
    g = 100 * (1 - rb / ra)
    return float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))


def bootstrap_intervals() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for base, specs in (("A", SPECS_VS_HAR), ("A1", SPECS_VS_HARX)):
        for spec in specs:
            for h in HORIZONS:
                got = _load(spec, h)
                if got is None:
                    continue
                yb, y = _load(base, h); ys, _ = got
                cm = yb.notna() & ys.notna() & y.notna()
                La = ((y - yb) ** 2).where(cm).mean(axis=1).dropna().values
                Lb = ((y - ys) ** 2).where(cm).mean(axis=1).dropna().values
                lo, hi = _block_boot_ratio(La, Lb, rng)
                rows.append({"baseline": base, "spec": spec, "h": h, "T": len(La),
                             "gain_pct": 100 * (1 - Lb.mean() / La.mean()), "ci_lo": lo, "ci_hi": hi})
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "bootstrap_intervals.csv", index=False)
    return df


def main() -> None:
    pd.set_option("display.width", 200)
    print("=" * 70); print("   MODULE 17: EVALUATION CHECKS"); print("=" * 70)
    bundle = load_bundle()
    print("\n[run-18] calendar"); print(calendar_report(bundle).T.to_string())
    print("\n[run-20] Clark-West decomposition"); print(cw_decomposition().round(4).to_string(index=False))
    print("\n[run-20] Clark-West size (Monte Carlo)"); print(cw_size_mc().round(3).to_string(index=False))
    print("\n[run-21] bootstrap intervals"); print(bootstrap_intervals().round(2).to_string(index=False))


if __name__ == "__main__":
    main()
