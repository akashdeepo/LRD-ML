"""
MODULE 24: Primary Family, Equivalence and Power, Clark-West Reconciliation,
Data Robustness (runs 34, 35, 36; JRFM round-1 revision)
=============================================================================
Registered in docs/EXPERIMENTS.md (sixth block, commit 4c4d1a4) before any
of these numbers existed.

  run-34  Primary family: model C vs HAR-X at h = 1, 5, 22. Panel DM (HLN)
          on squared error of log variance and on QLIKE (point-in-time
          smeared, as module 6), Holm across the three horizons for each
          loss. Equivalence by two one-sided tests at margins of 1% and 3%
          of HAR-X's MSE, read off 90% circular block-bootstrap intervals.
          Minimum detectable effect (5% two-sided, 80% power): 2.80 x the
          HAC standard error of the date-level mean loss differential, in
          percent of HAR-X's MSE.
  run-35  Why Clark-West rejects while realized loss does not fall:
          forecast-encompassing test of HAR-X by C (HLN 1998), the
          encompassing weight and the MSE of the combination it implies,
          dispersion of C's forecasts around HAR-X's, the stability of C's
          persistence coefficients across re-estimations, and in-sample
          (adjusted) R-squared gains against the out-of-sample gain.
  run-36  Primary family without TXN and AAPL; zero-range and missing days.

Outputs: results/intermediate/primary_family.csv, encompassing.csv,
         coef_stability.csv, insample_vs_oos.csv, data_robustness.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from modules.forecast_io import (
    HORIZONS, MODEL_FEATURES, aligned_xy, build_targets, load_bundle, n_train_rows,
    pit_level_correction, row_positions, stock_matrix,
)
from modules.module4_benchmarks import INIT_TRAIN_FRAC
from modules.module6_forecast_eval import diebold_mariano, qlike_loss, squared_loss
from modules.module12_incremental_tests import _hac_mean_test, holm

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
FCST = BASE / "results" / "intermediate" / "forecasts"
INTERM = BASE / "results" / "intermediate"

BLOCK, N_BOOT, SEED = 12, 5000, 0
MARGINS = (1.0, 3.0)            # equivalence margins, % of HAR-X's MSE
MDE_FACTOR = 1.96 + 0.8416      # 5% two-sided, 80% power
PERSISTENCE = [f for f in MODEL_FEATURES["C"] if f not in MODEL_FEATURES["A1"]]
REFIT_EVERY = 52                # origins between coefficient snapshots (run-35)


def _load(model: str, h: int):
    return (pd.read_csv(FCST / f"{model}_h{h:02d}_yhat.csv", index_col=0, parse_dates=True),
            pd.read_csv(FCST / f"{model}_h{h:02d}_y.csv", index_col=0, parse_dates=True))


def _boot_gain(La: np.ndarray, Lb: np.ndarray, rng, q=(5.0, 95.0)) -> tuple[float, float]:
    """Percentile interval for 100*(1 - mean(Lb)/mean(La)), circular block
    bootstrap over dates (module 17's scheme; 90% by default for TOST)."""
    T = len(La)
    nb = int(np.ceil(T / BLOCK))
    starts = rng.integers(0, T, size=(N_BOOT, nb))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(N_BOOT, -1)[:, :T] % T
    g = 100 * (1 - Lb[idx].mean(axis=1) / La[idx].mean(axis=1))
    return float(np.percentile(g, q[0])), float(np.percentile(g, q[1]))


def _pair(h: int, drop: tuple[str, ...] = ()):
    """HAR-X (b) and C (c) forecasts, target, on common cells."""
    yb, y = _load("A1", h)
    yc, _ = _load("C", h)
    cols = [c for c in y.columns if c not in drop]
    yb, yc, y = yb[cols], yc[cols], y[cols]
    cm = yb.notna() & yc.notna() & y.notna()
    return yb.where(cm), yc.where(cm), y.where(cm)


# ------------------------------------------------------------------ run-34
def primary_family(drop: tuple[str, ...] = ()) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for h in HORIZONS:
        yb, yc, y = _pair(h, drop)
        for loss_name in ("mse_log", "qlike"):
            if loss_name == "mse_log":
                Lb, Lc = squared_loss(yb, y), squared_loss(yc, y)
            else:
                Lb = qlike_loss(pit_level_correction(yb, y, h, kind="smear"), y)
                Lc = qlike_loss(pit_level_correction(yc, y, h, kind="smear"), y)
                cq = Lb.notna() & Lc.notna()
                Lb, Lc = Lb.where(cq), Lc.where(cq)
            _, t, p, T = diebold_mariano(Lb, Lc, h=h)
            db = Lb.mean(axis=1).dropna(); dc = Lc.mean(axis=1).dropna()
            db, dc = db.align(dc, join="inner")
            # QLIKE can be negative, so express the differential relative to
            # HAR-X's loss only for MSE; for QLIKE report the raw difference
            mb = float(db.mean())
            gain = 100 * (1 - dc.mean() / mb) if loss_name == "mse_log" else float(db.mean() - dc.mean())
            _, se, _, _ = _hac_mean_test((db - dc).values, h)
            r = {"loss": loss_name, "h": h, "T": T, "gain": gain, "dm_t": t, "dm_p": p}
            if loss_name == "mse_log":
                lo90, hi90 = _boot_gain(db.values, dc.values, rng)
                lo95, hi95 = _boot_gain(db.values, dc.values, rng, q=(2.5, 97.5))
                r.update({"ci90_lo": lo90, "ci90_hi": hi90, "ci95_lo": lo95, "ci95_hi": hi95,
                          "mde_pct": 100 * MDE_FACTOR * se / mb})
                for m in MARGINS:
                    r[f"equivalent_at_{m:g}pct"] = bool(lo90 > -m and hi90 < m)
            else:
                r["diff_se"] = se
            rows.append(r)
    df = pd.DataFrame(rows)
    df["dm_p_holm"] = np.nan
    for loss_name in ("mse_log", "qlike"):
        k = df["loss"] == loss_name
        df.loc[k, "dm_p_holm"] = holm(df.loc[k, "dm_p"].tolist())
    return df


# ------------------------------------------------------------------ run-35
def encompassing() -> pd.DataFrame:
    """HLN (1998): H0 HAR-X encompasses C, E[e_b (e_b - e_c)] = 0 against > 0.
    Weight lambda = sum e_b (e_b - e_c) / sum (e_b - e_c)^2 (pooled); the
    combination (1 - lambda) yhat_b + lambda yhat_c uses the full-sample
    weight and is therefore an optimistic in-sample bound."""
    from scipy import stats
    rows = []
    for h in HORIZONS:
        yb, yc, y = _pair(h)
        eb, ec = y - yb, y - yc
        f = (eb * (eb - ec)).mean(axis=1).dropna().values
        m, se, t, T = _hac_mean_test(f, h)
        num = float(np.nansum((eb * (eb - ec)).values))
        den = float(np.nansum(((eb - ec) ** 2).values))
        lam = num / den
        comb = (1 - lam) * yb + lam * yc
        mse_b = float(np.nanmean(squared_loss(yb, y).values))
        mse_comb = float(np.nanmean(squared_loss(comb, y).values))
        disp = float(np.nanmean(((yc - yb) ** 2).values))
        rows.append({"h": h, "T": T, "enc_t": t, "enc_p_one_sided": float(1 - stats.norm.cdf(t)),
                     "weight_on_C": lam, "comb_gain_vs_harx_pct": 100 * (1 - mse_comb / mse_b),
                     "dispersion_C_vs_harx": disp, "dispersion_over_target_var": disp / float(np.nanvar(y.values)),
                     "dispersion_over_mse_harx": disp / mse_b})
    return pd.DataFrame(rows)


def _ols_t(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    Xb = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    resid = y - Xb @ beta
    n, k = Xb.shape
    s2 = float(resid @ resid) / max(n - k, 1)
    cov = s2 * np.linalg.pinv(Xb.T @ Xb)
    tstat = beta / np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    adj = 1 - (1 - r2) * (n - 1) / max(n - k, 1)
    return beta, tstat, r2, adj


def coefficient_stability() -> tuple[pd.DataFrame, pd.DataFrame]:
    """C re-estimated at every REFIT_EVERY-th evaluation origin per stock, on
    the embargoed training rows; coefficients and OLS t-statistics of the
    persistence predictors. Also full-sample (all rows) R-squared and adjusted
    R-squared of C and HAR-X per stock."""
    bundle = load_bundle()
    targets = build_targets(bundle)
    sd = bundle.sample_dates
    init_date = sd[int(len(sd) * INIT_TRAIN_FRAC)]
    snaps, ins = [], []
    for h in HORIZONS:
        for tkr in bundle.panel.kept:
            Xc, y = aligned_xy(stock_matrix(bundle, tkr, "C", targets), h)
            if len(Xc) < 60:
                continue
            Xb = Xc[MODEL_FEATURES["A1"]]
            pos = row_positions(Xc.index, bundle.rv.index)
            t0 = max(int(np.searchsorted(Xc.index.values, np.datetime64(init_date))), 52)
            for t in range(t0, len(y), REFIT_EVERY):
                n = n_train_rows(pos, t, h)
                beta, tst, _, _ = _ols_t(Xc.values[:n], y.values[:n])
                for j, f in enumerate(Xc.columns):
                    if f in PERSISTENCE:
                        snaps.append({"h": h, "ticker": tkr, "origin": Xc.index[t], "feature": f,
                                      "coef": beta[j + 1], "t": tst[j + 1]})
            _, _, r2c, adjc = _ols_t(Xc.values, y.values)
            _, _, r2b, adjb = _ols_t(Xb.values, y.values)
            ins.append({"h": h, "ticker": tkr, "r2_C": r2c, "r2_harx": r2b, "adjr2_C": adjc, "adjr2_harx": adjb})
    snaps, ins = pd.DataFrame(snaps), pd.DataFrame(ins)
    rows = []
    for (h, f), g in snaps.groupby(["h", "feature"]):
        per = g.groupby("ticker")["coef"].agg(lambda c: bool((np.sign(c) != np.sign(c.iloc[0])).any()))
        rows.append({"h": h, "feature": f, "n_stocks": int(per.size),
                     "share_stocks_sign_change": float(per.mean()),
                     "median_abs_t": float(g["t"].abs().median()),
                     "share_abs_t_above_2": float((g["t"].abs() > 2).mean())})
    stab = pd.DataFrame(rows)
    summ = ins.groupby("h").agg(r2_gain=("r2_C", "mean"), r2_harx=("r2_harx", "mean"),
                                adjr2_C=("adjr2_C", "mean"), adjr2_harx=("adjr2_harx", "mean")).reset_index()
    summ["r2_gain"] = summ["r2_gain"] - summ["r2_harx"]
    summ["adjr2_gain"] = summ["adjr2_C"] - summ["adjr2_harx"]
    summ["share_stocks_adjr2_gain_pos"] = ins.assign(g=ins.adjr2_C > ins.adjr2_harx).groupby("h")["g"].mean().values
    oos_gain = []
    for h in HORIZONS:
        yb, yc, y = _pair(h)
        oos_gain.append(100 * (1 - np.nanmean(squared_loss(yc, y).values) / np.nanmean(squared_loss(yb, y).values)))
    summ["oos_gain_pct"] = oos_gain
    return stab, summ[["h", "r2_harx", "r2_gain", "adjr2_harx", "adjr2_gain", "share_stocks_adjr2_gain_pos", "oos_gain_pct"]]


# ------------------------------------------------------------------ run-36
def data_robustness() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = primary_family()
    drop = primary_family(drop=("TXN", "AAPL"))
    m = base[base.loss == "mse_log"][["h", "gain", "dm_t"]].merge(
        drop[drop.loss == "mse_log"][["h", "gain", "dm_t"]], on="h", suffixes=("_all", "_without_TXN_AAPL"))
    m["gain_change_pp"] = m["gain_without_TXN_AAPL"] - m["gain_all"]
    bundle = load_bundle()
    rv = bundle.rv
    rows = []
    for c in rv.columns:
        s = rv[c]
        first = s.first_valid_index()
        inside = s.loc[first:]
        rows.append({"ticker": c, "days_in_history": int(len(inside)),
                     "missing_days": int(inside.isna().sum()), "zero_range_days": int((inside == 0).sum())})
    counts = pd.DataFrame(rows)
    return m, counts


def main() -> None:
    pd.set_option("display.width", 220)
    print("=" * 70); print("   MODULE 24: PRIMARY FAMILY, EQUIVALENCE, CW RECONCILIATION"); print("=" * 70)
    pf = primary_family()
    pf.to_csv(INTERM / "primary_family.csv", index=False)
    print("\n[run-34] primary family"); print(pf.round(4).to_string(index=False))
    enc = encompassing()
    enc.to_csv(INTERM / "encompassing.csv", index=False)
    print("\n[run-35] encompassing and dispersion"); print(enc.round(4).to_string(index=False))
    stab, ins = coefficient_stability()
    stab.to_csv(INTERM / "coef_stability.csv", index=False)
    ins.to_csv(INTERM / "insample_vs_oos.csv", index=False)
    print("\n[run-35] coefficient stability"); print(stab.round(3).to_string(index=False))
    print("\n[run-35] in-sample vs out-of-sample"); print(ins.round(4).to_string(index=False))
    rob, counts = data_robustness()
    rob.to_csv(INTERM / "data_robustness.csv", index=False)
    counts.to_csv(INTERM / "missing_zero_days.csv", index=False)
    print("\n[run-36] without TXN and AAPL"); print(rob.round(4).to_string(index=False))
    print(f"\n[run-36] missing / zero-range days: total {counts.missing_days.sum()} missing, "
          f"{counts.zero_range_days.sum()} zero-range over {counts.days_in_history.sum()} stock-days; "
          f"max share missing {(counts.missing_days / counts.days_in_history).max():.4f}")


if __name__ == "__main__":
    main()
