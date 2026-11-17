"""
MODULE 30: Verification Checks for the Final Package (run-44; JRFM round-1
revision)
=========================================================================
Registered in docs/EXPERIMENTS.md (sixth block, commit 08383ba) before any
of these numbers existed. Nothing is refitted; saved forecasts only.

  (a) paired tests of the real-time combination of C against the same
      combination of the shuffled placebo, the noise placebo and C without
      the Hurst predictors;
  (b) timing census: the implemented weight recomputed with explicit
      trading-calendar maturity; first scored dates and sample sizes under
      MSE and QLIKE;
  (c) weight diagnostics, gains by period, and the run-41 sample census
      (per-date counts, the 24 stocks removed by the coverage filter).

    python -m modules.module30_verification --wrds <folder with SP500_WRDS_COMPLETE>
Outputs (results/intermediate/verification/): paired_combinations.csv,
timing_census.csv, weight_diagnostics.csv, combination_by_period.csv,
survivorship_census.csv, survivorship_excluded24.csv
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from modules.forecast_io import HORIZONS, load_bundle, pit_level_correction
from modules.module6_forecast_eval import diebold_mariano, qlike_loss
from modules.module19_matched_robustness import compare
from modules.module25_restricted_and_timing import COMB_WARMUP, REV, recursive_combination

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
INTERM = BASE / "results" / "intermediate"
FCST = INTERM / "forecasts"
OUT = INTERM / "verification"
WRDS_PKL = BASE / "bloomberg_pull" / "processed" / "wrds" / "survivorship_panel.pkl"
PERIODS = {"Jun 2013 - Dec 2019": ("2013-01-01", "2019-12-31"), "2020": ("2020-01-01", "2020-12-31"),
           "2021 - 2026": ("2021-01-01", "2026-12-31")}


def _rd(stem: Path):
    return (pd.read_csv(f"{stem}_yhat.csv", index_col=0, parse_dates=True),
            pd.read_csv(f"{stem}_y.csv", index_col=0, parse_dates=True))


# ------------------------------------------------------------------ weights
def lambda_implemented(yb, yc, y, h) -> pd.Series:
    """The weight inside recursive_combination (module 25), exposed."""
    lag = int(np.ceil(h / 5))
    cm = yb.notna() & yc.notna() & y.notna()
    eb, dc = (y - yb).where(cm), (yc - yb).where(cm)
    num, den = (eb * dc).sum(axis=1, min_count=1), (dc ** 2).sum(axis=1, min_count=1)
    used = num.notna().cumsum()
    lam = (num.fillna(0).cumsum() / den.fillna(0).cumsum()).shift(lag).clip(0, 1)
    lam[used.shift(lag) < COMB_WARMUP] = np.nan
    return lam


def lambda_calendar(yb, yc, y, h, pos: np.ndarray) -> tuple[pd.Series, int]:
    """Independent recomputation: origin s enters the weight at t only if its
    target window has ended, pos[s] + h <= pos[t] on the trading calendar.
    Returns the weight and the largest (pos[s] + h - pos[t]) over all errors
    used (must be <= 0)."""
    cm = yb.notna() & yc.notna() & y.notna()
    eb, dc = (y - yb).where(cm).values, (yc - yb).where(cm).values
    num, den = np.nansum(eb * dc, axis=1), np.nansum(dc ** 2, axis=1)
    has = cm.any(axis=1).values
    lam = np.full(len(pos), np.nan)
    worst = -10**9
    for t in range(len(pos)):
        S = np.flatnonzero(has & (pos + h <= pos[t]))
        if len(S) >= COMB_WARMUP:
            lam[t] = np.clip(num[S].sum() / den[S].sum(), 0, 1)
            worst = max(worst, int((pos[S] + h - pos[t]).max()))
    return pd.Series(lam, index=yb.index), worst


# ------------------------------------------------------------------ (a)
def paired() -> pd.DataFrame:
    rng = np.random.default_rng(8)
    rows = []
    for h in HORIZONS:
        yb, y = _rd(FCST / f"A1_h{h:02d}")
        comb = {k: recursive_combination(yb, p.reindex_like(yb), y, h) for k, p in
                {"C": _rd(FCST / f"C_h{h:02d}")[0], "C_perm": _rd(REV / f"C_perm_h{h:02d}")[0],
                 "C_noise": _rd(REV / f"C_noise_h{h:02d}")[0], "C_noH": _rd(REV / f"pk_C_noH_h{h:02d}")[0]}.items()}
        for other in ("C_perm", "C_noise", "C_noH"):
            both = comb["C"].notna() & comb[other].notna() & y.notna()
            r = compare(comb["C"].where(both), y, comb[other].where(both), y, h, rng=rng)
            rows.append({"h": h, "comparison": f"comb(C) vs comb({other})", "gain_pct": r["gain_pct"],
                         "dm_t": r["dm_t"], "dm_p": r["dm_p"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                         "T": r["T"], "n_stocks": r["n_stocks"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ (b), (c)
def timing_and_weights(pos_all: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(9)
    census, wdiag, period = [], [], []
    for h in HORIZONS:
        yb, y = _rd(FCST / f"A1_h{h:02d}")
        yc = _rd(FCST / f"C_h{h:02d}")[0].reindex_like(yb)
        pos = pos_all.get_indexer(yb.index)
        assert (pos >= 0).all()
        li = lambda_implemented(yb, yc, y, h)
        lc, worst = lambda_calendar(yb, yc, y, h, pos)
        both = li.notna() | lc.notna()
        diff = float(np.nanmax(np.abs(li[both].fillna(-9) - lc[both].fillna(-9))))
        rt = recursive_combination(yb, yc, y, h)
        first_eval = yb.dropna(how="all").index.min()
        mse_dates = (rt.notna() & y.notna()).any(axis=1)
        # QLIKE masks, as in run-34 (standalone) and run-43 (combination)
        cm = yb.notna() & yc.notna() & y.notna()
        qb0 = qlike_loss(pit_level_correction(yb.where(cm), y, h, kind="smear"), y)
        qc0 = qlike_loss(pit_level_correction(yc.where(cm), y, h, kind="smear"), y)
        q_std = (qb0.notna() & qc0.notna()).any(axis=1)
        cr = rt.notna() & yb.notna() & y.notna()
        qb1 = qlike_loss(pit_level_correction(yb.where(cr), y, h, kind="smear"), y)
        qc1 = qlike_loss(pit_level_correction(rt.where(cr), y, h, kind="smear"), y)
        q_comb = (qb1.notna() & qc1.notna()).any(axis=1)
        idx = list(yb.index)
        census.append({
            "h": h, "first_evaluation_origin": first_eval.date(),
            "last_evaluation_origin": yb.dropna(how="all").index.max().date(),
            "evaluation_origins": int(yb.notna().any(axis=1).sum()),
            "first_combination_origin": li.first_valid_index().date(),
            "combination_starts_after_n_origins": idx.index(li.first_valid_index()) - idx.index(first_eval),
            "combination_mse_dates": int(mse_dates.sum()),
            "first_qlike_origin_standalone": q_std[q_std].index.min().date(), "qlike_dates_standalone": int(q_std.sum()),
            "first_qlike_origin_combination": q_comb[q_comb].index.min().date(), "qlike_dates_combination": int(q_comb.sum()),
            "max_abs_diff_lambda_implemented_vs_calendar": diff,
            "max_target_end_minus_origin_days_in_weight": worst})
        w = li.dropna()
        wdiag.append({"h": h, "first_date": w.index.min().date(), "n": len(w), "min": w.min(),
                      "q05": w.quantile(.05), "q25": w.quantile(.25), "median": w.median(),
                      "q75": w.quantile(.75), "q95": w.quantile(.95), "max": w.max(),
                      "share_clipped_at_0": float((w <= 0).mean()), "share_clipped_at_1": float((w >= 1).mean()),
                      "mean_2020": w["2020"].mean(), "mean_other": w[w.index.year != 2020].mean()})
        for name, (a, b) in PERIODS.items():
            sl = slice(a, b)
            r = compare(rt.loc[sl], y.loc[sl], yb.where(rt.notna()).loc[sl], y.loc[sl], h, rng=rng)
            period.append({"h": h, "period": name, "gain_pct": r["gain_pct"], "dm_t": r["dm_t"], "T": r["T"]})
    return pd.DataFrame(census), pd.DataFrame(wdiag), pd.DataFrame(period)


def survivorship_census(wrds: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_pickle(WRDS_PKL)
    member, d = p["member"], p["feat"]["d_gph"]
    rows = []
    for h in HORIZONS:
        yc, y = _rd(REV / f"surv_C_h{h:02d}")
        yb, _ = _rd(REV / f"surv_A1_h{h:02d}")
        mw = member.reindex(yc.index).astype("boolean").fillna(False).astype(bool)[yc.columns]
        first, last = yc.dropna(how="all").index.min(), y.dropna(how="all").index.max()
        ev = slice(first, last)
        n_mem = mw.loc[ev].sum(axis=1)
        n_state = (mw & d.reindex(index=yc.index, columns=yc.columns).notna()).loc[ev].sum(axis=1)
        n_issued = (mw & yc.notna() & yb.notna()).loc[ev].sum(axis=1)
        n_score = (mw & yc.notna() & yb.notna() & y.notna()).loc[ev].sum(axis=1)
        contrib = (mw & yc.notna() & yb.notna() & y.notna()).any()
        in_win = mw.loc[ev].any()
        ever = mw.any()
        before = ~in_win & ever & mw.apply(lambda s: s[s].index.max() < first if s.any() else False)
        after = ~in_win & ever & mw.apply(lambda s: s[s].index.min() > last if s.any() else False)
        r = {"h": h, "first_origin": first.date(), "last_origin": last.date(), "eligible": mw.shape[1],
             "contribute": int(contrib.sum()), "members_only_before_window": int(before.sum()),
             "members_only_after_window": int(after.sum()),
             "members_never_on_grid": int((~ever).sum()),
             "members_in_window_without_scored_forecast": int((in_win & ~contrib).sum())}
        for k, s in (("members", n_mem), ("with_state", n_state), ("issued", n_issued), ("scorable", n_score)):
            r.update({f"{k}_min": int(s.min()), f"{k}_median": float(s.median()), f"{k}_max": int(s.max())})
        rows.append(r)
    census = pd.DataFrame(rows)
    excl = pd.DataFrame()
    if wrds is not None:
        D = wrds / "SP500_WRDS_COMPLETE"
        mem = pd.read_csv(D / "SP500_Membership_2001_2026.csv", parse_dates=["mbrstartdt", "mbrenddt"])
        names = pd.read_csv(D / "SP500_Historical_Names.csv", parse_dates=["namedt", "nameenddt"])
        kept_p = ({int(k[1:]) for k in member.columns if k.startswith("P") and k[1:].isdigit()}
                  | {int(v) for v in p["ours"].values()})
        drop = sorted(set(mem.permno.astype(int)) - kept_p)
        parts = []
        for f in sorted(D.glob("SP500_Daily_*.csv")):
            for ch in pd.read_csv(f, usecols=["permno", "dlycaldt", "dlyhigh", "dlylow"], chunksize=500_000):
                parts.append(ch[ch.permno.isin(drop)])
        c = pd.concat(parts); c["dlycaldt"] = pd.to_datetime(c["dlycaldt"])
        c["ok"] = (c.dlyhigh > 0) & (c.dlylow > 0) & (c.dlyhigh >= c.dlylow)
        cal = member.index
        out = []
        for pn in drop:
            sp = mem[mem.permno == pn]
            days = pd.DatetimeIndex(sorted(set().union(*[set(cal[(cal >= r.mbrstartdt) & (cal <= r.mbrenddt)]) for r in sp.itertuples()])))
            ok = c[(c.permno == pn) & c.ok].dlycaldt
            nm = names[names.permno == pn].sort_values("namedt").tail(1)
            out.append({"permno": pn, "ticker": nm.ticker.iloc[0] if len(nm) else "", "name": nm.comnam.iloc[0] if len(nm) and "comnam" in nm else "",
                        "member_from": sp.mbrstartdt.min().date(), "member_to": sp.mbrenddt.max().date(),
                        "member_trading_days": len(days), "valid_range_days_in_spells": int(ok.isin(days).sum()),
                        "coverage": int(ok.isin(days).sum()) / max(len(days), 1)})
        excl = pd.DataFrame(out)
    return census, excl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrds", type=Path, default=None)
    a = ap.parse_args()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    OUT.mkdir(parents=True, exist_ok=True)
    pa = paired(); pa.to_csv(OUT / "paired_combinations.csv", index=False); print(pa.round(4).to_string(index=False))
    b = load_bundle()
    tc, wd, pe = timing_and_weights(b.rv.index)
    for n, df in (("timing_census", tc), ("weight_diagnostics", wd), ("combination_by_period", pe)):
        df.to_csv(OUT / f"{n}.csv", index=False); print(n); print(df.round(4).to_string(index=False))
    sc, ex = survivorship_census(a.wrds)
    sc.to_csv(OUT / "survivorship_census.csv", index=False); print(sc.T.to_string())
    if len(ex):
        ex.to_csv(OUT / "survivorship_excluded24.csv", index=False); print(ex.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
