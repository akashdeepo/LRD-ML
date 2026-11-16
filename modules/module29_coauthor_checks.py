"""
MODULE 29: Co-author Checks Before Resubmission (run-43; JRFM round-1
revision)
====================================================================
Registered in docs/EXPERIMENTS.md (sixth block, commit 975971c) before any
of these numbers existed.

  (a) Matched proxy row: C without the Hurst predictors (C_noH, as run-39
      used under every alternative proxy) fitted on Parkinson variance, and
      evaluated against HAR-X with the real-time combination, on all 115
      stocks and on the 114 without AAPL (the Garman-Klass and
      Rogers-Satchell sample).
  (b) The real-time combination scored by QLIKE (point-in-time level
      correction of run-34), for C and the noise and shuffled placebos.
  (c) Survivorship panel counts: eligible stocks, contributors, and why the
      others contribute no evaluated forecast.

    python -m modules.module29_coauthor_checks --fit      (C_noH on Parkinson)
    python -m modules.module29_coauthor_checks --eval
Outputs: results/intermediate/matched_proxy.csv, combination_qlike.csv,
         survivorship_counts.csv
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from modules.forecast_io import HORIZONS, build_targets, load_bundle, pit_level_correction
from modules.module6_forecast_eval import diebold_mariano, qlike_loss
from modules.module12_incremental_tests import _hac_mean_test
from modules.module19_matched_robustness import compare
from modules.module25_restricted_and_timing import REV, fit_panel, recursive_combination
import modules.module26_proxies  # noqa: F401  (registers MODEL_FEATURES["C_noH"])

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
INTERM = BASE / "results" / "intermediate"
FCST = INTERM / "forecasts"
WRDS = BASE / "bloomberg_pull" / "processed" / "wrds" / "survivorship_panel.pkl"


def _rd(path_stem: Path):
    return (pd.read_csv(f"{path_stem}_yhat.csv", index_col=0, parse_dates=True),
            pd.read_csv(f"{path_stem}_y.csv", index_col=0, parse_dates=True))


def fit() -> None:
    b = load_bundle(); targets = build_targets(b)
    for h in HORIZONS:
        fit_panel(b, targets, "pk_C_noH", "C_noH", h)
        print(f"  Parkinson C_noH h={h} done", flush=True)


# ------------------------------------------------------------------ (a)
def matched_proxy() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for h in HORIZONS:
        A1 = _rd(FCST / f"A1_h{h:02d}")
        C = _rd(FCST / f"C_h{h:02d}")
        Cn = _rd(REV / f"pk_C_noH_h{h:02d}")
        for sample, drop in (("115 stocks", ()), ("114 stocks (no AAPL)", ("AAPL",))):
            cols = [c for c in A1[0].columns if c not in drop]
            yb, y = A1[0][cols], A1[1][cols]
            for spec, (yc, yy) in (("C", C), ("C_noH", Cn)):
                yc, yy = yc[cols].reindex_like(yb), yy[cols]
                solo = compare(yc, yy, yb, y, h)
                rt = recursive_combination(yb, yc, y, h)
                comb = compare(rt, y, yb.where(rt.notna()), y, h, rng=rng)
                rows.append({"sample": sample, "spec": spec, "h": h, "n_stocks": solo["n_stocks"],
                             "solo_gain_pct": solo["gain_pct"], "solo_t": solo["dm_t"],
                             "comb_rt_pct": comb["gain_pct"], "comb_rt_t": comb["dm_t"],
                             "comb_ci_lo": comb["ci_lo"], "comb_ci_hi": comb["ci_hi"]})
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "matched_proxy.csv", index=False)
    return df


# ------------------------------------------------------------------ (b)
def combination_qlike() -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        yb, y = _rd(FCST / f"A1_h{h:02d}")
        partners = {"C": _rd(FCST / f"C_h{h:02d}")[0], "C_noise": _rd(REV / f"C_noise_h{h:02d}")[0],
                    "C_perm": _rd(REV / f"C_perm_h{h:02d}")[0]}
        for name, yp in partners.items():
            rt = recursive_combination(yb, yp.reindex_like(yb), y, h)
            cm = rt.notna() & yb.notna() & y.notna()
            qb = qlike_loss(pit_level_correction(yb.where(cm), y, h, kind="smear"), y)
            qc = qlike_loss(pit_level_correction(rt.where(cm), y, h, kind="smear"), y)
            cq = qb.notna() & qc.notna()
            qb, qc = qb.where(cq), qc.where(cq)
            _, t, p, T = diebold_mariano(qb, qc, h=h)
            db, dc = qb.mean(axis=1).dropna().align(qc.mean(axis=1).dropna(), join="inner")
            _, se, _, _ = _hac_mean_test((db - dc).values, h)
            rows.append({"partner": name, "h": h, "T": T, "qlike_harx": float(db.mean()),
                         "qlike_comb": float(dc.mean()), "diff": float(db.mean() - dc.mean()),
                         "diff_se": se, "dm_t": t, "dm_p": p})
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "combination_qlike.csv", index=False)
    return df


# ------------------------------------------------------------------ (c)
def survivorship_counts() -> pd.DataFrame:
    p = pd.read_pickle(WRDS)
    member, rv = p["member"], p["rv"]
    rows = []
    for h in HORIZONS:
        yc, y = _rd(REV / f"surv_C_h{h:02d}")
        yb, _ = _rd(REV / f"surv_A1_h{h:02d}")
        mw = member.reindex(yc.index).astype("boolean").fillna(False).astype(bool)[yc.columns]
        contrib = (yc.notna() & yb.notna() & y.notna() & mw).any()
        first, last = yc.dropna(how="all").index.min(), y.dropna(how="all").index.max()
        in_eval = mw.loc[first:last].any()
        short = in_eval & ~contrib
        valid_before_exit = pd.Series({c: int((rv[c].loc[:mw[c][mw[c]].loc[first:last].index.max()] > 0).sum())
                                       for c in short[short].index}, dtype=float)
        rows.append({"h": h, "eligible": rv.shape[1], "first_origin": first.date(), "last_origin": last.date(),
                     "contribute": int(contrib.sum()), "members_only_before_evaluation": int((~in_eval).sum()),
                     "members_in_evaluation_without_forecast": int(short.sum()),
                     "median_valid_days_of_those": float(valid_before_exit.median()) if len(valid_before_exit) else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "survivorship_counts.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    if a.fit:
        fit()
    if a.eval:
        for f in (matched_proxy, combination_qlike, survivorship_counts):
            print(f.__name__); print(f().round(4).to_string(index=False))


if __name__ == "__main__":
    main()
