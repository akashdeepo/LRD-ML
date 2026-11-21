"""
MODULE 32: Round-2 co-author audits (run-49; JRFM round-2 revision)
====================================================================
Descriptive checks on saved forecasts and saved simulation records, requested
in the co-author review of 20 Nov 2026. NOT registered in advance and no
outcome predicted: none refits a model or changes a reported number; each
documents a sample, an aggregation or an implementation detail.

  (a) The equal-weight combination of Table 13 (all evaluation cells) and the
      fixed-0.50 row of Table 14A (after the real-time weight's warm-up):
      samples, origin and cell counts, gains.
  (b) Cell-pooled versus date-balanced relative reductions, main panel and
      historical-membership panel (run-41), and how often a scored target
      window is shortened (fewer than h valid post-origin days), in particular
      at the end of a stock's price history.
  (c) The real-time combination's QLIKE evaluation sample (second warm-up).
  (d) Run-48: whether replications with a positive event select the injected
      episode.

    python -m modules.module32_round2_audits            (a, c, d; seconds to minutes)
    python -m modules.module32_round2_audits --hist     (b, needs the run-41 CRSP panel)
Outputs: results/intermediate/verification/round2b_*.csv
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from modules.forecast_io import HORIZONS, pit_level_correction
from modules.module19_matched_robustness import compare
from modules.module25_restricted_and_timing import REV, recursive_combination
from modules.module6_forecast_eval import qlike_loss, squared_loss

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
INTERM = BASE / "results" / "intermediate"
FCST = INTERM / "forecasts"
OUT = INTERM / "verification"
WRDS_PKL = BASE / "bloomberg_pull" / "processed" / "wrds" / "survivorship_panel.pkl"


def _rd(stem: Path):
    return (pd.read_csv(f"{stem}_yhat.csv", index_col=0, parse_dates=True),
            pd.read_csv(f"{stem}_y.csv", index_col=0, parse_dates=True))


def _describe(f, y, yb, h, label) -> tuple[dict, pd.DataFrame]:
    r = compare(f, y, yb, y, h)
    Ls, Lb = squared_loss(f, y), squared_loss(yb, y)
    cm = Ls.notna() & Lb.notna()
    Ls, Lb = Ls.where(cm), Lb.where(cm)
    dates = cm.any(axis=1)
    n = cm.sum(axis=1)[dates]
    return ({"h": h, "comparison": label, "gain_cell_pooled": r["gain_pct"],
             "gain_date_balanced": 100 * (1 - Ls.mean(axis=1).mean() / Lb.mean(axis=1).mean()),
             "dm_t": r["dm_t"], "dm_p": r["dm_p"], "origins": int(dates.sum()), "cells": int(cm.values.sum()),
             "first": dates[dates].index.min().date(), "last": dates[dates].index.max().date(),
             "N_t_min": int(n.min()), "N_t_max": int(n.max())}, cm)


# ------------------------------------------------------------------ (a), (b) main panel
def equal_weight() -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        yb, y = _rd(FCST / f"A1_h{h:02d}")
        yc = _rd(FCST / f"C_h{h:02d}")[0].reindex_like(yb)
        cm = yb.notna() & yc.notna() & y.notna()
        f50 = 0.5 * yb + 0.5 * yc
        pub = recursive_combination(yb, yc, y, h)
        post = pub.notna() & y.notna()
        for label, f, m in (("fixed 0.5, all common cells (Table 13)", f50, cm),
                            ("fixed 0.5, after weight warm-up (Table 14A)", f50, post),
                            ("real-time combination", pub, post),
                            ("model C alone vs HAR-X (primary)", yc, cm),
                            ("model C alone, after warm-up", yc, post)):
            rows.append(_describe(f.where(m), y, yb.where(m), h, label)[0])
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "round2b_equal_weight_and_aggregation.csv", index=False)
    return df


# ------------------------------------------------------------------ (b) historical panel
def historical() -> pd.DataFrame:
    p = pd.read_pickle(WRDS_PKL)
    member, rv = p["member"], p["rv"]
    full, valid = rv.index, rv.notna().values
    last_pos = {c: full.get_loc(rv[c].last_valid_index()) for c in rv.columns if rv[c].notna().any()}
    rows = []
    for h in HORIZONS:
        S = {m: _rd(REV / f"surv_{m}_h{h:02d}") for m in ("A1", "C")}
        mw = member.reindex(S["A1"][0].index).astype("boolean").fillna(False).astype(bool)
        S = {m: (yh.where(mw[yh.columns]), yy.where(mw[yy.columns])) for m, (yh, yy) in S.items()}
        (yb, y), yc = S["A1"], S["C"][0].reindex_like(S["A1"][0])
        rt = recursive_combination(yb, yc, y, h)
        for label, f, m in (("model C vs HAR-X", yc, yb.notna() & yc.notna() & y.notna()),
                            ("real-time combination vs HAR-X", rt, rt.notna() & y.notna())):
            row, cm = _describe(f.where(m), y, yb.where(m), h, label)
            pos = full.get_indexer(cm.index)
            cols = [rv.columns.get_loc(c) for c in cm.columns]
            short = at_end = 0
            for i, j in zip(*np.nonzero(cm.values)):
                k, c = pos[i], cols[j]
                if valid[k + 1:k + 1 + h, c].sum() < h:
                    short += 1
                    at_end += last_pos.get(rv.columns[c], 10 ** 9) < k + h
            row.update(short_targets=short, short_pct=100 * short / row["cells"], short_at_history_end=at_end)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "round2b_historical_aggregation.csv", index=False)
    return df


# ------------------------------------------------------------------ (c)
def qlike_sample() -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        yb, y = _rd(FCST / f"A1_h{h:02d}")
        yc = _rd(FCST / f"C_h{h:02d}")[0].reindex_like(yb)
        rt = recursive_combination(yb, yc, y, h)
        cm = rt.notna() & yb.notna() & y.notna()
        qb = qlike_loss(pit_level_correction(yb.where(cm), y, h, kind="smear"), y)
        qc = qlike_loss(pit_level_correction(rt.where(cm), y, h, kind="smear"), y)
        for label, m in (("squared-error sample", cm), ("QLIKE sample", qb.notna() & qc.notna())):
            d = m.any(axis=1)
            rows.append({"h": h, "sample": label, "origins": int(d.sum()), "cells": int(m.values.sum()),
                         "first": d[d].index.min().date(), "last": d[d].index.max().date()})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "round2b_qlike_sample.csv", index=False)
    return df


# ------------------------------------------------------------------ (d)
def window_events(span: int = 300) -> pd.DataFrame:
    d = pd.read_csv(INTERM / "round2" / "run48_window_calibration_reps.csv")
    rows = []
    for s in ("arma+episode", "arfima+episode"):
        g = d[d.scenario == s]
        ep = g.apply(lambda r: any(0 <= int(e) - r.tau <= span for e in str(r.events).split(";")), axis=1)
        rows.append({"scenario": s, "reps": len(g), "reps_selecting_episode": int(ep.sum()),
                     "positive_reps": int(g.any_positive.sum()),
                     "positive_reps_selecting_episode": int((g.any_positive & ep).sum()),
                     "max_positive_events_per_rep": int(g.positive_events.max())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "round2b_window_events.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    if a.hist:
        print(historical().round(3).to_string(index=False))
    else:
        for f in (equal_weight, qlike_sample, window_events):
            print(f().round(3).to_string(index=False))
