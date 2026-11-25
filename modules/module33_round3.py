"""
MODULE 33: Round-3 checks (runs 50 and 51; JRFM round-3 reply)
==============================================================
Registered in docs/EXPERIMENTS.md before any result.

  run-50  The run-48 window-diagnostic calibration for the two scenarios with
          the 2020 episode injected, rerun with the same seeds, now recording
          the flag of every selected event, so that a positive replication can
          be checked to flag the injected episode itself. Needs only derived
          inputs (run48_calm_params.csv and the episode profile, exported with
          --export), so it can run on a cloud machine without the price panel.
  run-51  The volatility-managed portfolio exercise (Appendix E) with the
          real-time combination's weekly forecast, on the combination's own
          sample (after its warm-up). Table A4 is not changed.

    python -m modules.module33_round3 --export          (local: write the episode profile)
    python -m modules.module33_round3 --run 50 [--reps 300]
    python -m modules.module33_round3 --run 51          (local)
Outputs: results/intermediate/round3/
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
INTERM = BASE / "results" / "intermediate"
R2 = INTERM / "round2"
OUT = INTERM / "round3"
JOBS = int(os.environ.get("LRDML_JOBS", 12))
T_DAYS = 6136
SCEN = ["arma+episode", "arfima+episode"]


def export_profile() -> None:
    from modules.forecast_io import load_bundle
    from modules.module14_window_inclusion import episode_profile, pick_events
    b = load_bundle()
    assert len(b.rv.index) == T_DAYS
    covid = [e for e in pick_events(b.log_rv) if e.year == 2020][0]
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "run50_episode_profile.npy", episode_profile(b.log_rv, covid))
    print("profile saved", covid.date())


# ------------------------------------------------------------------ run-50
def run50(reps: int) -> pd.DataFrame:
    from joblib import Parallel, delayed
    from modules.module20_window_checks import simulate_arfima
    from modules.module31_round2 import _sim_one
    P = pd.read_csv(R2 / "run48_calm_params.csv")
    arma, d_s, sd_s = P[["ar", "ma", "s"]].values, P.d.values, P.sd.values
    profile = np.load(OUT / "run50_episode_profile.npy")
    sd_tab = {k: simulate_arfima(k, 1.0, 20000, np.random.default_rng(1)).std()
              for k in set(np.round(np.clip(d_s, 0, 0.49), 3).tolist()) | {0.30, 0.45}}
    fp = OUT / "run50_flags_reps.csv"
    for sc in SCEN:
        for lo in range(0, reps, 60):
            done = set(map(tuple, pd.read_csv(fp)[["scenario", "rep"]].values)) if fp.exists() else set()
            todo = [r for r in range(lo, min(lo + 60, reps)) if (sc, r) not in done]
            if not todo:
                continue
            t0 = time.time()
            res = Parallel(n_jobs=JOBS)(delayed(_sim_one)(sc, r, T_DAYS, arma, d_s, sd_s, profile, sd_tab)
                                        for r in todo)
            pd.DataFrame(res).to_csv(fp, mode="a", header=not fp.exists(), index=False)
            print(f"  run-50 {sc} reps {todo[0]}-{todo[-1]}: {time.time() - t0:.0f}s", flush=True)
    return summarize50()


def summarize50() -> pd.DataFrame:
    d = pd.read_csv(OUT / "run50_flags_reps.csv")
    ref = pd.read_csv(R2 / "run48_window_calibration_reps.csv")
    rows = []
    for sc in SCEN:
        g = d[d.scenario == sc].sort_values("rep")
        r = ref[ref.scenario == sc].set_index("rep").loc[g.rep]
        pos_ep = pos_other = 0
        for _, x in g.iterrows():
            ev = [int(e) for e in str(x.events).split(";")]
            fl = str(x.flags).split(";")
            for e, f in zip(ev, fl):
                if f == "1":
                    if x.tau <= e < x.tau + x.episode_len:
                        pos_ep += 1
                    else:
                        pos_other += 1
        rows.append({"scenario": sc, "reps": len(g), "positive_reps": int(g.any_positive.sum()),
                     "positive_events_episode": pos_ep, "positive_events_other": pos_other,
                     "matches_run48": bool((g.positive_events.values == r.positive_events.values).all()
                                           and (g.events.values == r.events.values).all())})
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "run50_flags_summary.csv", index=False)
    return s


# ------------------------------------------------------------------ run-51
def run51() -> pd.DataFrame:
    from scipy import stats as sstats
    from modules.forecast_io import load_bundle
    from modules import module11_economic as m11
    from modules.module12_incremental_tests import _circular_block_bootstrap_p, _sharpe_diff_hac
    from modules.module25_restricted_and_timing import recursive_combination
    F = INTERM / "forecasts"
    rd = lambda s: pd.read_csv(F / f"{s}_yhat.csv", index_col=0, parse_dates=True)
    yb = rd("A1_h05")
    y = pd.read_csv(F / "A1_h05_y.csv", index_col=0, parse_dates=True)
    yc = rd("C_h05").reindex_like(yb)
    comb = recursive_combination(yb, yc, y, 5)
    b = load_bundle()
    sd = b.sample_dates
    rows, tests = [], []
    for name, lag in (("same-close execution", 0), ("next-day execution", 1)):
        R = m11.five_day_simple_returns(b, sd, lag=lag, total=True)
        rf = m11.weekly_riskfree(b, sd)
        X = R.sub(rf, axis=0)
        s2 = {k: np.exp(v.reindex(index=sd, columns=X.columns)) for k, v in
              (("HAR-X-managed", yb), ("C-managed", yc), ("Combination-managed", comb))}
        cell = X.notna()
        for v in s2.values():
            cell &= v.notna()
        X = X.where(cell)
        out = {m11.UNMANAGED: X.mean(axis=1)}
        for k, v in s2.items():
            c = m11.realtime_constant(X, X / v)
            w = (c / v).where(cell)
            out[k] = (w * X).sum(axis=1, min_count=1) / w.notna().sum(axis=1)
        Pf = pd.DataFrame(out).dropna()
        for k in Pf.columns:
            rows.append({"variant": name, "portfolio": k, "weeks": len(Pf), "first": Pf.index.min().date(),
                         **m11.stats(Pf[k], rf)})
        for a, c2 in (("Combination-managed", "HAR-X-managed"), ("Combination-managed", m11.UNMANAGED),
                      ("C-managed", "HAR-X-managed")):
            r1, r2 = Pf[a].values, Pf[c2].values
            dlt, _, t = _sharpe_diff_hac(r1, r2, 6)
            tests.append({"variant": name, "pair": f"{a} vs {c2}", "delta_sharpe_ann": dlt, "hac_t": t,
                          "hac_p": float(2 * (1 - sstats.norm.cdf(abs(t)))),
                          "boot_p": _circular_block_bootstrap_p(r1, r2, 6, block=5, B=2000)})
    OUT.mkdir(parents=True, exist_ok=True)
    df, tdf = pd.DataFrame(rows), pd.DataFrame(tests)
    df.to_csv(OUT / "run51_portfolio_combination.csv", index=False)
    tdf.to_csv(OUT / "run51_portfolio_combination_tests.csv", index=False)
    print(df.round(3).to_string(index=False)); print(tdf.round(3).to_string(index=False))
    return tdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--run", type=int)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.export:
        export_profile()
    elif a.run == 50:
        print((summarize50() if a.summarize else run50(a.reps)).to_string(index=False))
    elif a.run == 51:
        run51()
