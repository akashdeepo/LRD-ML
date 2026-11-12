"""
MODULE 27: Window-Composition Placebo and Parameter Uncertainty (run-40;
JRFM round-1 revision)
=====================================================================
Registered in docs/EXPERIMENTS.md (sixth block, commit 4c4d1a4) before any
of these numbers existed.

  (a) Event dates chosen by a rule on the VIX alone, fixed in the ledger:
      the first close above 40 after at least 250 trading days below 30.
  (b) The exit step change of the cross-sectional mean of rolling d-hat
      (W = 750) at event + W, averaged over the rule's events, against 1000
      placebo draws of the same number of dates drawn uniformly from the
      eligible range: permutation p-value (one-sided, a fall).
  (c) The short- and long-memory nulls of run-24 re-simulated with each
      path's parameters drawn from the per-stock calm-period estimates (a
      bootstrap over stocks), so that the bands carry estimation error and
      cross-stock heterogeneity, not the median alone; the observed rise is
      compared with the 5-95% band of simulated rises.
  (d) The same exit test for rolling statistics of daily log variance over
      the same window (mean, 90th percentile, variance) and for one not
      driven by the episode's size (rolling median of daily returns).

Output: results/intermediate/window_placebo.csv, window_placebo_events.csv,
        window_param_uncertainty.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from modules.forecast_io import load_bundle
from modules.module2_lrd_estimation import local_whittle
from modules.module14_window_inclusion import CALM, episode_profile, pick_events, step_change, HALF
from modules.module20_window_checks import rolling_gph_path, simulate_arfima, simulate_arma

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
INTERM = BASE / "results" / "intermediate"
W, N_PLACEBO, N_REPS, SEED = 750, 1000, 200, 0
VIX_HIGH, VIX_CALM, CALM_DAYS, MIN_OBS = 40.0, 30.0, 250, 250


# ------------------------------------------------------------------ (a)
def rule_events(vix: pd.Series, runup: int = 0) -> list[pd.Timestamp]:
    """runup = 0: the rule as registered; runup = 20: the amended rule, whose
    250 calm days end 20 trading days before t (ledger amendment)."""
    v = vix.dropna()
    # every one of the CALM_DAYS closes before t below 30; then only the first close above 40 qualifies
    quiet = (v < VIX_CALM).rolling(CALM_DAYS).sum().shift(1 + runup) == CALM_DAYS
    hits = v[(v > VIX_HIGH) & quiet]
    # only the first close above 40 of an episode: drop hits within CALM_DAYS of an earlier one
    first, last = [], None
    for t in hits.index:
        if last is None or v.index.get_loc(t) - v.index.get_loc(last) > CALM_DAYS:
            first.append(t)
        last = t
    return first


# ------------------------------------------------------------------ (b), (d)
def _exit_test(vals: np.ndarray, ends: np.ndarray, event_pos: list[int], rng) -> dict:
    """Mean exit step change at event + W and its permutation p-value."""
    obs = [step_change(vals, ends, p + W) for p in event_pos]
    obs = float(np.nanmean(obs))
    lo, hi = int(ends.min()) + HALF - W, int(ends.max()) - HALF - W
    k = len(event_pos)
    draws = np.array([np.nanmean([step_change(vals, ends, int(x) + W)
                                  for x in rng.integers(lo, hi, size=k)]) for _ in range(N_PLACEBO)])
    return {"exit_change": obs, "p_one_sided": float((draws <= obs).mean()),
            "placebo_mean": float(np.nanmean(draws)), "placebo_p05": float(np.nanpercentile(draws, 5))}


def rolling_stats(b) -> dict[str, pd.Series]:
    """Cross-sectional means of rolling statistics over W trading days, at the
    weekly sample dates."""
    lr = b.log_rv
    roll = lr.rolling(W, min_periods=MIN_OBS)
    out = {"rolling d-hat (state)": b.cs["cs_mean_d"],
           "rolling mean of log variance": roll.mean().reindex(b.sample_dates).mean(axis=1),
           "rolling 90th percentile of log variance": roll.quantile(0.9).reindex(b.sample_dates).mean(axis=1),
           "rolling variance of log variance": roll.var().reindex(b.sample_dates).mean(axis=1),
           "rolling median of daily returns": b.returns_raw.rolling(W, min_periods=MIN_OBS).median()
                                               .reindex(b.sample_dates).mean(axis=1)}
    return out


def placebo(b, runup: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    full = b.rv.index
    vix = b.panel.market["VIX"].reindex(full).ffill()
    events = rule_events(vix, runup)
    ev = pd.DataFrame({"event_date": events, "vix": [vix[e] for e in events],
                       "exit_date": [full[min(full.get_loc(e) + W, len(full) - 1)] for e in events]})
    pos = [full.get_loc(e) for e in events if full.get_loc(e) + W + HALF < len(full)]
    rows = []
    for name, s in rolling_stats(b).items():
        s = s.dropna()
        ends = full.get_indexer(s.index)
        # standardise so that changes are comparable across statistics
        vals = ((s - s.mean()) / s.std()).values
        rows.append({"rule": "registered" if runup == 0 else "amended (20-day run-up)", "statistic": name,
                     "events_used": len(pos), **_exit_test(vals, ends, pos, rng)})
    return pd.DataFrame(rows), ev


# ------------------------------------------------------------------ (c)
def per_stock_params(log_rv: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arma, d, sd = [], [], []
    for c in log_rv.columns:
        x = log_rv[c].loc[CALM[0]:CALM[1]].dropna().values
        if len(x) < 1000:
            continue
        r = ARIMA(x, order=(1, 0, 1)).fit()
        arma.append((r.arparams[0], r.maparams[0], np.sqrt(r.params[-1])))
        d.append(local_whittle(x)[0]); sd.append(x.std())
    return np.array(arma), np.array(d), np.array(sd)


def param_uncertainty(b) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    log_rv = b.log_rv
    events = pick_events(log_rv)
    covid = [e for e in events if e.year == 2020][0]
    profile = episode_profile(log_rv, covid)
    arma, d_s, sd_s = per_stock_params(log_rv)
    unit_sd = {}
    pre = W + 120; T = pre + W + 300
    rows = []
    for kind in ("short_memory_arma", "long_memory_arfima"):
        rises, drops = [], []
        for _ in range(N_REPS):
            i = rng.integers(len(arma))
            if kind == "short_memory_arma":
                x = simulate_arma(*arma[i], T, rng)
            else:
                dd = float(np.clip(d_s[i], 0.0, 0.49))
                key = round(dd, 3)
                if key not in unit_sd:
                    unit_sd[key] = simulate_arfima(dd, 1.0, 20000, np.random.default_rng(1)).std()
                x = simulate_arfima(dd, sd_s[i] / unit_sd[key], T, rng)
            x = x.copy(); x[pre:pre + len(profile)] += profile
            rel, p = rolling_gph_path(x, W, pre)
            base = p[(rel >= -120) & (rel < 0)].mean(); plat = p[(rel >= 60) & (rel < W - 60)].mean()
            rises.append(plat - base)
            steps = np.array([step_change(p, rel, int(e)) for e in rel])
            drops.append(int(rel[np.nanargmin(steps)]))
        rises, drops = np.array(rises), np.array(drops)
        rows.append({"null": kind, "reps": N_REPS, "rise_mean": rises.mean(),
                     "rise_p05": np.percentile(rises, 5), "rise_p95": np.percentile(rises, 95),
                     "share_largest_drop_within_20d_of_exit": float((np.abs(drops - W) <= 20).mean())})
    obs = pd.read_csv(INTERM / "window_inclusion_sim.csv").set_index("W").loc[W, "obs_rise"]
    df = pd.DataFrame(rows)
    df["observed_rise"] = float(obs)
    df["observed_inside_band"] = (df.observed_rise >= df.rise_p05) & (df.observed_rise <= df.rise_p95)
    return df


def main() -> None:
    pd.set_option("display.width", 220)
    b = load_bundle()
    res = [placebo(b, r) for r in (0, 20)]
    pl = pd.concat([r[0] for r in res], ignore_index=True)
    ev = pd.concat([r[1].assign(rule="registered" if k == 0 else "amended") for k, r in zip((0, 20), res)], ignore_index=True)
    pl.to_csv(INTERM / "window_placebo.csv", index=False); ev.to_csv(INTERM / "window_placebo_events.csv", index=False)
    print("[run-40a] rule-selected events"); print(ev.to_string(index=False))
    print("\n[run-40b,d] exit test (standardised units)"); print(pl.round(4).to_string(index=False))
    pu = param_uncertainty(b)
    pu.to_csv(INTERM / "window_param_uncertainty.csv", index=False)
    print("\n[run-40c] parameter uncertainty"); print(pu.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
