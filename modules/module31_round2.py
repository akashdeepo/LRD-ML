"""
MODULE 31: Round-2 checks requested by Reviewer 1 (ledger runs 45-48)
=====================================================================
Registered in docs/EXPERIMENTS.md (commit 00a7539) before any result.

  run-45  combination robustness from saved forecasts: weight estimators
          (fixed, expanding, rolling), generic shrinkage (HAR-X toward lagged
          HAR-X, toward HAR), crisis exclusions, lag sensitivity, Holm over
          horizons, weight path.
  run-46  results by point-in-time size quintile on the run-41 CRSP panel.
  run-47  placebo distribution: persistence predictors shifted circularly in
          time (per-stock offsets, 200 draws; common offset, 100 draws).
  run-48  calibration of the window-length diagnostic by simulation.

Usage (from the repository root):
  python -m modules.module31_round2 --run 45
  python -m modules.module31_round2 --run 46 --wrds <folder holding SP500_WRDS_COMPLETE>
  python -m modules.module31_round2 --run 47 --variant a --draws 1-40    (resumable)
  python -m modules.module31_round2 --run 47 --summarize
  python -m modules.module31_round2 --run 48 --reps 500
Outputs: results/intermediate/round2/
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from modules.forecast_io import HORIZONS, build_targets, load_bundle
from modules.module4_benchmarks import INIT_TRAIN_FRAC, expanding_forecast
from modules.module12_incremental_tests import holm
from modules.module19_matched_robustness import compare
from modules.module25_restricted_and_timing import (COMB_WARMUP, PERSISTENCE, REV, _stock_xy,
                                                    recursive_combination)

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
INTERM = BASE / "results" / "intermediate"
FCST = INTERM / "forecasts"
OUT = INTERM / "round2"
WRDS_PKL = BASE / "bloomberg_pull" / "processed" / "wrds" / "survivorship_panel.pkl"
JOBS = 12


def _rd(stem: Path):
    return (pd.read_csv(f"{stem}_yhat.csv", index_col=0, parse_dates=True),
            pd.read_csv(f"{stem}_y.csv", index_col=0, parse_dates=True))


# ------------------------------------------------------------------ weights
def rt_weight(yb, yp, y, h, window: int | None = None) -> pd.Series:
    """Real-time pooled least-squares weight on the partner `yp` (Eq. 6).
    window=None: expanding (as published); window=K: the last K origin rows
    whose targets have matured. Same maturity lag and warm-up as module 25."""
    lag = int(np.ceil(h / 5))
    cm = yb.notna() & yp.notna() & y.notna()
    eb, dc = (y - yb).where(cm), (yp - yb).where(cm)
    num = (eb * dc).sum(axis=1, min_count=1)
    den = (dc ** 2).sum(axis=1, min_count=1)
    used = num.notna().cumsum()
    if window is None:
        N, D = num.fillna(0).cumsum(), den.fillna(0).cumsum()
    else:
        N, D = num.fillna(0).rolling(window, min_periods=1).sum(), den.fillna(0).rolling(window, min_periods=1).sum()
    lam = (N / D).shift(lag).clip(0, 1)
    lam[used.shift(lag) < COMB_WARMUP] = np.nan
    return lam


def combine(yb, yp, lam) -> pd.DataFrame:
    return yb.add((yp - yb).mul(lam, axis=0))


def _row(label, h, r, **extra) -> dict:
    return {"h": h, "variant": label, "gain_pct": r["gain_pct"], "dm_t": r["dm_t"], "dm_p": r["dm_p"],
            "T": r["T"], "n_stocks": r["n_stocks"], **extra}


# ------------------------------------------------------------------ run-45
def run45() -> pd.DataFrame:
    b = load_bundle()
    vix_all = b.panel.market["VIX"]
    rows, paths = [], {}
    for h in HORIZONS:
        yb, y = _rd(FCST / f"A1_h{h:02d}")
        yc = _rd(FCST / f"C_h{h:02d}")[0].reindex_like(yb)
        ya = _rd(FCST / f"A_h{h:02d}")[0].reindex_like(yb)
        pub = recursive_combination(yb, yc, y, h)
        lam = rt_weight(yb, yc, y, h)
        assert np.nanmax(np.abs((combine(yb, yc, lam) - pub).values)) < 1e-12, "rt_weight != published"
        paths[h] = lam
        mask = pub.notna() & y.notna()
        base = yb.where(mask)
        # (a) weight estimators
        variants = {f"fixed {w:.2f}": yb + w * (yc - yb) for w in (0.25, 0.5, 0.75)}
        variants["expanding real-time (published)"] = pub
        for K in (52, 156):
            variants[f"rolling real-time, {K} origins"] = combine(yb, yc, rt_weight(yb, yc, y, h, K))
        for name, f in variants.items():
            m = mask & f.notna()
            rows.append(_row(name, h, compare(f.where(m), y, yb.where(m), y, h), block="a weight estimator"))
        # (b) generic shrinkage with the same expanding weight
        for name, yp in {"HAR-X toward lagged HAR-X": yb.shift(1), "HAR-X toward HAR": ya}.items():
            f = combine(yb, yp, rt_weight(yb, yp, y, h))
            m = f.notna() & y.notna()
            rows.append(_row(name, h, compare(f.where(m), y, yb.where(m), y, h), block="b generic shrinkage"))
        # (c) crisis exclusions
        dates = mask.any(axis=1)
        vix = vix_all.reindex(yb.index)
        top = vix >= vix[dates].quantile(0.9)
        y2020 = pd.Series(yb.index.year == 2020, index=yb.index)
        for name, drop in {"all origins": pd.Series(False, index=yb.index), "excluding 2020": y2020,
                           "excluding top-decile VIX origins": top,
                           "excluding 2020 and top-decile VIX": y2020 | top}.items():
            m = mask.copy(); m.loc[drop.values] = False
            rows.append(_row(name, h, compare(pub.where(m), y, yb.where(m), y, h), block="c crisis exclusion",
                             dates=int(m.any(axis=1).sum())))
        # (d) lag sensitivity
        ybl, yl = _rd(REV / f"A1_lagidx_h{h:02d}")
        ycl = _rd(REV / f"C_lagidx_h{h:02d}")[0].reindex_like(ybl)
        f = recursive_combination(ybl, ycl, yl, h)
        m = f.notna() & yl.notna()
        rows.append(_row("indices lagged one day (C_lagidx with A1_lagidx)", h,
                         compare(f.where(m), yl, ybl.where(m), yl, h), block="d lag sensitivity"))
    df = pd.DataFrame(rows)
    pub_p = df[df.variant == "expanding real-time (published)"].sort_values("h")
    df["holm_p_published_family"] = np.nan
    df.loc[pub_p.index, "holm_p_published_family"] = holm(pub_p.dm_p.tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "run45_combination_robustness.csv", index=False)
    pd.DataFrame({f"h{h}": paths[h] for h in HORIZONS}).to_csv(OUT / "run45_weight_path.csv")
    _weight_figure(paths)
    return df


def _weight_figure(paths: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for h, c, lab in ((1, "#1f77b4", "one day"), (5, "black", "one week"), (22, "#d95f02", "one month")):
        s = paths[h].dropna()
        ax.plot(s.index, s.values, color=c, lw=1.1, label=lab)
    ax.set_ylim(0, 1); ax.set_ylabel(r"real-time weight $\hat\lambda_t$")
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"), color="0.9", zorder=0)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    (BASE / "results" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(BASE / "results" / "figures" / "fig_weight_path.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ run-46
def _market_cap(wrds: Path, dates: pd.DatetimeIndex, cols: list[str], ours: dict) -> pd.DataFrame:
    D = wrds / "SP500_WRDS_COMPLETE"
    want = set(dates)
    parts = []
    for f in sorted(D.glob("SP500_Daily_*.csv")):
        for ch in pd.read_csv(f, usecols=["permno", "dlycaldt", "dlycap"], chunksize=1_000_000):
            ch["dlycaldt"] = pd.to_datetime(ch["dlycaldt"])
            parts.append(ch[ch.dlycaldt.isin(want)])
    c = pd.concat(parts).drop_duplicates(["permno", "dlycaldt"])
    cap = c.pivot(index="dlycaldt", columns="permno", values="dlycap").reindex(dates)
    pn = {k: (int(k[1:]) if k.startswith("P") and k[1:].isdigit() else int(ours[k])) for k in cols}
    return pd.DataFrame({k: cap[v] if v in cap.columns else np.nan for k, v in pn.items()}, index=dates)


def run46(wrds: Path) -> pd.DataFrame:
    p = pd.read_pickle(WRDS_PKL)
    member, ours = p["member"], p["ours"]
    rows = []
    for h in HORIZONS:
        S = {m: _rd(REV / f"surv_{m}_h{h:02d}") for m in ("A1", "C")}
        mw = member.reindex(S["A1"][0].index).astype("boolean").fillna(False).astype(bool)
        S = {m: (yh.where(mw[yh.columns]), yy.where(mw[yy.columns])) for m, (yh, yy) in S.items()}
        (yb, y), yc = S["A1"], S["C"][0].reindex_like(S["A1"][0])
        rt = recursive_combination(yb, yc, y, h)                      # pooled weight, as Table 17
        scored = rt.notna() & yc.notna() & y.notna()
        cap = _market_cap(wrds, yb.index, list(yb.columns), ours)
        capm = cap.where(scored)
        q = np.ceil(capm.rank(axis=1, pct=True) * 5).clip(1, 5)
        for k in range(1, 6):
            m = scored & (q == k)
            rc = compare(yc.where(m), y, yb.where(m), y, h)
            rk = compare(rt.where(m), y, yb.where(m), y, h)
            ours_share = float((q[list(ours)] == k).sum().sum() / q[list(ours)].notna().sum().sum())
            rows.append({"h": h, "size_quintile": k, "n_stocks": rk["n_stocks"], "T": rk["T"],
                         "C_vs_harx_pct": rc["gain_pct"], "C_vs_harx_t": rc["dm_t"],
                         "comb_vs_harx_pct": rk["gain_pct"], "comb_vs_harx_t": rk["dm_t"],
                         "share_of_our115_cells": ours_share,
                         "cells_without_cap": int((scored & capm.isna()).sum().sum())})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "run46_size_quintiles.csv", index=False)
    return df


# ------------------------------------------------------------------ run-47
MIN_SHIFT = 150


def _design(h: int):
    b = load_bundle(); targets = build_targets(b)
    sd = b.sample_dates
    init_date = sd[int(len(sd) * INIT_TRAIN_FRAC)]
    jobs = []
    for tkr in b.panel.kept:
        X, y = _stock_xy(b, targets, tkr, "C", h, {}, {})
        if len(X) >= 60:
            jobs.append((tkr, X, y))
    return b, init_date, jobs


def _shifted_fit(X, y, off, init_date, h, full_index):
    X = X.copy()
    if off:
        X[PERSISTENCE] = np.roll(X[PERSISTENCE].values, off, axis=0)
    return expanding_forecast(X, y, init_date, h, full_index)


def _offsets(variant: str, draw: int, jobs) -> list[int]:
    if variant == "a":
        rng = np.random.default_rng([47, draw])
        return [int(rng.integers(MIN_SHIFT, len(X) - MIN_SHIFT + 1)) for _, X, _ in jobs]
    rng = np.random.default_rng([4700, draw])
    n_med = int(np.median([len(X) for _, X, _ in jobs]))
    k = int(rng.integers(MIN_SHIFT, n_med - MIN_SHIFT + 1))
    out = []
    for _, X, _ in jobs:
        n = len(X)
        out.append(k if MIN_SHIFT <= k <= n - MIN_SHIFT else MIN_SHIFT + k % max(n - 2 * MIN_SHIFT, 1))
    return out


def _panel(jobs, fits, index, cols) -> pd.DataFrame:
    yh = pd.DataFrame(index=index, columns=cols, dtype=float)
    for (tkr, _, _), f in zip(jobs, fits):
        yh.loc[f.index, tkr] = f.values
    return yh


def run47(variant: str, draws: list[int], check: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"run47_placebo_{variant}.csv"
    done = set(map(tuple, pd.read_csv(fp)[["draw", "h"]].values)) if fp.exists() else set()
    for h in HORIZONS:
        todo = [d for d in draws if (d, h) not in done]
        if not todo and not check:
            continue
        b, init_date, jobs = _design(h)
        yb, y = _rd(FCST / f"A1_h{h:02d}")
        full = b.rv.index
        if check:      # unit check: zero shift must reproduce the saved model C forecasts
            fits = Parallel(n_jobs=JOBS)(delayed(_shifted_fit)(X, yy, 0, init_date, h, full) for _, X, yy in jobs)
            saved = _rd(FCST / f"C_h{h:02d}")[0]
            new = _panel(jobs, fits, saved.index, saved.columns)
            print(f"  check h={h}: max abs diff vs saved C = {np.nanmax(np.abs((new - saved).values)):.2e}", flush=True)
            continue
        for d in todo:
            t0 = time.time()
            offs = _offsets(variant, d, jobs)
            fits = Parallel(n_jobs=JOBS)(delayed(_shifted_fit)(X, yy, o, init_date, h, full)
                                         for (_, X, yy), o in zip(jobs, offs))
            yp = _panel(jobs, fits, yb.index, yb.columns)
            f = recursive_combination(yb, yp, y, h)
            m = f.notna() & y.notna()
            r = compare(f.where(m), y, yb.where(m), y, h)
            solo = compare(yp.where(m), y, yb.where(m), y, h)
            row = pd.DataFrame([{"variant": variant, "draw": d, "h": h, "comb_gain_pct": r["gain_pct"],
                                 "comb_dm_t": r["dm_t"], "solo_gain_pct": solo["gain_pct"],
                                 "median_offset": float(np.median(offs)), "seconds": time.time() - t0}])
            row.to_csv(fp, mode="a", header=not fp.exists(), index=False)
            print(f"  run-47 {variant} h={h} draw {d}: comb {r['gain_pct']:+.3f}% ({time.time() - t0:.0f}s)", flush=True)


def summarize47() -> pd.DataFrame:
    real = {}
    for h in HORIZONS:
        yb, y = _rd(FCST / f"A1_h{h:02d}")
        yc = _rd(FCST / f"C_h{h:02d}")[0].reindex_like(yb)
        f = recursive_combination(yb, yc, y, h); m = f.notna() & y.notna()
        real[h] = compare(f.where(m), y, yb.where(m), y, h)["gain_pct"]
    rows = []
    for v in ("a", "b"):
        fp = OUT / f"run47_placebo_{v}.csv"
        if not fp.exists():
            continue
        d = pd.read_csv(fp)
        for h in HORIZONS:
            g = d[d.h == h].comb_gain_pct.values
            if len(g) == 0:
                continue
            rows.append({"variant": v, "h": h, "draws": len(g), "real_gain_pct": real[h],
                         "placebo_mean": g.mean(), "placebo_p50": np.median(g), "placebo_p95": np.percentile(g, 95),
                         "placebo_max": g.max(), "p_empirical": (1 + (g >= real[h]).sum()) / (1 + len(g))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "run47_placebo_summary.csv", index=False)
    return df


# ------------------------------------------------------------------ run-48
N_XS = 50          # simulated cross-section (amendment d2cb6e8)
WINDOWS = (500, 750, 1000)


def gph_rolling(x: np.ndarray, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling GPH d-hat at window ends W-1, W-1+stride, ... (batched; equals
    module2.gph window by window)."""
    from numpy.lib.stride_tricks import sliding_window_view
    from modules.module2_lrd_estimation import BANDWIDTH_POWER, ROLLING_STRIDE
    ends = np.arange(W - 1, len(x), ROLLING_STRIDE)
    win = sliding_window_view(x, W)[ends - W + 1]
    win = win - win.mean(axis=1, keepdims=True)
    m = int(np.floor(W ** BANDWIDTH_POWER))
    j = np.arange(1, m + 1)
    I = np.abs(np.fft.rfft(win, axis=1)[:, 1:m + 1]) ** 2 / (2 * np.pi * W)
    yv = np.log(I + 1e-12)
    X = np.log(4 * np.sin(np.pi * j / W) ** 2)
    Xd = X - X.mean()
    return ends, -((yv - yv.mean(axis=1, keepdims=True)) @ Xd) / (Xd ** 2).sum()


def _sim_one(scenario: str, rep: int, T: int, arma, d_s, sd_s, profile, sd_tab) -> dict:
    from scipy.signal import fftconvolve, lfilter
    from modules.module14_window_inclusion import N_EVENTS, EVENT_SEP, HALF, step_change
    from modules.module20_window_checks import FRAC_LAGS, frac_diff_weights
    rng = np.random.default_rng([48, sum(map(ord, scenario)), rep])
    usd = lambda d: sd_tab[round(float(d), 3)]
    idx = rng.integers(len(arma), size=N_XS)
    tau = int(rng.integers(1200, T - 1200))
    burn = 1000

    def frac(e, d):
        return fftconvolve(e, frac_diff_weights(d, FRAC_LAGS))[FRAC_LAGS:FRAC_LAGS + T + burn][burn:]

    panel = np.empty((N_XS, T))
    for k, i in enumerate(idx):
        if scenario.startswith("arma"):
            ar, ma, s = arma[i]
            e = rng.standard_normal(T + burn) * s
            panel[k] = lfilter([1.0, ma], [1.0, -ar], e)[burn:]
        else:
            e = rng.standard_normal(T + burn + FRAC_LAGS)
            if scenario.startswith("arfima"):
                dd = round(float(np.clip(d_s[i], 0.0, 0.49)), 3)
                x = frac(e, dd) * (sd_s[i] / usd(dd))
            else:
                x0, x1 = frac(e, 0.30), frac(e, 0.45)
                end = T if scenario == "d_change_permanent" else min(T, tau + 252)
                x = x0.copy(); x[tau:end] = x1[tau:end]
                x = x * (sd_s[i] / usd(0.30))
            panel[k] = x
    if scenario.endswith("+episode"):
        L = min(len(profile), T - tau)
        panel[:, tau:tau + L] += profile[:L]
    # event rule on the cross-sectional mean log variance
    xm = panel.mean(axis=0)
    ev = []
    for j in np.argsort(-xm):
        if all(abs(j - k) >= EVENT_SEP for k in ev):
            ev.append(int(j))
        if len(ev) == N_EVENTS:
            break
    paths = {}
    for W in WINDOWS:
        ends = None; acc = 0.0
        for k in range(N_XS):
            ends, d = gph_rolling(panel[k], W)
            acc = acc + d
        vals = acc / N_XS
        allc = np.array([step_change(vals, ends, int(z)) for z in ends]); allc = allc[~np.isnan(allc)]
        paths[W] = (ends, vals, allc)
    flags = []
    for e in ev:
        if any(e - HALF < W - 1 + HALF or e + W + HALF >= T for W in WINDOWS):
            flags.append(None); continue
        ok = True
        for W in WINDOWS:
            ends, vals, allc = paths[W]
            ent, ext = step_change(vals, ends, e), step_change(vals, ends, e + W)
            ok &= bool((allc <= ent).mean() >= 0.99 and (allc <= ext).mean() <= 0.01)
        flags.append(ok)
    usable = [f for f in flags if f is not None]
    return {"scenario": scenario, "rep": rep, "events_usable": len(usable), "positive_events": int(sum(usable)),
            "any_positive": bool(any(usable)), "tau": tau, "events": ";".join(map(str, ev))}


def run48(reps: int) -> pd.DataFrame:
    from modules.module14_window_inclusion import episode_profile, pick_events
    from modules.module20_window_checks import simulate_arfima
    from modules.module27_window_placebo import per_stock_params
    OUT.mkdir(parents=True, exist_ok=True)
    b = load_bundle()
    T = len(b.rv.index)
    pc = OUT / "run48_calm_params.csv"
    if pc.exists():
        P = pd.read_csv(pc); arma, d_s, sd_s = P[["ar", "ma", "s"]].values, P.d.values, P.sd.values
    else:
        arma, d_s, sd_s = per_stock_params(b.log_rv)
        pd.DataFrame({"ar": arma[:, 0], "ma": arma[:, 1], "s": arma[:, 2], "d": d_s, "sd": sd_s}).to_csv(pc, index=False)
    covid = [e for e in pick_events(b.log_rv) if e.year == 2020][0]
    profile = episode_profile(b.log_rv, covid)
    sd_tab = {k: simulate_arfima(k, 1.0, 20000, np.random.default_rng(1)).std()
              for k in set(np.round(np.clip(d_s, 0, 0.49), 3).tolist()) | {0.30, 0.45}}
    scenarios = ["arma", "arfima", "d_change_permanent", "d_change_one_year", "arma+episode", "arfima+episode"]
    fp = OUT / "run48_window_calibration_reps.csv"
    for sc in scenarios:
        for lo in range(0, reps, 60):
            done = set(map(tuple, pd.read_csv(fp)[["scenario", "rep"]].values)) if fp.exists() else set()
            todo = [r for r in range(lo, min(lo + 60, reps)) if (sc, r) not in done]
            if not todo:
                continue
            t0 = time.time()
            res = Parallel(n_jobs=JOBS)(delayed(_sim_one)(sc, r, T, arma, d_s, sd_s, profile, sd_tab) for r in todo)
            pd.DataFrame(res).to_csv(fp, mode="a", header=not fp.exists(), index=False)
            print(f"  run-48 {sc} reps {todo[0]}-{todo[-1]}: {time.time() - t0:.0f}s", flush=True)
    d = pd.read_csv(fp)
    s = d.groupby("scenario").agg(reps=("rep", "count"), any_positive_rate=("any_positive", "mean"),
                                  positive_events=("positive_events", "sum"), usable_events=("events_usable", "sum"))
    s["per_event_rate"] = s.positive_events / s.usable_events
    s = s.reindex(scenarios)
    s.to_csv(OUT / "run48_window_calibration.csv")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, required=True, choices=[45, 46, 47, 48])
    ap.add_argument("--wrds", type=Path)
    ap.add_argument("--variant", choices=["a", "b"], default="a")
    ap.add_argument("--draws", default="1-1")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--reps", type=int, default=500)
    a = ap.parse_args()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    if a.run == 45:
        print(run45().round(4).to_string(index=False))
    elif a.run == 46:
        print(run46(a.wrds).round(4).to_string(index=False))
    elif a.run == 47:
        if a.summarize:
            print(summarize47().round(4).to_string(index=False)); return
        lo, hi = map(int, a.draws.split("-"))
        run47(a.variant, list(range(lo, hi + 1)), check=a.check)
    else:
        print(run48(a.reps).round(4).to_string())


if __name__ == "__main__":
    main()
