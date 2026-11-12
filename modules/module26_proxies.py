"""
MODULE 26: Alternative Variance Proxies and Intraday Validation (run-39;
JRFM round-1 revision)
=====================================================================
Registered in docs/EXPERIMENTS.md (sixth block, commit 4c4d1a4; addendum
582bf9d) before any of these numbers existed.

Under each proxy the daily variance replaces Parkinson everywhere: in the
target (log mean over the next h days), the HAR terms, and the persistence
state (rolling GPH on log variance, 750-day window, at least 250 valid days,
every derived predictor rebuilt as in module 9). Proxies:
    gk  Garman-Klass   0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2
    rs  Rogers-Satchell ln(H/C) ln(H/O) + ln(L/C) ln(L/O)
    cc  close-to-close  r_t^2 (raw log return, overnight included)
Non-positive daily values are missing, as zero-range days are for Parkinson.
AAPL has no Bloomberg open price in the October 2026 pull and drops out of
gk and rs. Models: HAR (A), HAR-X (A1), C without the Hurst features (run-29
showed it within 0.1 pp of C) and A1cs; HAR and HAR-X also with HAR inputs
through t-1 (the misaligned timing convention). Fits are cached under
results/intermediate/forecasts/rev1/{proxy}_*.

Intraday: for the 20 stocks with 5-minute Bloomberg bars (13 Oct 2025 to
21 Apr 2026), daily realized variance from within-day 5-minute log returns,
compared with the same day's Parkinson, Garman-Klass and Rogers-Satchell
variance.

    python -m modules.module26_proxies --proxy gk     (then rs, cc)
    python -m modules.module26_proxies --eval
    python -m modules.module26_proxies --intraday
Outputs: results/intermediate/proxy_results.csv, proxy_exit.csv,
         intraday_validation.csv
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from modules.forecast_io import HORIZONS, MODEL_FEATURES, build_targets, load_bundle
from modules.io_v2 import log_variance
from modules.module2_lrd_estimation import gph
from modules.module3_feature_engineering import build_har
from modules.module9_robustness import _apply_d_overrides, _build_derived_from_d, _rolling_d_at_window
from modules.module15_timing import stale_inputs
from modules.module19_matched_robustness import compare
from modules.module25_restricted_and_timing import REV, fit_panel, recursive_combination

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
BB = BASE / "bloomberg_pull"
CLEAN = BB / "processed" / "clean_panel"
INTERM = BASE / "results" / "intermediate"

PROXIES = ("gk", "rs", "cc")
W = 750
EXIT_2020 = pd.Timestamp("2023-03-10")      # tau + 750 trading days for the 2020 episode (Section 4)
HALF = 20                                    # trading days either side, as module 14
MODEL_FEATURES["C_noH"] = [f for f in MODEL_FEATURES["C"] if f not in ("h", "delta_h")]


# ------------------------------------------------------------------ proxies
def ohlc() -> dict[str, pd.DataFrame]:
    rd = lambda p: pd.read_csv(p, index_col=0, parse_dates=True)
    c, h, l = rd(CLEAN / "prices_close.csv"), rd(CLEAN / "prices_high.csv"), rd(CLEAN / "prices_low.csv")
    o = rd(BB / "processed" / "prices_open.csv").reindex(index=c.index, columns=c.columns)
    return {"o": o, "h": h, "l": l, "c": c}


def proxy_variance(name: str, bundle=None) -> pd.DataFrame:
    if name == "cc":
        r = bundle.returns_raw if bundle is not None else load_bundle().returns_raw
        v = r ** 2
    else:
        p = ohlc()
        hl, co = np.log(p["h"] / p["l"]), np.log(p["c"] / p["o"])
        if name == "gk":
            v = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
        elif name == "rs":
            v = (np.log(p["h"] / p["c"]) * np.log(p["h"] / p["o"])
                 + np.log(p["l"] / p["c"]) * np.log(p["l"] / p["o"]))
        else:
            raise ValueError(name)
    return v.where(v > 0)


def proxy_bundle(name: str):
    """The standard bundle with every variance-derived input rebuilt on the
    proxy. The rolling d panel is cached (rev1/{name}_d_gph.csv)."""
    b = load_bundle()
    rv = proxy_variance(name, b)[b.panel.kept]
    b.rv, b.log_rv = rv, log_variance(rv)
    har = build_har(rv)
    for k in ("har_d", "har_w", "har_m"):
        b.feat[k] = har[k].reindex(b.sample_dates)
    fd = REV / f"{name}_d_gph.csv"
    if fd.exists():
        d = pd.read_csv(fd, index_col=0, parse_dates=True)
    else:
        d = _rolling_d_at_window(b, W, gph)
        REV.mkdir(parents=True, exist_ok=True)
        d.to_csv(fd)
    _apply_d_overrides(b, _build_derived_from_d(d, b.sectors))
    return b


def run_proxy(name: str) -> None:
    b = proxy_bundle(name)
    targets = build_targets(b)
    stale = stale_inputs(b)
    for model in ("A", "A1", "C_noH", "A1cs"):
        for h in HORIZONS:
            fit_panel(b, targets, f"{name}_{model}", model, h)
            print(f"  {name} {model} h={h} done", flush=True)
    for model in ("A", "A1"):
        for h in HORIZONS:
            fit_panel(b, targets, f"{name}_{model}_stale", model, h, overrides=stale)
            print(f"  {name} {model} (HAR inputs through t-1) h={h} done", flush=True)


# ------------------------------------------------------------------ evaluation
def _c(name, h):
    rd = lambda s: pd.read_csv(REV / f"{name}_h{h:02d}_{s}.csv", index_col=0, parse_dates=True)
    return rd("yhat"), rd("y")


def _saved(model, h):
    F = INTERM / "forecasts"
    rd = lambda s: pd.read_csv(F / f"{model}_h{h:02d}_{s}.csv", index_col=0, parse_dates=True)
    return rd("yhat"), rd("y")


def evaluate() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for proxy in ("parkinson",) + PROXIES:
        for h in HORIZONS:
            if proxy == "parkinson":
                A, A1, C, cs = _saved("A", h), _saved("A1", h), _saved("C", h), _saved("A1cs", h)
                As = A1s = None
            else:
                A, A1, C, cs = (_c(f"{proxy}_{m}", h) for m in ("A", "A1", "C_noH", "A1cs"))
                As, A1s = _c(f"{proxy}_A_stale", h), _c(f"{proxy}_A1_stale", h)
            harx = compare(A1[0], A1[1], A[0], A[1], h, rng=rng)
            cvx = compare(C[0], C[1], A1[0], A1[1], h, rng=rng)
            csx = compare(cs[0], cs[1], A1[0], A1[1], h)
            yb, y = A1
            rt = recursive_combination(yb, C[0].reindex_like(yb), y, h)
            comb = compare(rt, y, yb.where(rt.notna()), y, h, rng=rng)
            r = {"proxy": proxy, "h": h, "n_stocks": cvx["n_stocks"],
                 "harx_vs_har_pct": harx["gain_pct"], "harx_vs_har_t": harx["dm_t"],
                 "C_vs_harx_pct": cvx["gain_pct"], "C_vs_harx_t": cvx["dm_t"],
                 "C_ci_lo": cvx["ci_lo"], "C_ci_hi": cvx["ci_hi"],
                 "A1cs_vs_harx_pct": csx["gain_pct"], "A1cs_vs_harx_t": csx["dm_t"],
                 "comb_rt_pct": comb["gain_pct"], "comb_rt_t": comb["dm_t"],
                 "comb_ci_lo": comb["ci_lo"], "comb_ci_hi": comb["ci_hi"]}
            if As is not None:
                r["harx_vs_har_misaligned_pct"] = compare(A1s[0], A1s[1], As[0], As[1], h)["gain_pct"]
            else:                                   # Parkinson: module 15, run-16
                ta = pd.read_csv(INTERM / "timing_alignment.csv")
                r["harx_vs_har_misaligned_pct"] = float(ta[(ta.h == h) & (ta.alignment == "stale")].harx_gain_pct.iloc[0])
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "proxy_results.csv", index=False)
    return df


def exit_signature() -> pd.DataFrame:
    """Change in the cross-sectional mean of rolling d over the HALF trading
    days after minus before the 2020 episode's exit date at W = 750, and its
    percentile among the same statistic at every sample date."""
    rows = []
    b = load_bundle()
    series = {"parkinson": b.cs["cs_mean_d"]}
    for p in PROXIES:
        d = pd.read_csv(REV / f"{p}_d_gph.csv", index_col=0, parse_dates=True)
        series[p] = d.mean(axis=1)
    for name, s in series.items():
        s = s.dropna()
        step = HALF // 5                                # weekly grid
        stat = s.rolling(step).mean().shift(-step) - s.rolling(step).mean()
        k = s.index.get_indexer([EXIT_2020], method="nearest")[0]
        v = float(stat.iloc[k])
        rows.append({"proxy": name, "mean_d": float(s.mean()), "exit_change": v,
                     "exit_change_pct_rank": float((stat.dropna() <= v).mean() * 100)})
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "proxy_exit.csv", index=False)
    return df


# ------------------------------------------------------------------ intraday
def intraday_validation() -> pd.DataFrame:
    import openpyxl
    wb = openpyxl.load_workbook(BB / "intraday.xlsx", read_only=True, data_only=True)
    rows = list(wb["Sheet1"].iter_rows(values_only=True))
    names, starts = rows[2], [j for j, v in enumerate(rows[2]) if v]
    p = ohlc()
    pk = (np.log(p["h"] / p["l"]) ** 2) / (4 * np.log(2))
    gk, rs = proxy_variance("gk"), proxy_variance("rs")
    out = []
    for j in starts:
        tkr = str(names[j]).split()[0]
        if tkr == "SPX" or tkr not in pk.columns:
            continue
        recs = [(r[j], r[j + 2]) for r in rows[4:] if hasattr(r[j], "year") and r[j + 2] is not None]
        bars = pd.DataFrame(recs, columns=["t", "close"]).set_index("t").sort_index()["close"].astype(float)
        lr = np.log(bars).groupby(bars.index.date).diff()            # within-day returns only
        rv5 = (lr ** 2).groupby(lr.index.date).sum(min_count=10)
        rv5.index = pd.to_datetime(rv5.index)
        df = pd.DataFrame({"rv5": rv5, "pk": pk[tkr], "gk": gk[tkr], "rs": rs[tkr]}).dropna(subset=["rv5", "pk"])
        df = df[df.rv5 > 0]
        r = {"ticker": tkr, "days": len(df)}
        for prox in ("pk", "gk", "rs"):
            d = df[["rv5", prox]].dropna()
            d = d[d[prox] > 0]
            r[f"corr_log_{prox}"] = float(np.log(d["rv5"]).corr(np.log(d[prox])))
            r[f"ratio_mean_{prox}"] = float(d[prox].mean() / d["rv5"].mean())
        out.append(r)
    df = pd.DataFrame(out)
    df.to_csv(INTERM / "intraday_validation.csv", index=False)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--proxy", choices=PROXIES)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--intraday", action="store_true")
    a = p.parse_args()
    pd.set_option("display.width", 240)
    if a.proxy:
        run_proxy(a.proxy)
    if a.intraday:
        iv = intraday_validation()
        print(iv.round(3).to_string(index=False))
        print("\nmedian over stocks:"); print(iv.drop(columns=["ticker"]).median().round(3).to_string())
    if a.eval:
        print(evaluate().round(3).to_string(index=False))
        print(exit_signature().round(4).to_string(index=False))


if __name__ == "__main__":
    main()
