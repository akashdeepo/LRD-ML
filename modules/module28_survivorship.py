"""
MODULE 28: Survivorship, a Point-in-Time S&P 500 Panel (run-41; JRFM
round-1 revision)
====================================================================
Registered in docs/EXPERIMENTS.md (sixth block; design 082c0ed, end-date
amendment cf462d7) before any CRSP file was opened.

Data: CRSP via WRDS (Nicholas Appiah, November 2026), the raw extract
SP500_WRDS_COMPLETE: membership spells (crsp_a_indexes.dsp500list_v2),
daily OHLC and returns for the 1,055 stocks that were members at any time
from 2001 to 2025, identifier history and Compustat GICS sectors (current,
not point in time). The data end on 31 Dec 2025.

Panel: our 115 stocks keep their Bloomberg series; every other ever-member
gets CRSP's daily high and low (Parkinson variance) and its price return.
A stock enters if at least 70% of the trading days of its membership spells
have a valid range. Everything that defines the persistence state is
rebuilt on this panel with the code of modules 2 and 3: rolling GPH d-hat
and Hurst over 750 days (at least 250 valid), their dynamics, the HAR terms,
d x VIX and d x MOVE. Point in time: the cross-sectional mean and standard
deviation of d-hat and the leave-one-out sector mean use only the stocks
that are index members on that date; a stock's forecasts are evaluated only
on origins at which it is a member; its regressions use its own past data.

    python -m modules.module28_survivorship --build --wrds <folder with SP500_WRDS_COMPLETE>
    python -m modules.module28_survivorship --fit A        (then A1, C)
    python -m modules.module28_survivorship --eval
Cache: bloomberg_pull/processed/wrds/ (gitignored); fits in
results/intermediate/forecasts/rev1/surv_*. Output:
results/intermediate/survivorship.csv, survivorship_panel.csv
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from modules.forecast_io import HORIZONS, build_targets, load_bundle, pit_level_correction
from modules.io_v2 import WINSOR_LOWER, WINSOR_UPPER, _winsorize_pit, log_variance
from modules.module2_lrd_estimation import ROLLING_STRIDE, ROLLING_WINDOW, gph, hurst_scaling, rolling_panel
from modules.module3_feature_engineering import build_har, memory_dynamics, sector_mean_panel
from modules.module6_forecast_eval import diebold_mariano, qlike_loss, squared_loss
from modules.module14_window_inclusion import HALF, step_change
from modules.module19_matched_robustness import compare
from modules.module25_restricted_and_timing import REV, fit_panel, recursive_combination

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
CACHE = BASE / "bloomberg_pull" / "processed" / "wrds"
INTERM = BASE / "results" / "intermediate"
END = pd.Timestamp("2025-12-31")
COVERAGE = 0.70
GOOGL_CLASS_A = 90319
GICS_NAMES = {10: "Energy", 15: "Materials", 20: "Industrials", 25: "Consumer Discretionary",
              30: "Consumer Staples", 35: "Health Care", 40: "Financials", 45: "Information Technology",
              50: "Communication Services", 55: "Utilities", 60: "Real Estate"}
EXIT_2020 = pd.Timestamp("2023-03-10")


# ------------------------------------------------------------------ build
def _our_permnos(names: pd.DataFrame, kept: list[str]) -> dict[str, int]:
    asof = pd.Timestamp("2025-06-30")
    cur = names[(names.namedt <= asof) & (names.nameenddt.fillna(pd.Timestamp("2100-01-01")) >= asof)]
    alias = {"MRSH": ["MRSH", "MMC"]}         # Marsh & McLennan: CRSP still lists MMC at mid-2025
    out = {}
    for t in kept:
        p = sorted(cur[cur.ticker.isin(alias.get(t, [t]))].permno.unique())
        if t != "GOOGL" and len(p) != 1:
            raise ValueError(f"{t}: {len(p)} CRSP matches")
        out[t] = GOOGL_CLASS_A if t == "GOOGL" else int(p[0])
    return out


def _rolling(panel: pd.DataFrame, estimator, jobs: int = 12) -> pd.DataFrame:
    """rolling_panel (module 2) in parallel over blocks of columns."""
    blocks = np.array_split(np.arange(panel.shape[1]), jobs * 4)
    parts = Parallel(n_jobs=jobs)(delayed(rolling_panel)(panel.iloc[:, b], estimator, ROLLING_WINDOW,
                                                         ROLLING_STRIDE, "") for b in blocks if len(b))
    return pd.concat(parts, axis=1)[panel.columns]


def build(wrds: Path) -> None:
    D = wrds / "SP500_WRDS_COMPLETE"
    CACHE.mkdir(parents=True, exist_ok=True)
    b = load_bundle()
    cal = b.rv.index[b.rv.index <= END]
    # the full weekly grid, so that the training/evaluation split date (40% of
    # the grid) is the one of the main analysis; features are missing after END
    sd = b.sample_dates
    names = pd.read_csv(D / "SP500_Historical_Names.csv", parse_dates=["namedt", "nameenddt"])
    mem = pd.read_csv(D / "SP500_Membership_2001_2026.csv", parse_dates=["mbrstartdt", "mbrenddt"])
    gics = pd.read_csv(D / "SP500_Compustat_GICS.csv").dropna(subset=["permno", "gsector"])
    ours = _our_permnos(names, b.panel.kept)
    label = {p: t for t, p in ours.items()}
    lab = lambda p: label.get(int(p), f"P{int(p)}")

    # daily CRSP for the other ever-members
    parts = []
    for f in sorted(D.glob("SP500_Daily_*.csv")):
        for ch in pd.read_csv(f, usecols=["permno", "dlycaldt", "dlyhigh", "dlylow", "dlyretx"], chunksize=500_000):
            parts.append(ch[~ch.permno.isin(label)])
    c = pd.concat(parts)
    c["dlycaldt"] = pd.to_datetime(c["dlycaldt"])
    ok = (c.dlyhigh > 0) & (c.dlylow > 0) & (c.dlyhigh >= c.dlylow)
    c["pk"] = np.where(ok, np.log(c.dlyhigh / c.dlylow) ** 2 / (4 * np.log(2)), np.nan)
    c["lr"] = np.log1p(c.dlyretx)
    c["name"] = c.permno.map(lab)
    pk_new = c.pivot_table(index="dlycaldt", columns="name", values="pk", aggfunc="first").reindex(cal)
    lr_new = c.pivot_table(index="dlycaldt", columns="name", values="lr", aggfunc="first").reindex(cal)

    rv = pd.concat([b.rv.loc[cal, b.panel.kept], pk_new], axis=1)
    ret_raw = pd.concat([b.returns_raw.loc[cal, b.panel.kept], lr_new], axis=1)

    # membership masks (daily and on the weekly sample grid)
    member = pd.DataFrame(False, index=cal, columns=rv.columns)
    for r in mem.itertuples():
        n = lab(r.permno)
        if n in member.columns:
            member.loc[(cal >= r.mbrstartdt) & (cal <= r.mbrenddt), n] = True
    cover = (rv.where(member) > 0).sum() / member.sum().replace(0, np.nan)
    keep = sorted(cover[cover >= COVERAGE].index)
    print(f"  stocks: {rv.shape[1]} with data; kept {len(keep)} with >= {COVERAGE:.0%} valid ranges in their spells "
          f"({sum(k in ours for k in keep)} of our 115)")
    rv, ret_raw, member = rv[keep], ret_raw[keep], member[keep]
    log_rv = log_variance(rv)

    print("  rolling GPH d-hat ...", flush=True)
    d = _rolling(log_rv, gph).reindex(sd)
    print("  rolling Hurst ...", flush=True)
    h = _rolling(log_rv, hurst_scaling).reindex(sd)

    sectors = {t: s for t, s in b.sectors.items() if t in keep}
    g = gics.assign(permno=gics.permno.astype(int)).sort_values("linkdt").groupby("permno").gsector.last()
    for n in keep:
        if n not in sectors and n.startswith("P"):
            code = g.get(int(n[1:]))
            sectors[n] = GICS_NAMES.get(int(code), "Unknown") if pd.notna(code) else "Unknown"
    mw = member.reindex(sd).fillna(False)
    dm = d.where(mw)                                        # members only, for the common states
    cs = pd.DataFrame({"cs_mean_d": dm.mean(axis=1), "cs_std_d": dm.std(axis=1)})
    sec = sector_mean_panel(dm, sectors)
    dyn, dyn_h = memory_dynamics(d), memory_dynamics(h)
    har = {k: v.reindex(sd).ffill() for k, v in build_har(rv).items()}
    mk = b.market.reindex(sd)
    feat = {"d_gph": d, "delta_d_gph": dyn["delta"], "vol_d_gph": dyn["vol"], "trend_d_gph": dyn["trend"],
            "h": h, "delta_h": dyn_h["delta"], "sector_mean_d": sec,
            "d_x_vix": d.mul(mk["VIX"], axis=0), "d_x_move": d.mul(mk["MOVE"], axis=0), **har}
    print("  point-in-time winsorised returns ...", flush=True)
    ret_w = pd.concat([b.returns.loc[cal, [k for k in keep if k in ours]],
                       _winsorize_pit(ret_raw[[k for k in keep if k not in ours]], WINSOR_LOWER, WINSOR_UPPER)],
                      axis=1)[keep]
    pd.to_pickle({"rv": rv, "returns_raw": ret_raw, "returns": ret_w, "member": member, "feat": feat, "cs": cs,
                  "sectors": sectors, "sample_dates": sd, "ours": ours}, CACHE / "survivorship_panel.pkl")
    pd.DataFrame({"stock": keep, "ours": [k in ours for k in keep], "sector": [sectors[k] for k in keep],
                  "coverage_in_spells": cover[keep].values}).to_csv(INTERM / "survivorship_panel.csv", index=False)
    print(f"  saved; sectors unknown for {sum(v == 'Unknown' for v in sectors.values())} stocks")


def surv_bundle():
    p = pd.read_pickle(CACHE / "survivorship_panel.pkl")
    b = load_bundle()
    b.rv, b.log_rv = p["rv"], log_variance(p["rv"])
    b.returns_raw, b.returns = p["returns_raw"], p["returns"]
    b.feat, b.cs, b.sectors = p["feat"], p["cs"], p["sectors"]
    b.sample_dates = p["sample_dates"]
    b.market = b.market.reindex(b.sample_dates)          # unchanged grid; kept explicit
    b.panel.kept = list(p["rv"].columns)
    return b, p["member"]


def fit(model: str) -> None:
    b, _ = surv_bundle()
    targets = build_targets(b)
    for h in HORIZONS:
        fit_panel(b, targets, f"surv_{model}", model, h)
        print(f"  surv {model} h={h} done", flush=True)


# ------------------------------------------------------------------ evaluation
def _rd(path_stem: Path):
    return (pd.read_csv(f"{path_stem}_yhat.csv", index_col=0, parse_dates=True),
            pd.read_csv(f"{path_stem}_y.csv", index_col=0, parse_dates=True))


def _block(name: str, A, A1, C, h, rng) -> dict:
    harx = compare(A1[0], A1[1], A[0], A[1], h, rng=rng)
    cvx = compare(C[0], C[1], A1[0], A1[1], h, rng=rng)
    yb, y = A1
    cm = yb.notna() & C[0].notna() & y.notna()
    qb = qlike_loss(pit_level_correction(yb, y, h, kind="smear"), y)
    qc = qlike_loss(pit_level_correction(C[0], y, h, kind="smear"), y)
    cq = cm & qb.notna() & qc.notna()
    _, tq, pq, _ = diebold_mariano(qb.where(cq), qc.where(cq), h=h)
    rt = recursive_combination(yb, C[0].reindex_like(yb), y, h)
    comb = compare(rt, y, yb.where(rt.notna()), y, h, rng=rng)
    return {"panel": name, "h": h, "n_stocks": cvx["n_stocks"], "T": cvx["T"],
            "harx_vs_har_pct": harx["gain_pct"], "harx_vs_har_t": harx["dm_t"],
            "C_vs_harx_pct": cvx["gain_pct"], "C_vs_harx_t": cvx["dm_t"], "C_vs_harx_p": cvx["dm_p"],
            "C_ci_lo": cvx["ci_lo"], "C_ci_hi": cvx["ci_hi"], "qlike_t": tq, "qlike_p": pq,
            "comb_rt_pct": comb["gain_pct"], "comb_rt_t": comb["dm_t"], "comb_ci_lo": comb["ci_lo"], "comb_ci_hi": comb["ci_hi"]}


def evaluate() -> pd.DataFrame:
    rng = np.random.default_rng(6)
    _, member = surv_bundle()
    F = INTERM / "forecasts"
    rows = []
    for h in HORIZONS:
        S = {m: _rd(REV / f"surv_{m}_h{h:02d}") for m in ("A", "A1", "C")}
        mw = member.reindex(S["A"][0].index).fillna(False)
        S = {m: (yh.where(mw[yh.columns]), y.where(mw[y.columns])) for m, (yh, y) in S.items()}
        last = S["C"][1].dropna(how="all").index.max()                     # last origin with a target <= END
        O = {m: tuple(x.loc[:last] for x in _rd(F / f"{m}_h{h:02d}")) for m in ("A", "A1", "C")}
        rows.append(_block("115 stocks, to Dec 2025", O["A"], O["A1"], O["C"], h, rng))
        rows.append(_block("point-in-time members, to Dec 2025", S["A"], S["A1"], S["C"], h, rng))
        ours = [c for c in S["A"][0].columns if not c.startswith("P")]
        S115 = {m: (yh[ours], y[ours]) for m, (yh, y) in S.items()}
        rows.append(_block("our 115 inside the point-in-time panel", S115["A"], S115["A1"], S115["C"], h, rng))
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "survivorship.csv", index=False)
    return df


def state_check() -> pd.DataFrame:
    """Level of the cross-sectional state and its exit drop at the 2020 episode."""
    b0 = load_bundle()
    p = pd.read_pickle(CACHE / "survivorship_panel.pkl")
    series = {"115 stocks": b0.cs["cs_mean_d"].loc[:END], "point-in-time members": p["cs"]["cs_mean_d"]}
    full = b0.rv.index
    rows = []
    for name, s in series.items():
        s = s.dropna()
        ends = full.get_indexer(s.index)
        vals = s.values
        x = full.get_loc(EXIT_2020)
        ex = step_change(vals, ends, x)
        allc = np.array([step_change(vals, ends, int(e)) for e in ends])
        allc = allc[~np.isnan(allc)]
        rows.append({"panel": name, "mean_d": float(s.mean()), "exit_change_2020": ex,
                     "exit_pct_rank": float((allc <= ex).mean() * 100)})
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "survivorship_state.csv", index=False)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    p.add_argument("--wrds", type=Path)
    p.add_argument("--fit", choices=["A", "A1", "C"])
    p.add_argument("--eval", action="store_true")
    a = p.parse_args()
    pd.set_option("display.width", 240)
    if a.build:
        build(a.wrds)
    if a.fit:
        fit(a.fit)
    if a.eval:
        print(evaluate().round(3).to_string(index=False))
        print(state_check().round(4).to_string(index=False))


if __name__ == "__main__":
    main()
