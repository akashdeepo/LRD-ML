"""
MODULE 14: What the Rolling Persistence State Measures (window inclusion)
=========================================================================
The cross-sectional mean of the rolling GPH estimate d-hat of log Parkinson
variance rises in crises. This module tests whether that rise is a property
of the market or of the estimation window (ledger run-14, audit N1/N2):

  1. Rolling d-hat at windows W = 500, 750, 1000 (same estimator, bandwidth
     and stride as module 2); cross-sectional mean d-bar_t(W).
  2. Events, rule fixed in advance: the 3 largest days of the cross-sectional
     mean daily log variance, at least 250 trading days apart. For each event
     and W, the change in d-bar when the event day enters the window (tau)
     and when it leaves (tau + W), ranked among all such changes.
  3. R^2 of d-bar_t(W) on the maximum VIX over the same trailing W days
     versus on the current VIX.
  4. Matched short-memory null: ARMA(1,1) fitted to calm-period log variance
     (median parameters across stocks) with the observed one-year COVID excess
     profile injected; mean rolling d-hat path over replications.

Outputs:
    results/intermediate/window_inclusion_{events,r2,sim,paths}.csv
    results/tables/table14_window_inclusion.tex
    results/figures/fig10_window_inclusion.pdf
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402

from modules.forecast_io import load_bundle  # noqa: E402
from modules.module2_lrd_estimation import ROLLING_STRIDE, gph, rolling_panel  # noqa: E402

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
INTERM = BASE / "results" / "intermediate"
TABLES = BASE / "results" / "tables"
FIGS = BASE / "results" / "figures"

WINDOWS = (500, 750, 1000)
N_EVENTS, EVENT_SEP = 3, 250   # event rule (fixed before the run)
HALF = 20                      # trading days either side: 4 weekly grid points
CALM = ("2012-01-01", "2019-12-31")
EPISODE_DAYS = 252
N_REPS = 200
SEED = 0


# ------------------------------------------------------------------ events
def pick_events(log_rv: pd.DataFrame) -> list[pd.Timestamp]:
    """The N_EVENTS largest days of the cross-sectional mean log variance,
    at least EVENT_SEP trading days apart."""
    m = log_rv.mean(axis=1).dropna()
    pos = pd.Series(np.arange(len(log_rv)), index=log_rv.index)
    chosen: list[pd.Timestamp] = []
    for d in m.sort_values(ascending=False).index:
        if all(abs(pos[d] - pos[c]) >= EVENT_SEP for c in chosen):
            chosen.append(d)
        if len(chosen) == N_EVENTS:
            break
    return sorted(chosen)


def step_change(vals: np.ndarray, ends: np.ndarray, x: int) -> float:
    """Mean of d-bar over window ends in [x, x+HALF) minus over [x-HALF, x)."""
    after = vals[(ends >= x) & (ends < x + HALF)]
    before = vals[(ends >= x - HALF) & (ends < x)]
    if len(after) == 0 or len(before) == 0:
        return np.nan
    return float(np.nanmean(after) - np.nanmean(before))


def all_step_changes(vals: np.ndarray, ends: np.ndarray) -> np.ndarray:
    return np.array([step_change(vals, ends, int(e)) for e in ends])


def pct_rank(ref: np.ndarray, v: float) -> float:
    ref = ref[~np.isnan(ref)]
    return float(100 * (ref <= v).mean()) if np.isfinite(v) else np.nan


# ------------------------------------------------------------------ null model
def calm_arma(log_rv: pd.DataFrame) -> np.ndarray:
    """Median (ar, ma, sigma) of per-stock ARMA(1,1) fits on calm log variance."""
    rows = []
    for c in log_rv.columns:
        x = log_rv[c].loc[CALM[0]:CALM[1]].dropna().values
        if len(x) < 1000:
            continue
        r = ARIMA(x, order=(1, 0, 1)).fit()
        rows.append((r.arparams[0], r.maparams[0], np.sqrt(r.params[-1])))
    return np.median(np.array(rows), axis=0)


def episode_profile(log_rv: pd.DataFrame, tau: pd.Timestamp) -> np.ndarray:
    """Cross-sectional median of log variance minus each stock's calm mean,
    over EPISODE_DAYS from the event day."""
    calm_mean = log_rv.loc[CALM[0]:CALM[1]].mean()
    p = log_rv.index.get_loc(tau)
    seg = log_rv.iloc[p:p + EPISODE_DAYS] - calm_mean
    return seg.median(axis=1).values


def simulate_paths(arma: np.ndarray, profile: np.ndarray, W: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Mean rolling d-hat path of ARMA(1,1) + injected episode. Returns window-end
    positions relative to the episode start and the mean d-hat at each."""
    ar, ma, s = arma
    pre, burn = W + 120, 500
    T = pre + W + 300
    ends = np.arange(W - 1, T, ROLLING_STRIDE)
    paths = np.empty((N_REPS, len(ends)))
    for r in range(N_REPS):
        e = rng.standard_normal(T + burn) * s
        x = np.empty(T + burn)
        x[0] = e[0]
        for t in range(1, T + burn):
            x[t] = ar * x[t - 1] + e[t] + ma * e[t - 1]
        x = x[burn:]
        x[pre:pre + len(profile)] += profile
        paths[r] = [gph(x[end - W + 1:end + 1])[0] for end in ends]
    return ends - pre, paths.mean(axis=0)


def phase_means(rel: np.ndarray, d: np.ndarray, W: int) -> dict:
    """Baseline before entry, plateau while inside, level after exit."""
    def m(lo, hi):
        sel = (rel >= lo) & (rel < hi)
        return float(np.nanmean(d[sel])) if sel.any() else np.nan
    return {"baseline": m(-120, 0), "plateau": m(60, W - 60), "after_exit": m(W + 20, W + 120)}


# ------------------------------------------------------------------ main
def main() -> None:
    print("=" * 70)
    print("   MODULE 14: WINDOW INCLUSION")
    print("=" * 70)
    bundle = load_bundle()
    log_rv = bundle.log_rv
    idx = log_rv.index
    vix = bundle.panel.market["VIX"].reindex(idx).ffill()

    events = pick_events(log_rv)
    print("  events:", [d.date() for d in events])

    ev_rows, r2_rows, path_rows, sim_rows = [], [], [], []
    rng = np.random.default_rng(SEED)
    arma = calm_arma(log_rv)
    print(f"  calm ARMA(1,1) medians: ar={arma[0]:.3f} ma={arma[1]:.3f} sigma={arma[2]:.3f}")
    covid = [d for d in events if d.year == 2020][0]
    profile = episode_profile(log_rv, covid)

    for W in WINDOWS:
        dbar = rolling_panel(log_rv, gph, window=W, stride=ROLLING_STRIDE,
                             label=f"rolling GPH W={W}").mean(axis=1)
        ends = idx.get_indexer(dbar.index)
        vals = dbar.values
        ref = all_step_changes(vals, ends)
        for tau in events:
            p = idx.get_loc(tau)
            d_in, d_out = step_change(vals, ends, p), step_change(vals, ends, p + W)
            ev_rows.append({
                "W": W, "event": tau.date(),
                "entry_change": d_in, "entry_pct": pct_rank(ref, d_in),
                "exit_date": idx[p + W].date() if p + W < len(idx) else None,
                "exit_change": d_out, "exit_pct": pct_rank(ref, d_out),
            })
        mx = vix.rolling(W, min_periods=W // 2).max().reindex(dbar.index)
        df = pd.DataFrame({"d": vals, "max": mx.values,
                           "now": vix.reindex(dbar.index).values}).dropna()
        r2_rows.append({"W": W, "n": len(df),
                        "r2_max_vix_window": np.corrcoef(df["d"], df["max"])[0, 1] ** 2,
                        "r2_current_vix": np.corrcoef(df["d"], df["now"])[0, 1] ** 2})

        # observed vs simulated path around the 2020 event
        pc = idx.get_loc(covid)
        rel_obs = ends - pc
        keep = (rel_obs >= -200) & (rel_obs <= W + 250)
        rel_sim, d_sim = simulate_paths(arma, profile, W, rng)
        for r, v in zip(rel_obs[keep], vals[keep]):
            path_rows.append({"W": W, "kind": "observed", "rel_day": int(r), "d_bar": v})
        for r, v in zip(rel_sim, d_sim):
            path_rows.append({"W": W, "kind": "simulated", "rel_day": int(r), "d_bar": v})
        po, ps = phase_means(rel_obs, vals, W), phase_means(rel_sim, d_sim, W)
        sim_steps = np.array([step_change(d_sim, rel_sim, int(e)) for e in rel_sim])
        sim_rows.append({
            "W": W,
            **{f"obs_{k}": v for k, v in po.items()},
            **{f"sim_{k}": v for k, v in ps.items()},
            "obs_rise": po["plateau"] - po["baseline"],
            "sim_rise": ps["plateau"] - ps["baseline"],
            "sim_largest_drop_rel_day": int(rel_sim[np.nanargmin(sim_steps)]),
        })
        print(f"  W={W}: R2 max-VIX {r2_rows[-1]['r2_max_vix_window']:.2f} vs current "
              f"{r2_rows[-1]['r2_current_vix']:.2f}; rise obs {sim_rows[-1]['obs_rise']:+.3f} "
              f"sim {sim_rows[-1]['sim_rise']:+.3f}; sim largest drop at day "
              f"{sim_rows[-1]['sim_largest_drop_rel_day']}")

    ev, r2, sim = pd.DataFrame(ev_rows), pd.DataFrame(r2_rows), pd.DataFrame(sim_rows)
    paths = pd.DataFrame(path_rows)
    for name, df in (("events", ev), ("r2", r2), ("sim", sim), ("paths", paths)):
        df.to_csv(INTERM / f"window_inclusion_{name}.csv", index=False)
    print("\n" + ev.round(3).to_string(index=False))
    print("\n" + r2.round(3).to_string(index=False))
    print("\n" + sim.round(3).to_string(index=False))

    write_table(ev, r2, sim, arma, TABLES / "table14_window_inclusion.tex")
    write_figure(log_rv, events, paths, FIGS / "fig10_window_inclusion.pdf")


def write_table(ev, r2, sim, arma, fp: Path) -> None:
    with open(fp, "w", encoding="utf-8") as f:
        f.write("% Table 14: window inclusion (module 14)\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{What the rolling persistence state measures}\n")
        f.write("\\label{tab:window_inclusion}\n\\small\n")
        f.write("\\begin{tabular}{llrrlrr}\n\\toprule\n")
        f.write("\\multicolumn{7}{l}{\\emph{Panel A: change in $\\bar d_t(W)$ when the event day "
                "enters and leaves the window (percentile among all changes)}}\\\\\n")
        f.write("$W$ & Event & Entry & (pct) & Exit date & Exit & (pct) \\\\\n\\midrule\n")
        for _, r in ev.iterrows():
            f.write(f"{r.W} & {r.event} & {r.entry_change:+.3f} & ({r.entry_pct:.0f}) & "
                    f"{r.exit_date} & {r.exit_change:+.3f} & ({r.exit_pct:.0f}) \\\\\n")
        f.write("\\midrule\n\\multicolumn{7}{l}{\\emph{Panel B: $R^2$ of $\\bar d_t(W)$ on the "
                "maximum VIX in the same window and on the current VIX}}\\\\\n")
        f.write("$W$ & & Max VIX & & & Current VIX & \\\\\n\\midrule\n")
        for _, r in r2.iterrows():
            f.write(f"{r.W} & & {r.r2_max_vix_window:.2f} & & & {r.r2_current_vix:.2f} & \\\\\n")
        f.write("\\midrule\n\\multicolumn{7}{l}{\\emph{Panel C: 2020 episode, observed versus "
                "short-memory null (baseline, plateau, after exit)}}\\\\\n")
        f.write("$W$ & & Observed & & & Null & \\\\\n\\midrule\n")
        for _, r in sim.iterrows():
            f.write(f"{r.W} & & {r.obs_baseline:.2f} / {r.obs_plateau:.2f} / {r.obs_after_exit:.2f} & & & "
                    f"{r.sim_baseline:.2f} / {r.sim_plateau:.2f} / {r.sim_after_exit:.2f} & \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\begin{tablenotes}\\small\n\\item Notes: $\\bar d_t(W)$ is the cross-sectional "
                "mean of the rolling GPH estimate on log Parkinson variance, window of $W$ trading "
                "days ending on day $t$, bandwidth $W^{0.65}$, weekly grid. Events are the three "
                "largest days of cross-sectional mean log variance at least 250 days apart. Entry "
                "and exit changes compare the four grid points after with the four before. The "
                "null is an ARMA(1,1) with median calm-period (2012--2019) parameters "
                f"($\\phi={arma[0]:.2f}$, $\\theta={arma[1]:.2f}$), with the observed one-year 2020 "
                f"excess log variance added; mean over {N_REPS} replications.\n")
        f.write("\\end{tablenotes}\n\\end{table}\n")
    print(f"  saved {fp.name}")


def write_figure(log_rv, events, paths, fp: Path) -> None:
    colors = {500: "#2a78d6", 750: "#0b0b0b", 1000: "#eb6834"}
    covid = [d for d in events if d.year == 2020][0]
    idx = log_rv.index
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    obs = paths[paths["kind"] == "observed"]
    for W in WINDOWS:
        s = obs[obs["W"] == W]
        p = idx.get_loc(covid)
        dates = idx[np.clip(p + s["rel_day"].values, 0, len(idx) - 1)]
        ax.plot(dates, s["d_bar"].values, color=colors[W], lw=1.6, label=f"$W={W}$")
        if p + W < len(idx):
            ax.axvline(idx[p + W], color=colors[W], lw=1, ls="--")
    ax.axvline(covid, color="grey", lw=1)
    ax.set_ylabel(r"cross-sectional mean $\hat d_t$")
    ax.set_title(f"Observed: exits at {covid.date()} + $W$ (dashed)")
    ax.legend(frameon=False)
    ax = axes[1]
    W = 750
    for kind, ls in (("observed", "-"), ("simulated", "--")):
        s = paths[(paths["W"] == W) & (paths["kind"] == kind)]
        ax.plot(s["rel_day"], s["d_bar"], ls=ls, color="#0b0b0b" if kind == "observed" else "#eb6834",
                lw=1.6, label="observed" if kind == "observed" else "short-memory null + 2020 episode")
    for x in (0, W):
        ax.axvline(x, color="grey", lw=1)
    ax.set_xlabel("trading days since the 2020 event (window end)")
    ax.set_title("$W=750$: observed versus short-memory null")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fp.name}")


if __name__ == "__main__":
    main()
