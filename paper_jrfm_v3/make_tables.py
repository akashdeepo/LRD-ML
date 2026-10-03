"""Compact tables for the JRFM manuscript, built from the pipeline outputs in
results/intermediate (run from the repo root after the pipeline):

    python paper_jrfm_v3/make_tables.py

Every number in these tables is read from a CSV written by a module; nothing is
typed by hand. Figures are copied from results/figures.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
INT = REPO / "results" / "intermediate"
OUT = Path(__file__).resolve().parent / "tables"
FIG_OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
FIG_OUT.mkdir(exist_ok=True)


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.10 else ""


def write(name: str, body: str) -> None:
    (OUT / f"{name}.tex").write_text(body, encoding="utf-8")
    print(f"  wrote tables/{name}.tex")


def begin(caption: str, label: str, cols: str, size: str = "\\small") -> str:
    return (f"\\begin{{table}}[H]\n\\caption{{{caption}}}\n\\label{{{label}}}\n{size}\n"
            f"\\begin{{tabularx}}{{\\textwidth}}{{{cols}}}\n\\toprule\n")


def end(notes: str) -> str:
    return (f"\\bottomrule\n\\end{{tabularx}}\n\\noindent{{\\footnotesize{{\\textit{{Notes:}} {notes}}}}}\n"
            f"\\end{{table}}\n")


# ---------------------------------------------------------------- Table 1
def table_persistence() -> None:
    rg, rl = pd.read_csv(INT / "lrd_rv_gph.csv", index_col=0), pd.read_csv(INT / "lrd_rv_lw.csv", index_col=0)
    qg, ql = pd.read_csv(INT / "lrd_returns_gph.csv", index_col=0), pd.read_csv(INT / "lrd_returns_lw.csv", index_col=0)
    hu = pd.read_csv(INT / "hurst_rv_log.csv", index_col=0).iloc[:, 0]
    rows = [
        ("Returns", qg["d_hat"], ql["d_hat"]),
        ("Log Parkinson variance", rg["d_hat"], rl["d_hat"]),
    ]
    b = begin("Full-sample memory estimates across the 115 stocks.", "tab:memory",
              "Xcccccc")
    b += ("Series & Estimator & Mean & 10th pct & Median & 90th pct & Share $>0.5$ \\\\\n\\midrule\n")
    for lab, g, l in rows:
        for est, s in (("GPH", g), ("LW", l)):
            q = s.quantile([0.1, 0.5, 0.9])
            b += (f"{lab if est == 'GPH' else ''} & {est} & {s.mean():.3f} & {q[0.1]:.3f} & {q[0.5]:.3f} & "
                  f"{q[0.9]:.3f} & {100 * (s > 0.5).mean():.0f}\\% \\\\\n")
    q = hu.quantile([0.1, 0.5, 0.9])
    b += (f"Log Parkinson variance & Hurst ($q=2$) & {hu.mean():.3f} & {q[0.1]:.3f} & {q[0.5]:.3f} & "
          f"{q[0.9]:.3f} & -- \\\\\n")
    write("t1_memory", b + end(
        "Daily data, 29 November 2001 to 21 April 2026 (6136 trading days). GPH: log-periodogram "
        "estimator of the fractional parameter $d$; LW: local Whittle; bandwidth $m=T^{0.65}$. "
        "Hurst: scaling of second moments of increments of log variance over lags 1 to 21 days. "
        "Percentiles are across stocks."))


# ---------------------------------------------------------------- Table 2
def table_window() -> None:
    ev = pd.read_csv(INT / "window_inclusion_events.csv")
    r2 = pd.read_csv(INT / "window_inclusion_r2.csv")
    sim = pd.read_csv(INT / "window_inclusion_sim.csv")
    b = begin("What the rolling persistence state tracks.", "tab:window", "lXXXXXX", "\\footnotesize")
    b += ("\\multicolumn{7}{l}{\\textit{Panel A. Change in $\\bar d_t(W)$ when the event day enters and "
          "leaves the window (percentile among all changes)}} \\\\\n\\midrule\n")
    b += "Event & $W$ & Entry & Pct & Exit date & Exit & Pct \\\\\n\\midrule\n"
    for tau, g in ev.groupby("event", sort=True):
        for i, r in enumerate(g.sort_values("W").itertuples()):
            b += (f"{tau if i == 0 else ''} & {r.W} & {r.entry_change:+.3f} & {r.entry_pct:.1f} & "
                  f"{r.exit_date} & {r.exit_change:+.3f} & {r.exit_pct:.1f} \\\\\n")
    b += ("\\midrule\n\\multicolumn{7}{l}{\\textit{Panel B. $R^2$ of $\\bar d_t(W)$ on the VIX}} \\\\\n\\midrule\n"
          "& $W$ & \\multicolumn{2}{c}{Max VIX in window} & & \\multicolumn{2}{c}{Current VIX} \\\\\n\\midrule\n")
    for r in r2.itertuples():
        b += f"& {r.W} & \\multicolumn{{2}}{{c}}{{{r.r2_max_vix_window:.2f}}} & & \\multicolumn{{2}}{{c}}{{{r.r2_current_vix:.2f}}} \\\\\n"
    b += ("\\midrule\n\\multicolumn{7}{l}{\\textit{Panel C. The 2020 episode: data versus a short-memory null}} \\\\\n"
          "\\midrule\n& $W$ & Before & Inside & After exit & Rise & Largest drop \\\\\n\\midrule\n")
    for r in sim.itertuples():
        b += (f"Data & {r.W} & {r.obs_baseline:.2f} & {r.obs_plateau:.2f} & {r.obs_after_exit:.2f} & "
              f"{r.obs_rise:+.2f} & \\\\\n")
        b += (f"Null & {r.W} & {r.sim_baseline:.2f} & {r.sim_plateau:.2f} & {r.sim_after_exit:.2f} & "
              f"{r.sim_rise:+.2f} & day {r.sim_largest_drop_rel_day} \\\\\n")
    write("t2_window", b + end(
        "$\\bar d_t(W)$ is the cross-sectional mean of the GPH estimate on log Parkinson variance over "
        "the $W$ trading days ending on day $t$, on the weekly grid. Events are the three largest days "
        "of cross-sectional mean log variance at least 250 days apart (rule fixed in advance). Entry and "
        "exit changes compare the four weekly estimates after the date with the four before; Pct is the "
        "percentile of that change among all such changes. Panel C: mean $\\bar d_t$ in the 120 days "
        "before the event, from day 60 to day $W-60$, and from day $W+20$ to $W+120$. The null is an "
        "ARMA(1,1) with the median parameters of per-stock fits to 2012--2019 log variance plus the "
        "observed one-year 2020 excess log variance, 200 replications; Largest drop is the day "
        "(relative to the event) of its largest four-week fall."))


# ---------------------------------------------------------------- Table 4
def table_ladder() -> None:
    t5 = pd.read_csv(INT / "table5_raw.csv")
    names = {"A": "HAR", "A1": "HAR-X", "A2": "HAR + own persistence",
             "A3": "HAR + cross-sectional state", "A4": "HAR + sector state",
             "A5": "HAR + $d$, VIX, MOVE, $d\\times$VIX, $d\\times$MOVE", "C": "Full model $C$"}
    b = begin("Out-of-sample accuracy relative to HAR.", "tab:ladder", "Xrrrrrr", "\\footnotesize")
    b += ("& \\multicolumn{2}{c}{$h=1$} & \\multicolumn{2}{c}{$h=5$} & \\multicolumn{2}{c}{$h=22$} \\\\\n"
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
          "Model & $\\Delta$MSE & DM & $\\Delta$MSE & DM & $\\Delta$MSE & DM \\\\\n\\midrule\n")
    for m, lab in names.items():
        sub = t5[t5["model"] == m].set_index("h")
        cells = []
        for h in (1, 5, 22):
            r = sub.loc[h]
            if m == "A":
                cells += [f"{r.MSE_logRV:.4f}", ""]
            else:
                cells += [f"{r.imp_vs_A_pct:+.2f}\\%", f"{r.DM_t_HLN:+.2f}{stars(r.DM_p)}"]
        b += lab + " & " + " & ".join(cells) + " \\\\\n"
    write("t4_ladder", b + end(
        "Pooled out-of-sample MSE of log mean future Parkinson variance, 646 weekly origins "
        "(17 June 2013 to 14 April 2026) $\\times$ 115 stocks, common cells. The HAR row reports its MSE; "
        "other rows the percentage reduction relative to HAR (positive is better) and the panel "
        "Harvey--Leybourne--Newbold Diebold--Mariano statistic. $^{*}$, $^{**}$, $^{***}$: 10\\%, 5\\%, 1\\%."))


# ---------------------------------------------------------------- Table 5
def table_harx() -> None:
    df = pd.read_csv(INT / "harx_tests.csv")
    names = {"A1cs": "HAR-X + cross-sectional state$^{\\dagger}$",
             "A1sec": "HAR-X + sector state$^{\\dagger}$",
             "A1mod": "HAR-X + state $\\times$ HAR terms$^{\\dagger}$",
             "C": "Full model $C$"}
    b = begin("Incremental accuracy of the persistence specifications relative to HAR-X.", "tab:harx", "Xcrrrrr", "\\footnotesize")
    b += ("Specification & $h$ & $\\Delta$MSE & DM & CW & CW $p$ (Holm) & GW $p$ \\\\\n\\midrule\n")
    for s, lab in names.items():
        for i, r in enumerate(df[df["spec"] == s].sort_values("h").itertuples()):
            b += (f"{lab if i == 0 else ''} & {r.h} & {r.gain_vs_harx_pct:+.2f}\\% & "
                  f"{r.dm_t_vs_harx:+.2f}{stars(r.dm_p_vs_harx)} & {r.cw_t_vs_harx:+.2f}{stars(r.cw_p_vs_harx)} & "
                  f"{r.cw_p_holm:.3f} & {r.gw_p:.3f} \\\\\n")
        b += "\\midrule\n" if s != "C" else ""
    write("t5_harx", b + end(
        "$\\Delta$MSE: percentage reduction in pooled MSE relative to HAR-X (positive favors the "
        "specification). DM: panel HLN Diebold--Mariano statistic (two-sided). CW: Clark--West "
        "statistic for the nested comparison (one-sided). Holm adjusts the nine tests of the three "
        "registered specifications ($^{\\dagger}$) jointly and the three tests of model $C$ jointly. GW: "
        "Giacomini--White conditional predictive ability test with instruments $(1, \\bar\\delta_{t-k})$, "
        "$k=\\lceil h/5\\rceil$. All statistics use the cross-sectional mean loss differential per date and a "
        "Newey--West bandwidth of $\\max(\\lceil h/5\\rceil-1,\\lfloor 4(T/100)^{2/9}\\rfloor)=6$."))


# ---------------------------------------------------------------- Table 6
def table_timing() -> None:
    df = pd.read_csv(INT / "timing_alignment.csv")
    b = begin("Information timing and the apparent value of implied volatility.", "tab:timing", "Xcrrr")
    b += "HAR inputs & $h$ & MSE HAR & MSE HAR-X & HAR-X gain (DM) \\\\\n\\midrule\n"
    for align, lab in (("stale", "Through $t-1$"), ("aligned", "Through $t$")):
        for i, r in enumerate(df[df["alignment"] == align].sort_values("h").itertuples()):
            b += (f"{lab if i == 0 else ''} & {r.h} & {r.mse_har:.4f} & {r.mse_harx:.4f} & "
                  f"{r.harx_gain_pct:+.2f}\\% ({r.dm_t:+.2f}) \\\\\n")
        b += "\\midrule\n"
    for i, r in enumerate(df[df["alignment"] == "har_aligned_vs_stale"].sort_values("h").itertuples()):
        b += (f"{'HAR, $t$ vs $t-1$' if i == 0 else ''} & {r.h} & & & "
              f"{r.harx_gain_pct:+.2f}\\% ({r.dm_t:+.2f}) \\\\\n")
    write("t6_timing", b + end(
        "VIX and MOVE are closes of the origin day $t$ in every row. Through $t-1$: HAR terms built from "
        "variance through the day before the origin and the previous day's return. Through $t$: the "
        "aligned specification used everywhere else. Common cells of all four forecast panels; DM "
        "statistics in parentheses."))


# ---------------------------------------------------------------- Table 7
def table_designs() -> None:
    p = pd.read_csv(INT / "candidates_run05_pooled.csv")
    d = pd.read_csv(INT / "candidates_run06_duration.csv")
    m = pd.read_csv(INT / "candidates_run07_market.csv")
    rows = [("Pooled, fixed effects", p, "P-C vs P-A1", "Full model vs HAR-X"),
            ("Pooled, fixed effects", p, "P-A1cs vs P-A1", "HAR-X + cross-sectional state vs HAR-X"),
            ("Duration target", d, "C vs A1 (duration target)", "Full model vs HAR-X"),
            ("Duration target", d, "A1cs vs A1 (duration target)", "HAR-X + cross-sectional state vs HAR-X"),
            ("Market level", m, "M-C vs M-A1", "Full market model vs market HAR-X")]
    b = begin("Pooled, term-structure and market-level designs.", "tab:designs", "lXcrrr", "\\footnotesize")
    b += "Design & Comparison & $h$ & $\\Delta$MSE & DM & CW $p$ (Holm) \\\\\n\\midrule\n"
    prev = None
    for design, df, comp, lab in rows:
        if prev is not None and design != prev:
            b += "\\midrule\n"
        for i, r in enumerate(df[df["comparison"] == comp].sort_values("h").itertuples()):
            hp = f"{r.cw_p_holm:.3f}" if np.isfinite(r.cw_p_holm) else "--"
            first = i == 0 and design != prev
            b += (f"{design if first else ''} & {lab if i == 0 else ''} & {r.h} & {r.gain_pct:+.2f}\\% & "
                  f"{r.dm_t:+.2f}{stars(r.dm_p)} & {hp} \\\\\n")
        prev = design
    write("t7_designs", b + end(
        "Pooled: one regression across stocks with stock fixed effects (within transformation on "
        "training means), refit at every origin. Duration target: log ratio of mean future variance over "
        "22 and 5 days. Market level: one series, the cross-sectional mean of the stock targets, with "
        "market HAR terms, VIX, MOVE and the persistence state. Each design was specified before its "
        "results were seen."))


# ---------------------------------------------------------------- Table 8
def table_robust() -> None:
    t9 = pd.read_csv(INT / "table9_raw.csv").set_index("variant")
    t5 = pd.read_csv(INT / "table5_raw.csv")
    ht = pd.read_csv(INT / "harx_tests.csv")
    lab = {"headline": "Headline (GPH, $W=750$)", "estimator_LW": "Local Whittle instead of GPH",
           "window_500": "Window 500 days", "window_1000": "Window 1000 days",
           "target_sqret": "Target: squared returns", "liquidity_high_illiq": "Less liquid half",
           "liquidity_low_illiq": "More liquid half", "benchmark_garch11": "GARCH(1,1) instead of model $C$"}
    b = begin("Robustness of model $C$ at $h=5$, and the machine-learning estimators.", "tab:robust", "Xrr", "\\footnotesize")
    b += "\\multicolumn{3}{l}{\\textit{Panel A. Model $C$ versus HAR at $h=5$}} \\\\\n\\midrule\n"
    b += "Variant & $\\Delta$MSE vs HAR & DM \\\\\n\\midrule\n"
    for v, l in lab.items():
        r = t9.loc[v]
        b += f"{l} & {r.imp_pct:+.2f}\\% & {r.HLN_DM_t:+.2f}{stars(r.p)} \\\\\n"
    b += ("\\midrule\n\\multicolumn{3}{l}{\\textit{Panel B. Model $D$: model $C$'s predictors, other estimators "
          "($\\Delta$MSE vs HAR-X at $h=1$ / $5$ / $22$)}} \\\\\n\\midrule\n")
    est = {"D_lasso": "Lasso", "D_ridge": "Ridge", "D_en": "Elastic net", "D_rf": "Random forest",
           "D_gbm": "Gradient boosting"}
    b += "Estimator & $\\Delta$MSE & DM \\\\\n\\midrule\n"
    for k, l in est.items():
        s = ht[ht["spec"] == k].sort_values("h")
        g = " / ".join(f"{x:+.1f}\\%" for x in s["gain_vs_harx_pct"])
        t = " / ".join(f"{x:+.1f}" for x in s["dm_t_vs_harx"])
        b += f"{l} & {g} & {t} \\\\\n"
    write("t8_robust", b + end(
        "Panel A: percentage MSE reduction of model $C$ relative to HAR at $h=5$ under each variant; the "
        "GARCH row compares a level-corrected GARCH(1,1) forecast with HAR. Panel B: shrinkage "
        "hyperparameters and gradient-boosting tree size, leaf size and number of trees chosen by "
        "time-series cross-validation inside each training window; the random forest uses 200 trees with "
        "a minimum leaf of 20; refit every 20 weeks."))


# ---------------------------------------------------------------- Table 9
def table_portfolios() -> None:
    t10 = pd.read_csv(INT / "table10_raw.csv")
    cost = pd.read_csv(INT / "table10b_costs.csv")
    sh = pd.read_csv(INT / "sharpe_tests.csv")
    order = ["Equal-weight (unmanaged)", "Trailing-22d-RV-managed", "HAR-managed", "HAR-X-managed", "C-managed"]
    lab = {"Equal-weight (unmanaged)": "Equal weight", "Trailing-22d-RV-managed": "Trailing 22-day variance",
           "HAR-managed": "HAR", "HAR-X-managed": "HAR-X", "C-managed": "Model $C$"}
    b = begin("Volatility-managed portfolios (excess returns, weekly, 2014--2026).", "tab:portfolios", "Xrrrrr", "\\footnotesize")
    b += "Portfolio & Sharpe & Sharpe, 10 bp & Turnover & Sharpe, COVID & Max drawdown \\\\\n\\midrule\n"
    c = cost.set_index(cost.columns[0])
    for p in order:
        full = t10[(t10["regime"] == "Full sample") & (t10["portfolio"] == p)].iloc[0]
        cov = t10[(t10["regime"] == "COVID (2020)") & (t10["portfolio"] == p)].iloc[0]
        cr = c.loc[p]
        tcol = [k for k in c.columns if "turn" in k.lower()][0]
        s10 = [k for k in c.columns if "10" in k][0]
        b += (f"{lab[p]} & {full.sharpe:.2f} & {cr[s10]:.2f} & {cr[tcol]:.1f} & {cov.sharpe:.2f} & "
              f"{100 * full.max_dd:.1f}\\% \\\\\n")
    fs = sh[sh["regime"] == "Full sample"].set_index("pair")
    cv = sh[sh["regime"] == "COVID (2020)"].set_index("pair")
    note_tests = (f"Sharpe differences, Ledoit--Wolf HAC $p$ (block bootstrap $p$): model $C$ vs HAR-X "
                  f"{fs.loc['C vs A1', 'hac_p']:.2f} ({fs.loc['C vs A1', 'boot_p']:.2f}) full sample, "
                  f"{cv.loc['C vs A1', 'hac_p']:.2f} ({cv.loc['C vs A1', 'boot_p']:.2f}) COVID; model $C$ vs "
                  f"equal weight {fs.loc['C vs Unmanaged', 'hac_p']:.2f} ({fs.loc['C vs Unmanaged', 'boot_p']:.2f}).")
    write("t9_portfolios", b + end(
        "Each stock's position is scaled by $c_{i,t}/\\hat\\sigma^2_{i,t+5|t}$ with $c_{i,t}$ set in real time "
        "so that the managed and unmanaged positions have equal past variance; positions are funded at the "
        "three-month Treasury bill rate; 594 weeks after a 52-week warm-up. Turnover: notional traded per "
        "unit of capital per year, including drift. COVID: March to December 2020 (43 weeks). " + note_tests))


def copy_figures() -> None:
    for f in ("fig10_window_inclusion.pdf",):
        shutil.copy(REPO / "results" / "figures" / f, FIG_OUT / f)
        print(f"  copied figures/{f}")


def figure_cumulative() -> None:
    """Cumulative cross-sectional mean squared-error differential on common
    cells: HAR minus HAR-X (the value of implied volatility) and HAR-X minus
    model C (the value of persistence beyond it)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F = INT / "forecasts"
    rd = lambda m, h, s: pd.read_csv(F / f"{m}_h{h:02d}_{s}.csv", index_col=0, parse_dates=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, h in zip(axes, (1, 5, 22)):
        L = {m: (rd(m, h, "yhat") - rd(m, h, "y")) ** 2 for m in ("A", "A1", "C")}
        mask = L["A"].notna() & L["A1"].notna() & L["C"].notna()
        L = {m: v.where(mask) for m, v in L.items()}
        for (a, b), lab, col in (((("A", "A1")), "HAR minus HAR-X", "#2a78d6"),
                                 ((("A1", "C")), "HAR-X minus model $C$", "#eb6834")):
            d = (L[a] - L[b]).mean(axis=1).dropna().cumsum()
            ax.plot(d.index, d.values, color=col, lw=1.6, label=lab)
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-12-31"), color="0.85", lw=0)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"$h={h}$")
        if h == 1:
            ax.set_ylabel("cumulative loss differential")
            ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "fig_cumulative.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/fig_cumulative.pdf")


if __name__ == "__main__":
    table_persistence()
    table_window()
    table_ladder()
    table_harx()
    table_timing()
    table_designs()
    table_robust()
    table_portfolios()
    copy_figures()
    figure_cumulative()
