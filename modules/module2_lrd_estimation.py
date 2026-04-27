"""
MODULE 2: LRD and Roughness Estimation (v2)
============================================
Estimators implemented:
  - Geweke-Porter-Hudak (GPH) log-periodogram estimator
  - Local Whittle (LW) semiparametric estimator
  - Roughness / Hurst exponent via the scaling method on log realized variance

Outputs:
  - Cross-sectional one-shot estimates (Table 3, by sector)
  - Rolling weekly panels of (d_GPH, d_LW, H) per stock — saved as
    intermediate CSVs that feed Module 3 (feature engineering)
  - Figure 2 with four panels: distribution of d, rolling d for sample stocks,
    cross-sectional mean / dispersion of d over time, and d vs VIX.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from modules.io_v2 import build_clean_panel, sector_map

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
TABLES_DIR = BASE / "results" / "tables"
FIGURES_DIR = BASE / "results" / "figures"
INTERMEDIATE_DIR = BASE / "results" / "intermediate"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (12, 8), "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13, "font.family": "serif",
})

BANDWIDTH_POWER = 0.65
ROLLING_WINDOW = 750
MIN_OBS = 250
ROLLING_STRIDE = 5
HURST_LAGS = np.array([1, 2, 3, 5, 8, 13, 21], dtype=int)
HURST_Q = 2.0


def _periodogram(x: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    T = len(x)
    xd = x - x.mean()
    fft_x = np.fft.fft(xd)
    j = np.arange(1, m + 1)
    I = np.abs(fft_x[j]) ** 2 / (2 * np.pi * T)
    lam = 2 * np.pi * j / T
    return I, lam


def gph(x: np.ndarray, m: int | None = None) -> tuple[float, float, float]:
    x = x[~np.isnan(x)]
    T = len(x)
    if T < MIN_OBS:
        return np.nan, np.nan, np.nan
    if m is None:
        m = int(np.floor(T ** BANDWIDTH_POWER))
    I, lam = _periodogram(x, m)
    y = np.log(I + 1e-12)
    X = np.log(4 * np.sin(lam / 2) ** 2)
    Xd = X - X.mean()
    yd = y - y.mean()
    beta = (Xd * yd).sum() / (Xd ** 2).sum()
    d_hat = -beta / 2.0
    se = np.pi / np.sqrt(24 * m)
    p = 2 * (1 - stats.norm.cdf(abs(d_hat / se)))
    return d_hat, se, p


def local_whittle(x: np.ndarray, m: int | None = None) -> tuple[float, float, float]:
    x = x[~np.isnan(x)]
    T = len(x)
    if T < MIN_OBS:
        return np.nan, np.nan, np.nan
    if m is None:
        m = int(np.floor(T ** BANDWIDTH_POWER))
    I, lam = _periodogram(x, m)
    log_lam = np.log(lam)
    mean_log_lam = log_lam.mean()

    def Q(d: float) -> float:
        G = (lam ** (2 * d) * I).mean()
        return np.inf if G <= 0 else np.log(G) - 2 * d * mean_log_lam

    grid = np.linspace(-0.4, 0.8, 50)
    d0 = grid[int(np.argmin([Q(d) for d in grid]))]
    try:
        d_hat = minimize_scalar(
            Q, bounds=(max(-0.49, d0 - 0.2), min(0.99, d0 + 0.2)),
            method="bounded").x
    except Exception:
        d_hat = d0
    se = 1.0 / (2 * np.sqrt(m))
    p = 2 * (1 - stats.norm.cdf(abs(d_hat / se)))
    return d_hat, se, p


def hurst_scaling(x: np.ndarray, lags: np.ndarray = HURST_LAGS,
                  q: float = HURST_Q) -> float:
    """Estimate Hurst exponent of x via the scaling of qth absolute moments
    of increments. For a self-similar process,
        m(q, Δ) = E|x(t+Δ) - x(t)|^q ∝ Δ^{qH}.
    Slope of log m vs log Δ divided by q gives H. Used here on log RV."""
    x = x[~np.isnan(x)]
    if len(x) < lags.max() + MIN_OBS:
        return np.nan
    log_lags, log_m = [], []
    for lag in lags:
        diff = np.abs(x[lag:] - x[:-lag])
        if len(diff) == 0:
            continue
        log_lags.append(np.log(lag))
        log_m.append(np.log((diff ** q).mean() + 1e-30))
    if len(log_lags) < 3:
        return np.nan
    slope, _ = np.polyfit(log_lags, log_m, 1)
    return slope / q


def cross_sectional_estimates(data: pd.DataFrame,
                              estimator: callable) -> pd.DataFrame:
    rows = {}
    for col in data.columns:
        x = data[col].dropna().values
        if len(x) < MIN_OBS:
            continue
        d, se, p = estimator(x)
        rows[col] = {"d_hat": d, "se": se, "p_value": p, "T": len(x)}
    return pd.DataFrame(rows).T


def cross_sectional_hurst(data: pd.DataFrame) -> pd.Series:
    return pd.Series({c: hurst_scaling(data[c].dropna().values)
                      for c in data.columns}, name="H")


def rolling_panel(data: pd.DataFrame, estimator: callable,
                  window: int = ROLLING_WINDOW,
                  stride: int = ROLLING_STRIDE,
                  label: str = "") -> pd.DataFrame:
    """Rolling cross-sectional estimates. Returns (T_sample x N) DataFrame."""
    T, N = data.shape
    end_indices = np.arange(window, T, stride)
    sample_dates = data.index[end_indices]
    out = np.full((len(end_indices), N), np.nan)
    print(f"  {label}: {len(end_indices)} windows x {N} stocks "
          f"(window={window}, stride={stride})")
    for k, end in enumerate(end_indices):
        if k % 50 == 0 and k > 0:
            print(f"    {k}/{len(end_indices)}  ({sample_dates[k].date()})")
        block = data.iloc[end - window:end].values
        for j in range(N):
            x = block[:, j]
            x = x[~np.isnan(x)]
            if len(x) < MIN_OBS:
                continue
            try:
                if estimator is hurst_scaling:
                    out[k, j] = hurst_scaling(x)
                else:
                    out[k, j], _, _ = estimator(x)
            except Exception:
                pass
    return pd.DataFrame(out, index=sample_dates, columns=data.columns)


def export_table3(returns_gph: pd.DataFrame, returns_lw: pd.DataFrame,
                  rv_gph: pd.DataFrame, rv_lw: pd.DataFrame,
                  hurst: pd.Series, sectors: dict, fp: Path) -> None:
    for df, _ in [(returns_gph, "rg"), (returns_lw, "rl"),
                  (rv_gph, "vg"), (rv_lw, "vl")]:
        df["Sector"] = df.index.map(lambda t: sectors.get(t, "Other"))

    sec_index = sorted(set(returns_gph["Sector"]))

    def by_sector(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby("Sector")["d_hat"].agg(["mean", "std"]).round(3)

    rg = by_sector(returns_gph); rl = by_sector(returns_lw)
    vg = by_sector(rv_gph); vl = by_sector(rv_lw)
    sig_v = rv_gph.groupby("Sector").apply(
        lambda g: (g["p_value"] < 0.05).mean() * 100, include_groups=False)
    n_per_sector = rv_gph.groupby("Sector").size()

    hurst_with_sec = pd.DataFrame({"H": hurst})
    hurst_with_sec["Sector"] = hurst_with_sec.index.map(
        lambda t: sectors.get(t, "Other"))
    h_by = hurst_with_sec.groupby("Sector")["H"].agg(["mean", "std"]).round(3)

    with open(fp, "w") as f:
        f.write("% Table 3: LRD and Roughness Estimates by GICS Sector\n\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Long-Range Dependence and Roughness Estimates by Sector}\n")
        f.write("\\label{tab:lrd_estimates}\n\\small\n")
        f.write("\\begin{tabular}{lccccccc}\n\\toprule\n")
        f.write(" & \\multicolumn{2}{c}{Returns} "
                "& \\multicolumn{3}{c}{Parkinson RV} "
                "& Roughness & \\\\\n")
        f.write("\\cmidrule(lr){2-3} \\cmidrule(lr){4-6}\n")
        f.write("Sector & $\\bar d_{GPH}$ & $\\bar d_{LW}$ "
                "& $\\bar d_{GPH}$ & $\\bar d_{LW}$ & \\% Sig "
                "& $\\bar H$ & N \\\\\n\\midrule\n")

        for s in sec_index:
            n = int(n_per_sector.get(s, 0))
            f.write(
                f"{s} & {rg.loc[s,'mean']:.3f} & {rl.loc[s,'mean']:.3f} "
                f"& {vg.loc[s,'mean']:.3f} & {vl.loc[s,'mean']:.3f} "
                f"& {sig_v.loc[s]:.0f}\\% "
                f"& {h_by.loc[s,'mean']:.3f} & {n} \\\\\n"
            )

        f.write("\\midrule\n")
        f.write(
            f"\\textbf{{Overall}} "
            f"& {returns_gph['d_hat'].mean():.3f} & {returns_lw['d_hat'].mean():.3f} "
            f"& {rv_gph['d_hat'].mean():.3f} & {rv_lw['d_hat'].mean():.3f} "
            f"& {(rv_gph['p_value']<0.05).mean()*100:.0f}\\% "
            f"& {hurst.mean():.3f} & {len(rv_gph)} \\\\\n"
        )
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write(
            "\\item Notes: $d$ is the fractional differencing parameter. "
            "GPH is the Geweke--Porter--Hudak log-periodogram estimator "
            f"(bandwidth $m = T^{{{BANDWIDTH_POWER}}}$); "
            "LW is the local-Whittle semiparametric estimator. "
            "Returns are daily log returns; Parkinson RV is the range-based "
            "realized variance from daily H/L. $H$ is the Hurst exponent of "
            "$\\log\\mathrm{RV}^{PK}$ estimated by the scaling of the $q=2$ "
            "moment of increments over lags $\\{1,2,3,5,8,13,21\\}$. "
            "$H<0.5$ indicates rough behaviour. \\% Sig reports the share of "
            "stocks whose volatility $d_{GPH}$ is significant at 5\\%.\n"
        )
        f.write("\\end{tablenotes}\n\\end{table}\n")


def figure2(rv_gph: pd.DataFrame, rolling_d: pd.DataFrame,
            rolling_h: pd.DataFrame, market: pd.DataFrame, fp: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) cross-sectional distribution of d (volatility, GPH)
    ax = axes[0, 0]
    d = rv_gph["d_hat"].dropna()
    ax.hist(d, bins=30, density=True, alpha=0.7, color="darkred", edgecolor="black")
    ax.axvline(d.mean(), color="black", linestyle="--", linewidth=2,
               label=f"Mean = {d.mean():.3f}")
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel(r"$\hat d$ (memory parameter)")
    ax.set_ylabel("Density")
    ax.set_title(r"(a) Distribution of $\hat d$ for Parkinson RV (GPH)",
                 fontweight="bold")
    ax.legend(); ax.set_xlim(-0.1, 0.7)

    # (b) rolling d for a few stocks
    ax = axes[0, 1]
    sample = [t for t in ["AAPL", "JPM", "XOM", "JNJ", "PG"] if t in rolling_d.columns]
    for t in sample:
        ax.plot(rolling_d.index, rolling_d[t], label=t, alpha=0.85, linewidth=1)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel(r"$\hat d_t$")
    ax.set_title(f"(b) Rolling $\\hat d$ on Parkinson RV "
                 f"(window {ROLLING_WINDOW}, stride {ROLLING_STRIDE})",
                 fontweight="bold")
    ax.legend(loc="upper right", ncol=3)
    ax.axvspan(pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-30"),
               alpha=0.15, color="gray")
    ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30"),
               alpha=0.15, color="gray")

    # (c) cross-sectional mean and dispersion of d over time
    ax = axes[1, 0]
    cs_mean = rolling_d.mean(axis=1)
    cs_std = rolling_d.std(axis=1)
    ax.plot(cs_mean.index, cs_mean, color="navy", linewidth=1.5,
            label=r"Mean $\hat d_t$")
    ax.fill_between(cs_mean.index, cs_mean - cs_std, cs_mean + cs_std,
                    alpha=0.25, color="navy", label=r"$\pm 1$ std")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel(r"Cross-sectional $\hat d_t$")
    ax.set_title("(c) Cross-Sectional Memory Mean and Dispersion",
                 fontweight="bold")
    ax.legend()
    ax.axvspan(pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-30"),
               alpha=0.15, color="gray")
    ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30"),
               alpha=0.15, color="gray")

    # (d) cross-sectional mean d vs VIX
    ax = axes[1, 1]
    vix = market["VIX"].reindex(cs_mean.index)
    mask = vix.notna() & cs_mean.notna()
    ax.scatter(vix[mask], cs_mean[mask], alpha=0.5, s=20, c="darkred")
    if mask.sum() > 10:
        z = np.polyfit(vix[mask], cs_mean[mask], 1)
        xs = np.linspace(vix[mask].min(), vix[mask].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), "b--", linewidth=2)
        rho = np.corrcoef(vix[mask], cs_mean[mask])[0, 1]
        ax.text(0.05, 0.95, fr"$\rho$ = {rho:.3f}", transform=ax.transAxes,
                fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
    ax.set_xlabel("VIX")
    ax.set_ylabel(r"Cross-sectional mean $\hat d_t$")
    ax.set_title(r"(d) Memory Parameter vs VIX", fontweight="bold")

    plt.tight_layout()
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.savefig(str(fp).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("=" * 70)
    print("   MODULE 2: LRD AND ROUGHNESS ESTIMATION")
    print("=" * 70)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_clean_panel()
    sectors = sector_map(panel.metadata)
    print(f"  N={len(panel.kept)}  T={len(panel.prices)}")

    print("\n[1/5] Cross-sectional GPH on returns...")
    returns_gph = cross_sectional_estimates(panel.returns, gph)
    print(f"   mean d = {returns_gph['d_hat'].mean():.3f}, "
          f"% sig = {(returns_gph['p_value']<0.05).mean()*100:.1f}%")

    print("[2/5] Cross-sectional Local Whittle on returns...")
    returns_lw = cross_sectional_estimates(panel.returns, local_whittle)
    print(f"   mean d = {returns_lw['d_hat'].mean():.3f}, "
          f"% sig = {(returns_lw['p_value']<0.05).mean()*100:.1f}%")

    print("[3/5] Cross-sectional GPH on Parkinson RV...")
    rv_gph = cross_sectional_estimates(panel.rv_parkinson, gph)
    print(f"   mean d = {rv_gph['d_hat'].mean():.3f}, "
          f"% sig = {(rv_gph['p_value']<0.05).mean()*100:.1f}%")

    print("[4/5] Cross-sectional Local Whittle on Parkinson RV...")
    rv_lw = cross_sectional_estimates(panel.rv_parkinson, local_whittle)
    print(f"   mean d = {rv_lw['d_hat'].mean():.3f}, "
          f"% sig = {(rv_lw['p_value']<0.05).mean()*100:.1f}%")

    print("[5/5] Roughness (Hurst) on log Parkinson RV...")
    hurst = cross_sectional_hurst(panel.log_rv)
    print(f"   mean H = {hurst.mean():.3f}, std = {hurst.std():.3f}, "
          f"share H<0.5 = {(hurst<0.5).mean()*100:.1f}%")

    print("\n[Rolling] panels (this is the slow part)...")
    rolling_d_gph = rolling_panel(panel.rv_parkinson, gph, label="d_GPH on RV_PK")
    rolling_d_lw = rolling_panel(panel.rv_parkinson, local_whittle, label="d_LW on RV_PK")
    rolling_h = rolling_panel(panel.log_rv, hurst_scaling, label="H on log_RV")

    print("\nSaving intermediate panels...")
    returns_gph.to_csv(INTERMEDIATE_DIR / "lrd_returns_gph.csv")
    returns_lw.to_csv(INTERMEDIATE_DIR / "lrd_returns_lw.csv")
    rv_gph.to_csv(INTERMEDIATE_DIR / "lrd_rv_gph.csv")
    rv_lw.to_csv(INTERMEDIATE_DIR / "lrd_rv_lw.csv")
    hurst.to_csv(INTERMEDIATE_DIR / "hurst_rv_log.csv")
    rolling_d_gph.to_csv(INTERMEDIATE_DIR / "rolling_d_gph.csv")
    rolling_d_lw.to_csv(INTERMEDIATE_DIR / "rolling_d_lw.csv")
    rolling_h.to_csv(INTERMEDIATE_DIR / "rolling_hurst.csv")

    print("\nExporting Table 3 + Figure 2...")
    export_table3(returns_gph, returns_lw, rv_gph, rv_lw, hurst, sectors,
                  TABLES_DIR / "table3_lrd_estimates.tex")
    figure2(rv_gph, rolling_d_gph, rolling_h, panel.market,
            FIGURES_DIR / "fig2_lrd_estimates.pdf")

    print("\n" + "=" * 70)
    print("Outputs:")
    print(f"  {TABLES_DIR/'table3_lrd_estimates.tex'}")
    print(f"  {FIGURES_DIR/'fig2_lrd_estimates.pdf'} (+ .png)")
    print(f"  intermediate panels: rolling_d_gph.csv, rolling_d_lw.csv, "
          "rolling_hurst.csv (+ static cross-sectional CSVs)")


if __name__ == "__main__":
    main()
