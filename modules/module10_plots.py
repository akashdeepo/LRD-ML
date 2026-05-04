"""
MODULE 10: Plots for Section 8.2 / 8.4 / 8.5
============================================
Produces five figures from existing forecast and feature panels:

  fig4_cumulative_loss.pdf   item 4: cumulative SSE_A - SSE_C over time
  fig5_mse_improvement.pdf   items 5+6: bar chart of layer contributions (with
                             non-additivity caveat overlay)
  fig6_stock_distribution.pdf item 7: per-stock %-improvement histogram + box
  fig7_leadlag_d_vix.pdf     item 8: cross-correlation rho(d_t, VIX_{t+k})
  fig8_sector_heatmap.pdf    item 9: sector x horizon improvement heatmap

The bar chart deliberately reports "leave-one-block-in" (each layer's gain
over A) rather than naive additive contributions, since the blocks are
correlated. A Shapley decomposition would require fitting all 2^5 subsets,
which we do not produce; the layered-marginal numbers are the principled,
honest substitute.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.forecast_io import load_bundle

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
FCST = BASE / "results" / "intermediate" / "forecasts"
INTERM = BASE / "results" / "intermediate"
FIG_DIR = BASE / "results" / "figures"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 12.5,
    "axes.labelsize": 13.5,
    "axes.titlesize": 14.5,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11.5,
    "font.family": "serif",
})

LADDER = ["A1", "A2", "A3", "A4", "A5", "C"]
LAYER_LABEL = {
    "A1": r"$A_1$: HAR-X",
    "A2": r"$A_2$: own $\hat d$ block",
    "A3": r"$A_3$: cross-sec $\hat d$",
    "A4": r"$A_4$: sector-mean $\hat d$",
    "A5": r"$A_5$: $\hat d$ × stress",
    "C":  r"$C$: full union",
}
HORIZONS = [1, 5, 22]

LAYER_COLORS = {
    "A1": "#2E86AB", "A2": "#A23B72", "A3": "#F18F01",
    "A4": "#3CB371", "A5": "#8E44AD", "C": "#C0392B",
}


# ----------------------------------------------------- helpers
def _load_yhat_y(model: str, h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    yhat = pd.read_csv(FCST / f"{model}_h{h:02d}_yhat.csv",
                       index_col=0, parse_dates=True)
    y = pd.read_csv(FCST / f"{model}_h{h:02d}_y.csv",
                    index_col=0, parse_dates=True)
    return yhat, y


# ------------------------------------------------------- fig 4
def fig4_cumulative_loss(fp: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    handles_for_legend = []
    for i, (ax, h) in enumerate(zip(axes, HORIZONS)):
        yhat_A, y_A = _load_yhat_y("A", h)
        yhat_C, y_C = _load_yhat_y("C", h)
        L_A = ((yhat_A - y_A) ** 2).mean(axis=1)
        L_C = ((yhat_C - y_C) ** 2).mean(axis=1)
        diff = (L_A - L_C).dropna()
        cum = diff.cumsum()
        line, = ax.plot(cum.index, cum.values, lw=1.6, color="#1F4E79",
                        label=r"$\sum_t (L_{A,t}-L_{C,t})$")
        ax.axhline(0, color="black", lw=0.6)
        # Distinct, well-separated crisis colours: GFC slate-blue, COVID red.
        bands = [
            ("2008-09-01", "2009-12-31", "GFC",   "#5B6E91", 0.22),
            ("2020-03-01", "2020-12-31", "COVID", "#E74C3C", 0.22),
        ]
        for start, end, label, color, alpha in bands:
            sp = ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                            alpha=alpha, color=color, label=label)
            if i == 0:
                handles_for_legend.append(sp)
        ax.set_title(rf"$h={h}$")
        ax.set_xlabel("Date")
        if ax is axes[0]:
            ax.set_ylabel(r"$\sum_t \left[ L_{A,t} - L_{C,t} \right]$")

    # one shared legend at the bottom — does not collide with any line
    fig.legend(handles=handles_for_legend, labels=["GFC", "COVID"],
               loc="lower center", ncol=2, frameon=True, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Cumulative loss differential of Model C against Model A "
        "(rising line = C beats A)", y=1.02
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fp.name}")


# ------------------------------------------------------- fig 5
def fig5_mse_improvement(fp: Path) -> None:
    raw = pd.read_csv(INTERM / "table5_raw.csv")
    pivot = raw.pivot_table(index="model", columns="h",
                            values="imp_vs_A_pct").reindex(LADDER)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    width = 0.26
    xs = np.arange(len(LADDER))
    for i, h in enumerate(HORIZONS):
        offsets = (i - 1) * width
        bars = ax.bar(xs + offsets, pivot[h].values, width,
                      label=rf"$h={h}$",
                      color=plt.cm.viridis(0.15 + 0.30 * i),
                      edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, pivot[h].values):
            if pd.isna(v):
                continue
            # symmetric vertical offset; small near-zero values get an
            # unambiguous downward placement so they don't collide with the
            # zero line tick or with the larger neighbouring bar's label.
            offset = 0.20 if v >= 0 else -0.22
            ax.text(b.get_x() + b.get_width() / 2,
                    v + offset,
                    f"{v:+.1f}%",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=10,
                    fontweight="medium")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([LAYER_LABEL[m] for m in LADDER], rotation=10)
    ax.set_ylabel("% MSE improvement vs Model A")
    ax.set_title("Layered ablation: each block's gain over the HAR baseline",
                 pad=22)
    ymax = pivot.values[~np.isnan(pivot.values)].max()
    ymin = pivot.values[~np.isnan(pivot.values)].min()
    ax.set_ylim(ymin - 1.5, ymax + 2.2)
    # Three-column legend above the plot, between the title and the bars.
    ax.legend(loc="upper center", ncol=3, frameon=True, fontsize=10,
              bbox_to_anchor=(0.5, 1.06))
    # No in-figure footer text: the LaTeX caption already explains the
    # non-additivity / synergy point, so an in-figure repeat is redundant
    # and tends to overflow the axis.
    fig.tight_layout()
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fp.name}")


# ------------------------------------------------------- fig 6 (stock dist)
def fig6_stock_distribution(fp: Path) -> None:
    df = pd.read_csv(INTERM / "per_stock_improvement.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    for m, color in zip(["A1", "C", "D_lasso"],
                         ["#2E86AB", "#C0392B", "#27AE60"]):
        sub = df[(df["model"] == m) & (df["h"] == 5)]["imp_pct"].values
        sub = sub[~np.isnan(sub)]
        ax.hist(sub, bins=30, alpha=0.55, color=color, edgecolor="black",
                linewidth=0.5,
                label=f"{m}  ({(sub>0).mean()*100:.0f}% beat A; "
                      f"median {np.median(sub):+.2f}%)")
    ax.axvline(0, color="black", lw=1, linestyle="--")
    ax.set_xlabel(r"% MSE improvement vs A (per stock, $h=5$)")
    ax.set_ylabel("Number of stocks")
    ax.set_title("Per-stock improvement distribution")
    ax.legend(loc="upper left", fontsize=10.5)

    ax = axes[1]
    plot_models = ["A1", "A2", "A3", "A4", "A5", "C", "D_lasso", "D_rf", "D_gbm"]
    box_data = [df[(df["model"] == m) & (df["h"] == 5)]["imp_pct"].dropna().values
                for m in plot_models]
    bp = ax.boxplot(box_data, labels=plot_models, patch_artist=True,
                    showfliers=False)
    colors = (["#2E86AB"] + ["#A23B72"] * 4 + ["#C0392B"]
              + ["#27AE60", "#F39C12", "#8E44AD"])
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    ax.axhline(0, color="black", lw=0.7, linestyle="--")
    ax.set_ylabel(r"% MSE improvement vs A ($h=5$)")
    ax.set_title("Per-stock distribution by model")
    plt.setp(ax.get_xticklabels(), rotation=15)

    fig.suptitle(
        f"Per-stock forecast-improvement distribution (115 stocks, $h=5$)",
        y=1.02
    )
    fig.tight_layout()
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fp.name}")


# ------------------------------------------------------- fig 7 (lead-lag CCF)
def fig7_leadlag_d_vix(fp: Path) -> None:
    bundle = load_bundle()
    d = bundle.feat["d_gph"]                       # (T_sample x 115)
    vix = bundle.market["VIX"].reindex(d.index)
    d_bar = d.mean(axis=1).dropna()
    common = d_bar.index.intersection(vix.dropna().index)
    d_bar = d_bar.loc[common]
    vix = vix.loc[common]

    # CCF: rho( d_t , VIX_{t+k} ) with k in sample-stride steps.
    # Sample stride is 5 trading days, so k = +-12 steps ~ +-60 trading days.
    max_lag = 12
    lags = np.arange(-max_lag, max_lag + 1)
    rhos = []
    n = len(d_bar)
    for k in lags:
        if k >= 0:
            x = d_bar.values[: n - k]
            y = vix.values[k:]
        else:
            x = d_bar.values[-k:]
            y = vix.values[: n + k]
        if len(x) < 50:
            rhos.append(np.nan); continue
        r = np.corrcoef(x, y)[0, 1]
        rhos.append(r)
    rhos = np.array(rhos)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(lags * 5, rhos, width=4, color="#2E86AB",
           edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.7)
    # Bartlett 2-sigma band for white noise
    sigma = 1.96 / np.sqrt(n)
    ax.axhline(sigma, color="grey", lw=0.7, linestyle="--")
    ax.axhline(-sigma, color="grey", lw=0.7, linestyle="--",
               label=r"$\pm 1.96/\sqrt{T}$")
    ax.axvline(0, color="black", lw=0.5, linestyle=":")
    ax.set_xlabel(r"Lag $k$ (trading days). $k>0$: $\hat d_t$ leads VIX$_{t+k}$")
    ax.set_ylabel(r"Cross-correlation $\rho(\bar{\hat d}_t, \mathrm{VIX}_{t+k})$")
    ax.set_title(
        r"Lead-lag of cross-sectional mean memory $\bar{\hat d}_t$ vs VIX"
    )
    txt = ("Caveat: $\\hat d_t$ uses a 750-day backward-looking window, "
           "so persistence at small lags reflects window smoothing.")
    ax.text(0.99, -0.22, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=10, style="italic", color="#555")
    ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fp.name}")


# ------------------------------------------------------- fig 8 (sector heatmap)
def fig8_sector_heatmap(fp: Path) -> None:
    raw = pd.read_csv(INTERM / "table8_raw.csv")
    sub = raw[raw["model"] == "C"]
    pivot = sub.pivot_table(index="sector", columns="h",
                            values="imp_vs_A_pct")
    pivot = pivot.reindex(columns=HORIZONS)
    # sort sectors by h=5 improvement descending for readability
    pivot = pivot.sort_values(5, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-max(abs(np.nanmin(pivot.values)),
                             abs(np.nanmax(pivot.values))),
                   vmax=max(abs(np.nanmin(pivot.values)),
                            abs(np.nanmax(pivot.values))))
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([rf"$h={h}$" for h in HORIZONS])
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.isna(v):
                continue
            ax.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                    fontsize=11.5,
                    color="white" if abs(v) > 8 else "black")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% MSE improvement vs Model A")
    ax.set_title("Model C improvement by GICS sector and forecast horizon")
    fig.tight_layout()
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fp.name}")


# ----------------------------------------------------------- main
def main() -> None:
    print("=" * 70)
    print("   MODULE 10: PLOTS")
    print("=" * 70)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig4_cumulative_loss(FIG_DIR / "fig4_cumulative_loss.pdf")
    fig5_mse_improvement(FIG_DIR / "fig5_mse_improvement.pdf")
    fig6_stock_distribution(FIG_DIR / "fig6_stock_distribution.pdf")
    fig7_leadlag_d_vix(FIG_DIR / "fig7_leadlag_d_vix.pdf")
    fig8_sector_heatmap(FIG_DIR / "fig8_sector_heatmap.pdf")
    print(f"\nAll figures saved under {FIG_DIR}")


if __name__ == "__main__":
    main()
