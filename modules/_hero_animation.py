"""Hero visual for the README: the cross-sectional persistence state over time,
with the VIX beneath it on its own axis (no dual axis), revealed progressively
and written out as an animated GIF plus a static PNG of the final frame.

Reads the corrected rolling-GPH cross-sectional panel produced by module 3 and
the daily VIX from the clean panel. Run from the repo root:

    python -m modules._hero_animation
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "docs" / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- palette (dataviz reference instance, light mode) ------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"   # persistence state
SERIES_2 = "#eb6834"   # VIX

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.labelcolor": INK2, "axes.labelsize": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

# --- data --------------------------------------------------------------------
cs = pd.read_csv(BASE / "results" / "intermediate" / "features" / "feat_cross_section.csv",
                 index_col=0, parse_dates=True)
d_bar = cs["cs_mean_d"].dropna()
d_sd = cs["cs_std_d"].reindex(d_bar.index)
market = pd.read_csv(BASE / "bloomberg_pull" / "processed" / "clean_panel" / "market.csv",
                     index_col=0, parse_dates=True)
vix = market["VIX"].dropna()
vix = vix.loc[d_bar.index[0]:d_bar.index[-1]]

CALM = ("2013-01-01", "2014-12-31")
GFC = ("2008-07-01", "2009-12-31")
COVID = ("2020-03-01", "2020-12-31")
calm_level = d_bar.loc[CALM[0]:CALM[1]].mean()
gfc_level = d_bar.loc[GFC[0]:GFC[1]].mean()
covid_level = d_bar.loc[COVID[0]:COVID[1]].mean()
rho = np.corrcoef(d_bar.values, vix.reindex(d_bar.index, method="nearest").values)[0, 1]

# --- figure ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 5.9), dpi=100, sharex=True,
    gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12},
)
fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.09)

fig.text(0.075, 0.965, "Estimated volatility persistence rises in every major stress episode",
         fontsize=14, fontweight="semibold", color=INK, ha="left", va="top")
fig.text(0.075, 0.915,
         "Cross-sectional mean of the rolling long-memory parameter $\\hat d$ (GPH, 750-day window) "
         "of Parkinson variance\nacross 115 S&P 500 stocks, 2004–2026, with the VIX beneath on its own axis",
         fontsize=9.5, color=INK2, ha="left", va="top", linespacing=1.5)

for ax in (ax1, ax2):
    ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    for a, b in (GFC, COVID):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=INK, alpha=0.045, lw=0)

ax1.set_ylim(0.05, 0.75)
ax1.set_yticks([0.2, 0.4, 0.6])
ax1.set_ylabel("mean $\\hat d_t$ across stocks")
ax2.set_ylim(0, 90)
ax2.set_yticks([20, 40, 60, 80])
ax2.set_ylabel("VIX")
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.set_xlim(d_bar.index[0], d_bar.index[-1] + pd.Timedelta(days=200))

# crisis labels (context, in muted ink)
ax1.text(pd.Timestamp("2009-03-15"), 0.72, "GFC", color=MUTED, fontsize=9, ha="center", va="top")
ax1.text(pd.Timestamp("2020-08-01"), 0.72, "COVID", color=MUTED, fontsize=9, ha="center", va="top")

# calm baseline segment (a reference level, solid hairline)
ax1.hlines(calm_level, pd.Timestamp(CALM[0]), pd.Timestamp(CALM[1]),
           color=MUTED, linewidth=1)
ax1.text(pd.Timestamp("2014-01-01"), calm_level - 0.09,
         f"calm 2013–14: {calm_level:.2f}", color=MUTED, fontsize=8.5, ha="center", va="top")

(line1,) = ax1.plot([], [], color=SERIES_1, linewidth=2, solid_joinstyle="round",
                    solid_capstyle="round")
(line2,) = ax2.plot([], [], color=SERIES_2, linewidth=2, solid_joinstyle="round",
                    solid_capstyle="round")
band = [ax1.fill_between([], [], [], color=SERIES_1, alpha=0.10, lw=0)]
dot1 = ax1.scatter([], [], s=70, color=SERIES_1, edgecolor=SURFACE, linewidth=2, zorder=5)
dot2 = ax2.scatter([], [], s=70, color=SERIES_2, edgecolor=SURFACE, linewidth=2, zorder=5)
date_txt = fig.text(0.985, 0.955, "", fontsize=11, color=INK2, ha="right", va="top")
# selective direct labels, revealed once the cursor passes them
lab_gfc = ax1.text(pd.Timestamp("2009-03-15"), gfc_level + 0.105,
                   f"{gfc_level:.2f}  (+{100*(gfc_level/calm_level-1):.0f}% vs calm)",
                   color=INK2, fontsize=9, ha="center", va="bottom", visible=False)
lab_cov = ax1.text(pd.Timestamp("2020-08-01"), covid_level + 0.07,
                   f"{covid_level:.2f}  (+{100*(covid_level/calm_level-1):.0f}% vs calm)",
                   color=INK2, fontsize=9, ha="center", va="bottom", visible=False)
lab_rho = ax2.text(0.985, 0.90, f"corr. with mean $\\hat d_t$: {rho:+.2f}",
                   transform=ax2.transAxes, color=INK2, fontsize=9, ha="right", va="top",
                   visible=False)

N_FRAMES, HOLD = 140, 30
idx = np.linspace(1, len(d_bar), N_FRAMES).astype(int)
idx = np.concatenate([idx, np.repeat(len(d_bar), HOLD)])


def draw(frame: int):
    k = idx[frame]
    t = d_bar.index[k - 1]
    x = d_bar.index[:k]
    y = d_bar.values[:k]
    line1.set_data(x, y)
    band[0].remove()
    band[0] = ax1.fill_between(x, y - d_sd.values[:k], y + d_sd.values[:k],
                               color=SERIES_1, alpha=0.10, lw=0)
    v = vix.loc[:t]
    line2.set_data(v.index, v.values)
    dot1.set_offsets([[mdates.date2num(t), y[-1]]])
    dot2.set_offsets([[mdates.date2num(v.index[-1]), v.values[-1]]])
    date_txt.set_text(t.strftime("%b %Y"))
    lab_gfc.set_visible(t >= pd.Timestamp("2010-06-01"))
    lab_cov.set_visible(t >= pd.Timestamp("2021-06-01"))
    lab_rho.set_visible(frame >= N_FRAMES - 1)
    return line1, line2, dot1, dot2, date_txt


anim = FuncAnimation(fig, draw, frames=len(idx), interval=40, blit=False)
gif_path = OUT_DIR / "persistence_state.gif"
anim.save(gif_path, writer=PillowWriter(fps=25))
draw(len(idx) - 1)
png_path = OUT_DIR / "persistence_state.png"
fig.savefig(png_path, dpi=160)
print(f"wrote {gif_path} ({gif_path.stat().st_size/1e6:.2f} MB) and {png_path}")
print(f"calm={calm_level:.3f} gfc={gfc_level:.3f} covid={covid_level:.3f} rho={rho:.3f}")
