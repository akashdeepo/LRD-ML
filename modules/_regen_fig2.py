"""Regenerate Figure 2 from cached LRD panels with the new font sizes,
without rerunning the slow rolling-LRD estimation in module2.main()."""

from pathlib import Path
import pandas as pd

from modules.io_v2 import build_clean_panel
from modules.module2_lrd_estimation import figure2

BASE = Path(__file__).resolve().parent.parent
INTERM = BASE / "results" / "intermediate"
FIGS = BASE / "results" / "figures"

panel = build_clean_panel()

rv_gph = pd.read_csv(INTERM / "lrd_rv_gph.csv", index_col=0)
rolling_d = pd.read_csv(INTERM / "rolling_d_gph.csv",
                        index_col=0, parse_dates=True)
rolling_h = pd.read_csv(INTERM / "rolling_hurst.csv",
                        index_col=0, parse_dates=True)
market = panel.market

figure2(rv_gph, rolling_d, rolling_h, market, FIGS / "fig2_lrd_estimates.pdf")
print("fig2 regenerated.")
