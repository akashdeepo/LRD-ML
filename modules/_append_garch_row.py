"""Append the GARCH(1,1) row to table9_raw.csv (computed via
variant_garch) and regenerate table9_robustness.tex via the cached helper.

Avoids the ~30-minute module9 full rerun. Use after module4b_garch
forecasts land in results/intermediate/forecasts/G_h05_*.csv.
"""

from pathlib import Path
import pandas as pd

from modules.module9_robustness import variant_garch
from modules._regen_table9 import LABELS  # ensures import side-effect of label update

BASE = Path(__file__).resolve().parent.parent
INTERM = BASE / "results" / "intermediate"

# 1) compute GARCH variant numbers
fp = BASE / "results" / "intermediate" / "forecasts"
base_yhat = pd.read_csv(fp / "A_h05_yhat.csv", index_col=0, parse_dates=True)
base_y = pd.read_csv(fp / "A_h05_y.csv", index_col=0, parse_dates=True)
g = variant_garch(None, base_yhat, base_y)
g_row = {"variant": "benchmark_garch11", **g}
print(f"GARCH(1,1) row: {g_row}")

# 2) read existing raw CSV, drop any prior GARCH row, append, save
raw_fp = INTERM / "table9_raw.csv"
df = pd.read_csv(raw_fp)
df = df[df["variant"] != "benchmark_garch11"]
df = pd.concat([df, pd.DataFrame([g_row])], ignore_index=True)

# 3) reorder so GARCH sits right after inference_plain (paper-friendly order)
order = ["headline", "inference_plain", "benchmark_garch11",
         "liquidity_high_illiq", "liquidity_low_illiq",
         "estimator_LW", "window_500", "window_1000", "target_sqret",
         "regime_high_vix", "regime_low_vix", "regime_covid", "regime_gfc"]
df["_order"] = df["variant"].map({v: i for i, v in enumerate(order)}).fillna(99)
df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
df.to_csv(raw_fp, index=False)
print(f"Updated {raw_fp}")

# 4) regenerate the LaTeX via _regen_table9.py
import importlib
import modules._regen_table9 as r
importlib.reload(r)
print("Table 9 regenerated.")
