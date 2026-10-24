"""
Diff the 2026-10-22 data refresh (ledger runs 31, 32) against the results it
replaced, which were copied to results/_pre_refresh_2026-10-22/ before the
rerun. Checks the registered predictions:

  run-31  every relative MSE reduction moves by < 0.1 pp; no DM, CW or GW
          p-value crosses 0.05 or 0.10; price-return Sharpe ratios move < 0.01
  run-32  total returns raise annual excess returns by 1 to 2.5 pp and Sharpe
          by 0.05 to 0.15; ordering, signs and the next-day penalty unchanged

Run from the repo root:  python -m modules._diff_refresh
Writes results/intermediate/refresh_diff.txt and prints the same text.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
NEW = BASE / "results" / "intermediate"
OLD = BASE / "results" / "_pre_refresh_2026-10-22" / "intermediate"
OUT = NEW / "refresh_diff.txt"

# file -> (key columns, gain columns, p-value columns)
GAIN_FILES = {
    "harx_tests.csv": (["spec", "h"], ["gain_vs_har_pct", "gain_vs_harx_pct"],
                       ["dm_p_vs_har", "dm_p_vs_harx", "dm_p_vs_harx_qlike", "cw_p_vs_harx"]),
    "harx_tests_regimes.csv": (["spec", "h", "regime"], ["gain_vs_har_pct"], ["dm_p_vs_harx", "cw_p_vs_harx"]),
    "table5_raw.csv": (["model", "h"], ["imp_vs_A_pct"], ["DM_p"]),
    "table7_raw.csv": (["regime", "model", "h"], ["imp_vs_A_pct"], []),
    "table9_raw.csv": (["variant"], ["imp_pct"], ["p"]),
    "matched_robustness.csv": (["variant"], ["gain_pct"], ["dm_p"]),
    "balanced_panel.csv": (["comparison", "panel", "h"], ["gain_pct"], ["dm_p"]),
    "no_hurst_variant.csv": (["spec", "h"], ["gain_pct"], ["dm_p"]),
    "rolling_gw.csv": (["window", "spec", "h"], ["gain_vs_harx_pct"], ["dm_p", "gw_p"]),
    "candidates_run05_pooled.csv": (["comparison", "h"], ["gain_pct"], ["dm_p", "cw_p", "cw_p_holm"]),
    "candidates_run06_duration.csv": (["comparison", "h"], ["gain_pct"], ["dm_p", "cw_p", "cw_p_holm"]),
    "candidates_run07_market.csv": (["comparison", "h"], ["gain_pct"], ["dm_p", "cw_p", "cw_p_holm"]),
    "timing_alignment.csv": (["h", "alignment"], ["harx_gain_pct"], ["dm_p"]),
    "bootstrap_intervals.csv": (["baseline", "spec", "h"], ["gain_pct", "ci_lo", "ci_hi"], []),
}
THRESHOLDS = (0.05, 0.10)
PORT_ORDER = ["Equal-weight (unmanaged)", "HAR-managed", "HAR-X-managed",
              "Trailing-22d-RV-managed", "C-managed", "C-managed (full-sample c)"]


def _merge(name: str, keys: list[str]) -> pd.DataFrame | None:
    fo, fn = OLD / name, NEW / name
    if not (fo.exists() and fn.exists()):
        return None
    o, n = pd.read_csv(fo), pd.read_csv(fn)
    if n[keys].duplicated().any() or o[keys].duplicated().any():
        return None
    return o.merge(n, on=keys, suffixes=("_old", "_new"), how="inner")


def gains_and_pvalues(lines: list[str]) -> bool:
    ok = True
    lines.append("== run-31: relative MSE reductions and p-values, old vs new")
    for name, (keys, gains, pvals) in GAIN_FILES.items():
        m = _merge(name, keys)
        if m is None:
            lines.append(f"  {name}: not available on both sides (or keys not unique); skipped")
            continue
        worst = 0.0
        for g in gains:
            if f"{g}_old" in m:
                d = (m[f"{g}_new"] - m[f"{g}_old"]).abs()
                worst = max(worst, float(d.max()))
                i = int(d.idxmax())
                lines.append(f"  {name:28s} {g:18s} max |change| {d.max():.4f} pp  "
                             f"at {dict(zip(keys, m.loc[i, keys]))} ({m.loc[i, f'{g}_old']:.3f} -> {m.loc[i, f'{g}_new']:.3f})")
        for p in pvals:
            if f"{p}_old" not in m:
                continue
            for t in THRESHOLDS:
                crossed = m[((m[f"{p}_old"] < t) != (m[f"{p}_new"] < t))]
                for _, r in crossed.iterrows():
                    ok = False
                    lines.append(f"  !! {name} {p} crosses {t}: {dict(zip(keys, r[keys]))} "
                                 f"{r[f'{p}_old']:.4f} -> {r[f'{p}_new']:.4f}")
        if worst >= 0.1:
            ok = False
            lines.append(f"  !! {name}: a reduction moved by {worst:.3f} pp (prediction: < 0.1)")
    return ok


def memory_estimates(lines: list[str]) -> None:
    lines.append("== run-31: full-sample memory estimates (AAPL and the cross-section)")
    for name in ["lrd_rv_gph.csv", "lrd_rv_lw.csv", "lrd_returns_gph.csv", "lrd_returns_lw.csv", "hurst_rv_log.csv"]:
        fo, fn = OLD / name, NEW / name
        if not (fo.exists() and fn.exists()):
            continue
        o, n = pd.read_csv(fo, index_col=0), pd.read_csv(fn, index_col=0)
        col = "d_hat" if "d_hat" in n else n.columns[0]
        common = o.index.intersection(n.index)
        d = (n.loc[common, col] - o.loc[common, col]).abs()
        others = d.drop("AAPL", errors="ignore")
        lines.append(f"  {name:22s} AAPL {o.loc['AAPL', col]:.4f} -> {n.loc['AAPL', col]:.4f}; "
                     f"other stocks max |change| {others.max():.2e}; mean {o[col].mean():.4f} -> {n[col].mean():.4f}")


def portfolios(lines: list[str]) -> bool:
    ok = True
    fo, fn = OLD / "portfolio_variants.csv", NEW / "portfolio_variants.csv"
    if not (fo.exists() and fn.exists()):
        lines.append("== portfolios: portfolio_variants.csv missing on one side; skipped")
        return ok
    o = pd.read_csv(fo).set_index(["variant", "portfolio"])
    n = pd.read_csv(fn).set_index(["variant", "portfolio"])
    base, nxt = "baseline (same-close execution, no cap)", "next-day execution"
    pb, pn = "price returns, same-close execution", "price returns, next-day execution"

    lines.append("== run-31: price-return Sharpe ratios, April panel (run-25) vs refreshed panel")
    for old_v, new_v in [(base, pb), (nxt, pn)]:
        for p in PORT_ORDER:
            if (old_v, p) in o.index and (new_v, p) in n.index:
                a, b = o.loc[(old_v, p), "sharpe"], n.loc[(new_v, p), "sharpe"]
                flag = "" if abs(b - a) < 0.01 else "  !! moved >= 0.01"
                ok &= abs(b - a) < 0.01
                lines.append(f"  {old_v[:10]:10s} {p:28s} {a:.3f} -> {b:.3f} ({b - a:+.4f}){flag}")

    lines.append("== run-32: total returns (new default) vs price returns, same panel")
    hdr = f"  {'':10s} {'portfolio':28s} {'mean_ann':>17s} {'sharpe':>17s} {'max_dd':>17s}"
    lines.append(hdr)
    for tot_v, px_v in [(base, pb), (nxt, pn)]:
        for p in PORT_ORDER:
            if (tot_v, p) in n.index and (px_v, p) in n.index:
                t, x = n.loc[(tot_v, p)], n.loc[(px_v, p)]
                dm, ds = 100 * (t.mean_ann - x.mean_ann), t.sharpe - x.sharpe
                flag = ""
                if not (1.0 <= dm <= 2.5):
                    flag += "  !! mean change outside 1-2.5 pp"
                if not (0.05 <= ds <= 0.15):
                    flag += "  !! Sharpe change outside 0.05-0.15"
                ok &= flag == ""
                lines.append(f"  {tot_v[:10]:10s} {p:28s} {100*x.mean_ann:6.2f} -> {100*t.mean_ann:6.2f} "
                             f"{x.sharpe:7.3f} -> {t.sharpe:6.3f} {100*x.max_dd:7.1f} -> {100*t.max_dd:6.1f}{flag}")
    for v in [base, nxt]:
        if (v, "C-managed") in n.index:
            s = n.loc[v, "sharpe"]
            lines.append(f"  {v}: ordering by Sharpe {list(s.sort_values(ascending=False).index)}")
            lines.append(f"    C - HAR-X {s['C-managed'] - s['HAR-X-managed']:+.4f}; C - EW {s['C-managed'] - s['Equal-weight (unmanaged)']:+.4f}")
    if all((v, "C-managed") in n.index for v in [base, nxt]):
        for p in PORT_ORDER:
            pen = n.loc[(base, p), "sharpe"] - n.loc[(nxt, p), "sharpe"]
            lines.append(f"  next-day penalty {p:28s} {pen:+.3f}")
    ft = NEW / "portfolio_variants_tests.csv"
    if ft.exists():
        t = pd.read_csv(ft)
        lines.append("  Sharpe-difference tests (new):")
        for _, r in t.iterrows():
            flag = "  !! significant at 5%" if min(r.hac_p, r.boot_p) < 0.05 else ""
            ok &= flag == ""
            lines.append(f"    {r.variant[:34]:34s} {r.pair:44s} d={r.delta_sharpe_ann:+.3f} hac_p={r.hac_p:.3f} boot_p={r.boot_p:.3f}{flag}")
    return ok


def everything_else(lines: list[str]) -> None:
    """Every other CSV present on both sides: max absolute change of each
    numeric column when the frames align row for row, else 'structure changed'."""
    lines.append("== all other result files")
    done = set(GAIN_FILES) | {"portfolio_variants.csv", "portfolio_variants_tests.csv"}
    for fn in sorted(NEW.glob("*.csv")):
        if fn.name in done or not (OLD / fn.name).exists():
            continue
        o, n = pd.read_csv(OLD / fn.name), pd.read_csv(fn)
        if o.shape != n.shape or list(o.columns) != list(n.columns):
            lines.append(f"  {fn.name:34s} structure changed {o.shape} -> {n.shape}")
            continue
        num = [c for c in n.columns if pd.api.types.is_numeric_dtype(n[c]) and pd.api.types.is_numeric_dtype(o[c])]
        if not num:
            lines.append(f"  {fn.name:34s} no numeric columns")
            continue
        d = (n[num] - o[num]).abs()
        worst = d.max()
        c = worst.idxmax()
        lines.append(f"  {fn.name:34s} max |change| {worst.max():.3e} in {c}; columns changed: {int((worst > 0).sum())}/{len(num)}")


def main() -> None:
    lines = [f"Refresh diff: {OLD.relative_to(BASE)} -> {NEW.relative_to(BASE)}"]
    ok31 = gains_and_pvalues(lines)
    memory_estimates(lines)
    okp = portfolios(lines)
    everything_else(lines)
    lines.append(f"\nrun-31 predictions hold: {ok31 and okp}   (lines marked !! are the exceptions)")
    text = "\n".join(lines)
    OUT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
