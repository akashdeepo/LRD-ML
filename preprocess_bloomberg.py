"""
Convert bloomberg_pull/OHLCV.xlsx into clean daily panels, and
bloomberg_pull/TOT_RET_GROSS_DVD.xlsx into a total return index panel.

Fixes:
  - "JMP" sheet renamed to JPM (sheet-name typo in the April 2026 pull; the
    October 2026 pull spells it JPM).
  - Until the October 2026 pull the "AAPL" sheet held AAP (Advance Auto Parts)
    and Apple came from yfinance. The new sheet is genuine Apple but has a
    different header (no Open column; market cap in millions), so columns are
    matched by header name, not position.
  - The TXN sheet is empty in both pulls. Its yfinance series is frozen in
    bloomberg_pull/yfinance_fallback/TXN.csv (pulled 2026-04-26) so a rebuild
    does not depend on Yahoo revisions; a ticker with no cache is pulled and
    cached.
  - TOT_RET_GROSS_DVD.xlsx labels Bank of New York Mellon "BNY"; the OHLCV
    sheet and the pipeline call it BK.

Outputs (all under bloomberg_pull/processed/):
  - prices_close.csv, prices_open.csv, prices_high.csv, prices_low.csv (T x N)
  - volume.csv, mktcap.csv (T x N)
  - ohlcv_long.csv       long format with all fields
  - tot_return_index.csv gross total return index (T x N), same calendar
"""

from pathlib import Path
import pandas as pd
import openpyxl

BASE = Path(__file__).resolve().parent / "bloomberg_pull"
SRC = BASE / "OHLCV.xlsx"
TR_SRC = BASE / "TOT_RET_GROSS_DVD.xlsx"
OUT = BASE / "processed"
FALLBACK = BASE / "yfinance_fallback"
OUT.mkdir(exist_ok=True)
FALLBACK.mkdir(exist_ok=True)

SHEET_RENAMES = {"JMP": "JPM"}
TR_RENAMES = {"BNY": "BK"}
DEFAULT_START = "2001-11-29"
DEFAULT_END = "2026-04-22"
FIELDS = ["Close", "Open", "High", "Low", "Volume", "MktCap"]
# header substring -> field; headers are e.g. "Last Price  (R1)",
# "AAPL US Equity - Volume", "Current Market Capitalization of a Share Class  (L1)"
HEADER_PATTERNS = [("last price", "Close"), ("open price", "Open"), ("high price", "High"),
                   ("low price", "Low"), ("volume", "Volume"), ("market cap", "MktCap")]
# The AAPL sheet reports market cap in millions; every other sheet in dollars.
# mktcap.csv is not used downstream (size buckets come from stock_metadata.csv).
MKTCAP_SCALE = {"AAPL": 1e6}


def _field(header) -> str | None:
    h = str(header).lower()
    for pat, name in HEADER_PATTERNS:
        if pat in h:
            return name
    return None


def read_sheet(wb: openpyxl.Workbook, sheet: str) -> pd.DataFrame | None:
    """Sheet -> Date + FIELDS, columns matched by header name; None if empty."""
    rows = list(wb[sheet].iter_rows(values_only=True))
    if len(rows) <= 1:
        return None
    header, data = rows[0], rows[1:]
    names = ["Date"] + [_field(h) for h in header[1:]]
    unknown = [h for h, n in zip(header[1:], names[1:]) if n is None]
    if unknown:
        raise ValueError(f"{sheet}: unrecognised columns {unknown}")
    df = pd.DataFrame(data, columns=names)
    for f in FIELDS:
        if f not in df.columns:
            df[f] = pd.NA
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df[["Date"] + FIELDS]


def fallback_series(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Frozen yfinance series for a ticker whose Bloomberg sheet is empty."""
    cache = FALLBACK / f"{ticker}.csv"
    if cache.exists():
        print(f"    using frozen yfinance series {cache.name}")
        return pd.read_csv(cache, parse_dates=["Date"])
    import yfinance as yf
    print(f"    pulling {ticker} from yfinance ({start} to {end}) and freezing it in {cache.name}")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"index": "Date"})
    df["MktCap"] = pd.NA
    df = df[["Date"] + FIELDS]
    df.to_csv(cache, index=False)
    return df


def total_return_panel(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    tr = pd.read_excel(TR_SRC, index_col=0, parse_dates=True).sort_index()
    tr.columns = [TR_RENAMES.get(str(c).strip(), str(c).strip()) for c in tr.columns]
    tr.index.name = "Date"
    tr = tr.apply(pd.to_numeric, errors="coerce")
    if not tr.index.equals(calendar):
        raise ValueError("total return index calendar differs from the OHLCV calendar")
    return tr


def main():
    wb = openpyxl.load_workbook(SRC, read_only=True, data_only=True)
    panels = {}

    for sheet in wb.sheetnames:
        ticker = SHEET_RENAMES.get(sheet, sheet)
        df = read_sheet(wb, sheet)
        if df is None:
            print(f"  {sheet} -> {ticker}: empty Bloomberg sheet")
            df = fallback_series(ticker, DEFAULT_START, DEFAULT_END)
        if ticker in MKTCAP_SCALE:
            df["MktCap"] = pd.to_numeric(df["MktCap"], errors="coerce") * MKTCAP_SCALE[ticker]
        df["Ticker"] = ticker
        panels[ticker] = df

    wb.close()

    long_df = pd.concat(panels.values(), ignore_index=True)
    long_df = long_df[["Date", "Ticker"] + ["Open", "High", "Low", "Close", "Volume", "MktCap"]]
    long_df.to_csv(OUT / "ohlcv_long.csv", index=False)
    print(f"\nWrote {OUT/'ohlcv_long.csv'}  shape={long_df.shape}")

    calendar = None
    for field in FIELDS:
        wide = long_df.pivot(index="Date", columns="Ticker", values=field).sort_index()
        out = OUT / f"prices_{field.lower()}.csv" if field not in ("Volume", "MktCap") else OUT / f"{field.lower()}.csv"
        wide.to_csv(out)
        print(f"Wrote {out}  shape={wide.shape}")
        calendar = wide.index

    tr = total_return_panel(calendar)
    tr.to_csv(OUT / "tot_return_index.csv")
    absent = sorted(set(panels) - set(tr.columns))
    print(f"Wrote {OUT/'tot_return_index.csv'}  shape={tr.shape}; no total return index for {absent}")

    print(f"\nTickers ({len(panels)}): {sorted(panels.keys())}")


if __name__ == "__main__":
    main()
