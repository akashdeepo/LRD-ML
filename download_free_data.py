"""
LRD-ML Research: Free Data Download Script
Downloads VIX, market indices, and additional S&P 500 stock prices
Requires yfinance >= 1.0
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
START_DATE = "2000-01-01"
END_DATE = "2025-01-15"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Additional S&P 500 stocks (beyond your existing 30)
ADDITIONAL_TICKERS = [
    # Technology
    "IBM", "INTC", "ORCL", "CSCO", "ADBE", "CRM", "AVGO", "TXN", "QCOM", "AMD",
    # Finance
    "WFC", "C", "MS", "AXP", "BLK", "SCHW", "USB", "PNC", "TFC", "COF",
    # Healthcare
    "MRK", "ABBV", "LLY", "TMO", "ABT", "DHR", "BMY", "AMGN", "GILD", "CVS",
    # Consumer Discretionary
    "MCD", "NKE", "SBUX", "TJX", "LOW", "TGT", "BKNG", "MAR", "GM", "F",
    # Consumer Staples
    "WMT", "COST", "PM", "MO", "CL", "KMB", "GIS", "K", "SYY", "ADM",
    # Energy
    "SLB", "EOG", "PSX", "VLO", "MPC", "OXY", "HAL", "KMI", "WMB",
    # Industrials
    "HON", "GE", "RTX", "LMT", "MMM", "DE", "EMR", "ITW", "ETN", "FDX",
    # Utilities
    "SO", "D", "AEP", "EXC", "SRE", "XEL", "PEG", "ED", "WEC", "ES",
    # Materials
    "LIN", "APD", "SHW", "ECL", "NEM", "NUE", "VMC", "MLM", "DD", "PPG",
    # Real Estate
    "SPG", "EQIX", "PSA", "O", "WELL", "AVB", "EQR", "DLR", "VTR", "BXP",
    # Communication
    "DIS", "CMCSA", "NFLX", "T", "VZ", "TMUS", "CHTR", "EA"
]

EXISTING_TICKERS = [
    "AAPL", "MSFT", "NVDA", "JPM", "BAC", "GS", "JNJ", "PFE", "UNH", "AMZN",
    "TSLA", "HD", "PG", "KO", "PEP", "XOM", "CVX", "COP", "BA", "CAT",
    "UNP", "NEE", "DUK", "DOW", "FCX", "GOOGL", "META", "AMT", "PLD", "V"
]


def download_single(ticker, start, end):
    """Download single ticker, returns Close prices as Series"""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df is not None and len(df) > 0:
            print(f"  {ticker}: {len(df)} rows")
            # Handle MultiIndex columns from yfinance 1.0
            close = df['Close']
            if isinstance(close, pd.DataFrame):
                # MultiIndex: get the first (only) column
                close = close.iloc[:, 0]
            close = pd.Series(close.values, index=close.index, name=ticker)
            return close
        else:
            print(f"  {ticker}: NO DATA")
            return None
    except Exception as e:
        print(f"  {ticker}: ERROR - {e}")
        return None


def download_batch(tickers, start, end, desc="stocks"):
    """Download multiple tickers at once"""
    print(f"\n{'='*50}")
    print(f"Downloading {desc} ({len(tickers)} tickers)...")
    print(f"{'='*50}")

    try:
        df = yf.download(tickers, start=start, end=end, progress=True, group_by='ticker')

        # Extract just Close prices
        close_data = {}
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    close_data[ticker] = df['Close']
                else:
                    if ticker in df.columns.get_level_values(0):
                        close_data[ticker] = df[ticker]['Close']
            except:
                pass

        result = pd.DataFrame(close_data)
        print(f"\nGot {len(result.columns)} tickers, {len(result)} rows")

        # Coverage stats
        coverage = result.notna().sum()
        print(f"Coverage: min={coverage.min()}, max={coverage.max()}, mean={coverage.mean():.0f}")

        return result

    except Exception as e:
        print(f"Batch download error: {e}")
        print("Trying individual downloads...")

        # Fallback to individual downloads
        close_data = {}
        for i, ticker in enumerate(tickers):
            try:
                print(f"  [{i+1}/{len(tickers)}] {ticker}...", end=" ")
                data = yf.download(ticker, start=start, end=end, progress=False)
                if len(data) > 0:
                    close_data[ticker] = data['Close']
                    print(f"OK ({len(data)} rows)")
                else:
                    print("NO DATA")
            except Exception as ex:
                print(f"ERROR: {ex}")

        return pd.DataFrame(close_data) if close_data else None


def main():
    print("\n" + "="*60)
    print("   LRD-ML RESEARCH: FREE DATA DOWNLOAD")
    print(f"   yfinance version: {yf.__version__}")
    print("="*60)
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Output directory: {OUTPUT_DIR}")

    # 1. Download market indices
    print("\n[1/5] MARKET INDICES")
    market_tickers = {"^GSPC": "SP500", "^DJI": "DJIA", "^IXIC": "NASDAQ", "^RUT": "Russell2000"}
    market_series = []
    for ticker, name in market_tickers.items():
        series = download_single(ticker, START_DATE, END_DATE)
        if series is not None:
            series.name = name
            market_series.append(series)

    if market_series:
        market_df = pd.concat(market_series, axis=1)
        market_df.to_csv(os.path.join(OUTPUT_DIR, "market_indices.csv"))
        print(f"Saved market_indices.csv ({len(market_df)} rows)")

    # 2. Download VIX
    print("\n[2/5] VIX")
    vix_series = download_single("^VIX", START_DATE, END_DATE)
    if vix_series is not None:
        vix_df = pd.DataFrame({"VIX": vix_series})
        vix_df.to_csv(os.path.join(OUTPUT_DIR, "vix_data.csv"))
        print(f"Saved vix_data.csv ({len(vix_df)} rows)")

    # 3. Download additional stocks
    print("\n[3/5] ADDITIONAL STOCKS")
    additional_df = download_batch(ADDITIONAL_TICKERS, START_DATE, END_DATE, "additional S&P 500 stocks")
    if additional_df is not None and len(additional_df) > 0:
        additional_df.to_csv(os.path.join(OUTPUT_DIR, "additional_stocks.csv"))
        print(f"Saved additional_stocks.csv")

    # 4. Download volume for existing stocks
    print("\n[4/5] VOLUME DATA")
    try:
        vol_df = yf.download(EXISTING_TICKERS, start=START_DATE, end=END_DATE, progress=True, group_by='ticker')
        volume_data = {}
        for ticker in EXISTING_TICKERS:
            try:
                if ticker in vol_df.columns.get_level_values(0):
                    volume_data[ticker] = vol_df[ticker]['Volume']
            except:
                pass
        if volume_data:
            volume_df = pd.DataFrame(volume_data)
            volume_df.to_csv(os.path.join(OUTPUT_DIR, "volume_data.csv"))
            print(f"Saved volume_data.csv ({len(volume_df)} rows, {len(volume_df.columns)} tickers)")
    except Exception as e:
        print(f"Volume download error: {e}")

    # 5. Commodities and FX
    print("\n[5/5] COMMODITIES & FX")
    commodity_tickers = {"GC=F": "Gold", "CL=F": "Oil_WTI", "EURUSD=X": "EUR_USD"}
    commodity_series = []
    for ticker, name in commodity_tickers.items():
        series = download_single(ticker, START_DATE, END_DATE)
        if series is not None:
            series.name = name
            commodity_series.append(series)

    if commodity_series:
        commodity_df = pd.concat(commodity_series, axis=1)
        commodity_df.to_csv(os.path.join(OUTPUT_DIR, "commodities_fx.csv"))
        print(f"Saved commodities_fx.csv ({len(commodity_df)} rows)")

    # Summary
    print("\n" + "="*60)
    print("   DOWNLOAD SUMMARY")
    print("="*60)

    print("\nFiles created:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.csv'):
            fpath = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(fpath) / 1024
            print(f"  - {f} ({size:.1f} KB)")

    print("\n" + "="*60)
    print("   BLOOMBERG DATA STILL NEEDED")
    print("="*60)
    print("""
    For Realized Volatility (critical for the research):

    1. 5-MINUTE INTRADAY DATA
       - Tickers: Your 30 stocks (or subset)
       - Period: 2010-2025 (or whatever is available)
       - Bloomberg: Use INTRADAY_BAR or GetIntradayBar

    Export format: CSV with columns [Date, Time, Open, High, Low, Close, Volume]
    """)


if __name__ == "__main__":
    main()
