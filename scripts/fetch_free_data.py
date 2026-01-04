#!/usr/bin/env python3
"""
Fetch FREE historical futures data.

Sources:
1. Yahoo Finance - Free, limited futures data (ES=F, NQ=F)
2. Databento - $125 free credits, professional CME data
3. Synthetic - Generate realistic fake data for testing

Usage:
    python scripts/fetch_free_data.py --source yahoo --symbol ES=F --days 365
    python scripts/fetch_free_data.py --source databento --symbol MES --days 30
    python scripts/fetch_free_data.py --source synthetic --symbol MES --days 365
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np


def fetch_yahoo_finance(symbol: str, days: int, interval: str = "1h") -> pd.DataFrame:
    """
    Fetch data from Yahoo Finance (FREE).

    Symbols:
        ES=F  - E-mini S&P 500 Futures
        NQ=F  - E-mini Nasdaq-100 Futures
        YM=F  - E-mini Dow Futures
        RTY=F - E-mini Russell 2000 Futures
        GC=F  - Gold Futures
        CL=F  - Crude Oil Futures

    Note: Yahoo doesn't have MES/MNQ micros, use ES=F/NQ=F instead.
    """
    import yfinance as yf

    print(f"Fetching {symbol} from Yahoo Finance...")

    # Yahoo Finance limitations:
    # - 1m data: last 7 days
    # - 5m data: last 60 days
    # - 1h data: last 730 days
    # - 1d data: full history

    end = datetime.now()
    start = end - timedelta(days=days)

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    # Standardize columns
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]

    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
    elif "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})

    # Keep only OHLCV
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    print(f"Fetched {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def fetch_databento(symbol: str, days: int, api_key: str) -> pd.DataFrame:
    """
    Fetch data from Databento ($125 free credits).

    Symbols: MES, ES, NQ, MNQ, etc.

    Sign up at: https://databento.com/signup
    """
    try:
        import databento as db
    except ImportError:
        print("Installing databento...")
        import subprocess
        subprocess.run(["pip", "install", "databento"], check=True)
        import databento as db

    print(f"Fetching {symbol} from Databento...")

    client = db.Historical(api_key)

    end = datetime.now()
    start = end - timedelta(days=days)

    # Fetch OHLCV bars
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",  # CME Globex
        symbols=[symbol],
        schema="ohlcv-1m",  # 1-minute bars
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )

    df = data.to_df()

    # Standardize columns
    df = df.reset_index()
    df = df.rename(columns={
        "ts_event": "timestamp",
    })

    # Convert prices from fixed-point
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col] / 1e9  # Databento uses 9 decimal places

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    print(f"Fetched {len(df)} bars")

    return df


def generate_synthetic(symbol: str, days: int, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic futures data."""
    from scripts.generate_sample_data import generate_mes_data

    end = datetime.now()
    start = end - timedelta(days=days)

    df = generate_mes_data(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        timeframe="1min",
        seed=seed
    )

    print(f"Generated {len(df)} synthetic bars")

    return df


def save_data(df: pd.DataFrame, symbol: str, source: str, output_dir: Path):
    """Save data to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    start_str = df["timestamp"].min().strftime("%Y%m%d")
    end_str = df["timestamp"].max().strftime("%Y%m%d")

    # Clean symbol for filename
    clean_symbol = symbol.replace("=", "").replace("/", "")

    filename = f"{clean_symbol}_{source}_{start_str}_{end_str}.csv"
    filepath = output_dir / filename

    df.to_csv(filepath, index=False)
    print(f"\nSaved to: {filepath}")

    # Print summary stats
    print(f"\nData Summary:")
    print(f"  Bars: {len(df):,}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"  Avg volume: {df['volume'].mean():,.0f}")

    return filepath


def main():
    parser = argparse.ArgumentParser(description="Fetch free historical futures data")
    parser.add_argument("--source", default="yahoo",
                        choices=["yahoo", "databento", "synthetic"],
                        help="Data source")
    parser.add_argument("--symbol", default="ES=F",
                        help="Symbol (ES=F, NQ=F for Yahoo; MES, ES for Databento)")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--interval", default="1h",
                        help="Interval for Yahoo (1m, 5m, 15m, 1h, 1d)")
    parser.add_argument("--databento-key", help="Databento API key")
    parser.add_argument("--output", default="data/historical", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic")

    args = parser.parse_args()

    try:
        if args.source == "yahoo":
            df = fetch_yahoo_finance(args.symbol, args.days, args.interval)

        elif args.source == "databento":
            if not args.databento_key:
                print("Error: Databento requires --databento-key")
                print("Sign up for $125 free credits at: https://databento.com/signup")
                return
            df = fetch_databento(args.symbol, args.days, args.databento_key)

        elif args.source == "synthetic":
            df = generate_synthetic(args.symbol, args.days, args.seed)

        save_data(df, args.symbol, args.source, Path(args.output))

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
