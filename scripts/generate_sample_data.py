#!/usr/bin/env python3
"""
Generate synthetic futures data for backtesting.

Creates realistic MES (Micro E-mini S&P 500) data with:
- Proper tick sizes
- Volume patterns
- Trend and mean reversion dynamics
- Session patterns (overnight, regular hours)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse


def generate_mes_data(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    timeframe: str = "1min",
    initial_price: float = 5000.0,
    volatility: float = 0.0002,
    trend: float = 0.0001,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic MES futures data.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timeframe: Bar timeframe (1min, 5min, 15min, 1hour)
        initial_price: Starting price
        volatility: Per-bar volatility
        trend: Long-term trend component
        seed: Random seed for reproducibility

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(seed)

    # Parse dates
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Create date range based on timeframe
    freq_map = {
        "1min": "1min",
        "5min": "5min",
        "15min": "15min",
        "1hour": "1h"
    }
    freq = freq_map.get(timeframe, "1min")

    # Generate timestamps (exclude weekends, only trading hours)
    all_timestamps = pd.date_range(start, end, freq=freq)

    # Filter to CME Globex hours (Sunday 5pm - Friday 4pm CT, with daily break 4-5pm)
    def is_trading_hour(ts):
        # Skip weekends (Saturday all day, Sunday before 5pm)
        if ts.dayofweek == 5:  # Saturday
            return False
        if ts.dayofweek == 6 and ts.hour < 17:  # Sunday before 5pm
            return False
        # Skip daily maintenance (4pm-5pm CT)
        if ts.hour == 16:
            return False
        return True

    timestamps = [ts for ts in all_timestamps if is_trading_hour(ts)]

    if len(timestamps) == 0:
        raise ValueError("No valid trading timestamps in range")

    # Generate price series using geometric Brownian motion with mean reversion
    n_bars = len(timestamps)
    returns = np.random.normal(trend, volatility, n_bars)

    # Clip extreme returns to prevent overflow
    returns = np.clip(returns, -0.05, 0.05)

    # Add mean reversion
    mean_price = initial_price
    reversion_strength = 0.0001  # Reduced for stability

    prices = [initial_price]
    for i in range(1, n_bars):
        # Mean reversion component
        reversion = reversion_strength * (mean_price - prices[-1])

        # Session-based volatility (higher at open/close)
        hour = timestamps[i].hour
        if hour in [9, 15, 16]:  # Market open/close hours
            session_mult = 1.5
        elif hour in [12, 13]:  # Lunch lull
            session_mult = 0.7
        else:
            session_mult = 1.0

        # Calculate new price with bounds checking
        change = returns[i] * session_mult + reversion
        change = np.clip(change, -0.01, 0.01)  # Max 1% per bar
        new_price = prices[-1] * (1 + change)

        # Keep price in reasonable range
        new_price = max(min(new_price, initial_price * 2), initial_price * 0.5)

        # Round to tick size (0.25 for MES)
        new_price = round(new_price / 0.25) * 0.25
        prices.append(new_price)

    # Generate OHLC from close prices
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        # Generate intrabar volatility
        bar_volatility = volatility * np.random.uniform(0.5, 2.0)

        # OHLC with realistic patterns
        if i > 0:
            open_price = prices[i-1] + np.random.normal(0, volatility * prices[i-1])
        else:
            open_price = close

        open_price = round(open_price / 0.25) * 0.25

        # High and low
        high = max(open_price, close) + abs(np.random.normal(0, bar_volatility * close))
        low = min(open_price, close) - abs(np.random.normal(0, bar_volatility * close))

        high = round(high / 0.25) * 0.25
        low = round(low / 0.25) * 0.25

        # Ensure OHLC consistency
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Volume (higher during regular hours, spikes at open/close)
        base_volume = 1000
        if 9 <= ts.hour <= 15:
            volume_mult = 3.0
        elif ts.hour in [8, 16]:
            volume_mult = 2.0
        else:
            volume_mult = 1.0

        if ts.hour == 9 and ts.minute < 30:
            volume_mult *= 2.0  # Opening spike
        if ts.hour == 15 and ts.minute >= 45:
            volume_mult *= 1.5  # Closing spike

        volume = int(base_volume * volume_mult * np.random.uniform(0.5, 1.5))

        data.append({
            "timestamp": ts,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        })

    df = pd.DataFrame(data)
    return df


def save_ninjatrader_format(df: pd.DataFrame, filepath: Path, symbol: str = "MES"):
    """
    Save data in NinjaTrader semicolon-separated format.

    Format: yyyyMMdd HHmmss;Open;High;Low;Close;Volume
    """
    with open(filepath, "w") as f:
        for _, row in df.iterrows():
            date_str = row["timestamp"].strftime("%Y%m%d")
            time_str = row["timestamp"].strftime("%H%M%S")
            line = f"{date_str} {time_str};{row['open']:.2f};{row['high']:.2f};{row['low']:.2f};{row['close']:.2f};{row['volume']}\n"
            f.write(line)

    print(f"Saved {len(df)} bars to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic futures data")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", default="1min", choices=["1min", "5min", "15min", "1hour"])
    parser.add_argument("--symbol", default="MES", help="Symbol name")
    parser.add_argument("--output", default="data/historical", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print(f"Generating {args.symbol} data from {args.start} to {args.end} ({args.timeframe})")

    df = generate_mes_data(
        start_date=args.start,
        end_date=args.end,
        timeframe=args.timeframe,
        seed=args.seed
    )

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    csv_path = output_dir / f"{args.symbol}_{args.timeframe}_{args.start}_{args.end}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save as NinjaTrader format
    nt_path = output_dir / f"{args.symbol}_{args.timeframe}_{args.start}_{args.end}.txt"
    save_ninjatrader_format(df, nt_path, args.symbol)

    # Print summary
    print(f"\nData Summary:")
    print(f"  Total bars: {len(df):,}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"  Avg volume: {df['volume'].mean():,.0f}")


if __name__ == "__main__":
    main()
