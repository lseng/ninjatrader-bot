#!/usr/bin/env python3
"""
Backtesting Script for ML Trading Strategy

Simulates trading with the trained model and provides detailed performance metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def create_features(df):
    """Create features matching training."""
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["range"] = (df["high"] - df["low"]) / df["close"]
    df["body"] = (df["close"] - df["open"]) / df["close"]

    for period in [5, 10, 20]:
        df[f"sma_{period}"] = df["close"].rolling(period).mean()
        df[f"close_vs_sma_{period}"] = (df["close"] - df[f"sma_{period}"]) / df[f"sma_{period}"]

    for period in [3, 5, 10]:
        df[f"roc_{period}"] = df["close"].pct_change(period)

    df["volatility_5"] = df["returns"].rolling(5).std()
    df["volatility_10"] = df["returns"].rolling(10).std()

    if "volume" in df.columns:
        df["volume_sma_5"] = df["volume"].rolling(5).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_5"]

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

    return df


def run_backtest(df, model, initial_capital=1000, contract_value=5, commission=2.00):
    """
    Run backtest simulation.

    Args:
        df: DataFrame with OHLCV data
        model: Trained model
        initial_capital: Starting capital in dollars
        contract_value: Value per point (MES = $5)
        commission: Round-trip commission per trade

    Returns:
        dict with backtest results
    """
    # Create features
    df = create_features(df)
    df = df.dropna().reset_index(drop=True)

    feature_cols = [c for c in df.columns
                    if c not in ["timestamp", "open", "high", "low", "close", "volume"]]

    X = df[feature_cols].values
    prices = df["close"].values
    timestamps = df["timestamp"].values if "timestamp" in df.columns else range(len(df))

    # Get predictions
    predictions = model.predict(X)

    # Simulation
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]

    for i in range(1, len(predictions)):
        pred = predictions[i]
        current_price = prices[i]
        prev_price = prices[i-1]

        # Calculate P&L for current position
        if position != 0:
            point_change = current_price - prev_price
            pnl = position * point_change * contract_value
            capital += pnl

        # Check for position change
        if pred != position:
            # Close existing position
            if position != 0:
                trade_pnl = (current_price - entry_price) * position * contract_value - commission
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": timestamps[i],
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "direction": "LONG" if position == 1 else "SHORT",
                    "pnl": trade_pnl,
                    "bars_held": i - entry_bar
                })

            # Open new position
            if pred != 0:
                position = pred
                entry_price = current_price
                entry_time = timestamps[i]
                entry_bar = i
                capital -= commission / 2  # Entry commission
            else:
                position = 0

        equity_curve.append(capital)

    # Close final position if open
    if position != 0:
        trade_pnl = (prices[-1] - entry_price) * position * contract_value - commission
        trades.append({
            "entry_time": entry_time,
            "exit_time": timestamps[-1],
            "entry_price": entry_price,
            "exit_price": prices[-1],
            "direction": "LONG" if position == 1 else "SHORT",
            "pnl": trade_pnl,
            "bars_held": len(prices) - entry_bar
        })

    # Calculate statistics
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    if len(trades_df) > 0:
        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] < 0]

        stats = {
            "initial_capital": initial_capital,
            "final_capital": equity_curve[-1],
            "total_return_pct": (equity_curve[-1] / initial_capital - 1) * 100,
            "total_pnl": equity_curve[-1] - initial_capital,
            "total_trades": len(trades_df),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
            "avg_win": winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0,
            "avg_loss": losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0,
            "largest_win": trades_df["pnl"].max(),
            "largest_loss": trades_df["pnl"].min(),
            "avg_bars_held": trades_df["bars_held"].mean(),
            "profit_factor": abs(winning_trades["pnl"].sum() / losing_trades["pnl"].sum()) if len(losing_trades) > 0 and losing_trades["pnl"].sum() != 0 else float("inf"),
            "max_equity": max(equity_curve),
            "min_equity": min(equity_curve),
            "max_drawdown": min(0, min(equity_curve) - max(equity_curve[:equity_curve.index(min(equity_curve))+1])) if len(equity_curve) > 1 else 0
        }

        # Long vs Short breakdown
        long_trades = trades_df[trades_df["direction"] == "LONG"]
        short_trades = trades_df[trades_df["direction"] == "SHORT"]

        stats["long_trades"] = len(long_trades)
        stats["long_pnl"] = long_trades["pnl"].sum() if len(long_trades) > 0 else 0
        stats["short_trades"] = len(short_trades)
        stats["short_pnl"] = short_trades["pnl"].sum() if len(short_trades) > 0 else 0

    else:
        stats = {
            "initial_capital": initial_capital,
            "final_capital": initial_capital,
            "total_return_pct": 0,
            "total_pnl": 0,
            "total_trades": 0
        }

    return {
        "stats": stats,
        "trades": trades,
        "equity_curve": equity_curve
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ML Trading Strategy Backtester")
    parser.add_argument("--data", default="data/historical/MES_1m.parquet",
                       help="Path to price data")
    parser.add_argument("--model", default="models/ultra_rf/model.joblib",
                       help="Path to trained model")
    parser.add_argument("--capital", type=float, default=1000,
                       help="Initial capital")
    parser.add_argument("--sample-rate", type=int, default=20,
                       help="Sample rate (1 = every bar, 20 = every 20th bar)")
    parser.add_argument("--holdout-only", action="store_true",
                       help="Only backtest on holdout period - last 15 percent")
    args = parser.parse_args()

    log("=" * 60)
    log("ML TRADING STRATEGY BACKTESTER")
    log("=" * 60)

    # Load model
    log(f"Loading model from {args.model}...")
    model = joblib.load(args.model)

    # Load data
    log(f"Loading data from {args.data}...")
    if args.data.endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data, parse_dates=["timestamp"])

    # Sample data
    if args.sample_rate > 1:
        df = df.iloc[::args.sample_rate].reset_index(drop=True)

    # Use holdout period only
    if args.holdout_only:
        holdout_start = int(len(df) * 0.85)
        df = df.iloc[holdout_start:].reset_index(drop=True)
        log(f"Using holdout period only: {len(df):,} bars")
    else:
        log(f"Total bars: {len(df):,}")

    # Run backtest
    log("Running backtest...")
    results = run_backtest(df, model, initial_capital=args.capital)

    # Print results
    stats = results["stats"]

    log("")
    log("=" * 60)
    log("BACKTEST RESULTS")
    log("=" * 60)
    log("")
    log("CAPITAL")
    log(f"  Initial: ${stats['initial_capital']:,.2f}")
    log(f"  Final:   ${stats['final_capital']:,.2f}")
    log(f"  P&L:     ${stats['total_pnl']:,.2f} ({stats['total_return_pct']:.2f}%)")
    log("")
    log("TRADES")
    log(f"  Total:   {stats['total_trades']}")
    log(f"  Winners: {stats.get('winning_trades', 0)} ({stats.get('win_rate', 0):.1f}%)")
    log(f"  Losers:  {stats.get('losing_trades', 0)}")
    log("")

    if stats['total_trades'] > 0:
        log("PERFORMANCE")
        log(f"  Avg Win:      ${stats.get('avg_win', 0):,.2f}")
        log(f"  Avg Loss:     ${stats.get('avg_loss', 0):,.2f}")
        log(f"  Largest Win:  ${stats.get('largest_win', 0):,.2f}")
        log(f"  Largest Loss: ${stats.get('largest_loss', 0):,.2f}")
        log(f"  Profit Factor: {stats.get('profit_factor', 0):.2f}")
        log(f"  Avg Bars Held: {stats.get('avg_bars_held', 0):.1f}")
        log("")
        log("DIRECTION BREAKDOWN")
        log(f"  Long Trades:  {stats.get('long_trades', 0)} (P&L: ${stats.get('long_pnl', 0):,.2f})")
        log(f"  Short Trades: {stats.get('short_trades', 0)} (P&L: ${stats.get('short_pnl', 0):,.2f})")
        log("")
        log("RISK")
        log(f"  Max Drawdown: ${stats.get('max_drawdown', 0):,.2f}")
        log(f"  Peak Equity:  ${stats.get('max_equity', 0):,.2f}")

    # Save results
    output_file = Path(args.model).parent / "backtest_results.json"
    with open(output_file, "w") as f:
        json.dump({"stats": stats, "config": vars(args)}, f, indent=2, default=str)
    log(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
