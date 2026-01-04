#!/usr/bin/env python3
"""
Deep dive analysis of short signals to determine if they add value.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime


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
        df["volume_ratio"] = df["volume"] / df["volume_sma_5"].replace(0, 1)

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

    return df


def analyze_by_direction(prices, predictions, contract_value=5, commission=2.0):
    """Analyze long and short trades separately."""

    capital = 1000
    position = 0
    entry_price = 0
    entry_idx = 0

    long_trades = []
    short_trades = []
    equity = [capital]

    for i in range(1, len(predictions)):
        pred = predictions[i]
        current_price = prices[i]
        prev_price = prices[i-1]

        if position != 0:
            capital += position * (current_price - prev_price) * contract_value

        if pred != position:
            if position != 0:
                trade_pnl = (current_price - entry_price) * position * contract_value - commission
                bars_held = i - entry_idx

                trade = {
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "pnl": trade_pnl,
                    "bars_held": bars_held,
                    "entry_idx": entry_idx,
                    "exit_idx": i
                }

                if position == 1:
                    long_trades.append(trade)
                else:
                    short_trades.append(trade)

            if pred != 0:
                position = pred
                entry_price = current_price
                entry_idx = i
                capital -= commission / 2
            else:
                position = 0

        equity.append(capital)

    # Close final position
    if position != 0:
        trade_pnl = (prices[-1] - entry_price) * position * contract_value - commission
        trade = {
            "entry_price": entry_price,
            "exit_price": prices[-1],
            "pnl": trade_pnl,
            "bars_held": len(prices) - entry_idx
        }
        if position == 1:
            long_trades.append(trade)
        else:
            short_trades.append(trade)

    return {
        "long_trades": pd.DataFrame(long_trades) if long_trades else pd.DataFrame(),
        "short_trades": pd.DataFrame(short_trades) if short_trades else pd.DataFrame(),
        "equity": equity,
        "final_capital": equity[-1]
    }


def trade_stats(trades_df, name):
    """Calculate statistics for a set of trades."""
    if len(trades_df) == 0:
        return {"count": 0}

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]

    return {
        "count": len(trades_df),
        "total_pnl": trades_df["pnl"].sum(),
        "avg_pnl": trades_df["pnl"].mean(),
        "win_rate": len(wins) / len(trades_df) * 100,
        "avg_win": wins["pnl"].mean() if len(wins) > 0 else 0,
        "avg_loss": losses["pnl"].mean() if len(losses) > 0 else 0,
        "largest_win": trades_df["pnl"].max(),
        "largest_loss": trades_df["pnl"].min(),
        "avg_bars_held": trades_df["bars_held"].mean(),
        "profit_factor": abs(wins["pnl"].sum() / losses["pnl"].sum()) if len(losses) > 0 and losses["pnl"].sum() != 0 else float("inf")
    }


def main():
    log("=" * 70)
    log("SHORT SIGNAL ANALYSIS")
    log("=" * 70)

    # Load data
    log("Loading data...")
    df_raw = pd.read_parquet("data/historical/MES_1m.parquet")

    # Load model
    model = joblib.load("models/shallow_rf/model.joblib")

    # Get holdout data at different timeframes
    holdout_start = int(len(df_raw) * 0.85)
    df_holdout = df_raw.iloc[holdout_start:].reset_index(drop=True)

    timeframes = [15, 20, 30, 45, 60]

    for tf in timeframes:
        log(f"\n{'='*70}")
        log(f"TIMEFRAME: {tf} minutes")
        log(f"{'='*70}")

        df_sampled = df_holdout.iloc[::tf].reset_index(drop=True)
        df_feat = create_features(df_sampled).dropna()

        feature_cols = [c for c in df_feat.columns
                       if c not in ["timestamp", "open", "high", "low", "close", "volume"]]

        X = df_feat[feature_cols].values
        prices = df_feat["close"].values
        predictions = model.predict(X)

        # Analyze by direction
        results = analyze_by_direction(prices, predictions)

        long_stats = trade_stats(results["long_trades"], "LONG")
        short_stats = trade_stats(results["short_trades"], "SHORT")

        log(f"\nLONG TRADES:")
        log(f"  Count: {long_stats['count']}")
        if long_stats['count'] > 0:
            log(f"  Total P&L: ${long_stats['total_pnl']:,.2f}")
            log(f"  Win Rate: {long_stats['win_rate']:.1f}%")
            log(f"  Avg P&L: ${long_stats['avg_pnl']:,.2f}")
            log(f"  Profit Factor: {long_stats['profit_factor']:.2f}")
            log(f"  Avg Bars Held: {long_stats['avg_bars_held']:.1f}")

        log(f"\nSHORT TRADES:")
        log(f"  Count: {short_stats['count']}")
        if short_stats['count'] > 0:
            log(f"  Total P&L: ${short_stats['total_pnl']:,.2f}")
            log(f"  Win Rate: {short_stats['win_rate']:.1f}%")
            log(f"  Avg P&L: ${short_stats['avg_pnl']:,.2f}")
            log(f"  Profit Factor: {short_stats['profit_factor']:.2f}")
            log(f"  Avg Bars Held: {short_stats['avg_bars_held']:.1f}")

        # Combined
        total_pnl = long_stats.get('total_pnl', 0) + short_stats.get('total_pnl', 0)
        long_only_pnl = long_stats.get('total_pnl', 0)

        log(f"\nCOMBINED:")
        log(f"  Total P&L (Long+Short): ${total_pnl:,.2f}")
        log(f"  Long-Only P&L: ${long_only_pnl:,.2f}")
        log(f"  Short Contribution: ${short_stats.get('total_pnl', 0):,.2f}")

        if short_stats['count'] > 0:
            short_value = "ADDS VALUE" if short_stats.get('total_pnl', 0) > 0 else "DESTROYS VALUE"
            log(f"  Short Assessment: {short_value}")

    # Summary recommendation
    log("\n" + "=" * 70)
    log("RECOMMENDATION")
    log("=" * 70)


if __name__ == "__main__":
    main()
