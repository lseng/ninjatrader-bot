#!/usr/bin/env python3
"""
Analyze performance across different timeframes and modes.
Identifies optimal trading frequency for the model.
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
        df["volume_ratio"] = df["volume"] / df["volume_sma_5"].replace(0, 1)

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

    return df


def run_backtest(df, model, long_only=False, initial_capital=1000, contract_value=5, commission=2.00):
    """Run backtest with optional long-only mode."""
    df = create_features(df)
    df = df.dropna().reset_index(drop=True)

    feature_cols = [c for c in df.columns
                    if c not in ["timestamp", "open", "high", "low", "close", "volume"]]

    X = df[feature_cols].values
    prices = df["close"].values

    predictions = model.predict(X)

    # Apply long-only filter
    if long_only:
        predictions = np.where(predictions == 1, 1, 0)

    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]

    long_pnl = 0
    short_pnl = 0

    for i in range(1, len(predictions)):
        pred = predictions[i]
        current_price = prices[i]
        prev_price = prices[i-1]

        if position != 0:
            point_change = current_price - prev_price
            pnl = position * point_change * contract_value
            capital += pnl

        if pred != position:
            if position != 0:
                trade_pnl = (current_price - entry_price) * position * contract_value - commission
                if position == 1:
                    long_pnl += trade_pnl
                else:
                    short_pnl += trade_pnl
                trades.append({
                    "direction": "LONG" if position == 1 else "SHORT",
                    "pnl": trade_pnl,
                    "bars_held": i - entry_bar
                })

            if pred != 0:
                position = pred
                entry_price = current_price
                entry_bar = i
                capital -= commission / 2
            else:
                position = 0

        equity_curve.append(capital)

    if position != 0:
        trade_pnl = (prices[-1] - entry_price) * position * contract_value - commission
        if position == 1:
            long_pnl += trade_pnl
        else:
            short_pnl += trade_pnl
        trades.append({
            "direction": "LONG" if position == 1 else "SHORT",
            "pnl": trade_pnl,
            "bars_held": len(prices) - entry_bar
        })

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    if len(trades_df) > 0:
        winning = trades_df[trades_df["pnl"] > 0]
        losing = trades_df[trades_df["pnl"] < 0]

        return {
            "total_return_pct": (equity_curve[-1] / initial_capital - 1) * 100,
            "total_pnl": equity_curve[-1] - initial_capital,
            "total_trades": len(trades_df),
            "win_rate": len(winning) / len(trades_df) * 100,
            "profit_factor": abs(winning["pnl"].sum() / losing["pnl"].sum()) if len(losing) > 0 and losing["pnl"].sum() != 0 else float("inf"),
            "max_drawdown": min(0, min(equity_curve) - max(equity_curve[:equity_curve.index(min(equity_curve))+1])),
            "long_pnl": long_pnl,
            "short_pnl": short_pnl,
            "long_trades": len(trades_df[trades_df["direction"] == "LONG"]),
            "short_trades": len(trades_df[trades_df["direction"] == "SHORT"]),
            "avg_bars_held": trades_df["bars_held"].mean()
        }
    return None


def main():
    log("=" * 70)
    log("TIMEFRAME & MODE ANALYSIS")
    log("=" * 70)

    # Load data
    log("Loading data...")
    df = pd.read_parquet("data/historical/MES_1m.parquet")
    holdout_start = int(len(df) * 0.85)
    df_holdout = df.iloc[holdout_start:].reset_index(drop=True)
    log(f"Holdout: {len(df_holdout):,} bars")

    # Load models
    models = {
        "shallow_rf": joblib.load("models/shallow_rf/model.joblib"),
        "ultra_rf": joblib.load("models/ultra_rf/model.joblib"),
    }

    # Test different sample rates (effectively different timeframes)
    sample_rates = [1, 5, 10, 15, 20, 30, 60]

    results = []

    for model_name, model in models.items():
        log(f"\n{'='*70}")
        log(f"MODEL: {model_name}")
        log(f"{'='*70}")

        for sr in sample_rates:
            df_sampled = df_holdout.iloc[::sr].reset_index(drop=True)
            timeframe = f"{sr}m"

            for long_only in [False, True]:
                mode = "LONG_ONLY" if long_only else "FULL"

                result = run_backtest(df_sampled, model, long_only=long_only)

                if result:
                    results.append({
                        "model": model_name,
                        "timeframe": timeframe,
                        "sample_rate": sr,
                        "mode": mode,
                        **result
                    })

                    log(f"{timeframe:>4} {mode:>10}: P&L={result['total_return_pct']:>8.1f}%, "
                        f"Trades={result['total_trades']:>5}, WR={result['win_rate']:.1f}%, "
                        f"PF={result['profit_factor']:.2f}")

    # Create summary
    results_df = pd.DataFrame(results)

    log("\n" + "=" * 70)
    log("BEST CONFIGURATIONS")
    log("=" * 70)

    # Best by P&L
    best_pnl = results_df.loc[results_df["total_return_pct"].idxmax()]
    log(f"\nBest P&L: {best_pnl['model']} @ {best_pnl['timeframe']} ({best_pnl['mode']})")
    log(f"  Return: {best_pnl['total_return_pct']:.1f}%")
    log(f"  Trades: {best_pnl['total_trades']}")
    log(f"  Win Rate: {best_pnl['win_rate']:.1f}%")
    log(f"  Profit Factor: {best_pnl['profit_factor']:.2f}")
    log(f"  Max Drawdown: ${best_pnl['max_drawdown']:,.0f}")

    # Best by Profit Factor (with min trades filter)
    valid = results_df[results_df["total_trades"] >= 50]
    if len(valid) > 0:
        best_pf = valid.loc[valid["profit_factor"].idxmax()]
        log(f"\nBest Profit Factor (min 50 trades): {best_pf['model']} @ {best_pf['timeframe']} ({best_pf['mode']})")
        log(f"  Profit Factor: {best_pf['profit_factor']:.2f}")
        log(f"  Return: {best_pf['total_return_pct']:.1f}%")
        log(f"  Trades: {best_pf['total_trades']}")

    # Compare long-only vs full
    log("\n" + "=" * 70)
    log("LONG-ONLY vs FULL COMPARISON (20m timeframe)")
    log("=" * 70)

    for model_name in models.keys():
        df_20m = results_df[(results_df["model"] == model_name) & (results_df["sample_rate"] == 20)]
        if len(df_20m) == 2:
            full = df_20m[df_20m["mode"] == "FULL"].iloc[0]
            long_only = df_20m[df_20m["mode"] == "LONG_ONLY"].iloc[0]
            log(f"\n{model_name}:")
            log(f"  FULL:      P&L={full['total_return_pct']:>7.1f}%, WR={full['win_rate']:.1f}%, PF={full['profit_factor']:.2f}")
            log(f"  LONG_ONLY: P&L={long_only['total_return_pct']:>7.1f}%, WR={long_only['win_rate']:.1f}%, PF={long_only['profit_factor']:.2f}")
            log(f"  Improvement: {long_only['total_return_pct'] - full['total_return_pct']:+.1f}%")

    # Save results
    output_path = Path("models/timeframe_analysis.json")
    with open(output_path, "w") as f:
        json.dump({
            "results": results,
            "best_pnl": best_pnl.to_dict(),
            "analysis_time": datetime.now().isoformat()
        }, f, indent=2, default=str)

    log(f"\nResults saved to {output_path}")

    return results_df


if __name__ == "__main__":
    main()
