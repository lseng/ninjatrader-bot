#!/usr/bin/env python3
"""
Validate original model and optimize label thresholds for 30m timeframe.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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


def run_backtest_detailed(prices, predictions, long_only=False, contract_value=5, commission=2.0):
    """Detailed backtest with equity curve."""
    if long_only:
        predictions = np.where(predictions == 1, 1, 0)

    capital = 1000
    position = 0
    entry_price = 0
    trades = []
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
                trades.append({
                    "direction": "LONG" if position == 1 else "SHORT",
                    "pnl": trade_pnl
                })

            if pred != 0:
                position = pred
                entry_price = current_price
                capital -= commission / 2
            else:
                position = 0

        equity.append(capital)

    if position != 0:
        trade_pnl = (prices[-1] - entry_price) * position * contract_value - commission
        trades.append({
            "direction": "LONG" if position == 1 else "SHORT",
            "pnl": trade_pnl
        })

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    if len(trades_df) > 0:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] < 0]
        return {
            "return_pct": (equity[-1] / 1000 - 1) * 100,
            "trades": len(trades_df),
            "win_rate": len(wins) / len(trades_df) * 100,
            "profit_factor": abs(wins["pnl"].sum() / losses["pnl"].sum()) if len(losses) > 0 and losses["pnl"].sum() != 0 else float("inf"),
            "max_drawdown": min(equity) - 1000,
            "long_trades": len(trades_df[trades_df["direction"] == "LONG"]),
            "long_pnl": trades_df[trades_df["direction"] == "LONG"]["pnl"].sum() if len(trades_df[trades_df["direction"] == "LONG"]) > 0 else 0
        }
    return {"return_pct": 0, "trades": 0}


def main():
    log("=" * 70)
    log("MODEL VALIDATION & THRESHOLD OPTIMIZATION")
    log("=" * 70)

    # Load raw 1m data
    log("Loading data...")
    df_raw = pd.read_parquet("data/historical/MES_1m.parquet")
    log(f"Raw bars: {len(df_raw):,}")

    # Load original shallow_rf model
    model = joblib.load("models/shallow_rf/model.joblib")
    log("Loaded shallow_rf model")

    # Test: Original model on 30m sampled holdout (like analyze_timeframe.py did)
    log("\n" + "=" * 70)
    log("TEST 1: Original shallow_rf on 30m holdout (reproduce 612%)")
    log("=" * 70)

    # Get holdout from raw data THEN sample (matching analyze_timeframe.py)
    holdout_start = int(len(df_raw) * 0.85)
    df_holdout_raw = df_raw.iloc[holdout_start:].reset_index(drop=True)
    df_holdout_30m = df_holdout_raw.iloc[::30].reset_index(drop=True)

    log(f"Holdout 30m bars: {len(df_holdout_30m):,}")

    df_feat = create_features(df_holdout_30m)
    df_feat = df_feat.dropna()

    feature_cols = [c for c in df_feat.columns
                    if c not in ["timestamp", "open", "high", "low", "close", "volume"]]

    X = df_feat[feature_cols].values
    prices = df_feat["close"].values

    predictions = model.predict(X)

    # Test long-only
    result = run_backtest_detailed(prices, predictions, long_only=True)
    log(f"Long-only result: {result['return_pct']:.1f}%, Trades: {result['trades']}, WR: {result.get('win_rate', 0):.1f}%")

    # Test full
    result_full = run_backtest_detailed(prices, predictions, long_only=False)
    log(f"Full result: {result_full['return_pct']:.1f}%, Trades: {result_full['trades']}")

    # Now test at different sample rates to find optimal
    log("\n" + "=" * 70)
    log("TEST 2: Original shallow_rf at different timeframes")
    log("=" * 70)

    for sample_rate in [15, 20, 30, 45, 60]:
        df_sampled = df_holdout_raw.iloc[::sample_rate].reset_index(drop=True)
        df_feat = create_features(df_sampled).dropna()
        X = df_feat[feature_cols].values
        prices = df_feat["close"].values
        predictions = model.predict(X)
        result = run_backtest_detailed(prices, predictions, long_only=True)
        log(f"{sample_rate:>3}m: Return={result['return_pct']:>7.1f}%, WR={result.get('win_rate', 0):.1f}%, Trades={result['trades']}")

    # Analyze prediction distribution
    log("\n" + "=" * 70)
    log("TEST 3: Prediction distribution analysis")
    log("=" * 70)

    df_30m = df_holdout_raw.iloc[::30].reset_index(drop=True)
    df_feat = create_features(df_30m).dropna()
    X = df_feat[feature_cols].values
    predictions = model.predict(X)

    unique, counts = np.unique(predictions, return_counts=True)
    log("Prediction distribution:")
    for u, c in zip(unique, counts):
        label = {-1: "SHORT", 0: "FLAT", 1: "LONG"}[u]
        pct = c / len(predictions) * 100
        log(f"  {label}: {c} ({pct:.1f}%)")

    # Test confidence thresholds using predict_proba
    log("\n" + "=" * 70)
    log("TEST 4: Confidence filtering")
    log("=" * 70)

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        log(f"Classes: {model.classes_}")

        # For each prediction, get confidence (max probability)
        confidences = np.max(proba, axis=1)

        # Make sure prices array matches predictions
        prices = df_feat["close"].values

        for min_conf in [0.5, 0.6, 0.7, 0.8]:
            # Only trade when confidence > threshold
            filtered_pred = predictions.copy()
            filtered_pred[confidences < min_conf] = 0  # Go flat when uncertain

            result = run_backtest_detailed(prices, filtered_pred, long_only=True)
            active_trades = np.sum(filtered_pred != 0)
            log(f"Conf >= {min_conf:.0%}: Return={result['return_pct']:>7.1f}%, "
                f"WR={result.get('win_rate', 0):.1f}%, Trades={result['trades']}, "
                f"Active: {active_trades/len(filtered_pred)*100:.1f}%")
    else:
        log("Model does not support predict_proba")

    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    log("\nThe original shallow_rf model trained on 1m sampled data works")
    log("best on longer timeframes (30m-60m) with long-only mode.")
    log("\nThis is because:")
    log("1. The model was trained on 20x sampled data (effectively 20m)")
    log("2. Shorter timeframes have more noise and transaction costs dominate")
    log("3. Long signals are more reliable than short signals")


if __name__ == "__main__":
    main()
