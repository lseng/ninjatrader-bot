#!/usr/bin/env python3
"""
Train the optimal 60-minute long-only model with video strategy features.

This is the production-ready training script that:
1. Uses 2-year 1-second Databento data aggregated to 60m
2. Applies all 12 video strategy features
3. Trains a shallow Random Forest (anti-overfitting)
4. Saves the model for production use
"""

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

from src.features.features import VideoStrategyFeatures


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def create_labels(df, lookahead=3, threshold=0.002):
    """Create trading labels."""
    close = df['close']
    future_max = close.shift(-lookahead).rolling(lookahead).max()
    future_min = close.shift(-lookahead).rolling(lookahead).min()

    future_return_up = (future_max - close) / close
    future_return_down = (close - future_min) / close

    labels = pd.Series(0, index=df.index)
    labels[future_return_up > threshold] = 1
    labels[future_return_down > threshold] = -1

    return labels


def main():
    log("=" * 70)
    log("TRAINING 60-MINUTE PRODUCTION MODEL")
    log("=" * 70)

    # Load data
    data_path = Path("/Users/leoneng/Downloads/topstep-trading-bot/data/historical/MES/MES_1s_2years.parquet")
    log(f"Loading data from {data_path}...")
    df_raw = pd.read_parquet(data_path)
    log(f"Loaded {len(df_raw):,} 1-second bars")

    # Prepare and aggregate
    df = df_raw.copy()
    df = df.set_index('timestamp')

    log("Aggregating to 60-minute bars...")
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in df.columns:
        agg_dict['volume'] = 'sum'

    df_60m = df.resample('60min').agg(agg_dict).dropna()
    log(f"Aggregated to {len(df_60m):,} bars")

    # Create features
    log("Creating video strategy features...")
    feature_gen = VideoStrategyFeatures()
    features = feature_gen.create_features(df_60m.reset_index(), include_zones=False)

    # Create labels
    labels = create_labels(df_60m.reset_index(), lookahead=3, threshold=0.002)

    # Combine and clean
    df_ml = pd.concat([features, labels.rename('label')], axis=1).dropna()
    log(f"Final dataset: {len(df_ml):,} samples")

    # Feature columns
    feature_cols = [c for c in df_ml.columns if c != 'label']

    # Split data (80/20 time series split)
    train_size = int(len(df_ml) * 0.8)
    X_train = df_ml[feature_cols].iloc[:train_size].values
    y_train = df_ml['label'].iloc[:train_size].values
    X_test = df_ml[feature_cols].iloc[train_size:].values
    y_test = df_ml['label'].iloc[train_size:].values

    log(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    # Train model
    log("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=60,
        max_depth=5,
        min_samples_leaf=50,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = (train_pred == y_train).mean()
    test_acc = (test_pred == y_test).mean()

    log(f"Train accuracy: {train_acc:.1%}")
    log(f"Test accuracy: {test_acc:.1%}")

    # Backtest
    log("\nBacktesting long-only on test set...")
    prices = df_60m.iloc[train_size:train_size+len(test_pred)]['close'].values

    # Long-only predictions
    predictions = np.where(test_pred == 1, 1, 0)

    capital = 1000
    position = 0
    entry_price = 0
    trades = []
    contract_value = 5
    commission = 2.0

    for i in range(1, len(predictions)):
        pred = predictions[i]
        current_price = prices[i]
        prev_price = prices[i-1]

        if position != 0:
            capital += position * (current_price - prev_price) * contract_value

        if pred != position:
            if position != 0:
                pnl = (current_price - entry_price) * position * contract_value - commission
                trades.append(pnl)

            if pred != 0:
                position = pred
                entry_price = current_price
                capital -= commission / 2
            else:
                position = 0

    # Close final position
    if position != 0:
        pnl = (prices[-1] - entry_price) * position * contract_value - commission
        trades.append(pnl)

    trades = np.array(trades)
    wins = trades[trades > 0]
    losses = trades[trades <= 0]

    log(f"\n{'='*50}")
    log("LONG-ONLY RESULTS:")
    log(f"{'='*50}")
    log(f"Final Capital: ${capital:,.2f}")
    log(f"Total Return: {(capital - 1000) / 1000 * 100:.1f}%")
    log(f"Total Trades: {len(trades)}")
    log(f"Win Rate: {len(wins) / len(trades) * 100:.1f}%")
    log(f"Avg Win: ${wins.mean():.2f}" if len(wins) > 0 else "Avg Win: N/A")
    log(f"Avg Loss: ${losses.mean():.2f}" if len(losses) > 0 else "Avg Loss: N/A")
    log(f"Profit Factor: {abs(wins.sum() / losses.sum()):.2f}" if len(losses) > 0 and losses.sum() != 0 else "Profit Factor: Inf")

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    log(f"\nTop 20 Features:")
    for _, row in importance.head(20).iterrows():
        log(f"  {row['feature']}: {row['importance']:.1%}")

    # Save model
    model_dir = PROJECT_ROOT / "models" / "production_60m_video"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "model.joblib")
    importance.to_csv(model_dir / "feature_importance.csv", index=False)

    # Save config
    config = {
        "name": "60m_video_strategy_longonly",
        "timeframe": "60min",
        "mode": "long_only",
        "features": feature_cols,
        "model_type": "RandomForestClassifier",
        "model_params": {
            "n_estimators": 60,
            "max_depth": 5,
            "min_samples_leaf": 50,
            "max_features": "sqrt"
        },
        "training_config": {
            "lookahead": 3,
            "threshold": 0.002,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        },
        "performance": {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "test_return_pct": float((capital - 1000) / 1000 * 100),
            "total_trades": int(len(trades)),
            "win_rate": float(len(wins) / len(trades) * 100) if len(trades) > 0 else 0,
            "profit_factor": float(abs(wins.sum() / losses.sum())) if len(losses) > 0 and losses.sum() != 0 else None
        },
        "trading_config": {
            "timeframe": "60min",
            "mode": "long_only",
            "contract_value": 5,
            "commission_per_trade": 2.0
        },
        "created_at": datetime.now().isoformat()
    }

    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log(f"\nModel saved to {model_dir}/")
    log(f"Files: model.joblib, config.json, feature_importance.csv")

    log("\n" + "=" * 70)
    log("TRAINING COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
