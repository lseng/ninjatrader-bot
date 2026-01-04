#!/usr/bin/env python3
"""
Train ML model with Video Strategy Features

Uses the 2-year 1-second Databento data and all 12 video strategy features.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

# Import our video strategy features
from src.features.features import VideoStrategyFeatures, aggregate_to_timeframe


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_databento_data():
    """Load the 2-year 1-second Databento data."""
    data_path = Path("/Users/leoneng/Downloads/topstep-trading-bot/data/historical/MES/MES_1s_2years.parquet")

    if not data_path.exists():
        log(f"ERROR: Data file not found at {data_path}")
        return None

    log(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    log(f"Loaded {len(df):,} rows")
    log(f"Columns: {list(df.columns)}")
    log(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def create_labels(df, lookahead=30, threshold=0.003):
    """
    Create trading labels for ML model.

    Label:
        1 = Price goes up > threshold in next N bars
        -1 = Price goes down > threshold in next N bars
        0 = Price stays within range

    Args:
        df: DataFrame with 'close' column
        lookahead: Number of bars to look ahead
        threshold: Minimum price change for signal
    """
    close = df['close']
    future_max = close.shift(-lookahead).rolling(lookahead).max()
    future_min = close.shift(-lookahead).rolling(lookahead).min()

    future_return_up = (future_max - close) / close
    future_return_down = (close - future_min) / close

    labels = pd.Series(0, index=df.index)
    labels[future_return_up > threshold] = 1
    labels[future_return_down > threshold] = -1

    return labels


def train_model(X_train, y_train, X_test, y_test, feature_cols):
    """Train and evaluate a Random Forest model."""

    log(f"Training Random Forest on {len(X_train):,} samples...")

    model = RandomForestClassifier(
        n_estimators=60,
        max_depth=5,  # Shallow to prevent overfitting
        min_samples_leaf=50,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    # Evaluate
    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()

    log(f"Train accuracy: {train_acc:.1%}")
    log(f"Test accuracy: {test_acc:.1%}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    log("\nTop 15 Most Important Features:")
    for _, row in importance.head(15).iterrows():
        log(f"  {row['feature']}: {row['importance']:.1%}")

    return model, importance


def backtest_model(df, predictions, contract_value=5, commission=2.0):
    """Backtest the model predictions."""
    prices = df['close'].values
    capital = 1000
    position = 0
    entry_price = 0

    trades = []
    equity = [capital]

    for i in range(1, len(predictions)):
        pred = predictions[i]
        current_price = prices[i]
        prev_price = prices[i-1]

        # Update P&L for existing position
        if position != 0:
            capital += position * (current_price - prev_price) * contract_value

        # Position change
        if pred != position:
            if position != 0:
                # Close existing position
                pnl = (current_price - entry_price) * position * contract_value - commission
                trades.append({
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'pnl': pnl,
                    'entry': entry_price,
                    'exit': current_price
                })

            if pred != 0:
                position = pred
                entry_price = current_price
                capital -= commission / 2
            else:
                position = 0

        equity.append(capital)

    # Close final position
    if position != 0:
        pnl = (prices[-1] - entry_price) * position * contract_value - commission
        trades.append({
            'direction': 'LONG' if position == 1 else 'SHORT',
            'pnl': pnl,
            'entry': entry_price,
            'exit': prices[-1]
        })

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    return {
        'final_capital': equity[-1],
        'total_return': (equity[-1] - 1000) / 1000 * 100,
        'trades': trades_df,
        'equity': equity
    }


def main():
    log("=" * 70)
    log("TRAINING WITH VIDEO STRATEGY FEATURES")
    log("=" * 70)

    # Load data
    df_raw = load_databento_data()
    if df_raw is None:
        return

    # Prepare data
    df = df_raw.copy()

    # Rename columns if needed
    if 'ts_event' in df.columns:
        df = df.rename(columns={'ts_event': 'timestamp'})

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Set timestamp as index for resampling
    df = df.set_index('timestamp')

    # Test different timeframes
    timeframes = [15, 30, 60]  # Minutes

    for tf in timeframes:
        log(f"\n{'='*70}")
        log(f"TIMEFRAME: {tf} MINUTES")
        log(f"{'='*70}")

        # Aggregate to timeframe
        log(f"Aggregating to {tf}-minute bars...")
        agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
        if 'volume' in df.columns:
            agg_dict['volume'] = 'sum'

        df_tf = df.resample(f'{tf}min').agg(agg_dict).dropna()
        log(f"Aggregated to {len(df_tf):,} bars")

        # Create features
        log("Creating video strategy features...")
        feature_gen = VideoStrategyFeatures()
        features = feature_gen.create_features(df_tf.reset_index(), include_zones=False)

        # Create labels
        labels = create_labels(df_tf.reset_index(), lookahead=3, threshold=0.002)

        # Combine and drop NaN
        df_ml = pd.concat([features, labels.rename('label')], axis=1).dropna()
        log(f"Final dataset: {len(df_ml):,} samples")

        # Label distribution
        label_counts = df_ml['label'].value_counts()
        log(f"Label distribution:")
        log(f"  FLAT (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(df_ml)*100:.1f}%)")
        log(f"  LONG (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(df_ml)*100:.1f}%)")
        log(f"  SHORT (-1): {label_counts.get(-1, 0):,} ({label_counts.get(-1, 0)/len(df_ml)*100:.1f}%)")

        # Split data (time series split)
        feature_cols = [c for c in df_ml.columns if c != 'label']
        X = df_ml[feature_cols].values
        y = df_ml['label'].values

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        log(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

        # Train model
        model, importance = train_model(X_train, y_train, X_test, y_test, feature_cols)

        # Backtest on test set
        log("\nBacktesting on test set...")
        predictions = model.predict(X_test)

        # Long-only backtest
        predictions_long = np.where(predictions == 1, 1, 0)
        results_long = backtest_model(df_tf.iloc[train_size:train_size+len(predictions)], predictions_long)

        log(f"\nLONG-ONLY RESULTS:")
        log(f"  Final Capital: ${results_long['final_capital']:,.2f}")
        log(f"  Total Return: {results_long['total_return']:.1f}%")
        log(f"  Total Trades: {len(results_long['trades'])}")

        if len(results_long['trades']) > 0:
            trades_df = results_long['trades']
            wins = trades_df[trades_df['pnl'] > 0]
            win_rate = len(wins) / len(trades_df) * 100
            avg_pnl = trades_df['pnl'].mean()
            log(f"  Win Rate: {win_rate:.1f}%")
            log(f"  Avg P&L per Trade: ${avg_pnl:.2f}")

        # Full strategy backtest
        results_full = backtest_model(df_tf.iloc[train_size:train_size+len(predictions)], predictions)

        log(f"\nFULL STRATEGY (LONG+SHORT) RESULTS:")
        log(f"  Final Capital: ${results_full['final_capital']:,.2f}")
        log(f"  Total Return: {results_full['total_return']:.1f}%")
        log(f"  Total Trades: {len(results_full['trades'])}")

        if len(results_full['trades']) > 0:
            trades_df = results_full['trades']
            wins = trades_df[trades_df['pnl'] > 0]
            win_rate = len(wins) / len(trades_df) * 100
            avg_pnl = trades_df['pnl'].mean()
            log(f"  Win Rate: {win_rate:.1f}%")
            log(f"  Avg P&L per Trade: ${avg_pnl:.2f}")

            # Long vs Short breakdown
            longs = trades_df[trades_df['direction'] == 'LONG']
            shorts = trades_df[trades_df['direction'] == 'SHORT']
            log(f"\n  Long trades: {len(longs)}, P&L: ${longs['pnl'].sum():.2f}")
            log(f"  Short trades: {len(shorts)}, P&L: ${shorts['pnl'].sum():.2f}")

        # Save best model
        if tf == 30 and results_long['total_return'] > 0:
            model_dir = PROJECT_ROOT / "models" / "video_strategy_30m"
            model_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(model, model_dir / "model.joblib")
            importance.to_csv(model_dir / "feature_importance.csv", index=False)

            log(f"\nModel saved to {model_dir}")

    log("\n" + "=" * 70)
    log("TRAINING COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
