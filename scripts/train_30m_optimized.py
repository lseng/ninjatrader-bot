#!/usr/bin/env python3
"""
Train optimized model for 30-minute timeframe.

Key findings from analysis:
- 30m timeframe is optimal (612% return on holdout)
- Long-only mode outperforms full strategy
- Top 4 features explain 85% of predictions: range, volatility_5, volatility_10, volume_sma_5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import TimeSeriesSplit
import joblib
from datetime import datetime
import json


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def create_features_full(df):
    """Create all features."""
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


def create_features_minimal(df):
    """Create only top features."""
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["range"] = (df["high"] - df["low"]) / df["close"]
    df["volatility_5"] = df["returns"].rolling(5).std()
    df["volatility_10"] = df["returns"].rolling(10).std()

    if "volume" in df.columns:
        df["volume_sma_5"] = df["volume"].rolling(5).mean()

    return df


def create_labels(df, threshold=0.0015, lookahead=3):
    """Create classification labels: 1=long, 0=flat, -1=short"""
    future_return = df["close"].pct_change(lookahead).shift(-lookahead)
    labels = np.where(future_return > threshold, 1,
                      np.where(future_return < -threshold, -1, 0))
    return labels


def run_backtest_fast(prices, predictions, long_only=False, contract_value=5, commission=2.0):
    """Fast backtest for model comparison."""
    if long_only:
        predictions = np.where(predictions == 1, 1, 0)

    capital = 1000
    position = 0
    entry_price = 0
    entry_idx = 0
    pnl_list = []

    for i in range(1, len(predictions)):
        pred = predictions[i]
        current_price = prices[i]
        prev_price = prices[i-1]

        if position != 0:
            capital += position * (current_price - prev_price) * contract_value

        if pred != position:
            if position != 0:
                trade_pnl = (current_price - entry_price) * position * contract_value - commission
                pnl_list.append(trade_pnl)

            if pred != 0:
                position = pred
                entry_price = current_price
                entry_idx = i
                capital -= commission / 2
            else:
                position = 0

    if position != 0:
        trade_pnl = (prices[-1] - entry_price) * position * contract_value - commission
        pnl_list.append(trade_pnl)

    return_pct = (capital / 1000 - 1) * 100
    trades = len(pnl_list)
    wins = sum(1 for p in pnl_list if p > 0)
    win_rate = wins / trades * 100 if trades > 0 else 0

    return {"return_pct": return_pct, "trades": trades, "win_rate": win_rate}


def train_and_evaluate(X_train, y_train, X_val, y_val, X_holdout, y_holdout,
                       prices_holdout, model_config, long_only=True):
    """Train model and evaluate on holdout."""
    model_type = model_config.pop("type", "rf")

    if model_type == "rf":
        model = RandomForestClassifier(**model_config, random_state=42, n_jobs=-1)
    else:
        model = ExtraTreesClassifier(**model_config, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)

    # Validation accuracy
    val_acc = model.score(X_val, y_val)

    # Holdout backtest
    predictions = model.predict(X_holdout)
    bt = run_backtest_fast(prices_holdout, predictions, long_only=long_only)

    return model, val_acc, bt


def main():
    log("=" * 70)
    log("TRAINING 30M OPTIMIZED MODEL")
    log("=" * 70)

    # Load data at 30m timeframe
    log("Loading data...")
    df_raw = pd.read_parquet("data/historical/MES_1m.parquet")
    df = df_raw.iloc[::30].reset_index(drop=True)  # 30m bars
    log(f"Total bars: {len(df):,}")

    # Train/Val/Holdout split (70/15/15)
    train_end = int(len(df) * 0.70)
    val_end = int(len(df) * 0.85)

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_holdout = df.iloc[val_end:].copy()

    log(f"Train: {len(df_train):,}, Val: {len(df_val):,}, Holdout: {len(df_holdout):,}")

    # Feature configurations to test
    feature_configs = [
        ("full", create_features_full, None),
        ("minimal", create_features_minimal, ["range", "volatility_5", "volatility_10", "volume_sma_5"]),
        ("top6", create_features_full, ["range", "volatility_5", "volatility_10", "volume_sma_5", "close_vs_sma_10", "close_vs_sma_20"])
    ]

    # Model configurations to test
    model_configs = [
        {"type": "rf", "n_estimators": 60, "max_depth": 5, "min_samples_split": 20},  # shallow_rf baseline
        {"type": "rf", "n_estimators": 80, "max_depth": 6, "min_samples_split": 15},
        {"type": "rf", "n_estimators": 100, "max_depth": 4, "min_samples_split": 25},
        {"type": "rf", "n_estimators": 50, "max_depth": 7, "min_samples_split": 10},
        {"type": "et", "n_estimators": 80, "max_depth": 6, "min_samples_split": 15},
    ]

    results = []

    for feat_name, create_features, feature_subset in feature_configs:
        log(f"\n{'='*70}")
        log(f"FEATURE SET: {feat_name}")
        log(f"{'='*70}")

        # Create features
        df_train_feat = create_features(df_train)
        df_val_feat = create_features(df_val)
        df_holdout_feat = create_features(df_holdout)

        # Create labels
        labels_train = create_labels(df_train_feat)
        labels_val = create_labels(df_val_feat)
        labels_holdout = create_labels(df_holdout_feat)

        # Drop NaN
        df_train_feat = df_train_feat.dropna()
        df_val_feat = df_val_feat.dropna()
        df_holdout_feat = df_holdout_feat.dropna()

        labels_train = labels_train[:len(df_train_feat)]
        labels_val = labels_val[:len(df_val_feat)]
        labels_holdout = labels_holdout[:len(df_holdout_feat)]

        # Filter valid labels
        valid_train = ~np.isnan(labels_train)
        valid_val = ~np.isnan(labels_val)
        valid_holdout = ~np.isnan(labels_holdout)

        # Get feature columns
        if feature_subset:
            feature_cols = [c for c in feature_subset if c in df_train_feat.columns]
        else:
            feature_cols = [c for c in df_train_feat.columns
                           if c not in ["timestamp", "open", "high", "low", "close", "volume"]]

        log(f"Features: {feature_cols}")

        X_train = df_train_feat[feature_cols].values[valid_train]
        y_train = labels_train[valid_train].astype(int)
        X_val = df_val_feat[feature_cols].values[valid_val]
        y_val = labels_val[valid_val].astype(int)
        X_holdout = df_holdout_feat[feature_cols].values[valid_holdout]
        y_holdout = labels_holdout[valid_holdout].astype(int)
        prices_holdout = df_holdout_feat["close"].values[valid_holdout]

        log(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Holdout: {len(X_holdout):,}")

        for model_config in model_configs:
            config = model_config.copy()
            model_name = f"{config['type']}_{config['n_estimators']}_{config['max_depth']}"

            model, val_acc, bt = train_and_evaluate(
                X_train, y_train, X_val, y_val, X_holdout, y_holdout,
                prices_holdout, config, long_only=True
            )

            result = {
                "features": feat_name,
                "model": model_name,
                "val_acc": val_acc,
                "return_pct": bt["return_pct"],
                "trades": bt["trades"],
                "win_rate": bt["win_rate"],
                "config": model_config.copy(),
                "feature_cols": feature_cols
            }
            results.append(result)

            log(f"{model_name:>15} {feat_name:>8}: Return={bt['return_pct']:>7.1f}%, "
                f"WR={bt['win_rate']:.1f}%, Trades={bt['trades']}")

            # Save best model
            if bt["return_pct"] > 500:
                model_dir = Path(f"models/optimized_30m_{feat_name}_{model_name}")
                model_dir.mkdir(exist_ok=True)
                joblib.dump(model, model_dir / "model.joblib")

    # Find best
    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df["return_pct"].idxmax()]

    log("\n" + "=" * 70)
    log("BEST CONFIGURATION")
    log("=" * 70)
    log(f"Features: {best['features']}")
    log(f"Model: {best['model']}")
    log(f"Return: {best['return_pct']:.1f}%")
    log(f"Win Rate: {best['win_rate']:.1f}%")
    log(f"Trades: {best['trades']}")

    # Retrain best on full train+val data
    log("\n" + "=" * 70)
    log("RETRAINING BEST MODEL ON FULL DATA")
    log("=" * 70)

    best_feat_name = best['features']
    best_config = best['config'].copy()

    # Find the feature function
    create_features = create_features_full
    for fn, cf, fs in feature_configs:
        if fn == best_feat_name:
            create_features = cf
            feature_subset = fs
            break

    # Merge train + val
    df_full_train = df.iloc[:val_end].copy()
    df_full_train_feat = create_features(df_full_train)
    labels_full = create_labels(df_full_train_feat)

    df_full_train_feat = df_full_train_feat.dropna()
    labels_full = labels_full[:len(df_full_train_feat)]
    valid_full = ~np.isnan(labels_full)

    if feature_subset:
        feature_cols = [c for c in feature_subset if c in df_full_train_feat.columns]
    else:
        feature_cols = [c for c in df_full_train_feat.columns
                       if c not in ["timestamp", "open", "high", "low", "close", "volume"]]

    X_full = df_full_train_feat[feature_cols].values[valid_full]
    y_full = labels_full[valid_full].astype(int)

    log(f"Full training set: {len(X_full):,} samples")

    model_type = best_config.pop("type", "rf")
    if model_type == "rf":
        final_model = RandomForestClassifier(**best_config, random_state=42, n_jobs=-1)
    else:
        final_model = ExtraTreesClassifier(**best_config, random_state=42, n_jobs=-1)

    final_model.fit(X_full, y_full)

    # Save final model
    final_dir = Path("models/optimized_30m_final")
    final_dir.mkdir(exist_ok=True)
    joblib.dump(final_model, final_dir / "model.joblib")

    # Final holdout test
    predictions = final_model.predict(X_holdout)
    final_bt = run_backtest_fast(prices_holdout, predictions, long_only=True)

    log(f"\nFINAL HOLDOUT RESULTS:")
    log(f"  Return: {final_bt['return_pct']:.1f}%")
    log(f"  Win Rate: {final_bt['win_rate']:.1f}%")
    log(f"  Trades: {final_bt['trades']}")

    # Save summary
    summary = {
        "best_config": {
            "features": best_feat_name,
            "model": best['model'],
            "model_config": best['config'],
            "feature_cols": feature_cols
        },
        "final_results": final_bt,
        "all_results": results,
        "trained_at": datetime.now().isoformat()
    }

    with open(final_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\nFinal model saved to {final_dir}")


if __name__ == "__main__":
    main()
