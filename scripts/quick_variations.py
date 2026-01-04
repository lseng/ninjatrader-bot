#!/usr/bin/env python3
"""
Quick Variations - Try different feature sets and model configurations
to potentially find better strategies than the ultra-fast baseline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, accuracy_score


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


def load_data(data_path: str, sample_rate: int = 20):
    """Load and sample data."""
    log(f"Loading {data_path}...")
    df = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path, parse_dates=["timestamp"])
    log(f"Full: {len(df):,} bars")
    df = df.iloc[::sample_rate].reset_index(drop=True)
    log(f"Sampled (1/{sample_rate}): {len(df):,} bars")
    return df


def create_features_v1(df):
    """Original features (baseline)."""
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


def create_features_v2(df):
    """Extended features with more indicators."""
    df = create_features_v1(df)

    # Add more features
    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = (ema12 - ema26) / df["close"]

    # ATR
    df["atr"] = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift(1)),
        abs(df["low"] - df["close"].shift(1))
    ], axis=1).max(axis=1).rolling(10).mean() / df["close"]

    # Higher timeframe momentum
    df["roc_20"] = df["close"].pct_change(20)

    # Trend strength
    df["trend"] = df["close"].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)

    return df


def create_features_v3(df):
    """Minimal features (reduced complexity)."""
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["range"] = (df["high"] - df["low"]) / df["close"]

    # Just key moving averages
    df["sma_10"] = df["close"].rolling(10).mean()
    df["close_vs_sma_10"] = (df["close"] - df["sma_10"]) / df["sma_10"]

    # Momentum
    df["roc_5"] = df["close"].pct_change(5)

    # Volatility
    df["volatility"] = df["returns"].rolling(10).std()

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

    return df


def add_labels(df, lookahead=3, threshold=0.001):
    """Add labels to dataframe."""
    future_ret = df["close"].shift(-lookahead) / df["close"] - 1
    df["label"] = 0
    df.loc[future_ret > threshold, "label"] = 1
    df.loc[future_ret < -threshold, "label"] = -1
    return df


def run_experiment(df, feature_fn, model_class, model_params, name, output_dir):
    """Run a single experiment."""
    log(f"\n{'='*60}")
    log(f"EXPERIMENT: {name}")
    log(f"{'='*60}")

    # Create features
    df_feat = feature_fn(df.copy())
    df_feat = add_labels(df_feat, lookahead=3, threshold=0.001)
    df_feat = df_feat.dropna().reset_index(drop=True)

    # Get feature columns
    exclude = ["timestamp", "open", "high", "low", "close", "volume", "label"]
    feature_cols = [c for c in df_feat.columns if c not in exclude]
    X = df_feat[feature_cols].values
    y = df_feat["label"].values

    log(f"Features: {X.shape}, Labels dist: {np.bincount(y + 1)}")

    # Split
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_holdout, y_holdout = X[val_end:], y[val_end:]
    df_holdout = df_feat.iloc[val_end:].copy()

    log(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Holdout: {len(X_holdout):,}")

    # Train
    model = model_class(**model_params)
    log("Training...")
    X_dev = np.vstack([X_train, X_val])
    y_dev = np.hstack([y_train, y_val])
    model.fit(X_dev, y_dev)

    # Evaluate
    preds = model.predict(X_holdout)
    accuracy = accuracy_score(y_holdout, preds)
    f1 = f1_score(y_holdout, preds, average="weighted", zero_division=0)

    # P&L
    returns = df_holdout["close"].pct_change().values[1:]
    position = 0
    trades = 0
    wins = 0
    total_pnl = 0

    for i, pred in enumerate(preds[:-1]):
        if pred != position:
            trades += 1
            position = pred
        if position != 0 and i < len(returns):
            pnl = position * returns[i]
            total_pnl += pnl
            if pnl > 0:
                wins += 1

    win_rate = wins / max(trades, 1)

    log(f"Results: Acc={accuracy:.4f}, F1={f1:.4f}, P&L={total_pnl*100:.2f}%, WR={win_rate:.2%}, Trades={trades}")

    # Save
    exp_dir = output_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": name,
        "timestamp": datetime.now().isoformat(),
        "model_params": {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v for k, v in model_params.items()},
        "holdout_metrics": {
            "accuracy": accuracy,
            "f1": f1,
            "total_pnl_pct": total_pnl * 100,
            "trades": trades,
            "win_rate": win_rate
        }
    }

    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    joblib.dump(model, exp_dir / "model.joblib")

    return results


def main():
    log("="*60)
    log("QUICK VARIATIONS")
    log("="*60)

    df = load_data("data/historical/MES_1m.parquet", sample_rate=20)
    output_dir = Path("models")

    experiments = [
        # Baseline with best params
        ("baseline_rf", create_features_v1, RandomForestClassifier,
         {"n_estimators": 38, "max_depth": 8, "min_samples_split": 15, "random_state": 42, "n_jobs": -1}),

        # Extra Trees (often better than RF)
        ("extra_trees", create_features_v1, ExtraTreesClassifier,
         {"n_estimators": 50, "max_depth": 8, "min_samples_split": 15, "random_state": 42, "n_jobs": -1}),

        # Deeper forest
        ("deep_rf", create_features_v1, RandomForestClassifier,
         {"n_estimators": 50, "max_depth": 12, "min_samples_split": 10, "random_state": 42, "n_jobs": -1}),

        # Shallow forest (more conservative)
        ("shallow_rf", create_features_v1, RandomForestClassifier,
         {"n_estimators": 60, "max_depth": 5, "min_samples_split": 20, "random_state": 42, "n_jobs": -1}),

        # Extended features
        ("extended_features", create_features_v2, RandomForestClassifier,
         {"n_estimators": 38, "max_depth": 8, "min_samples_split": 15, "random_state": 42, "n_jobs": -1}),

        # Minimal features (simpler model)
        ("minimal_features", create_features_v3, RandomForestClassifier,
         {"n_estimators": 50, "max_depth": 6, "min_samples_split": 20, "random_state": 42, "n_jobs": -1}),

        # Large ensemble
        ("large_ensemble", create_features_v1, RandomForestClassifier,
         {"n_estimators": 100, "max_depth": 8, "min_samples_split": 15, "random_state": 42, "n_jobs": -1}),
    ]

    all_results = []
    for name, feat_fn, model_cls, params in experiments:
        try:
            result = run_experiment(df, feat_fn, model_cls, params, name, output_dir)
            all_results.append(result)
        except Exception as e:
            log(f"Experiment {name} failed: {e}")

    # Summary
    log(f"\n{'='*60}")
    log("QUICK VARIATIONS COMPLETE")
    log(f"{'='*60}")

    if all_results:
        all_results.sort(key=lambda x: x["holdout_metrics"]["total_pnl_pct"], reverse=True)
        log("\nRanked by P&L:")
        for i, r in enumerate(all_results, 1):
            m = r["holdout_metrics"]
            log(f"  {i}. {r['experiment']}: P&L={m['total_pnl_pct']:.2f}%, WR={m['win_rate']:.2%}")

        with open(output_dir / "quick_variations_summary.json", "w") as f:
            json.dump({"experiments": all_results, "best": all_results[0]}, f, indent=2)


if __name__ == "__main__":
    main()
