#!/usr/bin/env python3
"""
Aggressive Training - Try different thresholds and features
to potentially find higher profit strategies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


def load_data(data_path: str, sample_rate: int = 10):
    """Load data with moderate sampling."""
    log(f"Loading {data_path}...")

    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, parse_dates=["timestamp"])

    log(f"Full: {len(df):,} bars")

    # Sample for faster training
    df = df.iloc[::sample_rate].reset_index(drop=True)
    log(f"Sampled (1/{sample_rate}): {len(df):,} bars")

    return df


def create_features_with_threshold(df, lookahead=3, threshold=0.001):
    """Create features with configurable threshold."""
    df = df.copy()

    # Price features
    df["returns"] = df["close"].pct_change()
    df["range"] = (df["high"] - df["low"]) / df["close"]
    df["body"] = (df["close"] - df["open"]) / df["close"]
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]

    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f"sma_{period}"] = df["close"].rolling(period).mean()
        df[f"close_vs_sma_{period}"] = (df["close"] - df[f"sma_{period}"]) / df[f"sma_{period}"]

    # EMA
    for period in [5, 10, 20]:
        df[f"ema_{period}"] = df["close"].ewm(span=period).mean()
        df[f"close_vs_ema_{period}"] = (df["close"] - df[f"ema_{period}"]) / df[f"ema_{period}"]

    # Momentum
    for period in [1, 3, 5, 10, 20]:
        df[f"roc_{period}"] = df["close"].pct_change(period)

    # Volatility
    for period in [5, 10, 20]:
        df[f"volatility_{period}"] = df["returns"].rolling(period).std()

    # ATR-like
    df["true_range"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        )
    )
    df["atr_10"] = df["true_range"].rolling(10).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Volume features (if available)
    if "volume" in df.columns:
        df["volume_sma_5"] = df["volume"].rolling(5).mean()
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_5"]

    # Labels with configurable threshold
    future_ret = df["close"].shift(-lookahead) / df["close"] - 1
    df["label"] = 0
    df.loc[future_ret > threshold, "label"] = 1
    df.loc[future_ret < -threshold, "label"] = -1

    # Drop NaN
    df = df.dropna().reset_index(drop=True)

    # Feature columns
    exclude = ["timestamp", "open", "high", "low", "close", "volume", "label",
               "sma_5", "sma_10", "sma_20", "sma_50", "ema_5", "ema_10", "ema_20",
               "bb_mid", "bb_std", "bb_upper", "bb_lower", "true_range"]
    feature_cols = [c for c in df.columns if c not in exclude]

    return df[feature_cols].values, df["label"].values, df


def run_threshold_experiment(df, threshold, lookahead, n_trials=20, output_dir=Path("models")):
    """Run experiment with specific threshold."""
    name = f"thresh_{int(threshold*10000)}bp_look{lookahead}"
    log(f"\n{'='*60}")
    log(f"EXPERIMENT: {name}")
    log(f"Threshold: {threshold*100:.2f}%, Lookahead: {lookahead} bars")
    log(f"{'='*60}")

    # Create features
    X, y, df_proc = create_features_with_threshold(df.copy(), lookahead, threshold)
    log(f"Features: {X.shape}, Labels dist: {np.bincount(y + 1)}")

    # Split: 70% train, 15% val, 15% holdout
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_holdout, y_holdout = X[val_end:], y[val_end:]
    df_holdout = df_proc.iloc[val_end:].copy()

    log(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Holdout: {len(X_holdout):,}")

    def objective(trial):
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 20, 80),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_samples_split=trial.suggest_int("min_samples_split", 5, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 15),
            n_jobs=4,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return f1_score(y_val, preds, average="weighted", zero_division=0)

    # Run optimization
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log(f"Best score: {study.best_value:.4f}")

    # Train final model on train+val
    X_dev = np.vstack([X_train, X_val])
    y_dev = np.hstack([y_train, y_val])

    bp = study.best_params
    final_model = RandomForestClassifier(
        n_estimators=bp["n_estimators"],
        max_depth=bp["max_depth"],
        min_samples_split=bp["min_samples_split"],
        min_samples_leaf=bp["min_samples_leaf"],
        n_jobs=-1,
        random_state=42
    )
    final_model.fit(X_dev, y_dev)

    # Evaluate on holdout
    preds = final_model.predict(X_holdout)
    accuracy = accuracy_score(y_holdout, preds)
    f1 = f1_score(y_holdout, preds, average="weighted", zero_division=0)

    # P&L calculation
    returns = df_holdout["close"].pct_change().values[1:]
    position = 0
    trades = 0
    wins = 0
    total_pnl = 0
    pnl_curve = [0]

    for i, pred in enumerate(preds[:-1]):
        if pred != position:
            trades += 1
            position = pred
        if position != 0 and i < len(returns):
            pnl = position * returns[i]
            total_pnl += pnl
            if pnl > 0:
                wins += 1
        pnl_curve.append(total_pnl)

    win_rate = wins / max(trades, 1)
    max_drawdown = min(0, min(pnl_curve) - max(pnl_curve[:pnl_curve.index(min(pnl_curve))+1]))

    log(f"Holdout: Acc={accuracy:.4f}, F1={f1:.4f}, P&L={total_pnl*100:.2f}%, WR={win_rate:.2%}, Trades={trades}")

    # Save results
    exp_dir = output_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": name,
        "timestamp": datetime.now().isoformat(),
        "config": {"threshold": threshold, "lookahead": lookahead},
        "best_params": bp,
        "best_val_score": study.best_value,
        "holdout_metrics": {
            "accuracy": accuracy,
            "f1": f1,
            "total_pnl_pct": total_pnl * 100,
            "trades": trades,
            "win_rate": win_rate,
            "max_drawdown_pct": max_drawdown * 100
        }
    }

    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    import joblib
    joblib.dump(final_model, exp_dir / "model.joblib")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/historical/MES_1m.parquet")
    parser.add_argument("--sample-rate", type=int, default=10)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output", default="models")
    args = parser.parse_args()

    log("="*60)
    log("AGGRESSIVE THRESHOLD EXPLORATION")
    log("="*60)

    # Load data
    df = load_data(args.data, args.sample_rate)

    output_dir = Path(args.output)

    # Test different configurations
    configs = [
        # (threshold, lookahead)
        (0.0005, 3),   # 5bp threshold, 3-bar look
        (0.0015, 3),   # 15bp threshold, 3-bar look
        (0.002, 3),    # 20bp threshold, 3-bar look
        (0.001, 5),    # 10bp threshold, 5-bar look
        (0.001, 10),   # 10bp threshold, 10-bar look
        (0.002, 5),    # 20bp threshold, 5-bar look
    ]

    all_results = []

    for threshold, lookahead in configs:
        try:
            result = run_threshold_experiment(
                df, threshold, lookahead,
                n_trials=args.trials,
                output_dir=output_dir
            )
            all_results.append(result)
        except Exception as e:
            log(f"Experiment failed: {e}")

    # Summary
    log(f"\n{'='*60}")
    log("AGGRESSIVE TRAINING COMPLETE")
    log(f"{'='*60}")

    if all_results:
        # Sort by P&L
        all_results.sort(key=lambda x: x["holdout_metrics"]["total_pnl_pct"], reverse=True)

        log("\nTop 3 Strategies by P&L:")
        for i, r in enumerate(all_results[:3], 1):
            m = r["holdout_metrics"]
            log(f"  {i}. {r['experiment']}")
            log(f"     P&L: {m['total_pnl_pct']:.2f}%, WR: {m['win_rate']:.2%}, Trades: {m['trades']}")

        # Save summary
        summary = {
            "completed_at": datetime.now().isoformat(),
            "experiments": all_results,
            "best": all_results[0]
        }
        with open(output_dir / "aggressive_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    log(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
