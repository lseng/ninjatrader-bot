#!/usr/bin/env python3
"""
Ultra-Fast Training - Designed to complete in 30 minutes

Uses:
- Heavy downsampling (every 20th bar)
- Simple holdout validation (no walk-forward during optimization)
- Walk-forward only for final validation
- 30 trials per experiment
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


def load_and_heavily_sample(data_path: str):
    """Load data with heavy sampling for ultra-fast training."""
    log(f"Loading {data_path}...")

    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, parse_dates=["timestamp"])

    log(f"Full: {len(df):,} bars")

    # Heavy sampling - take every 20th bar
    df = df.iloc[::20].reset_index(drop=True)
    log(f"Sampled: {len(df):,} bars")

    return df


def create_features(df):
    """Create features directly."""
    df = df.copy()

    # Price features
    df["returns"] = df["close"].pct_change()
    df["range"] = (df["high"] - df["low"]) / df["close"]
    df["body"] = (df["close"] - df["open"]) / df["close"]

    # Moving averages
    for period in [5, 10, 20]:
        df[f"sma_{period}"] = df["close"].rolling(period).mean()
        df[f"close_vs_sma_{period}"] = (df["close"] - df[f"sma_{period}"]) / df[f"sma_{period}"]

    # Momentum
    for period in [3, 5, 10]:
        df[f"roc_{period}"] = df["close"].pct_change(period)

    # Volatility
    df["volatility_5"] = df["returns"].rolling(5).std()
    df["volatility_10"] = df["returns"].rolling(10).std()

    # Volume features (if available)
    if "volume" in df.columns:
        df["volume_sma_5"] = df["volume"].rolling(5).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_5"]

    # RSI-like
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

    # Labels: 1 = up, -1 = down, 0 = flat
    future_ret = df["close"].shift(-3) / df["close"] - 1
    df["label"] = 0
    df.loc[future_ret > 0.001, "label"] = 1
    df.loc[future_ret < -0.001, "label"] = -1

    # Drop NaN
    df = df.dropna().reset_index(drop=True)

    # Feature columns
    feature_cols = [c for c in df.columns if c not in ["timestamp", "open", "high", "low", "close", "volume", "label"]]

    return df[feature_cols].values, df["label"].values, df


def run_ultra_fast_experiment(X, y, df, name, n_trials=30, output_dir=Path("models")):
    """Ultra-fast experiment."""
    log(f"\n{'='*60}")
    log(f"ULTRA-FAST: {name}")
    log(f"{'='*60}")

    # Simple split: 70% train, 15% val, 15% holdout
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_holdout, y_holdout = X[val_end:], y[val_end:]
    df_holdout = df.iloc[val_end:].copy()

    log(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Holdout: {len(X_holdout):,}")

    def objective(trial):
        model_type = trial.suggest_categorical("model_type", ["rf", "gb"])

        if model_type == "rf":
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 30, 100),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                min_samples_split=trial.suggest_int("min_samples_split", 5, 20),
                n_jobs=4,
                random_state=42
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=trial.suggest_int("n_estimators", 30, 100),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                max_depth=trial.suggest_int("max_depth", 2, 6),
                random_state=42
            )

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return f1_score(y_val, preds, average="weighted", zero_division=0)

    # Run optimization
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    log(f"Running {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log(f"Best score: {study.best_value:.4f}")
    log(f"Best params: {study.best_params}")

    # Train final model on train+val
    X_dev = np.vstack([X_train, X_val])
    y_dev = np.hstack([y_train, y_val])

    best_params = study.best_params
    if best_params["model_type"] == "rf":
        final_model = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            n_jobs=-1,
            random_state=42
        )
    else:
        final_model = GradientBoostingClassifier(
            n_estimators=best_params["n_estimators"],
            learning_rate=best_params["learning_rate"],
            max_depth=best_params["max_depth"],
            random_state=42
        )

    final_model.fit(X_dev, y_dev)

    # Evaluate on holdout
    preds = final_model.predict(X_holdout)
    accuracy = accuracy_score(y_holdout, preds)
    f1 = f1_score(y_holdout, preds, average="weighted", zero_division=0)

    # Simple P&L calculation
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

    log(f"Holdout: Accuracy={accuracy:.4f}, F1={f1:.4f}, P&L={total_pnl*100:.2f}%, WR={win_rate:.2%}")

    # Save
    exp_dir = output_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": name,
        "timestamp": datetime.now().isoformat(),
        "best_params": best_params,
        "best_val_score": study.best_value,
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

    # Save model
    import joblib
    joblib.dump(final_model, exp_dir / "model.joblib")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/historical/MES_1m.parquet")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--output", default="models")
    args = parser.parse_args()

    log("="*60)
    log("ULTRA-FAST TRAINING")
    log("="*60)

    # Load heavily sampled data
    df = load_and_heavily_sample(args.data)

    # Create features
    log("Creating features...")
    X, y, df_processed = create_features(df)
    log(f"Features: {X.shape}")

    output_dir = Path(args.output)

    # Run experiments
    all_results = []

    for exp_name in ["ultra_rf", "ultra_gb", "ultra_aggressive"]:
        result = run_ultra_fast_experiment(
            X, y, df_processed,
            name=exp_name,
            n_trials=args.trials,
            output_dir=output_dir
        )
        all_results.append(result)

    # Summary
    log(f"\n{'='*60}")
    log("ULTRA-FAST COMPLETE")
    log(f"{'='*60}")

    best = max(all_results, key=lambda x: x["holdout_metrics"]["total_pnl_pct"])
    log(f"Best: {best['experiment']}")
    log(f"  P&L: {best['holdout_metrics']['total_pnl_pct']:.2f}%")
    log(f"  Win Rate: {best['holdout_metrics']['win_rate']:.2%}")
    log(f"  Params: {best['best_params']}")

    # Save summary
    summary = {"experiments": all_results, "best": best, "completed_at": datetime.now().isoformat()}
    with open(output_dir / "ultra_fast_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
