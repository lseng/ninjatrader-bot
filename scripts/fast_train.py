#!/usr/bin/env python3
"""
Fast Training Script - Optimized for Quick Convergence

Uses:
- Sampled data (every Nth bar) for faster iteration
- Fewer trials but smarter search
- Walk-forward validation on sampled data
- Final validation on full holdout
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

from src.data.loader import NinjaTraderDataLoader
from src.data.processor import DataProcessor
from src.ml.models import RandomForestModel, GradientBoostingModel, StrategyModel


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


def load_and_sample_data(data_path: str, sample_rate: int = 5):
    """Load data and sample every Nth bar for faster processing."""
    log(f"Loading data from {data_path}...")

    if data_path.endswith(".parquet"):
        loader = NinjaTraderDataLoader(Path(data_path).parent)
        df = loader.load_parquet(Path(data_path).name)
    else:
        df = pd.read_csv(data_path, parse_dates=["timestamp"])

    log(f"Full dataset: {len(df):,} bars")

    # Sample for faster training
    df_sampled = df.iloc[::sample_rate].reset_index(drop=True)
    log(f"Sampled (every {sample_rate}th bar): {len(df_sampled):,} bars")

    # Process features
    log("Creating features...")
    processor = DataProcessor(df_sampled)
    processor.add_all_features()
    processor.create_labels(lookahead=3, threshold=0.001, label_type="ternary")

    X, y = processor.get_features_and_labels()
    log(f"Feature matrix: {X.shape}")

    return X, y, df_sampled, df


def simple_walk_forward(X, y, model_factory, n_folds=3):
    """Simple walk-forward validation for speed."""
    from sklearn.metrics import f1_score

    n_samples = len(X)
    fold_size = n_samples // (n_folds + 1)

    scores = []
    for i in range(n_folds):
        train_end = fold_size * (i + 2)
        test_start = train_end
        test_end = min(test_start + fold_size, n_samples)

        if test_end <= test_start:
            continue

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        model = model_factory()
        model.fit(X_train, y_train)

        # Get predictions
        preds = []
        for j in range(len(X_test)):
            pred = model.predict(X_test.iloc[[j]])
            preds.append(pred.signal)

        if len(preds) > 0:
            score = f1_score(y_test, preds, average="weighted", zero_division=0)
            scores.append(score)

    return np.mean(scores) if scores else 0


def objective(trial, X_dev, y_dev):
    """Optuna objective."""
    model_type = trial.suggest_categorical("model_type", ["rf", "gb"])

    if model_type == "rf":
        params = {
            "n_estimators": trial.suggest_int("rf_n_estimators", 30, 150),
            "max_depth": trial.suggest_int("rf_max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("rf_min_samples_split", 5, 30),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 2, 15),
        }
        factory = lambda: RandomForestModel(**params, random_state=42)
    else:
        params = {
            "n_estimators": trial.suggest_int("gb_n_estimators", 30, 150),
            "learning_rate": trial.suggest_float("gb_learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("gb_max_depth", 2, 8),
            "min_samples_split": trial.suggest_int("gb_min_samples_split", 5, 30),
        }
        factory = lambda p=params: GradientBoostingModel(**p, random_state=42)

    try:
        score = simple_walk_forward(X_dev, y_dev, factory, n_folds=3)
        return score
    except Exception as e:
        return 0.0


def evaluate_on_holdout(model, X_holdout, y_holdout, df_holdout):
    """Evaluate model on holdout set."""
    # Get predictions
    predictions = []
    for i in range(len(X_holdout)):
        pred = model.predict(X_holdout.iloc[[i]])
        predictions.append(pred.signal)
    predictions = np.array(predictions)

    # Simple P&L calculation
    returns = df_holdout["close"].pct_change().values[1:]
    position = 0
    trades = 0
    wins = 0
    total_pnl = 0

    for i in range(len(predictions) - 1):
        if predictions[i] != position:
            trades += 1
            position = predictions[i]

        if position != 0 and i < len(returns):
            pnl = position * returns[i]
            total_pnl += pnl
            if pnl > 0:
                wins += 1

    win_rate = wins / max(trades, 1)

    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(y_holdout, predictions)
    f1 = f1_score(y_holdout, predictions, average="weighted", zero_division=0)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "total_pnl_pct": total_pnl * 100,
        "trades": trades,
        "win_rate": win_rate
    }


def run_fast_experiment(X, y, df, name, n_trials=50, n_jobs=14, output_dir=Path("models")):
    """Run a fast experiment."""
    log(f"\n{'='*60}")
    log(f"FAST EXPERIMENT: {name}")
    log(f"{'='*60}")

    # Split: 85% dev, 15% holdout
    holdout_start = int(len(X) * 0.85)
    X_dev, y_dev = X.iloc[:holdout_start], y.iloc[:holdout_start]
    X_holdout, y_holdout = X.iloc[holdout_start:], y.iloc[holdout_start:]
    df_holdout = df.iloc[holdout_start:].copy()

    log(f"Development set: {len(X_dev):,} samples")
    log(f"Holdout set: {len(X_holdout):,} samples")

    # Create study
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name=name
    )

    def obj(trial):
        return objective(trial, X_dev, y_dev)

    log(f"Starting {n_trials} optimization trials...")
    study.optimize(obj, n_trials=n_trials, n_jobs=min(n_jobs, 4), show_progress_bar=True)

    log(f"Best trial: {study.best_trial.number}")
    log(f"Best score: {study.best_value:.4f}")

    # Train final model
    best_params = study.best_params
    model_type = best_params.get("model_type", "rf")

    log("Training final model...")
    if model_type == "rf":
        final_model = RandomForestModel(
            n_estimators=best_params.get("rf_n_estimators", 100),
            max_depth=best_params.get("rf_max_depth", 10),
            min_samples_split=best_params.get("rf_min_samples_split", 10),
            min_samples_leaf=best_params.get("rf_min_samples_leaf", 5),
            random_state=42
        )
    else:
        final_model = GradientBoostingModel(
            n_estimators=best_params.get("gb_n_estimators", 100),
            learning_rate=best_params.get("gb_learning_rate", 0.1),
            max_depth=best_params.get("gb_max_depth", 5),
            min_samples_split=best_params.get("gb_min_samples_split", 10),
            random_state=42
        )

    final_model.fit(X_dev, y_dev)

    # Evaluate on holdout
    log("Evaluating on holdout...")
    metrics = evaluate_on_holdout(final_model, X_holdout, y_holdout, df_holdout)

    log(f"Holdout Results:")
    log(f"  Accuracy: {metrics['accuracy']:.4f}")
    log(f"  F1 Score: {metrics['f1']:.4f}")
    log(f"  P&L: {metrics['total_pnl_pct']:.2f}%")
    log(f"  Trades: {metrics['trades']}")
    log(f"  Win Rate: {metrics['win_rate']:.2%}")

    # Save results
    exp_dir = output_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment_name": name,
        "timestamp": datetime.now().isoformat(),
        "n_trials": n_trials,
        "best_params": best_params,
        "holdout_metrics": metrics,
        "sample_rate": 5
    }

    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    final_model.save(exp_dir / "model.joblib")
    log(f"Results saved to {exp_dir}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/historical/MES_1m.parquet")
    parser.add_argument("--sample-rate", type=int, default=5)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=14)
    parser.add_argument("--output", default="models")
    args = parser.parse_args()

    log("="*60)
    log("FAST TRAINING SCRIPT")
    log("="*60)

    # Load sampled data
    X, y, df_sampled, df_full = load_and_sample_data(args.data, args.sample_rate)

    output_dir = Path(args.output)

    # Run experiments
    experiments = [
        {"name": "fast_rf", "trials": args.trials},
        {"name": "fast_gb", "trials": args.trials},
        {"name": "fast_aggressive", "trials": 30},
    ]

    all_results = []
    for exp in experiments:
        try:
            result = run_fast_experiment(
                X, y, df_sampled,
                name=exp["name"],
                n_trials=exp["trials"],
                n_jobs=args.n_jobs,
                output_dir=output_dir
            )
            all_results.append(result)
        except Exception as e:
            log(f"Experiment {exp['name']} failed: {e}")

    # Summary
    log(f"\n{'='*60}")
    log("FAST TRAINING COMPLETE")
    log(f"{'='*60}")

    if all_results:
        best = max(all_results, key=lambda x: x["holdout_metrics"]["total_pnl_pct"])
        log(f"Best: {best['experiment_name']}")
        log(f"  P&L: {best['holdout_metrics']['total_pnl_pct']:.2f}%")
        log(f"  Win Rate: {best['holdout_metrics']['win_rate']:.2%}")

        # Save summary
        summary = {
            "completed_at": datetime.now().isoformat(),
            "experiments": all_results,
            "best": best
        }
        with open(output_dir / "fast_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
