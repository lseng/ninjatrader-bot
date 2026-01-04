#!/usr/bin/env python3
"""
Master Training Script - 8 Hour Comprehensive Run

This script runs comprehensive ML training with the goal of maximizing profitability.
It will:
1. Run 1000+ hyperparameter trials across multiple model types
2. Use walk-forward validation to prevent overfitting
3. Run backtests with the best strategies
4. Track all results and P&L over time
5. Save detailed reports

Usage:
    python scripts/master_train.py --hours 8
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler

from src.data.loader import NinjaTraderDataLoader
from src.data.processor import DataProcessor
from src.ml.models import RandomForestModel, GradientBoostingModel, StrategyModel
from src.ml.walk_forward import (
    WalkForwardOptimizer,
    OutOfSampleHoldout,
    AntiOverfitPipeline
)
from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.backtesting.metrics import PerformanceMetrics


def log(msg):
    """Print with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


def load_data(data_path: str) -> tuple:
    """Load and process data."""
    log(f"Loading data from {data_path}...")

    if data_path.endswith(".parquet"):
        loader = NinjaTraderDataLoader(Path(data_path).parent)
        df = loader.load_parquet(Path(data_path).name)
    else:
        df = pd.read_csv(data_path, parse_dates=["timestamp"])

    log(f"Loaded {len(df):,} bars")

    # Process features
    log("Creating features...")
    processor = DataProcessor(df)
    processor.add_all_features()
    processor.create_labels(lookahead=5, threshold=0.001, label_type="ternary")

    X, y = processor.get_features_and_labels()

    log(f"Feature matrix: {X.shape}")
    log(f"Label distribution: {y.value_counts().to_dict()}")

    return X, y, df


def create_model(params: dict, seed: int = 42):
    """Create model from parameters."""
    model_type = params.get("model_type", "rf")

    if model_type == "rf":
        return RandomForestModel(
            n_estimators=params.get("rf_n_estimators", 100),
            max_depth=params.get("rf_max_depth", 10),
            min_samples_split=params.get("rf_min_samples_split", 10),
            min_samples_leaf=params.get("rf_min_samples_leaf", 5),
            random_state=seed
        )
    elif model_type == "gb":
        return GradientBoostingModel(
            n_estimators=params.get("gb_n_estimators", 100),
            learning_rate=params.get("gb_learning_rate", 0.1),
            max_depth=params.get("gb_max_depth", 5),
            min_samples_split=params.get("gb_min_samples_split", 10),
            random_state=seed
        )
    else:  # ensemble
        return StrategyModel(
            models=[
                RandomForestModel(
                    n_estimators=params.get("rf_n_estimators", 100),
                    max_depth=params.get("rf_max_depth", 10),
                    random_state=seed
                ),
                GradientBoostingModel(
                    n_estimators=params.get("gb_n_estimators", 100),
                    learning_rate=params.get("gb_learning_rate", 0.1),
                    random_state=seed
                )
            ],
            confidence_threshold=params.get("confidence_threshold", 0.6)
        )


def run_backtest_with_model(model, X_test, y_test, df_test, config):
    """Run backtest with trained model and return metrics."""
    from src.strategies.ml_strategy import MLStrategy

    # Get predictions
    predictions = []
    for i in range(len(X_test)):
        pred = model.predict(X_test.iloc[[i]])
        predictions.append(pred.signal)

    predictions = np.array(predictions)

    # Calculate accuracy metrics
    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    # Simple P&L calculation
    # Assume we trade based on predictions: +1 = long, -1 = short, 0 = flat
    # P&L = direction * price_change
    returns = df_test["close"].pct_change().values[1:]
    pred_returns = []

    position = 0
    trades = 0
    wins = 0
    total_pnl = 0

    for i in range(len(predictions) - 1):
        if predictions[i] != position:
            trades += 1
            position = predictions[i]

        if position != 0:
            pnl = position * returns[i] if i < len(returns) else 0
            total_pnl += pnl
            if pnl > 0:
                wins += 1
            pred_returns.append(pnl)

    win_rate = wins / max(trades, 1)

    # Calculate Sharpe ratio
    if len(pred_returns) > 0 and np.std(pred_returns) > 0:
        sharpe = np.mean(pred_returns) / np.std(pred_returns) * np.sqrt(252 * 390)  # Annualized for 1-min bars
    else:
        sharpe = 0

    return {
        "accuracy": accuracy,
        "f1": f1,
        "total_pnl_pct": total_pnl * 100,
        "trades": trades,
        "win_rate": win_rate,
        "sharpe": sharpe
    }


def objective(trial, X_dev, y_dev, df_dev):
    """Optuna objective for hyperparameter optimization."""
    # Sample hyperparameters
    model_type = trial.suggest_categorical("model_type", ["rf", "gb", "ensemble"])

    params = {"model_type": model_type}

    if model_type in ["rf", "ensemble"]:
        params["rf_n_estimators"] = trial.suggest_int("rf_n_estimators", 50, 300)
        params["rf_max_depth"] = trial.suggest_int("rf_max_depth", 3, 20)
        params["rf_min_samples_split"] = trial.suggest_int("rf_min_samples_split", 2, 30)
        params["rf_min_samples_leaf"] = trial.suggest_int("rf_min_samples_leaf", 1, 20)

    if model_type in ["gb", "ensemble"]:
        params["gb_n_estimators"] = trial.suggest_int("gb_n_estimators", 50, 300)
        params["gb_learning_rate"] = trial.suggest_float("gb_learning_rate", 0.01, 0.3, log=True)
        params["gb_max_depth"] = trial.suggest_int("gb_max_depth", 3, 12)
        params["gb_min_samples_split"] = trial.suggest_int("gb_min_samples_split", 2, 30)

    if model_type == "ensemble":
        params["confidence_threshold"] = trial.suggest_float("confidence_threshold", 0.5, 0.8)

    # Walk-forward validation
    wfo = WalkForwardOptimizer(
        initial_train_size=0.4,
        test_size=0.1,
        step_size=0.1,
        embargo_bars=100
    )

    try:
        result = wfo.optimize(X_dev, y_dev, lambda: create_model(params, seed=42))

        # Multi-objective scoring
        score = result.mean_score

        # Penalties
        stability_penalty = min(result.std_score * 2, 0.1)
        significance_penalty = 0 if result.is_statistically_significant else 0.05
        trade_penalty = 0.1 if result.total_trades < 100 else 0

        final_score = score - stability_penalty - significance_penalty - trade_penalty

        # Store metrics
        trial.set_user_attr("mean_score", result.mean_score)
        trial.set_user_attr("std_score", result.std_score)
        trial.set_user_attr("total_trades", result.total_trades)
        trial.set_user_attr("is_significant", result.is_statistically_significant)
        trial.set_user_attr("mean_pnl", result.mean_pnl)

        return final_score

    except Exception as e:
        return 0.0


def run_experiment(
    X, y, df,
    experiment_name: str,
    n_trials: int,
    n_jobs: int,
    output_dir: Path
):
    """Run a single experiment with the given configuration."""
    log(f"\n{'='*60}")
    log(f"EXPERIMENT: {experiment_name}")
    log(f"{'='*60}")

    # Split holdout
    holdout = OutOfSampleHoldout(holdout_pct=0.15)
    X_dev, y_dev, X_holdout, y_holdout = holdout.split(X, y)

    # Get corresponding dataframes
    holdout_start = len(X_dev)
    df_dev = df.iloc[:holdout_start].copy()
    df_holdout = df.iloc[holdout_start:holdout_start + len(X_holdout)].copy()

    log(f"Development set: {len(X_dev)} samples")
    log(f"Holdout set: {len(X_holdout)} samples")

    # Create study
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name=experiment_name
    )

    # Objective with data
    def obj(trial):
        return objective(trial, X_dev, y_dev, df_dev)

    log(f"Starting {n_trials} optimization trials with {n_jobs} parallel jobs...")

    study.optimize(
        obj,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
        gc_after_trial=True
    )

    log(f"\nOptimization complete!")
    log(f"Best trial: {study.best_trial.number}")
    log(f"Best score: {study.best_value:.4f}")

    # Train final model on all dev data
    best_params = study.best_params
    log(f"\nTraining final model with best params...")

    final_model = create_model(best_params, seed=42)
    final_model.fit(X_dev, y_dev)

    # Evaluate on holdout
    log(f"\nEvaluating on holdout set...")
    holdout_metrics = run_backtest_with_model(
        final_model, X_holdout, y_holdout, df_holdout, None
    )

    log(f"Holdout Results:")
    log(f"  Accuracy: {holdout_metrics['accuracy']:.4f}")
    log(f"  F1 Score: {holdout_metrics['f1']:.4f}")
    log(f"  Total P&L: {holdout_metrics['total_pnl_pct']:.2f}%")
    log(f"  Trades: {holdout_metrics['trades']}")
    log(f"  Win Rate: {holdout_metrics['win_rate']:.2%}")
    log(f"  Sharpe: {holdout_metrics['sharpe']:.2f}")

    # Save results
    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "n_trials": n_trials,
        "best_trial": study.best_trial.number,
        "best_score": study.best_value,
        "best_params": best_params,
        "holdout_metrics": holdout_metrics,
        "trial_history": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs
            }
            for t in study.trials if t.value is not None
        ]
    }

    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save model
    final_model.save(exp_dir / "model.joblib")

    log(f"Results saved to {exp_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Master Training Script")
    parser.add_argument("--data", default="data/historical/MES_1m.parquet")
    parser.add_argument("--hours", type=float, default=8, help="Hours to run")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs")
    parser.add_argument("--output", default="models", help="Output directory")

    args = parser.parse_args()

    # Get CPU count
    import multiprocessing
    n_jobs = args.n_jobs if args.n_jobs > 0 else multiprocessing.cpu_count()

    log(f"="*60)
    log(f"MASTER TRAINING SCRIPT")
    log(f"="*60)
    log(f"CPUs: {n_jobs}")
    log(f"Duration: {args.hours} hours")
    log(f"Data: {args.data}")
    log(f"Output: {args.output}")

    # Load data
    X, y, df = load_data(args.data)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate time budget
    end_time = datetime.now() + timedelta(hours=args.hours)

    # Run experiments
    all_results = []
    experiment_num = 0

    # Experiment configurations
    experiments = [
        {"name": "rf_focused", "trials": 300, "description": "Random Forest focused"},
        {"name": "gb_focused", "trials": 300, "description": "Gradient Boosting focused"},
        {"name": "ensemble_focused", "trials": 300, "description": "Ensemble focused"},
        {"name": "aggressive", "trials": 200, "description": "Aggressive parameters"},
        {"name": "conservative", "trials": 200, "description": "Conservative parameters"},
    ]

    log(f"\nPlanned experiments: {len(experiments)}")
    log(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    for exp_config in experiments:
        if datetime.now() >= end_time:
            log(f"\nTime limit reached. Stopping.")
            break

        remaining_time = (end_time - datetime.now()).total_seconds() / 3600
        log(f"\nRemaining time: {remaining_time:.2f} hours")

        experiment_num += 1
        exp_name = f"exp{experiment_num:03d}_{exp_config['name']}"

        try:
            result = run_experiment(
                X, y, df,
                experiment_name=exp_name,
                n_trials=exp_config["trials"],
                n_jobs=n_jobs,
                output_dir=output_dir
            )
            all_results.append(result)

        except Exception as e:
            log(f"Experiment {exp_name} failed: {e}")
            continue

        # If we have time, run more trials on the best experiment
        if datetime.now() < end_time:
            time.sleep(5)  # Brief pause

    # Summary
    log(f"\n{'='*60}")
    log(f"TRAINING COMPLETE")
    log(f"{'='*60}")
    log(f"Total experiments: {len(all_results)}")

    if all_results:
        # Find best experiment
        best_exp = max(all_results, key=lambda x: x["holdout_metrics"]["total_pnl_pct"])

        log(f"\nBEST EXPERIMENT: {best_exp['experiment_name']}")
        log(f"  P&L: {best_exp['holdout_metrics']['total_pnl_pct']:.2f}%")
        log(f"  Win Rate: {best_exp['holdout_metrics']['win_rate']:.2%}")
        log(f"  Sharpe: {best_exp['holdout_metrics']['sharpe']:.2f}")
        log(f"  Best Params: {best_exp['best_params']}")

        # Save summary
        summary = {
            "completed_at": datetime.now().isoformat(),
            "total_experiments": len(all_results),
            "best_experiment": best_exp,
            "all_results": [
                {
                    "name": r["experiment_name"],
                    "pnl": r["holdout_metrics"]["total_pnl_pct"],
                    "win_rate": r["holdout_metrics"]["win_rate"],
                    "sharpe": r["holdout_metrics"]["sharpe"]
                }
                for r in all_results
            ]
        }

        with open(output_dir / "master_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        log(f"\nSummary saved to {output_dir / 'master_summary.json'}")

    log(f"\nDone!")


if __name__ == "__main__":
    main()
