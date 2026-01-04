#!/usr/bin/env python3
"""
Parallel Training Script for RunPod

This script is designed to run distributed hyperparameter optimization
across multiple CPUs on RunPod or any cloud infrastructure.

Features:
1. Distributed Optuna optimization with SQLite/PostgreSQL storage
2. Walk-forward validation during optimization
3. Anti-overfitting checks at each trial
4. Checkpointing and resume capability
5. Multi-node support via Optuna storage

Usage:
    # Single node (local or RunPod)
    python scripts/parallel_train.py --data data/historical/MES_1m.parquet --n-jobs 8

    # Multi-node with shared storage
    python scripts/parallel_train.py --data data/historical/MES_1m.parquet \
        --storage postgresql://user:pass@host/db --study-name mes_strategy_v1

    # Resume existing study
    python scripts/parallel_train.py --data data/historical/MES_1m.parquet \
        --storage postgresql://user:pass@host/db --study-name mes_strategy_v1 --resume
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import joblib
from concurrent.futures import ProcessPoolExecutor

from src.data.loader import NinjaTraderDataLoader
from src.data.processor import DataProcessor
from src.ml.models import RandomForestModel, GradientBoostingModel, StrategyModel
from src.ml.walk_forward import (
    WalkForwardOptimizer,
    PurgedKFold,
    OutOfSampleHoldout,
    AntiOverfitPipeline
)
from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.backtesting.metrics import PerformanceMetrics


def load_and_process_data(data_path: str, cache_dir: str = "cache") -> tuple:
    """Load data and create features with caching."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    # Create cache key from file path and modification time
    data_file = Path(data_path)
    cache_key = f"{data_file.stem}_{data_file.stat().st_mtime}"
    cache_file = cache_path / f"{cache_key}_features.parquet"
    labels_file = cache_path / f"{cache_key}_labels.parquet"

    if cache_file.exists() and labels_file.exists():
        print(f"Loading cached features from {cache_file}")
        X = pd.read_parquet(cache_file)
        y = pd.read_parquet(labels_file)["label"]
        return X, y

    print(f"Loading data from {data_path}...")

    if data_path.endswith(".parquet"):
        loader = NinjaTraderDataLoader(Path(data_path).parent)
        df = loader.load_parquet(Path(data_path).name)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
    else:
        loader = NinjaTraderDataLoader(Path(data_path).parent)
        df = loader.load_auto_detect(Path(data_path).name)

    print(f"Loaded {len(df):,} bars")

    # Process features
    print("Creating features...")
    processor = DataProcessor(df)
    processor.add_all_features()
    processor.create_labels(lookahead=5, threshold=0.001, label_type="ternary")

    X, y = processor.get_features_and_labels()

    print(f"Feature matrix: {X.shape}")
    print(f"Label distribution: {y.value_counts().to_dict()}")

    # Cache
    X.to_parquet(cache_file)
    y.to_frame("label").to_parquet(labels_file)

    return X, y


def create_model_from_params(params: dict, seed: int = 42):
    """Create model from hyperparameters."""
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

    elif model_type == "ensemble":
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

    raise ValueError(f"Unknown model type: {model_type}")


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Optuna objective function with walk-forward validation.

    Uses purged cross-validation to prevent overfitting.
    """
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

    # Walk-forward optimization
    wfo = WalkForwardOptimizer(
        initial_train_size=0.4,
        test_size=0.1,
        step_size=0.1,
        embargo_bars=100  # ~100 bars gap
    )

    try:
        result = wfo.optimize(
            X, y,
            lambda: create_model_from_params(params, seed=42)
        )

        # Multi-objective: score + stability + significance
        score = result.mean_score

        # Penalty for instability
        stability_penalty = min(result.std_score * 2, 0.1)

        # Penalty for not being significant
        significance_penalty = 0 if result.is_statistically_significant else 0.05

        # Penalty for too few trades
        if result.total_trades < 100:
            trade_penalty = 0.1
        else:
            trade_penalty = 0

        final_score = score - stability_penalty - significance_penalty - trade_penalty

        # Store additional metrics
        trial.set_user_attr("mean_score", result.mean_score)
        trial.set_user_attr("std_score", result.std_score)
        trial.set_user_attr("total_trades", result.total_trades)
        trial.set_user_attr("is_significant", result.is_statistically_significant)
        trial.set_user_attr("p_value", result.p_value)

        return final_score

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0


def run_optimization(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 100,
    n_jobs: int = 1,
    storage: str = None,
    study_name: str = None,
    resume: bool = False,
    timeout: int = None
) -> optuna.Study:
    """
    Run distributed hyperparameter optimization.

    Args:
        X, y: Feature matrix and labels
        n_trials: Number of optimization trials
        n_jobs: Number of parallel jobs
        storage: Optuna storage URL (for distributed training)
        study_name: Study name for distributed training
        resume: Whether to resume existing study
        timeout: Maximum optimization time in seconds

    Returns:
        Completed Optuna study
    """
    if storage:
        # Distributed mode with shared storage
        if resume:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage
            )
            print(f"Resumed study '{study_name}' with {len(study.trials)} existing trials")
        else:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="maximize",
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_startup_trials=10),
                load_if_exists=True
            )
    else:
        # Local mode
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10)
        )

    # Create objective with data closure
    def objective_fn(trial):
        return objective(trial, X, y)

    print(f"\nStarting optimization with {n_trials} trials, {n_jobs} parallel jobs...")

    study.optimize(
        objective_fn,
        n_trials=n_trials,
        n_jobs=n_jobs,
        timeout=timeout,
        show_progress_bar=True,
        gc_after_trial=True
    )

    return study


def final_validation(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict,
    output_dir: Path
) -> dict:
    """
    Run final validation with anti-overfitting pipeline.

    This is the definitive test that determines if the strategy is viable.
    """
    print("\n" + "=" * 60)
    print("FINAL VALIDATION WITH ANTI-OVERFITTING PIPELINE")
    print("=" * 60)

    pipeline = AntiOverfitPipeline(
        holdout_pct=0.15,
        wfo_initial_train=0.3,
        wfo_test_size=0.1,
        n_mc_permutations=1000,
        n_seeds=10,
        min_trades=100,
        min_sharpe=0.5,
        max_drawdown_pct=0.20,
        min_win_rate=0.45
    )

    def model_factory(seed):
        return create_model_from_params(best_params, seed)

    result = pipeline.validate(X, y, model_factory)

    # Save results
    validation_results = {
        "is_valid": result.is_valid,
        "rejection_reasons": result.rejection_reasons,
        "walk_forward": {
            "mean_score": result.walk_forward_result.mean_score,
            "std_score": result.walk_forward_result.std_score,
            "total_trades": result.walk_forward_result.total_trades,
            "is_significant": result.walk_forward_result.is_statistically_significant,
            "p_value": result.walk_forward_result.p_value
        },
        "monte_carlo": result.monte_carlo_result,
        "multi_seed": result.multi_seed_result,
        "holdout": result.holdout_result,
        "best_params": best_params,
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(validation_results, f, indent=2, default=str)

    return validation_results


def main():
    parser = argparse.ArgumentParser(description="Parallel Training for NinjaTrader ML Strategy")
    parser.add_argument("--data", required=True, help="Path to data file (parquet or csv)")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs (-1 = all CPUs)")
    parser.add_argument("--storage", default=None, help="Optuna storage URL for distributed training")
    parser.add_argument("--study-name", default=None, help="Study name for distributed training")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    parser.add_argument("--timeout", type=int, default=None, help="Max optimization time in seconds")
    parser.add_argument("--output", default="models", help="Output directory for models")
    parser.add_argument("--skip-validation", action="store_true", help="Skip final validation")
    parser.add_argument("--cache-dir", default="cache", help="Cache directory for processed data")

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and process data
    X, y = load_and_process_data(args.data, args.cache_dir)

    # Handle n_jobs
    n_jobs = args.n_jobs
    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
        print(f"Using all {n_jobs} CPUs")

    # Generate study name if not provided
    study_name = args.study_name or f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Run optimization
    study = run_optimization(
        X, y,
        n_trials=args.n_trials,
        n_jobs=n_jobs,
        storage=args.storage,
        study_name=study_name,
        resume=args.resume,
        timeout=args.timeout
    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Additional metrics from best trial
    if study.best_trial.user_attrs:
        print("\nBest trial metrics:")
        for key, value in study.best_trial.user_attrs.items():
            print(f"  {key}: {value}")

    # Save best params
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

    # Save study
    joblib.dump(study, output_dir / "optuna_study.joblib")

    # Final validation
    if not args.skip_validation:
        validation_results = final_validation(X, y, study.best_params, output_dir)

        if validation_results["is_valid"]:
            print("\n" + "=" * 60)
            print("TRAINING FINAL MODEL")
            print("=" * 60)

            # Train final model on all development data
            holdout = OutOfSampleHoldout(holdout_pct=0.15)
            X_dev, y_dev, X_holdout, y_holdout = holdout.split(X, y)

            final_model = create_model_from_params(study.best_params, seed=42)
            final_model.fit(X_dev, y_dev)
            final_model.save(output_dir / "final_model.joblib")

            print(f"Final model saved to {output_dir / 'final_model.joblib'}")
        else:
            print("\nStrategy failed validation. Not saving final model.")
            print("Rejection reasons:")
            for reason in validation_results["rejection_reasons"]:
                print(f"  - {reason}")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
