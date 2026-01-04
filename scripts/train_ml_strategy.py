#!/usr/bin/env python3
"""
Train an ML-based trading strategy.

This script demonstrates the full ML pipeline:
1. Load historical data
2. Generate features
3. Create labels
4. Train models
5. Optimize hyperparameters
6. Evaluate and save

Example:
    python scripts/train_ml_strategy.py --data data/historical/MES_1min.csv --optimize
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.data.processor import DataProcessor
from src.ml.models import RandomForestModel, GradientBoostingModel, StrategyModel
from src.ml.trainer import ModelTrainer
from src.ml.optimizer import StrategyOptimizer


def main():
    parser = argparse.ArgumentParser(description="Train ML trading strategy")
    parser.add_argument("--data", required=True, help="Path to data file")
    parser.add_argument("--model", default="ensemble", choices=["rf", "gb", "ensemble"])
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--n-trials", type=int, default=50, help="Optimization trials")
    parser.add_argument("--output", default="models/strategy", help="Output path for model")
    parser.add_argument("--lookahead", type=int, default=5, help="Prediction horizon (bars)")
    parser.add_argument("--threshold", type=float, default=0.001, help="Min return for signal")

    args = parser.parse_args()

    print("=" * 60)
    print("ML STRATEGY TRAINING")
    print("=" * 60)

    # Load data
    print(f"\n1. Loading data from {args.data}...")
    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    print(f"   Loaded {len(df):,} bars")

    # Process features
    print("\n2. Generating features...")
    processor = DataProcessor(df)
    processor.add_all_features()
    processor.create_labels(
        lookahead=args.lookahead,
        threshold=args.threshold,
        label_type="ternary"
    )

    X, y = processor.get_features_and_labels(dropna=True)
    print(f"   Features: {len(X.columns)}")
    print(f"   Samples: {len(X):,}")
    print(f"   Label distribution:")
    for label in sorted(y.unique()):
        count = (y == label).sum()
        pct = count / len(y) * 100
        print(f"     {label:>2}: {count:>6,} ({pct:.1f}%)")

    # Create model
    print(f"\n3. Creating {args.model} model...")

    if args.model == "rf":
        model = RandomForestModel()
    elif args.model == "gb":
        model = GradientBoostingModel()
    else:
        model = StrategyModel(
            models=[RandomForestModel(), GradientBoostingModel()],
            confidence_threshold=0.6
        )

    # Optimize if requested
    if args.optimize:
        print(f"\n4. Running hyperparameter optimization ({args.n_trials} trials)...")
        optimizer = StrategyOptimizer(
            n_trials=args.n_trials,
            n_cv_splits=5,
            metric="f1"
        )

        if args.model == "rf":
            result = optimizer.optimize_random_forest(X, y)
        elif args.model == "gb":
            result = optimizer.optimize_gradient_boosting(X, y)
        else:
            result = optimizer.optimize_strategy(X, y)

        print(f"\n   Best parameters: {result.best_params}")
        print(f"   Best score: {result.best_score:.4f}")

        # Recreate model with best params
        if args.model == "rf":
            model = RandomForestModel(**result.best_params)
        elif args.model == "gb":
            model = GradientBoostingModel(**result.best_params)

    # Train and evaluate
    print("\n5. Training and cross-validating...")
    trainer = ModelTrainer(n_splits=5)
    cv_result = trainer.cross_validate(model, X, y)
    trainer.print_results(cv_result)

    # Final training on all data
    print("\n6. Final training on full dataset...")
    model.fit(X, y)

    # Feature importance (for tree models)
    if hasattr(model, "feature_importance"):
        importance = model.feature_importance()
        print("\n   Top 10 features:")
        for _, row in importance.head(10).iterrows():
            print(f"     {row['feature']}: {row['importance']:.4f}")

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, StrategyModel):
        model.save(output_path)
        print(f"\n7. Model saved to {output_path}/")
    else:
        model.save(f"{output_path}.joblib")
        print(f"\n7. Model saved to {output_path}.joblib")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
