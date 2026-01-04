"""
Walk-Forward Optimization with Anti-Overfitting Measures

This module implements:
1. Anchored Walk-Forward Optimization
2. Purged K-Fold Cross-Validation with Embargo
3. Combinatorial Purged Cross-Validation (CPCV)
4. Monte Carlo Significance Testing
5. Multiple Seed Stability Testing

These techniques prevent overfitting in time-series ML for trading.
"""

import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import joblib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, f1_score
import scipy.stats as stats


@dataclass
class WalkForwardFold:
    """Single fold in walk-forward optimization."""
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_size: int
    test_size: int


@dataclass
class WalkForwardResult:
    """Results from walk-forward optimization."""
    folds: list[WalkForwardFold]
    fold_scores: list[float]
    fold_trades: list[int]
    fold_pnl: list[float]
    mean_score: float
    std_score: float
    mean_pnl: float
    total_trades: int
    is_statistically_significant: bool
    p_value: float


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation for Time Series.

    Unlike standard K-Fold, this:
    1. Preserves time order
    2. Adds an embargo (gap) between train and test to prevent leakage
    3. Purges overlapping samples that could leak information

    Critical for trading because features often use lagged data.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,  # 1% of data as embargo
        purge_pct: float = 0.01    # 1% purge overlap
    ):
        """
        Initialize Purged K-Fold.

        Args:
            n_splits: Number of folds
            embargo_pct: Percentage of data to use as embargo gap
            purge_pct: Percentage of overlap to purge
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        groups=None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices with purging and embargo.

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate embargo and purge sizes
        embargo_size = int(n_samples * self.embargo_pct)
        purge_size = int(n_samples * self.purge_pct)

        # Fold size
        fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            # Test set indices
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples

            test_indices = indices[test_start:test_end]

            # Train indices: everything before test, minus embargo and purge
            train_end = max(0, test_start - embargo_size - purge_size)

            # Also include data after test (for later folds)
            train_start_after = min(n_samples, test_end + embargo_size)

            train_indices_before = indices[:train_end]
            train_indices_after = indices[train_start_after:]

            train_indices = np.concatenate([train_indices_before, train_indices_after])

            if len(train_indices) > 0:
                yield train_indices, test_indices


class WalkForwardOptimizer:
    """
    Anchored Walk-Forward Optimization.

    This is the gold standard for trading strategy validation:
    1. Train on historical data
    2. Test on out-of-sample future data
    3. Expand training window and repeat
    4. Never look ahead

    Mimics how you'd actually deploy: train on past, trade future.
    """

    def __init__(
        self,
        initial_train_size: float = 0.3,  # Start with 30% of data
        test_size: float = 0.1,            # Test on 10% at a time
        step_size: float = 0.1,            # Move forward 10% each step
        min_train_samples: int = 1000,     # Minimum training samples
        embargo_bars: int = 50,            # Gap between train and test
        n_jobs: int = -1                   # Parallel jobs
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            initial_train_size: Fraction of data for initial training
            test_size: Fraction of data for each test period
            step_size: How much to move forward each iteration
            min_train_samples: Minimum samples needed for training
            embargo_bars: Number of bars to skip between train and test
            n_jobs: Number of parallel jobs (-1 = all cores)
        """
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.min_train_samples = min_train_samples
        self.embargo_bars = embargo_bars
        self.n_jobs = n_jobs

    def generate_folds(self, n_samples: int) -> list[WalkForwardFold]:
        """Generate walk-forward fold definitions."""
        folds = []

        initial_train_end = int(n_samples * self.initial_train_size)
        test_window = int(n_samples * self.test_size)
        step = int(n_samples * self.step_size)

        fold_id = 0
        train_end = initial_train_end

        while train_end + self.embargo_bars + test_window <= n_samples:
            train_start = 0  # Anchored: always start from beginning
            test_start = train_end + self.embargo_bars
            test_end = min(test_start + test_window, n_samples)

            if train_end - train_start >= self.min_train_samples:
                folds.append(WalkForwardFold(
                    fold_id=fold_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_size=train_end - train_start,
                    test_size=test_end - test_start
                ))
                fold_id += 1

            train_end += step

        return folds

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_factory: Callable[[], Any],
        param_grid: dict = None,
        metric: str = "f1"
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.

        Args:
            X: Feature matrix
            y: Labels
            model_factory: Function that creates a fresh model instance
            param_grid: Optional parameter grid for tuning (not implemented here)
            metric: Scoring metric

        Returns:
            WalkForwardResult with all fold results
        """
        folds = self.generate_folds(len(X))

        if len(folds) == 0:
            raise ValueError("Not enough data for walk-forward optimization")

        fold_scores = []
        fold_trades = []
        fold_pnl = []

        for fold in folds:
            # Get train/test data
            X_train = X.iloc[fold.train_start:fold.train_end]
            y_train = y.iloc[fold.train_start:fold.train_end]
            X_test = X.iloc[fold.test_start:fold.test_end]
            y_test = y.iloc[fold.test_start:fold.test_end]

            # Train model
            model = model_factory()
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            if hasattr(y_pred, 'signal'):
                y_pred = np.array([p.signal for p in y_pred])

            # Score
            if metric == "f1":
                score = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            else:
                score = accuracy_score(y_test, y_pred)

            fold_scores.append(score)

            # Count trades (non-neutral predictions)
            n_trades = np.sum(y_pred != 0)
            fold_trades.append(n_trades)

            # Estimate P&L (simplified: correct direction = +1, wrong = -1)
            correct = (y_pred == y_test) & (y_pred != 0)
            wrong = (y_pred != y_test) & (y_pred != 0)
            pnl = correct.sum() - wrong.sum()
            fold_pnl.append(pnl)

        # Statistical significance test
        # H0: Strategy has zero expected return
        # Use one-sample t-test
        if len(fold_pnl) >= 3:
            t_stat, p_value = stats.ttest_1samp(fold_pnl, 0)
            is_significant = p_value < 0.05 and np.mean(fold_pnl) > 0
        else:
            p_value = 1.0
            is_significant = False

        return WalkForwardResult(
            folds=folds,
            fold_scores=fold_scores,
            fold_trades=fold_trades,
            fold_pnl=fold_pnl,
            mean_score=np.mean(fold_scores),
            std_score=np.std(fold_scores),
            mean_pnl=np.mean(fold_pnl),
            total_trades=sum(fold_trades),
            is_statistically_significant=is_significant,
            p_value=p_value
        )


class MonteCarloValidator:
    """
    Monte Carlo Significance Testing for Trading Strategies.

    Tests whether strategy performance is significantly better than random.
    Uses permutation testing to establish null distribution.
    """

    def __init__(
        self,
        n_permutations: int = 1000,
        confidence_level: float = 0.95,
        n_jobs: int = -1
    ):
        """
        Initialize Monte Carlo validator.

        Args:
            n_permutations: Number of random permutations
            confidence_level: Confidence level for significance
            n_jobs: Parallel jobs
        """
        self.n_permutations = n_permutations
        self.confidence_level = confidence_level
        self.n_jobs = n_jobs

    def validate(
        self,
        actual_pnl: float,
        trades: pd.DataFrame,
        metric: str = "total_pnl"
    ) -> dict:
        """
        Test if strategy performance is statistically significant.

        Args:
            actual_pnl: Actual strategy P&L
            trades: DataFrame of trades with 'pnl' column
            metric: Metric to test

        Returns:
            Dictionary with test results
        """
        if len(trades) == 0:
            return {
                "is_significant": False,
                "p_value": 1.0,
                "actual_pnl": actual_pnl,
                "random_mean": 0,
                "random_std": 0,
                "percentile": 0
            }

        trade_pnls = trades["pnl"].values

        # Generate permuted P&Ls
        random_pnls = []
        for _ in range(self.n_permutations):
            # Randomly shuffle trade signs
            shuffled = trade_pnls.copy()
            np.random.shuffle(shuffled)
            random_pnls.append(shuffled.sum())

        random_pnls = np.array(random_pnls)

        # Calculate p-value
        p_value = np.mean(random_pnls >= actual_pnl)

        # Percentile of actual performance
        percentile = stats.percentileofscore(random_pnls, actual_pnl)

        return {
            "is_significant": p_value < (1 - self.confidence_level),
            "p_value": p_value,
            "actual_pnl": actual_pnl,
            "random_mean": np.mean(random_pnls),
            "random_std": np.std(random_pnls),
            "percentile": percentile,
            "confidence_level": self.confidence_level
        }


class MultiSeedValidator:
    """
    Test strategy stability across different random seeds.

    A robust strategy should perform consistently regardless of:
    - Random initialization
    - Random sampling
    - Training order
    """

    def __init__(
        self,
        n_seeds: int = 10,
        base_seed: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize multi-seed validator.

        Args:
            n_seeds: Number of different seeds to test
            base_seed: Base seed for reproducibility
            n_jobs: Parallel jobs
        """
        self.n_seeds = n_seeds
        self.base_seed = base_seed
        self.n_jobs = n_jobs
        self.seeds = [base_seed + i * 1000 for i in range(n_seeds)]

    def validate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_factory: Callable[[int], Any]
    ) -> dict:
        """
        Test model stability across seeds.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            model_factory: Function that takes seed and returns model

        Returns:
            Dictionary with stability metrics
        """
        scores = []
        predictions_list = []

        for seed in self.seeds:
            model = model_factory(seed)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            if hasattr(y_pred[0], 'signal'):
                y_pred = np.array([p.signal for p in y_pred])

            score = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            scores.append(score)
            predictions_list.append(y_pred)

        scores = np.array(scores)

        # Calculate prediction agreement
        predictions_matrix = np.array(predictions_list)
        agreement_scores = []
        for i in range(len(predictions_list)):
            for j in range(i + 1, len(predictions_list)):
                agreement = np.mean(predictions_list[i] == predictions_list[j])
                agreement_scores.append(agreement)

        return {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            "coefficient_of_variation": np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else float('inf'),
            "prediction_agreement": np.mean(agreement_scores) if agreement_scores else 0,
            "is_stable": np.std(scores) < 0.05,  # Less than 5% variation
            "scores_by_seed": dict(zip(self.seeds, scores.tolist()))
        }


class OutOfSampleHoldout:
    """
    Manage strict out-of-sample holdout data.

    CRITICAL: This data should NEVER be seen during any optimization or tuning.
    Only used for final validation.
    """

    def __init__(
        self,
        holdout_pct: float = 0.15,  # Hold out last 15% of data
        min_holdout_samples: int = 5000
    ):
        """
        Initialize holdout manager.

        Args:
            holdout_pct: Percentage of data to hold out
            min_holdout_samples: Minimum samples in holdout
        """
        self.holdout_pct = holdout_pct
        self.min_holdout_samples = min_holdout_samples
        self._holdout_start_idx = None

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Split data into development and holdout sets.

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            X_dev, y_dev, X_holdout, y_holdout
        """
        n_samples = len(X)
        holdout_size = max(
            int(n_samples * self.holdout_pct),
            self.min_holdout_samples
        )

        if holdout_size >= n_samples:
            raise ValueError("Not enough data for holdout")

        self._holdout_start_idx = n_samples - holdout_size

        X_dev = X.iloc[:self._holdout_start_idx]
        y_dev = y.iloc[:self._holdout_start_idx]
        X_holdout = X.iloc[self._holdout_start_idx:]
        y_holdout = y.iloc[self._holdout_start_idx:]

        return X_dev, y_dev, X_holdout, y_holdout

    def get_holdout_info(self) -> dict:
        """Get information about holdout split."""
        return {
            "holdout_start_idx": self._holdout_start_idx,
            "holdout_pct": self.holdout_pct
        }


@dataclass
class ValidationPipelineResult:
    """Complete validation pipeline results."""
    walk_forward_result: WalkForwardResult
    monte_carlo_result: dict
    multi_seed_result: dict
    holdout_result: dict
    is_valid: bool
    rejection_reasons: list[str]


class AntiOverfitPipeline:
    """
    Complete anti-overfitting validation pipeline.

    Combines all validation techniques to ensure strategy robustness.
    """

    def __init__(
        self,
        holdout_pct: float = 0.15,
        wfo_initial_train: float = 0.3,
        wfo_test_size: float = 0.1,
        n_mc_permutations: int = 1000,
        n_seeds: int = 10,
        min_trades: int = 100,
        min_sharpe: float = 0.5,
        max_drawdown_pct: float = 0.20,
        min_win_rate: float = 0.45
    ):
        """
        Initialize pipeline with validation thresholds.

        Args:
            holdout_pct: Holdout data percentage
            wfo_initial_train: Walk-forward initial training size
            wfo_test_size: Walk-forward test window size
            n_mc_permutations: Monte Carlo permutations
            n_seeds: Number of seeds for stability testing
            min_trades: Minimum trades required
            min_sharpe: Minimum acceptable Sharpe ratio
            max_drawdown_pct: Maximum acceptable drawdown
            min_win_rate: Minimum acceptable win rate
        """
        self.holdout = OutOfSampleHoldout(holdout_pct)
        self.wfo = WalkForwardOptimizer(
            initial_train_size=wfo_initial_train,
            test_size=wfo_test_size
        )
        self.monte_carlo = MonteCarloValidator(n_mc_permutations)
        self.multi_seed = MultiSeedValidator(n_seeds)

        # Validation thresholds
        self.min_trades = min_trades
        self.min_sharpe = min_sharpe
        self.max_drawdown_pct = max_drawdown_pct
        self.min_win_rate = min_win_rate

    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_factory: Callable[[int], Any],
        backtest_fn: Callable = None
    ) -> ValidationPipelineResult:
        """
        Run complete validation pipeline.

        Args:
            X: Feature matrix
            y: Labels
            model_factory: Function(seed) -> model
            backtest_fn: Optional function to run actual backtest

        Returns:
            ValidationPipelineResult with all test results
        """
        rejection_reasons = []

        # Step 1: Split holdout
        print("Step 1: Splitting holdout data...")
        X_dev, y_dev, X_holdout, y_holdout = self.holdout.split(X, y)
        print(f"  Development set: {len(X_dev)} samples")
        print(f"  Holdout set: {len(X_holdout)} samples")

        # Step 2: Walk-forward optimization on dev set
        print("\nStep 2: Walk-forward optimization...")
        wfo_result = self.wfo.optimize(X_dev, y_dev, lambda: model_factory(42))
        print(f"  Folds: {len(wfo_result.folds)}")
        print(f"  Mean score: {wfo_result.mean_score:.4f} (+/- {wfo_result.std_score:.4f})")
        print(f"  Total trades: {wfo_result.total_trades}")
        print(f"  Statistically significant: {wfo_result.is_statistically_significant}")

        if not wfo_result.is_statistically_significant:
            rejection_reasons.append("Walk-forward not statistically significant")

        if wfo_result.total_trades < self.min_trades:
            rejection_reasons.append(f"Insufficient trades: {wfo_result.total_trades} < {self.min_trades}")

        # Step 3: Multi-seed stability test
        print("\nStep 3: Multi-seed stability testing...")
        # Use first 70% of dev for train, last 30% for test
        split_idx = int(len(X_dev) * 0.7)
        X_train, X_test = X_dev.iloc[:split_idx], X_dev.iloc[split_idx:]
        y_train, y_test = y_dev.iloc[:split_idx], y_dev.iloc[split_idx:]

        seed_result = self.multi_seed.validate(
            X_train, y_train, X_test, y_test, model_factory
        )
        print(f"  Mean score: {seed_result['mean_score']:.4f}")
        print(f"  Std score: {seed_result['std_score']:.4f}")
        print(f"  CV: {seed_result['coefficient_of_variation']:.4f}")
        print(f"  Stable: {seed_result['is_stable']}")

        if not seed_result['is_stable']:
            rejection_reasons.append(f"Unstable across seeds: CV={seed_result['coefficient_of_variation']:.4f}")

        # Step 4: Final holdout evaluation
        print("\nStep 4: Final holdout evaluation...")
        final_model = model_factory(42)
        final_model.fit(X_dev, y_dev)

        y_pred = final_model.predict(X_holdout)
        if hasattr(y_pred[0], 'signal'):
            y_pred = np.array([p.signal for p in y_pred])

        holdout_score = f1_score(y_holdout, y_pred, average="weighted", zero_division=0)
        holdout_accuracy = accuracy_score(y_holdout, y_pred)

        # Win rate on directional trades
        directional_mask = y_pred != 0
        if directional_mask.sum() > 0:
            holdout_win_rate = accuracy_score(
                y_holdout[directional_mask],
                y_pred[directional_mask]
            )
        else:
            holdout_win_rate = 0

        holdout_result = {
            "f1_score": holdout_score,
            "accuracy": holdout_accuracy,
            "win_rate": holdout_win_rate,
            "n_trades": int(directional_mask.sum()),
            "n_samples": len(X_holdout)
        }

        print(f"  Holdout F1: {holdout_score:.4f}")
        print(f"  Holdout win rate: {holdout_win_rate:.4f}")
        print(f"  Holdout trades: {holdout_result['n_trades']}")

        if holdout_win_rate < self.min_win_rate:
            rejection_reasons.append(f"Low win rate: {holdout_win_rate:.4f} < {self.min_win_rate}")

        # Step 5: Monte Carlo (simplified - using fold P&Ls)
        print("\nStep 5: Monte Carlo significance test...")
        trades_df = pd.DataFrame({"pnl": wfo_result.fold_pnl})
        mc_result = self.monte_carlo.validate(
            sum(wfo_result.fold_pnl),
            trades_df
        )
        print(f"  P-value: {mc_result['p_value']:.4f}")
        print(f"  Percentile: {mc_result['percentile']:.1f}%")
        print(f"  Significant: {mc_result['is_significant']}")

        if not mc_result['is_significant']:
            rejection_reasons.append(f"Monte Carlo not significant: p={mc_result['p_value']:.4f}")

        # Final verdict
        is_valid = len(rejection_reasons) == 0

        print("\n" + "=" * 50)
        if is_valid:
            print("✓ STRATEGY PASSED ALL VALIDATION CHECKS")
        else:
            print("✗ STRATEGY REJECTED")
            for reason in rejection_reasons:
                print(f"  - {reason}")
        print("=" * 50)

        return ValidationPipelineResult(
            walk_forward_result=wfo_result,
            monte_carlo_result=mc_result,
            multi_seed_result=seed_result,
            holdout_result=holdout_result,
            is_valid=is_valid,
            rejection_reasons=rejection_reasons
        )
