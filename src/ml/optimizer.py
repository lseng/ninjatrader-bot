"""
Strategy Optimizer

Hyperparameter optimization for trading strategies using Optuna.
"""

import pandas as pd
import numpy as np
from typing import Any, Callable
from dataclasses import dataclass
import optuna
from optuna.samplers import TPESampler

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

from .models import RandomForestModel, GradientBoostingModel, StrategyModel


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization."""
    best_params: dict[str, Any]
    best_score: float
    study: optuna.Study
    n_trials: int


class StrategyOptimizer:
    """
    Optimize trading strategy hyperparameters.

    Uses Optuna for Bayesian optimization with time-series cross-validation.
    """

    def __init__(
        self,
        n_trials: int = 100,
        n_cv_splits: int = 5,
        metric: str = "f1",
        random_state: int = 42,
        timeout: int | None = None
    ):
        """
        Initialize optimizer.

        Args:
            n_trials: Number of optimization trials
            n_cv_splits: Number of CV folds
            metric: Optimization metric ("accuracy", "f1", "precision")
            random_state: Random seed
            timeout: Maximum optimization time in seconds
        """
        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.metric = metric
        self.random_state = random_state
        self.timeout = timeout

    def optimize_random_forest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Optimize Random Forest hyperparameters.

        Args:
            X: Feature matrix
            y: Labels
            verbose: Show progress

        Returns:
            OptimizationResult with best parameters
        """
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }

            model = RandomForestModel(**params, random_state=self.random_state)
            score = self._cross_validate(model, X, y)

            return score

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state)
        )

        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=verbose
        )

        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            study=study,
            n_trials=len(study.trials)
        )

    def optimize_gradient_boosting(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Optimize Gradient Boosting hyperparameters.

        Args:
            X: Feature matrix
            y: Labels
            verbose: Show progress

        Returns:
            OptimizationResult with best parameters
        """
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            }

            model = GradientBoostingModel(**params, random_state=self.random_state)
            score = self._cross_validate(model, X, y)

            return score

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state)
        )

        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=verbose
        )

        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            study=study,
            n_trials=len(study.trials)
        )

    def optimize_strategy(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Optimize full strategy including model selection and thresholds.

        Args:
            X: Feature matrix
            y: Labels
            verbose: Show progress

        Returns:
            OptimizationResult with best configuration
        """
        def objective(trial: optuna.Trial) -> float:
            # Model selection
            model_type = trial.suggest_categorical("model_type", ["rf", "gb", "ensemble"])

            if model_type == "rf":
                model = RandomForestModel(
                    n_estimators=trial.suggest_int("rf_n_estimators", 50, 200),
                    max_depth=trial.suggest_int("rf_max_depth", 3, 15),
                    min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 15),
                    random_state=self.random_state
                )

            elif model_type == "gb":
                model = GradientBoostingModel(
                    n_estimators=trial.suggest_int("gb_n_estimators", 50, 200),
                    learning_rate=trial.suggest_float("gb_learning_rate", 0.01, 0.2, log=True),
                    max_depth=trial.suggest_int("gb_max_depth", 3, 8),
                    random_state=self.random_state
                )

            else:  # ensemble
                confidence_threshold = trial.suggest_float("confidence_threshold", 0.5, 0.8)
                model = StrategyModel(
                    models=[
                        RandomForestModel(
                            n_estimators=100,
                            max_depth=10,
                            random_state=self.random_state
                        ),
                        GradientBoostingModel(
                            n_estimators=100,
                            learning_rate=0.1,
                            random_state=self.random_state
                        )
                    ],
                    confidence_threshold=confidence_threshold
                )

            score = self._cross_validate(model, X, y)
            return score

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state)
        )

        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=verbose
        )

        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            study=study,
            n_trials=len(study.trials)
        )

    def _cross_validate(
        self,
        model: RandomForestModel | GradientBoostingModel | StrategyModel,
        X: pd.DataFrame,
        y: pd.Series
    ) -> float:
        """Perform time-series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Clone and train model
            model_clone = type(model)()
            if hasattr(model, 'models'):  # StrategyModel
                model_clone.confidence_threshold = model.confidence_threshold
            model_clone.fit(X_train, y_train)

            # Predict
            predictions = []
            for i in range(len(X_test)):
                pred = model_clone.predict(X_test.iloc[[i]])
                predictions.append(pred.signal)

            predictions = np.array(predictions)

            # Score
            if self.metric == "accuracy":
                score = accuracy_score(y_test, predictions)
            elif self.metric == "f1":
                score = f1_score(y_test, predictions, average="weighted", zero_division=0)
            else:
                score = accuracy_score(y_test, predictions)

            scores.append(score)

        return np.mean(scores)

    def plot_optimization_history(self, result: OptimizationResult):
        """Plot optimization history."""
        try:
            import plotly.express as px

            fig = optuna.visualization.plot_optimization_history(result.study)
            fig.show()
        except ImportError:
            print("Install plotly for visualization: pip install plotly")

    def plot_param_importances(self, result: OptimizationResult):
        """Plot parameter importances."""
        try:
            fig = optuna.visualization.plot_param_importances(result.study)
            fig.show()
        except ImportError:
            print("Install plotly for visualization: pip install plotly")
