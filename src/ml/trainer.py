"""
Model Trainer

Handles training, validation, and cross-validation of trading models.
"""

import pandas as pd
import numpy as np
from typing import Any
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from .models import BaseModel, StrategyModel


@dataclass
class TrainingResult:
    """Results from model training."""
    train_accuracy: float
    val_accuracy: float
    train_f1: float
    val_f1: float
    precision: dict[int, float]
    recall: dict[int, float]
    confusion_matrix: np.ndarray
    classification_report: str
    cv_scores: list[float] | None = None


class ModelTrainer:
    """
    Train and validate trading models.

    Uses time-series cross-validation to prevent look-ahead bias.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.2,
        gap: int = 0
    ):
        """
        Initialize trainer.

        Args:
            n_splits: Number of CV splits
            test_size: Fraction of data for testing
            gap: Number of samples to skip between train/test
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train/test sets preserving time order.

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            X_train, X_test, y_train, y_test
        """
        split_idx = int(len(X) * (1 - self.test_size))

        X_train = X.iloc[:split_idx - self.gap]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx - self.gap]
        y_test = y.iloc[split_idx:]

        return X_train, X_test, y_train, y_test

    def train(
        self,
        model: BaseModel | StrategyModel,
        X: pd.DataFrame,
        y: pd.Series,
        validate: bool = True
    ) -> TrainingResult:
        """
        Train a model and evaluate performance.

        Args:
            model: Model to train
            X: Feature matrix
            y: Labels
            validate: Whether to validate on held-out data

        Returns:
            TrainingResult with metrics
        """
        if validate:
            X_train, X_test, y_train, y_test = self.train_test_split(X, y)
        else:
            X_train, y_train = X, y
            X_test, y_test = X, y

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        train_preds = self._batch_predict(model, X_train)
        test_preds = self._batch_predict(model, X_test)

        # Metrics
        train_accuracy = accuracy_score(y_train, train_preds)
        test_accuracy = accuracy_score(y_test, test_preds)

        train_f1 = f1_score(y_train, train_preds, average="weighted", zero_division=0)
        test_f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)

        # Per-class metrics
        classes = sorted(y.unique())
        precision = {}
        recall = {}

        for cls in classes:
            precision[cls] = precision_score(
                y_test == cls, test_preds == cls, zero_division=0
            )
            recall[cls] = recall_score(
                y_test == cls, test_preds == cls, zero_division=0
            )

        cm = confusion_matrix(y_test, test_preds, labels=classes)
        report = classification_report(y_test, test_preds, zero_division=0)

        return TrainingResult(
            train_accuracy=train_accuracy,
            val_accuracy=test_accuracy,
            train_f1=train_f1,
            val_f1=test_f1,
            precision=precision,
            recall=recall,
            confusion_matrix=cm,
            classification_report=report
        )

    def cross_validate(
        self,
        model: BaseModel | StrategyModel,
        X: pd.DataFrame,
        y: pd.Series
    ) -> TrainingResult:
        """
        Perform time-series cross-validation.

        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Labels

        Returns:
            TrainingResult with CV scores
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap)

        cv_scores = []
        all_test_preds = []
        all_test_labels = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Clone model for each fold
            model_clone = type(model)()
            model_clone.fit(X_train, y_train)

            test_preds = self._batch_predict(model_clone, X_test)
            score = accuracy_score(y_test, test_preds)
            cv_scores.append(score)

            all_test_preds.extend(test_preds)
            all_test_labels.extend(y_test.tolist())

        # Aggregate metrics
        all_test_preds = np.array(all_test_preds)
        all_test_labels = np.array(all_test_labels)

        classes = sorted(y.unique())
        precision = {}
        recall = {}

        for cls in classes:
            precision[cls] = precision_score(
                all_test_labels == cls, all_test_preds == cls, zero_division=0
            )
            recall[cls] = recall_score(
                all_test_labels == cls, all_test_preds == cls, zero_division=0
            )

        cm = confusion_matrix(all_test_labels, all_test_preds, labels=classes)
        report = classification_report(all_test_labels, all_test_preds, zero_division=0)

        return TrainingResult(
            train_accuracy=np.mean(cv_scores),
            val_accuracy=np.mean(cv_scores),
            train_f1=f1_score(all_test_labels, all_test_preds, average="weighted", zero_division=0),
            val_f1=f1_score(all_test_labels, all_test_preds, average="weighted", zero_division=0),
            precision=precision,
            recall=recall,
            confusion_matrix=cm,
            classification_report=report,
            cv_scores=cv_scores
        )

    def _batch_predict(
        self,
        model: BaseModel | StrategyModel,
        X: pd.DataFrame
    ) -> np.ndarray:
        """Get predictions as numpy array."""
        predictions = []
        for i in range(len(X)):
            pred = model.predict(X.iloc[[i]])
            predictions.append(pred.signal)
        return np.array(predictions)

    def print_results(self, result: TrainingResult):
        """Print formatted training results."""
        print("\n" + "=" * 50)
        print("TRAINING RESULTS")
        print("=" * 50)

        print(f"\nAccuracy:  Train={result.train_accuracy:.4f}  Val={result.val_accuracy:.4f}")
        print(f"F1 Score:  Train={result.train_f1:.4f}  Val={result.val_f1:.4f}")

        if result.cv_scores:
            print(f"\nCV Scores: {[f'{s:.4f}' for s in result.cv_scores]}")
            print(f"CV Mean: {np.mean(result.cv_scores):.4f} (+/- {np.std(result.cv_scores):.4f})")

        print("\nPer-Class Metrics:")
        for cls in result.precision:
            print(f"  Class {cls}: Precision={result.precision[cls]:.4f}, Recall={result.recall[cls]:.4f}")

        print("\nConfusion Matrix:")
        print(result.confusion_matrix)

        print("\nClassification Report:")
        print(result.classification_report)
