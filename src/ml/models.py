"""
ML Strategy Models

Various machine learning models for trading signal generation.
"""

import pandas as pd
import numpy as np
from typing import Literal, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class Prediction:
    """Model prediction with confidence."""
    signal: int  # 1 = long, -1 = short, 0 = neutral
    confidence: float  # 0.0 to 1.0
    probabilities: dict[int, float] | None = None


class BaseModel(ABC):
    """Abstract base class for trading models."""

    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Prediction:
        """Generate prediction for single sample."""
        pass

    def predict_batch(self, X: pd.DataFrame) -> list[Prediction]:
        """Generate predictions for multiple samples."""
        predictions = []
        for i in range(len(X)):
            pred = self.predict(X.iloc[[i]])
            predictions.append(pred)
        return predictions

    def save(self, path: str | Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "name": self.name
        }, path)

    def load(self, path: str | Path) -> "BaseModel":
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.name = data["name"]
        self.is_fitted = True
        return self


class RandomForestModel(BaseModel):
    """Random Forest classifier for trading signals."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 10,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        random_state: int = 42
    ):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        """Train the Random Forest model."""
        self.feature_names = list(X.columns)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> Prediction:
        """Generate prediction."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[0]
        classes = self.model.classes_

        pred_class = classes[np.argmax(proba)]
        confidence = np.max(proba)

        return Prediction(
            signal=int(pred_class),
            confidence=float(confidence),
            probabilities={int(c): float(p) for c, p in zip(classes, proba)}
        )

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        return importance


class GradientBoostingModel(BaseModel):
    """Gradient Boosting classifier for trading signals."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        min_samples_split: int = 10,
        random_state: int = 42
    ):
        super().__init__("GradientBoosting")
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingModel":
        """Train the Gradient Boosting model."""
        self.feature_names = list(X.columns)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> Prediction:
        """Generate prediction."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[0]
        classes = self.model.classes_

        pred_class = classes[np.argmax(proba)]
        confidence = np.max(proba)

        return Prediction(
            signal=int(pred_class),
            confidence=float(confidence),
            probabilities={int(c): float(p) for c, p in zip(classes, proba)}
        )


class StrategyModel:
    """
    High-level strategy model that combines multiple ML models.

    Supports ensemble methods and confidence thresholds.
    """

    def __init__(
        self,
        models: list[BaseModel] | None = None,
        confidence_threshold: float = 0.6,
        ensemble_method: Literal["voting", "weighted"] = "voting"
    ):
        """
        Initialize strategy model.

        Args:
            models: List of base models to use
            confidence_threshold: Minimum confidence to generate signal
            ensemble_method: How to combine model predictions
        """
        self.models = models or [RandomForestModel(), GradientBoostingModel()]
        self.confidence_threshold = confidence_threshold
        self.ensemble_method = ensemble_method
        self.weights: list[float] = [1.0] * len(self.models)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StrategyModel":
        """Train all models."""
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> Prediction:
        """
        Generate ensemble prediction.

        Uses voting or weighted averaging based on ensemble_method.
        """
        predictions = [model.predict(X) for model in self.models]

        if self.ensemble_method == "voting":
            # Majority voting
            signals = [p.signal for p in predictions]
            signal = max(set(signals), key=signals.count)

            # Average confidence for the winning signal
            confidences = [p.confidence for p in predictions if p.signal == signal]
            confidence = np.mean(confidences)

        else:  # weighted
            # Weighted average of probabilities
            all_probs = {}
            for pred, weight in zip(predictions, self.weights):
                if pred.probabilities:
                    for cls, prob in pred.probabilities.items():
                        all_probs[cls] = all_probs.get(cls, 0) + prob * weight

            total_weight = sum(self.weights)
            all_probs = {k: v / total_weight for k, v in all_probs.items()}

            signal = max(all_probs, key=all_probs.get)
            confidence = all_probs[signal]

        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            signal = 0

        return Prediction(signal=signal, confidence=confidence)

    def save(self, path: str | Path):
        """Save all models."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for i, model in enumerate(self.models):
            model.save(path / f"model_{i}.joblib")

        # Save ensemble config
        joblib.dump({
            "confidence_threshold": self.confidence_threshold,
            "ensemble_method": self.ensemble_method,
            "weights": self.weights,
            "n_models": len(self.models)
        }, path / "ensemble_config.joblib")

    def load(self, path: str | Path) -> "StrategyModel":
        """Load all models."""
        path = Path(path)

        config = joblib.load(path / "ensemble_config.joblib")
        self.confidence_threshold = config["confidence_threshold"]
        self.ensemble_method = config["ensemble_method"]
        self.weights = config["weights"]

        self.models = []
        for i in range(config["n_models"]):
            # Detect model type from saved file
            model = RandomForestModel()  # Default
            model.load(path / f"model_{i}.joblib")
            self.models.append(model)

        return self
