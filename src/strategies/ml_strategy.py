"""
ML-Based Trading Strategy

Uses trained ML models to generate trading signals.
"""

import pandas as pd
from typing import Literal

from .base import BaseStrategy, Signal
from ..ml.models import StrategyModel, Prediction
from ..data.processor import DataProcessor


class MLStrategy(BaseStrategy):
    """
    Machine learning based trading strategy.

    Uses a trained model to predict market direction and generate signals.
    """

    def __init__(
        self,
        model: StrategyModel,
        confidence_threshold: float = 0.6,
        use_stops: bool = True,
        atr_stop_mult: float = 2.0,
        atr_target_mult: float = 3.0,
        name: str = "MLStrategy"
    ):
        """
        Initialize ML strategy.

        Args:
            model: Trained StrategyModel
            confidence_threshold: Minimum confidence to enter trade
            use_stops: Whether to use stop loss/take profit
            atr_stop_mult: ATR multiplier for stop loss
            atr_target_mult: ATR multiplier for take profit
            name: Strategy name
        """
        super().__init__(name)
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.use_stops = use_stops
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult

        # Feature columns expected by model
        self.feature_columns = model.models[0].feature_names if model.models else []

    def generate_signal(
        self,
        current_bar: pd.Series,
        history: pd.DataFrame
    ) -> Signal | None:
        """Generate signal using ML model."""
        # Need enough history for features
        if len(history) < self.warmup_period():
            return None

        # Process data to generate features
        processor = DataProcessor(history)
        processor.add_all_features()
        features_df = processor.get_dataframe()

        # Get features for current bar
        if len(features_df) == 0:
            return None

        # Filter to model's expected features
        available_features = [c for c in self.feature_columns if c in features_df.columns]
        if len(available_features) < len(self.feature_columns) * 0.8:
            # Missing too many features
            return None

        # Fill missing features with 0
        current_features = features_df.iloc[[-1]][available_features].fillna(0)

        # Get prediction
        try:
            prediction = self.model.predict(current_features)
        except Exception:
            return None

        # Check confidence
        if prediction.confidence < self.confidence_threshold:
            return None

        # Determine direction
        if prediction.signal == 1:
            direction = "long"
        elif prediction.signal == -1:
            direction = "short"
        else:
            return None  # No signal

        # Calculate stops if enabled
        stop_loss = None
        take_profit = None

        if self.use_stops:
            atr = (history["high"] - history["low"]).rolling(14).mean().iloc[-1]
            current_price = current_bar["close"]

            if direction == "long":
                stop_loss = current_price - (atr * self.atr_stop_mult)
                take_profit = current_price + (atr * self.atr_target_mult)
            else:
                stop_loss = current_price + (atr * self.atr_stop_mult)
                take_profit = current_price - (atr * self.atr_target_mult)

        return Signal(
            direction=direction,
            confidence=prediction.confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=f"ML prediction: {prediction.signal} with {prediction.confidence:.2%} confidence"
        )

    def warmup_period(self) -> int:
        """ML strategy needs more history for features."""
        return 100
