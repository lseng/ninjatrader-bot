"""
Sample Trading Strategies

Classic technical analysis strategies for backtesting and learning.
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


class SMAStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.

    Goes long when fast SMA crosses above slow SMA.
    Goes short when fast SMA crosses below slow SMA.
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        atr_stop_mult: float = 2.0,
        atr_target_mult: float = 3.0
    ):
        super().__init__("SMA_Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult

    def generate_signal(
        self,
        current_bar: pd.Series,
        history: pd.DataFrame
    ) -> Signal | None:
        if len(history) < self.slow_period + 2:
            return None

        close = history["close"]
        fast_sma = close.rolling(self.fast_period).mean()
        slow_sma = close.rolling(self.slow_period).mean()

        # Current values
        fast_now = fast_sma.iloc[-1]
        slow_now = slow_sma.iloc[-1]
        fast_prev = fast_sma.iloc[-2]
        slow_prev = slow_sma.iloc[-2]

        # Detect crossover
        cross_up = fast_prev <= slow_prev and fast_now > slow_now
        cross_down = fast_prev >= slow_prev and fast_now < slow_now

        if not cross_up and not cross_down:
            return None

        # Calculate ATR for stops
        atr = (history["high"] - history["low"]).rolling(14).mean().iloc[-1]
        price = current_bar["close"]

        if cross_up and self.position != "long":
            return Signal(
                direction="long",
                confidence=0.7,
                stop_loss=price - (atr * self.atr_stop_mult),
                take_profit=price + (atr * self.atr_target_mult),
                reason=f"SMA crossover: {self.fast_period} > {self.slow_period}"
            )

        elif cross_down and self.position != "short":
            return Signal(
                direction="short",
                confidence=0.7,
                stop_loss=price + (atr * self.atr_stop_mult),
                take_profit=price - (atr * self.atr_target_mult),
                reason=f"SMA crossover: {self.fast_period} < {self.slow_period}"
            )

        return None

    def warmup_period(self) -> int:
        return self.slow_period + 5


class RSIStrategy(BaseStrategy):
    """
    RSI Mean Reversion Strategy.

    Goes long when RSI is oversold (< 30).
    Goes short when RSI is overbought (> 70).
    Exits when RSI returns to neutral zone.
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        exit_level: float = 50,
        atr_stop_mult: float = 1.5
    ):
        super().__init__("RSI_Reversion")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_level = exit_level
        self.atr_stop_mult = atr_stop_mult

    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """Calculate RSI."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(self.period).mean()
        avg_loss = loss.rolling(self.period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def generate_signal(
        self,
        current_bar: pd.Series,
        history: pd.DataFrame
    ) -> Signal | None:
        if len(history) < self.period + 5:
            return None

        rsi = self._calculate_rsi(history["close"])
        current_rsi = rsi.iloc[-1]

        atr = (history["high"] - history["low"]).rolling(14).mean().iloc[-1]
        price = current_bar["close"]

        # Entry signals
        if self.position == "flat":
            if current_rsi < self.oversold:
                return Signal(
                    direction="long",
                    confidence=min(0.9, (self.oversold - current_rsi) / 30 + 0.5),
                    stop_loss=price - (atr * self.atr_stop_mult),
                    take_profit=price + (atr * 2),
                    reason=f"RSI oversold: {current_rsi:.1f}"
                )

            elif current_rsi > self.overbought:
                return Signal(
                    direction="short",
                    confidence=min(0.9, (current_rsi - self.overbought) / 30 + 0.5),
                    stop_loss=price + (atr * self.atr_stop_mult),
                    take_profit=price - (atr * 2),
                    reason=f"RSI overbought: {current_rsi:.1f}"
                )

        # Exit signals
        elif self.position == "long" and current_rsi > self.exit_level:
            return Signal(
                direction="flat",
                confidence=0.8,
                reason=f"RSI exit: {current_rsi:.1f} > {self.exit_level}"
            )

        elif self.position == "short" and current_rsi < self.exit_level:
            return Signal(
                direction="flat",
                confidence=0.8,
                reason=f"RSI exit: {current_rsi:.1f} < {self.exit_level}"
            )

        return None

    def warmup_period(self) -> int:
        return self.period + 10


class MACDStrategy(BaseStrategy):
    """
    MACD Trend Following Strategy.

    Goes long when MACD crosses above signal line and histogram is positive.
    Goes short when MACD crosses below signal line and histogram is negative.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        atr_stop_mult: float = 2.0,
        atr_target_mult: float = 3.0
    ):
        super().__init__("MACD_Trend")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult

    def _calculate_macd(self, close: pd.Series) -> tuple:
        """Calculate MACD, signal, and histogram."""
        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()

        macd = fast_ema - slow_ema
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd - signal

        return macd, signal, histogram

    def generate_signal(
        self,
        current_bar: pd.Series,
        history: pd.DataFrame
    ) -> Signal | None:
        if len(history) < self.slow_period + self.signal_period + 5:
            return None

        macd, signal, histogram = self._calculate_macd(history["close"])

        macd_now = macd.iloc[-1]
        signal_now = signal.iloc[-1]
        hist_now = histogram.iloc[-1]

        macd_prev = macd.iloc[-2]
        signal_prev = signal.iloc[-2]

        # Detect crossover
        cross_up = macd_prev <= signal_prev and macd_now > signal_now
        cross_down = macd_prev >= signal_prev and macd_now < signal_now

        atr = (history["high"] - history["low"]).rolling(14).mean().iloc[-1]
        price = current_bar["close"]

        # Long signal: MACD crosses above signal with positive histogram
        if cross_up and hist_now > 0 and self.position != "long":
            return Signal(
                direction="long",
                confidence=0.75,
                stop_loss=price - (atr * self.atr_stop_mult),
                take_profit=price + (atr * self.atr_target_mult),
                reason=f"MACD bullish crossover, histogram: {hist_now:.4f}"
            )

        # Short signal: MACD crosses below signal with negative histogram
        elif cross_down and hist_now < 0 and self.position != "short":
            return Signal(
                direction="short",
                confidence=0.75,
                stop_loss=price + (atr * self.atr_stop_mult),
                take_profit=price - (atr * self.atr_target_mult),
                reason=f"MACD bearish crossover, histogram: {hist_now:.4f}"
            )

        return None

    def warmup_period(self) -> int:
        return self.slow_period + self.signal_period + 10
