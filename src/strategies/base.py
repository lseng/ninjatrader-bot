"""
Base Strategy Class

Abstract base class for all trading strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
import pandas as pd


@dataclass
class Signal:
    """Trading signal from a strategy."""
    direction: Literal["long", "short", "flat"]
    confidence: float  # 0.0 to 1.0
    stop_loss: float | None = None
    take_profit: float | None = None
    size: int = 1
    reason: str = ""


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    All strategies must implement the generate_signal method.
    """

    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.position: Literal["long", "short", "flat"] = "flat"
        self.entry_price: float | None = None

    @abstractmethod
    def generate_signal(
        self,
        current_bar: pd.Series,
        history: pd.DataFrame
    ) -> Signal | None:
        """
        Generate a trading signal based on current market data.

        Args:
            current_bar: Current OHLCV bar
            history: Historical bars including current

        Returns:
            Signal if action should be taken, None otherwise
        """
        pass

    def on_trade_entry(self, price: float, direction: Literal["long", "short"]):
        """Called when a trade is entered."""
        self.position = direction
        self.entry_price = price

    def on_trade_exit(self):
        """Called when a trade is exited."""
        self.position = "flat"
        self.entry_price = None

    def warmup_period(self) -> int:
        """
        Number of bars needed before strategy can generate signals.

        Override in subclass if needed.
        """
        return 50

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
