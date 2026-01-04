"""
Backtesting Engine

Event-driven backtesting engine for strategy evaluation.
Designed for futures trading with proper handling of:
- Commission and slippage
- Position sizing
- Risk management
- Performance tracking
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Literal
from enum import Enum


class Side(Enum):
    LONG = "long"
    SHORT = "short"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: datetime
    exit_time: datetime
    side: Side
    entry_price: float
    exit_price: float
    size: int
    pnl: float
    commission: float
    slippage: float
    bars_held: int
    exit_reason: str = ""


@dataclass
class Position:
    """Current open position."""
    side: Side
    entry_price: float
    size: int
    entry_time: datetime
    entry_bar: int
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100_000.0
    commission_per_contract: float = 2.50  # Round trip
    slippage_ticks: float = 1.0
    tick_size: float = 0.25  # MES tick size
    tick_value: float = 1.25  # MES tick value ($5 for ES, $1.25 for MES)
    max_position_size: int = 10
    risk_per_trade: float = 0.02  # 2% risk per trade
    use_stops: bool = True


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: list[Trade]
    equity_curve: pd.Series
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    total_bars: int


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Supports:
    - Long and short positions
    - Stop loss and take profit orders
    - Commission and slippage modeling
    - Position sizing based on risk
    """

    def __init__(self, config: BacktestConfig | None = None):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        self.reset()

    def reset(self):
        """Reset engine state for new backtest."""
        self.capital = self.config.initial_capital
        self.position: Position | None = None
        self.trades: list[Trade] = []
        self.equity_history: list[float] = []
        self.current_bar = 0
        self.current_time: datetime | None = None

    def _calculate_slippage(self, side: Side, is_entry: bool) -> float:
        """Calculate slippage in price terms."""
        slippage = self.config.slippage_ticks * self.config.tick_size
        # Slippage works against us
        if (side == Side.LONG and is_entry) or (side == Side.SHORT and not is_entry):
            return slippage  # Pay more
        return -slippage  # Receive less

    def _calculate_pnl(self, side: Side, entry: float, exit: float, size: int) -> float:
        """Calculate P&L for a trade."""
        ticks = (exit - entry) / self.config.tick_size
        if side == Side.SHORT:
            ticks = -ticks
        return ticks * self.config.tick_value * size

    def enter_long(
        self,
        price: float,
        size: int = 1,
        stop_loss: float | None = None,
        take_profit: float | None = None
    ) -> bool:
        """
        Enter a long position.

        Args:
            price: Entry price
            size: Number of contracts
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            True if order filled
        """
        if self.position is not None:
            return False

        if size > self.config.max_position_size:
            size = self.config.max_position_size

        slippage = self._calculate_slippage(Side.LONG, True)
        fill_price = price + slippage

        self.position = Position(
            side=Side.LONG,
            entry_price=fill_price,
            size=size,
            entry_time=self.current_time,
            entry_bar=self.current_bar,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        # Deduct commission
        self.capital -= self.config.commission_per_contract * size

        return True

    def enter_short(
        self,
        price: float,
        size: int = 1,
        stop_loss: float | None = None,
        take_profit: float | None = None
    ) -> bool:
        """
        Enter a short position.

        Args:
            price: Entry price
            size: Number of contracts
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            True if order filled
        """
        if self.position is not None:
            return False

        if size > self.config.max_position_size:
            size = self.config.max_position_size

        slippage = self._calculate_slippage(Side.SHORT, True)
        fill_price = price + slippage

        self.position = Position(
            side=Side.SHORT,
            entry_price=fill_price,
            size=size,
            entry_time=self.current_time,
            entry_bar=self.current_bar,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        # Deduct commission
        self.capital -= self.config.commission_per_contract * size

        return True

    def exit_position(self, price: float, reason: str = "") -> Trade | None:
        """
        Exit current position.

        Args:
            price: Exit price
            reason: Reason for exit

        Returns:
            Completed Trade object
        """
        if self.position is None:
            return None

        pos = self.position
        slippage = self._calculate_slippage(pos.side, False)
        fill_price = price + slippage

        pnl = self._calculate_pnl(
            pos.side,
            pos.entry_price,
            fill_price,
            pos.size
        )

        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=self.current_time,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=fill_price,
            size=pos.size,
            pnl=pnl,
            commission=self.config.commission_per_contract * pos.size * 2,  # Round trip
            slippage=abs(self._calculate_slippage(pos.side, True)) + abs(slippage),
            bars_held=self.current_bar - pos.entry_bar,
            exit_reason=reason
        )

        self.trades.append(trade)
        self.capital += pnl
        self.position = None

        return trade

    def check_stops(self, high: float, low: float) -> Trade | None:
        """
        Check if stop loss or take profit was hit.

        Args:
            high: Current bar high
            low: Current bar low

        Returns:
            Trade if position was closed
        """
        if self.position is None:
            return None

        pos = self.position

        # Check stop loss
        if pos.stop_loss is not None:
            if pos.side == Side.LONG and low <= pos.stop_loss:
                return self.exit_position(pos.stop_loss, "stop_loss")
            elif pos.side == Side.SHORT and high >= pos.stop_loss:
                return self.exit_position(pos.stop_loss, "stop_loss")

        # Check take profit
        if pos.take_profit is not None:
            if pos.side == Side.LONG and high >= pos.take_profit:
                return self.exit_position(pos.take_profit, "take_profit")
            elif pos.side == Side.SHORT and low <= pos.take_profit:
                return self.exit_position(pos.take_profit, "take_profit")

        return None

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L for open position."""
        if self.position is None:
            return 0.0

        return self._calculate_pnl(
            self.position.side,
            self.position.entry_price,
            current_price,
            self.position.size
        )

    def get_equity(self, current_price: float) -> float:
        """Get current equity including unrealized P&L."""
        return self.capital + self.get_unrealized_pnl(current_price)

    def run(
        self,
        data: pd.DataFrame,
        strategy: Callable[["BacktestEngine", pd.Series, pd.DataFrame], None]
    ) -> BacktestResult:
        """
        Run backtest with a strategy function.

        Args:
            data: OHLCV DataFrame with at least: timestamp, open, high, low, close, volume
            strategy: Strategy function that receives (engine, current_bar, history)

        Returns:
            BacktestResult with all trades and metrics
        """
        self.reset()

        for i in range(len(data)):
            self.current_bar = i
            self.current_time = data.iloc[i]["timestamp"]

            current = data.iloc[i]
            history = data.iloc[:i+1]

            # Check stops first
            if self.position is not None:
                self.check_stops(current["high"], current["low"])

            # Run strategy
            strategy(self, current, history)

            # Record equity
            self.equity_history.append(self.get_equity(current["close"]))

        # Close any remaining position at last close
        if self.position is not None:
            self.exit_position(data.iloc[-1]["close"], "end_of_data")

        return BacktestResult(
            trades=self.trades,
            equity_curve=pd.Series(self.equity_history, index=data["timestamp"]),
            config=self.config,
            start_date=data.iloc[0]["timestamp"],
            end_date=data.iloc[-1]["timestamp"],
            total_bars=len(data)
        )


def example_strategy(engine: BacktestEngine, bar: pd.Series, history: pd.DataFrame):
    """
    Example mean reversion strategy.

    This is a simple example - real strategies should be more sophisticated.
    """
    if len(history) < 20:
        return

    # Simple SMA crossover
    close = history["close"]
    sma_fast = close.rolling(5).mean().iloc[-1]
    sma_slow = close.rolling(20).mean().iloc[-1]

    current_price = bar["close"]
    atr = (history["high"] - history["low"]).rolling(14).mean().iloc[-1]

    if engine.position is None:
        # Entry logic
        if sma_fast > sma_slow:
            engine.enter_long(
                current_price,
                size=1,
                stop_loss=current_price - 2 * atr,
                take_profit=current_price + 3 * atr
            )
        elif sma_fast < sma_slow:
            engine.enter_short(
                current_price,
                size=1,
                stop_loss=current_price + 2 * atr,
                take_profit=current_price - 3 * atr
            )
