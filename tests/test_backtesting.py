"""Tests for backtesting engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.engine import BacktestEngine, BacktestConfig, Side
from src.backtesting.metrics import PerformanceMetrics


def generate_test_data(n_bars: int = 100, trend: float = 0.0001) -> pd.DataFrame:
    """Generate simple test data."""
    np.random.seed(42)

    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n_bars)]
    prices = [5000.0]

    for _ in range(n_bars - 1):
        change = np.random.normal(trend, 0.001)
        prices.append(prices[-1] * (1 + change))

    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        volatility = abs(np.random.normal(0, 0.001)) * close
        data.append({
            "timestamp": ts,
            "open": close if i == 0 else prices[i-1],
            "high": close + volatility,
            "low": close - volatility,
            "close": close,
            "volume": np.random.randint(100, 1000)
        })

    return pd.DataFrame(data)


class TestBacktestEngine:
    """Test backtest engine functionality."""

    def test_initialization(self):
        """Test engine initializes correctly."""
        config = BacktestConfig(initial_capital=50000)
        engine = BacktestEngine(config)

        assert engine.capital == 50000
        assert engine.position is None
        assert len(engine.trades) == 0

    def test_enter_long(self):
        """Test entering a long position."""
        engine = BacktestEngine()
        engine.current_time = datetime.now()
        engine.current_bar = 0

        result = engine.enter_long(5000.0, size=1)

        assert result is True
        assert engine.position is not None
        assert engine.position.side == Side.LONG
        assert engine.position.size == 1

    def test_enter_short(self):
        """Test entering a short position."""
        engine = BacktestEngine()
        engine.current_time = datetime.now()
        engine.current_bar = 0

        result = engine.enter_short(5000.0, size=2)

        assert result is True
        assert engine.position is not None
        assert engine.position.side == Side.SHORT
        assert engine.position.size == 2

    def test_no_double_entry(self):
        """Test that we can't enter when already in position."""
        engine = BacktestEngine()
        engine.current_time = datetime.now()
        engine.current_bar = 0

        engine.enter_long(5000.0)
        result = engine.enter_long(5000.0)

        assert result is False

    def test_exit_position(self):
        """Test exiting a position."""
        engine = BacktestEngine()
        engine.current_time = datetime.now()
        engine.current_bar = 0

        engine.enter_long(5000.0, size=1)
        engine.current_bar = 5
        trade = engine.exit_position(5010.0, "test_exit")

        assert trade is not None
        assert engine.position is None
        assert trade.side == Side.LONG
        assert trade.exit_reason == "test_exit"
        assert len(engine.trades) == 1

    def test_stop_loss_long(self):
        """Test stop loss triggers for long position."""
        engine = BacktestEngine()
        engine.current_time = datetime.now()
        engine.current_bar = 0

        engine.enter_long(5000.0, size=1, stop_loss=4990.0)
        engine.current_bar = 1

        trade = engine.check_stops(high=5005.0, low=4985.0)

        assert trade is not None
        assert trade.exit_reason == "stop_loss"
        assert engine.position is None

    def test_take_profit_long(self):
        """Test take profit triggers for long position."""
        engine = BacktestEngine()
        engine.current_time = datetime.now()
        engine.current_bar = 0

        engine.enter_long(5000.0, size=1, take_profit=5020.0)
        engine.current_bar = 1

        trade = engine.check_stops(high=5025.0, low=5010.0)

        assert trade is not None
        assert trade.exit_reason == "take_profit"

    def test_run_backtest(self):
        """Test running a full backtest."""
        data = generate_test_data(100)
        engine = BacktestEngine()

        def simple_strategy(eng, bar, history):
            if len(history) < 10:
                return
            if eng.position is None:
                eng.enter_long(bar["close"], size=1)
            elif eng.current_bar > 50:
                eng.exit_position(bar["close"])

        result = engine.run(data, simple_strategy)

        assert result is not None
        assert len(result.equity_curve) == len(data)
        assert result.total_bars == len(data)


class TestPerformanceMetrics:
    """Test performance metrics calculation."""

    def test_empty_trades(self):
        """Test metrics with no trades."""
        metrics = PerformanceMetrics._empty_metrics()

        assert metrics.total_trades == 0
        assert metrics.win_rate == 0

    def test_metrics_calculation(self):
        """Test basic metrics are calculated correctly."""
        data = generate_test_data(200, trend=0.0002)
        config = BacktestConfig(initial_capital=100000)
        engine = BacktestEngine(config)

        def trend_strategy(eng, bar, history):
            if len(history) < 20:
                return
            sma = history["close"].rolling(10).mean().iloc[-1]
            if eng.position is None and bar["close"] > sma:
                eng.enter_long(bar["close"], size=1)
            elif eng.position is not None and bar["close"] < sma:
                eng.exit_position(bar["close"])

        result = engine.run(data, trend_strategy)
        metrics = PerformanceMetrics.from_backtest(result)

        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.win_rate, float)
        assert metrics.total_trades >= 0

    def test_summary_output(self):
        """Test summary string generation."""
        metrics = PerformanceMetrics._empty_metrics()
        summary = metrics.summary()

        assert "BACKTEST PERFORMANCE SUMMARY" in summary
        assert "Total Return" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
