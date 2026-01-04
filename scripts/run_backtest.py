#!/usr/bin/env python3
"""
Run a backtest with sample strategies.

Example usage:
    python scripts/run_backtest.py --data data/historical/MES_1min.csv --strategy sma
    python scripts/run_backtest.py --data data/historical/MES_1min.csv --strategy ml --model models/strategy.joblib
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.data.loader import NinjaTraderDataLoader
from src.data.processor import DataProcessor
from src.backtesting.engine import BacktestEngine, BacktestConfig, Side
from src.backtesting.metrics import PerformanceMetrics
from src.strategies.sample_strategies import SMAStrategy, RSIStrategy, MACDStrategy


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV, Parquet, or NT format."""
    filepath = Path(filepath)

    if filepath.suffix == ".csv":
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
    elif filepath.suffix == ".parquet":
        loader = NinjaTraderDataLoader(filepath.parent)
        df = loader.load_parquet(filepath.name)
    else:
        # Assume NinjaTrader format
        loader = NinjaTraderDataLoader(filepath.parent)
        df = loader.load_auto_detect(filepath.name)

    return df


def create_strategy(name: str, **kwargs):
    """Create a strategy by name."""
    strategies = {
        "sma": SMAStrategy,
        "rsi": RSIStrategy,
        "macd": MACDStrategy
    }

    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")

    return strategies[name](**kwargs)


def run_backtest(
    data: pd.DataFrame,
    strategy,
    config: BacktestConfig
) -> tuple:
    """Run backtest with a strategy."""
    engine = BacktestEngine(config)

    # Strategy wrapper for backtest engine
    def strategy_fn(eng: BacktestEngine, bar: pd.Series, history: pd.DataFrame):
        signal = strategy.generate_signal(bar, history)

        if signal is None:
            return

        # Handle entry signals
        if signal.direction == "long" and eng.position is None:
            eng.enter_long(
                bar["close"],
                size=signal.size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            strategy.on_trade_entry(bar["close"], "long")

        elif signal.direction == "short" and eng.position is None:
            eng.enter_short(
                bar["close"],
                size=signal.size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            strategy.on_trade_entry(bar["close"], "short")

        # Handle exit signals
        elif signal.direction == "flat" and eng.position is not None:
            eng.exit_position(bar["close"], signal.reason)
            strategy.on_trade_exit()

        # Handle direction changes
        elif signal.direction == "long" and eng.position and eng.position.side == Side.SHORT:
            eng.exit_position(bar["close"], "direction_change")
            strategy.on_trade_exit()
            eng.enter_long(
                bar["close"],
                size=signal.size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            strategy.on_trade_entry(bar["close"], "long")

        elif signal.direction == "short" and eng.position and eng.position.side == Side.LONG:
            eng.exit_position(bar["close"], "direction_change")
            strategy.on_trade_exit()
            eng.enter_short(
                bar["close"],
                size=signal.size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            strategy.on_trade_entry(bar["close"], "short")

    result = engine.run(data, strategy_fn)
    metrics = PerformanceMetrics.from_backtest(result)

    return result, metrics


def main():
    parser = argparse.ArgumentParser(description="Run trading strategy backtest")
    parser.add_argument("--data", required=True, help="Path to data file")
    parser.add_argument("--strategy", default="sma", choices=["sma", "rsi", "macd"])
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--commission", type=float, default=2.50, help="Commission per contract")
    parser.add_argument("--slippage", type=float, default=1, help="Slippage in ticks")

    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    data = load_data(args.data)
    print(f"Loaded {len(data):,} bars")

    print(f"\nCreating {args.strategy.upper()} strategy...")
    strategy = create_strategy(args.strategy)

    config = BacktestConfig(
        initial_capital=args.capital,
        commission_per_contract=args.commission,
        slippage_ticks=args.slippage
    )

    print(f"\nRunning backtest...")
    result, metrics = run_backtest(data, strategy, config)

    print(metrics.summary())

    # Save equity curve
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    result.equity_curve.to_csv(output_dir / "equity_curve.csv")
    print(f"\nEquity curve saved to output/equity_curve.csv")

    # Save trade log
    trades_df = pd.DataFrame([{
        "entry_time": t.entry_time,
        "exit_time": t.exit_time,
        "side": t.side.value,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "size": t.size,
        "pnl": t.pnl,
        "exit_reason": t.exit_reason
    } for t in result.trades])

    trades_df.to_csv(output_dir / "trades.csv", index=False)
    print(f"Trade log saved to output/trades.csv")


if __name__ == "__main__":
    main()
