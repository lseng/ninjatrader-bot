#!/usr/bin/env python3
"""
Paper Trading Simulator

Simulates live trading using historical data.
Useful for testing before going live.

Usage:
    python scripts/paper_trade.py --capital 1000 --speed 10
"""

import sys
from pathlib import Path
import argparse
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

# Import directly from file to avoid package issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "williams_fractals_strategy",
    Path(src_path) / "strategies" / "williams_fractals_strategy.py"
)
wfs_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wfs_module)
WilliamsFractalsStrategy = wfs_module.WilliamsFractalsStrategy
TradeSignal = wfs_module.TradeSignal

spec2 = importlib.util.spec_from_file_location(
    "config",
    Path(src_path) / "config.py"
)
config_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(config_module)
get_position_size = config_module.get_position_size


class PaperTrader:
    """Paper trading simulator."""

    def __init__(
        self,
        initial_capital: float = 1000,
        risk_per_trade: float = 0.02,
        max_contracts: int = 5,
        speed: float = 1.0,  # Bars per second
        verbose: bool = True
    ):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_contracts = max_contracts
        self.speed = speed
        self.verbose = verbose

        # Strategy
        self.strategy = WilliamsFractalsStrategy()

        # State
        self.position = 0  # -1, 0, 1
        self.position_size = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0

        # Tracking
        self.trades = []
        self.equity_curve = [initial_capital]
        self.max_equity = initial_capital
        self.max_drawdown = 0

    def log(self, msg: str):
        """Print if verbose."""
        if self.verbose:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] {msg}")

    def run(self, data: pd.DataFrame):
        """Run paper trading simulation."""
        self.log("=" * 60)
        self.log("PAPER TRADING SIMULATOR")
        self.log(f"Capital: ${self.capital:,.2f}")
        self.log(f"Risk per trade: {self.risk_per_trade * 100:.1f}%")
        self.log(f"Speed: {self.speed} bars/sec")
        self.log("=" * 60)
        self.log("")

        # Need warmup
        warmup = 200

        for i in range(warmup, len(data)):
            # Simulate delay
            if self.speed > 0:
                time.sleep(1 / self.speed)

            # Get history window
            window = data.iloc[max(0, i-300):i+1].copy()
            current_bar = data.iloc[i]

            # Check for exit first
            if self.position != 0:
                self._check_exit(current_bar)

            # Check for entry
            if self.position == 0:
                self._check_entry(window, current_bar)

            # Update equity curve
            self.equity_curve.append(self.capital)

            # Update drawdown
            if self.capital > self.max_equity:
                self.max_equity = self.capital
            dd = (self.max_equity - self.capital) / self.max_equity * 100
            if dd > self.max_drawdown:
                self.max_drawdown = dd

            # Check for blowup
            if self.capital < self.initial_capital * 0.2:
                self.log("ACCOUNT BLEW UP!")
                break

        # Print summary
        self._print_summary()

        return self.trades

    def _check_entry(self, history: pd.DataFrame, bar: pd.Series):
        """Check for entry signal."""
        signal = self.strategy.generate_signal(history)

        if signal.direction == 0:
            return

        # Calculate position size
        size = get_position_size(
            capital=self.capital,
            entry=signal.entry_price,
            stop=signal.stop_loss,
            risk_pct=self.risk_per_trade,
            contract_value=5.0,
            max_contracts=self.max_contracts
        )

        # Enter position
        self.position = signal.direction
        self.position_size = size
        self.entry_price = signal.entry_price
        self.stop_loss = signal.stop_loss
        self.take_profit = signal.take_profit

        direction = "LONG" if signal.direction == 1 else "SHORT"
        self.log(f"ENTRY: {direction} {size} @ {signal.entry_price:.2f}")
        self.log(f"  Stop: {signal.stop_loss:.2f} | Target: {signal.take_profit:.2f}")
        self.log(f"  Reason: {signal.reason}")

    def _check_exit(self, bar: pd.Series):
        """Check for exit conditions."""
        exit_price = None
        reason = None

        if self.position == 1:  # Long
            if bar['low'] <= self.stop_loss:
                exit_price = self.stop_loss
                reason = "Stop Loss"
            elif bar['high'] >= self.take_profit:
                exit_price = self.take_profit
                reason = "Take Profit"
        else:  # Short
            if bar['high'] >= self.stop_loss:
                exit_price = self.stop_loss
                reason = "Stop Loss"
            elif bar['low'] <= self.take_profit:
                exit_price = self.take_profit
                reason = "Take Profit"

        if exit_price:
            self._exit_position(exit_price, reason)

    def _exit_position(self, exit_price: float, reason: str):
        """Exit current position."""
        # Calculate P&L
        gross_pnl = (exit_price - self.entry_price) * self.position * 5.0 * self.position_size
        commission = 2.50 * self.position_size
        net_pnl = gross_pnl - commission

        # Update capital
        self.capital += net_pnl

        # Record trade
        trade = {
            'entry': self.entry_price,
            'exit': exit_price,
            'direction': 'LONG' if self.position == 1 else 'SHORT',
            'size': self.position_size,
            'gross_pnl': gross_pnl,
            'commission': commission,
            'net_pnl': net_pnl,
            'reason': reason,
            'balance': self.capital
        }
        self.trades.append(trade)

        result = "WIN" if net_pnl > 0 else "LOSS"
        self.log(f"EXIT: {reason} | {result} | P&L: ${net_pnl:,.2f} | Balance: ${self.capital:,.2f}")

        # Reset state
        self.position = 0
        self.position_size = 0
        self.entry_price = 0

    def _print_summary(self):
        """Print trading summary."""
        self.log("")
        self.log("=" * 60)
        self.log("PAPER TRADING SUMMARY")
        self.log("=" * 60)

        if not self.trades:
            self.log("No trades executed")
            return

        trades_df = pd.DataFrame(self.trades)
        winners = trades_df[trades_df['net_pnl'] > 0]
        losers = trades_df[trades_df['net_pnl'] <= 0]

        total_pnl = trades_df['net_pnl'].sum()
        win_pnl = winners['net_pnl'].sum() if len(winners) > 0 else 0
        loss_pnl = losers['net_pnl'].sum() if len(losers) > 0 else 0

        self.log(f"Total Trades: {len(trades_df)}")
        self.log(f"Winners: {len(winners)} ({len(winners)/len(trades_df)*100:.1f}%)")
        self.log(f"Losers: {len(losers)} ({len(losers)/len(trades_df)*100:.1f}%)")
        self.log(f"Profit Factor: {abs(win_pnl/loss_pnl):.2f}" if loss_pnl != 0 else "N/A")
        self.log(f"Max Drawdown: {self.max_drawdown:.1f}%")
        self.log("")
        self.log(f"Starting Capital: ${self.initial_capital:,.2f}")
        self.log(f"Ending Capital: ${self.capital:,.2f}")
        self.log(f"Net P&L: ${total_pnl:,.2f}")
        self.log(f"Return: {(self.capital - self.initial_capital) / self.initial_capital * 100:.1f}%")

        if len(winners) > 0:
            self.log(f"Avg Win: ${winners['net_pnl'].mean():,.2f}")
        if len(losers) > 0:
            self.log(f"Avg Loss: ${losers['net_pnl'].mean():,.2f}")


def load_sample_data(path: str = None) -> pd.DataFrame:
    """Load sample data for paper trading."""
    # Try different data sources
    paths = [
        Path(path) if path else None,
        Path("/Users/leoneng/Downloads/ninjatrader-bot/data/historical/MES_5min.parquet"),
        Path("/Users/leoneng/Downloads/topstep-trading-bot/data/historical/MES/MES_1s_2years.parquet"),
    ]

    for p in paths:
        if p and p.exists():
            print(f"Loading data from {p}...")
            df = pd.read_parquet(p)

            # Resample to 5-min if needed
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')

            if len(df) > 500000:  # 1-second data
                print("Resampling to 5-minute bars...")
                df = df.resample('5min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()

            df = df.reset_index()
            print(f"Loaded {len(df):,} bars")
            return df

    raise FileNotFoundError("No data file found. Please provide a path to historical data.")


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Simulator")
    parser.add_argument("--capital", type=float, default=1000, help="Starting capital")
    parser.add_argument("--risk", type=float, default=0.02, help="Risk per trade, e.g. 0.02 for 2 percent")
    parser.add_argument("--max-contracts", type=int, default=5, help="Max contracts")
    parser.add_argument("--speed", type=float, default=100, help="Bars per second (0 = instant)")
    parser.add_argument("--data", type=str, default=None, help="Path to data file")
    parser.add_argument("--bars", type=int, default=5000, help="Number of bars to simulate")

    args = parser.parse_args()

    # Load data
    df = load_sample_data(args.data)

    # Take subset
    if len(df) > args.bars + 300:
        # Start from a random point for variety
        start = np.random.randint(0, len(df) - args.bars - 300)
        df = df.iloc[start:start + args.bars + 300].reset_index(drop=True)

    print(f"Simulating {len(df):,} bars...")
    print()

    # Run paper trading
    trader = PaperTrader(
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        max_contracts=args.max_contracts,
        speed=args.speed
    )

    trades = trader.run(df)

    # Save trades to CSV
    if trades:
        output_path = Path(__file__).parent.parent / "output" / "paper_trades.csv"
        pd.DataFrame(trades).to_csv(output_path, index=False)
        print(f"\nTrades saved to {output_path}")


if __name__ == "__main__":
    main()
