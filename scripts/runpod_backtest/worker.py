#!/usr/bin/env python3
"""
RunPod Backtesting Worker

Each worker processes a batch of strategy configurations and reports results.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime, time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import *


@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    direction: int  # 1=long, -1=short
    size: int
    stop_loss: float
    take_profit: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: str = ""


class ScalpingStrategy:
    """Base class for scalping strategies implementing video concepts."""

    def __init__(self, params: dict):
        self.params = params

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals. Override in subclass."""
        raise NotImplementedError


class WilliamsFractalsStrategy(ScalpingStrategy):
    """
    Strategy 5: Williams Fractals Scalping

    Entry: Fractal + MA alignment + pullback
    Best for: 1-minute timeframe (as mentioned in video)
    """

    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        period = self.params.get('period', 2)
        ma_fast = self.params.get('ma_fast', 20)
        ma_medium = self.params.get('ma_medium', 50)
        ma_slow = self.params.get('ma_slow', 100)

        # Calculate MAs
        sma_fast = df['close'].rolling(ma_fast).mean()
        sma_medium = df['close'].rolling(ma_medium).mean()
        sma_slow = df['close'].rolling(ma_slow).mean()

        # MA alignment
        bullish_align = (sma_fast > sma_medium) & (sma_medium > sma_slow)
        bearish_align = (sma_fast < sma_medium) & (sma_medium < sma_slow)

        # Williams Fractals
        bullish_frac = pd.Series(False, index=df.index)
        bearish_frac = pd.Series(False, index=df.index)

        for i in range(period, len(df) - period):
            # Bullish fractal: low is lowest
            if df['low'].iloc[i] == df['low'].iloc[i-period:i+period+1].min():
                bullish_frac.iloc[i] = True
            # Bearish fractal: high is highest
            if df['high'].iloc[i] == df['high'].iloc[i-period:i+period+1].max():
                bearish_frac.iloc[i] = True

        # Pullback condition
        pullback_to_fast = df['close'] < sma_fast
        rally_to_fast = df['close'] > sma_fast

        # Critical rule from video: Don't trade if price crosses 100 MA against trend
        above_slow = df['close'] > sma_slow
        below_slow = df['close'] < sma_slow

        # Signals
        long_signal = bullish_frac & bullish_align & pullback_to_fast & above_slow
        short_signal = bearish_frac & bearish_align & rally_to_fast & below_slow

        return long_signal, short_signal


class MACDStrategy(ScalpingStrategy):
    """
    Strategy 6: MACD + 200 MA (claimed 86% win rate)

    Entry: MACD cross + trend filter
    """

    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        fast = self.params.get('fast', 12)
        slow = self.params.get('slow', 26)
        signal_period = self.params.get('signal', 9)
        ma_period = self.params.get('ma_period', 200)

        # MACD
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # 200 MA filter
        ma_200 = df['close'].rolling(ma_period).mean()

        # MACD cross
        macd_cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        macd_cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

        # Signals: Cross must happen on correct side of zero + trend filter
        long_signal = macd_cross_up & (macd_line < 0) & (df['close'] > ma_200)
        short_signal = macd_cross_down & (macd_line > 0) & (df['close'] < ma_200)

        return long_signal, short_signal


class TripleSuperTrendStrategy(ScalpingStrategy):
    """
    Strategy 8: Triple SuperTrend

    Entry: All 3 SuperTrends agree
    """

    def _supertrend(self, df: pd.DataFrame, atr_period: int, multiplier: float) -> pd.Series:
        hl2 = (df['high'] + df['low']) / 2

        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()

        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr

        trend = pd.Series(1, index=df.index)
        for i in range(1, len(df)):
            if df['close'].iloc[i] > upper.iloc[i-1]:
                trend.iloc[i] = 1
            elif df['close'].iloc[i] < lower.iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1]

        return trend

    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        st1 = self._supertrend(df, self.params.get('atr1', 12), self.params.get('mult1', 3.0))
        st2 = self._supertrend(df, self.params.get('atr2', 10), self.params.get('mult2', 1.0))
        st3 = self._supertrend(df, self.params.get('atr3', 11), self.params.get('mult3', 2.0))

        # All bullish or all bearish
        all_bullish = (st1 == 1) & (st2 == 1) & (st3 == 1)
        all_bearish = (st1 == -1) & (st2 == -1) & (st3 == -1)

        # Signal on state change
        long_signal = all_bullish & ~all_bullish.shift(1).fillna(False)
        short_signal = all_bearish & ~all_bearish.shift(1).fillna(False)

        return long_signal, short_signal


class LiquiditySweepStrategy(ScalpingStrategy):
    """
    Strategy 2: Liquidity Hunting

    Entry: Sweep of highs/lows + reversal
    """

    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        lookback = self.params.get('lookback', 20)
        confirm = self.params.get('confirmation_bars', 2)

        recent_high = df['high'].rolling(lookback).max().shift(1)
        recent_low = df['low'].rolling(lookback).min().shift(1)

        # Sweep: wick beyond level, close inside
        sweep_high = (df['high'] > recent_high) & (df['close'] < recent_high)
        sweep_low = (df['low'] < recent_low) & (df['close'] > recent_low)

        # Confirmation: opposite color candle after sweep
        bullish_confirm = (df['close'] > df['open']).rolling(confirm).sum() == confirm
        bearish_confirm = (df['close'] < df['open']).rolling(confirm).sum() == confirm

        long_signal = sweep_low.shift(1) & bullish_confirm
        short_signal = sweep_high.shift(1) & bearish_confirm

        return long_signal, short_signal


class BacktestEngine:
    """
    Backtest engine with EOD flatten and proper P&L tracking.
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        contract_value: float = CONTRACT_VALUE,
        commission: float = COMMISSION_PER_TRADE,
        slippage_ticks: int = SLIPPAGE_TICKS,
        daily_loss_limit: float = DAILY_LOSS_LIMIT,
        flatten_hour: int = TRADING_END_HOUR,
    ):
        self.initial_capital = initial_capital
        self.contract_value = contract_value
        self.commission = commission
        self.slippage = slippage_ticks * 0.25 * contract_value  # MES tick = 0.25
        self.daily_loss_limit = daily_loss_limit
        self.flatten_hour = flatten_hour

    def run(
        self,
        df: pd.DataFrame,
        long_signals: pd.Series,
        short_signals: pd.Series,
        mode: str = "both",
        exit_strategy: dict = None,
        atr_series: pd.Series = None,
    ) -> Dict:
        """
        Run backtest with signals.

        Args:
            df: OHLCV DataFrame with datetime index
            long_signals: Boolean series for long entries
            short_signals: Boolean series for short entries
            mode: "long_only", "short_only", or "both"
            exit_strategy: Exit configuration
            atr_series: ATR values for dynamic stops

        Returns:
            Dictionary with performance metrics
        """
        if exit_strategy is None:
            exit_strategy = {"type": "fixed_rr", "target_rr": 1.5}

        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_idx = 0
        stop_loss = 0
        take_profit = 0

        trades = []
        equity = [capital]
        daily_pnl = 0
        current_day = None

        # Get timestamps
        if isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index
        elif 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
        else:
            timestamps = pd.date_range(start='2023-01-01', periods=len(df), freq='1min')

        # ATR for stops
        if atr_series is None:
            tr = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            ], axis=1).max(axis=1)
            atr_series = tr.rolling(14).mean()

        for i in range(1, len(df)):
            current_time = timestamps[i]
            current_price = df['close'].iloc[i]
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            atr = atr_series.iloc[i] if not pd.isna(atr_series.iloc[i]) else 1.0

            # Reset daily P&L at day change
            if hasattr(current_time, 'date'):
                day = current_time.date()
                if current_day != day:
                    current_day = day
                    daily_pnl = 0

            # EOD Flatten
            if hasattr(current_time, 'hour') and current_time.hour >= self.flatten_hour:
                if position != 0:
                    exit_price = current_price - self.slippage * np.sign(position)
                    pnl = (exit_price - entry_price) * position * self.contract_value - self.commission
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': 'LONG' if position > 0 else 'SHORT',
                        'pnl': pnl,
                        'exit_reason': 'EOD_FLATTEN'
                    })
                    capital += pnl
                    daily_pnl += pnl
                    position = 0
                equity.append(capital)
                continue

            # Check daily loss limit
            if daily_pnl <= -self.daily_loss_limit:
                if position != 0:
                    exit_price = current_price - self.slippage * np.sign(position)
                    pnl = (exit_price - entry_price) * position * self.contract_value - self.commission
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': 'LONG' if position > 0 else 'SHORT',
                        'pnl': pnl,
                        'exit_reason': 'DAILY_LOSS_LIMIT'
                    })
                    capital += pnl
                    position = 0
                equity.append(capital)
                continue

            # Check exits for open position
            if position != 0:
                exit_price = None
                exit_reason = ""

                # Stop loss
                if position > 0 and low <= stop_loss:
                    exit_price = stop_loss - self.slippage
                    exit_reason = "STOP_LOSS"
                elif position < 0 and high >= stop_loss:
                    exit_price = stop_loss + self.slippage
                    exit_reason = "STOP_LOSS"

                # Take profit
                if exit_price is None:
                    if position > 0 and high >= take_profit:
                        exit_price = take_profit - self.slippage
                        exit_reason = "TAKE_PROFIT"
                    elif position < 0 and low <= take_profit:
                        exit_price = take_profit + self.slippage
                        exit_reason = "TAKE_PROFIT"

                # Time-based exit
                if exit_price is None and exit_strategy.get('type') == 'time_based':
                    max_bars = exit_strategy.get('max_bars', 30)
                    if i - entry_idx >= max_bars:
                        exit_price = current_price - self.slippage * np.sign(position)
                        exit_reason = "TIME_EXIT"

                # Execute exit
                if exit_price is not None:
                    pnl = (exit_price - entry_price) * position * self.contract_value - self.commission
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': 'LONG' if position > 0 else 'SHORT',
                        'pnl': pnl,
                        'exit_reason': exit_reason
                    })
                    capital += pnl
                    daily_pnl += pnl
                    position = 0

            # Check for new entries (only if flat)
            if position == 0:
                long_sig = long_signals.iloc[i] if i < len(long_signals) else False
                short_sig = short_signals.iloc[i] if i < len(short_signals) else False

                # Filter by mode
                if mode == "long_only":
                    short_sig = False
                elif mode == "short_only":
                    long_sig = False

                if long_sig:
                    position = 1
                    entry_price = current_price + self.slippage
                    entry_idx = i

                    # Stop and target based on ATR
                    atr_mult = exit_strategy.get('atr_mult', 2.0) if 'atr_mult' in exit_strategy else 2.0
                    stop_loss = entry_price - atr_mult * atr
                    target_rr = exit_strategy.get('target_rr', 1.5)
                    take_profit = entry_price + target_rr * (entry_price - stop_loss)

                elif short_sig:
                    position = -1
                    entry_price = current_price - self.slippage
                    entry_idx = i

                    atr_mult = exit_strategy.get('atr_mult', 2.0) if 'atr_mult' in exit_strategy else 2.0
                    stop_loss = entry_price + atr_mult * atr
                    target_rr = exit_strategy.get('target_rr', 1.5)
                    take_profit = entry_price - target_rr * (stop_loss - entry_price)

            equity.append(capital)

        # Close any remaining position
        if position != 0:
            exit_price = df['close'].iloc[-1] - self.slippage * np.sign(position)
            pnl = (exit_price - entry_price) * position * self.contract_value - self.commission
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': len(df) - 1,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': 'LONG' if position > 0 else 'SHORT',
                'pnl': pnl,
                'exit_reason': 'END_OF_DATA'
            })
            capital += pnl

        # Calculate metrics
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return self._calculate_metrics(trades_df, equity, capital)

    def _calculate_metrics(self, trades_df: pd.DataFrame, equity: list, final_capital: float) -> Dict:
        """Calculate comprehensive performance metrics."""
        if len(trades_df) == 0:
            return {
                'total_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'long_pnl': 0,
                'short_pnl': 0,
                'final_capital': final_capital,
            }

        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        longs = trades_df[trades_df['direction'] == 'LONG']
        shorts = trades_df[trades_df['direction'] == 'SHORT']

        # Drawdown
        equity_series = pd.Series(equity)
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) * 100

        # Sharpe (assuming daily returns)
        returns = pd.Series(equity).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0

        return {
            'total_return': (final_capital - self.initial_capital) / self.initial_capital * 100,
            'total_trades': len(trades_df),
            'win_rate': len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
            'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf'),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'avg_pnl': trades_df['pnl'].mean(),
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
            'long_pnl': longs['pnl'].sum() if len(longs) > 0 else 0,
            'short_pnl': shorts['pnl'].sum() if len(shorts) > 0 else 0,
            'long_trades': len(longs),
            'short_trades': len(shorts),
            'final_capital': final_capital,
            'trades': trades_df.to_dict('records') if len(trades_df) < 100 else [],
        }


def run_single_backtest(config: dict) -> dict:
    """
    Run a single backtest configuration.

    Args:
        config: Dictionary with:
            - df: DataFrame
            - strategy: Strategy class
            - params: Strategy parameters
            - timeframe: Timeframe in minutes
            - mode: Trading mode
            - exit_strategy: Exit configuration

    Returns:
        Results dictionary
    """
    df = config['df']
    strategy_class = config['strategy']
    params = config['params']
    mode = config['mode']
    exit_strategy = config['exit_strategy']

    # Initialize strategy
    strategy = strategy_class(params)

    # Generate signals
    long_signals, short_signals = strategy.generate_signals(df)

    # Run backtest
    engine = BacktestEngine()
    results = engine.run(df, long_signals, short_signals, mode, exit_strategy)

    # Add metadata
    results['strategy'] = strategy_class.__name__
    results['params'] = params
    results['mode'] = mode
    results['exit_strategy'] = exit_strategy
    results['timeframe'] = config.get('timeframe', 1)

    return results


def worker_main(worker_id: int, task_queue: list, results_path: str):
    """
    Main worker function for RunPod.

    Args:
        worker_id: Unique worker identifier
        task_queue: List of backtest configurations
        results_path: Path to save results
    """
    print(f"[Worker {worker_id}] Starting with {len(task_queue)} tasks")

    results = []
    for i, config in enumerate(task_queue):
        try:
            result = run_single_backtest(config)
            results.append(result)

            if (i + 1) % 10 == 0:
                print(f"[Worker {worker_id}] Completed {i+1}/{len(task_queue)}")

        except Exception as e:
            print(f"[Worker {worker_id}] Error on task {i}: {e}")
            results.append({'error': str(e), 'config': str(config.get('params', {}))})

    # Save results
    output_file = Path(results_path) / f"results_worker_{worker_id}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"[Worker {worker_id}] Completed. Results saved to {output_file}")
    return results


if __name__ == "__main__":
    # Test locally
    print("Testing worker locally...")

    # Create sample data
    np.random.seed(42)
    n = 10000
    prices = 5000 + np.cumsum(np.random.randn(n) * 0.5)

    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 1),
        'low': prices - np.abs(np.random.randn(n) * 1),
        'close': prices + np.random.randn(n) * 0.5,
        'volume': np.random.randint(100, 1000, n)
    })

    # Test Williams Fractals
    config = {
        'df': df,
        'strategy': WilliamsFractalsStrategy,
        'params': {'period': 2, 'ma_fast': 20, 'ma_medium': 50, 'ma_slow': 100},
        'mode': 'both',
        'exit_strategy': {'type': 'fixed_rr', 'target_rr': 1.5},
        'timeframe': 1,
    }

    result = run_single_backtest(config)
    print(f"\nTest Results:")
    print(f"  Return: {result['total_return']:.1f}%")
    print(f"  Trades: {result['total_trades']}")
    print(f"  Win Rate: {result['win_rate']:.1f}%")
    print(f"  Long P&L: ${result['long_pnl']:.2f}")
    print(f"  Short P&L: ${result['short_pnl']:.2f}")
