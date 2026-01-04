#!/usr/bin/env python3
"""
Confluence Strategy - Combines ALL Video Strategies

Instead of using a single strategy, this combines signals from:
1. Williams Fractals (Strategy 5) - PRIMARY signal generator
2. MACD + 200 MA (Strategy 6) - Trend confirmation
3. Triple SuperTrend (Strategy 8) - Trend confirmation
4. Supply & Demand Zones (Strategy 4) - Entry optimization
5. Liquidity Sweep (Strategy 2) - Smart money confirmation
6. Fibonacci Levels (Strategy 10) - Key level filter
7. Heikin Ashi (Strategy 7) - Trend clarity
8. Candlestick Context (Strategy 3) - Entry quality

Trade only when MULTIPLE strategies agree (confluence).
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from datetime import datetime


@dataclass
class ConfluenceSignal:
    direction: int  # 1=long, -1=short, 0=flat
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    score: int  # How many strategies agree
    reasons: list


class ConfluenceStrategy:
    """
    Production strategy combining ALL video strategies.

    Confluence Levels:
    - Score 1-2: No trade (weak signal)
    - Score 3-4: Normal position size
    - Score 5+: Full position size (high confidence)
    """

    def __init__(
        self,
        min_confluence: int = 3,  # Minimum strategies that must agree
        # Fractal params (Strategy 5)
        fractal_period: int = 3,
        ma_fast: int = 20,
        ma_medium: int = 50,
        ma_slow: int = 100,
        # MACD params (Strategy 6)
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        ma_200: int = 200,
        # SuperTrend params (Strategy 8)
        st_atr1: int = 12,
        st_mult1: float = 3.0,
        st_atr2: int = 10,
        st_mult2: float = 1.0,
        st_atr3: int = 11,
        st_mult3: float = 2.0,
        # Risk params
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
        target_rr: float = 1.5,
        contract_value: float = 5.0,
        flatten_hour: int = 15,
    ):
        self.min_confluence = min_confluence

        # Fractal params
        self.fractal_period = fractal_period
        self.ma_fast = ma_fast
        self.ma_medium = ma_medium
        self.ma_slow = ma_slow

        # MACD params
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ma_200 = ma_200

        # SuperTrend params
        self.st_atr1 = st_atr1
        self.st_mult1 = st_mult1
        self.st_atr2 = st_atr2
        self.st_mult2 = st_mult2
        self.st_atr3 = st_atr3
        self.st_mult3 = st_mult3

        # Risk params
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.target_rr = target_rr
        self.contract_value = contract_value
        self.flatten_hour = flatten_hour

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for ALL strategies."""
        df = df.copy()
        close = df['close']
        high = df['high']
        low = df['low']
        open_ = df['open']

        # ============================================================
        # 1. WILLIAMS FRACTALS (Strategy 5)
        # ============================================================
        df['sma_fast'] = close.rolling(self.ma_fast).mean()
        df['sma_medium'] = close.rolling(self.ma_medium).mean()
        df['sma_slow'] = close.rolling(self.ma_slow).mean()

        # MA Alignment
        df['bullish_ma_align'] = (df['sma_fast'] > df['sma_medium']) & (df['sma_medium'] > df['sma_slow'])
        df['bearish_ma_align'] = (df['sma_fast'] < df['sma_medium']) & (df['sma_medium'] < df['sma_slow'])

        # Fractals
        window = 2 * self.fractal_period + 1
        df['rolling_low_min'] = low.rolling(window, center=True).min()
        df['rolling_high_max'] = high.rolling(window, center=True).max()
        df['bullish_fractal'] = low == df['rolling_low_min']
        df['bearish_fractal'] = high == df['rolling_high_max']

        # Pullback conditions
        df['pullback_long'] = close < df['sma_fast']
        df['pullback_short'] = close > df['sma_fast']
        df['above_slow'] = close > df['sma_slow']
        df['below_slow'] = close < df['sma_slow']

        # ============================================================
        # 2. MACD + 200 MA (Strategy 6)
        # ============================================================
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['ma_200'] = close.rolling(self.ma_200).mean()

        # MACD crossovers
        df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))

        # MACD signals (cross below zero + price above 200 MA for long)
        df['macd_long'] = df['macd_cross_up'] & (df['macd'] < 0) & (close > df['ma_200'])
        df['macd_short'] = df['macd_cross_down'] & (df['macd'] > 0) & (close < df['ma_200'])

        # Also track general MACD trend (not just crossovers)
        df['macd_bullish'] = (df['macd'] > df['macd_signal']) & (close > df['ma_200'])
        df['macd_bearish'] = (df['macd'] < df['macd_signal']) & (close < df['ma_200'])

        # ============================================================
        # 3. TRIPLE SUPERTREND (Strategy 8)
        # ============================================================
        def calc_supertrend(high, low, close, atr_period, mult):
            hl2 = (high + low) / 2
            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)
            atr = tr.rolling(atr_period).mean()

            upper = hl2 + mult * atr
            lower = hl2 - mult * atr

            trend = pd.Series(1, index=close.index)
            for i in range(1, len(close)):
                if close.iloc[i] > upper.iloc[i-1]:
                    trend.iloc[i] = 1
                elif close.iloc[i] < lower.iloc[i-1]:
                    trend.iloc[i] = -1
                else:
                    trend.iloc[i] = trend.iloc[i-1]
            return trend

        df['st1'] = calc_supertrend(high, low, close, self.st_atr1, self.st_mult1)
        df['st2'] = calc_supertrend(high, low, close, self.st_atr2, self.st_mult2)
        df['st3'] = calc_supertrend(high, low, close, self.st_atr3, self.st_mult3)

        df['supertrend_bullish'] = (df['st1'] == 1) & (df['st2'] == 1) & (df['st3'] == 1)
        df['supertrend_bearish'] = (df['st1'] == -1) & (df['st2'] == -1) & (df['st3'] == -1)

        # ============================================================
        # 4. SUPPLY & DEMAND ZONES (Strategy 4)
        # ============================================================
        # Detect impulse moves (strong directional candles)
        returns = close.pct_change()
        df['impulse_up'] = returns > returns.rolling(20).std() * 2
        df['impulse_down'] = returns < -returns.rolling(20).std() * 2

        # Demand zone: price near recent consolidation before up move
        demand_low = low.rolling(20).min()
        demand_high = low.rolling(20).min() + (high.rolling(20).max() - low.rolling(20).min()) * 0.3
        df['at_demand_zone'] = (close >= demand_low) & (close <= demand_high)

        # Supply zone: price near recent consolidation before down move
        supply_high = high.rolling(20).max()
        supply_low = high.rolling(20).max() - (high.rolling(20).max() - low.rolling(20).min()) * 0.3
        df['at_supply_zone'] = (close >= supply_low) & (close <= supply_high)

        # ============================================================
        # 5. LIQUIDITY SWEEP (Strategy 2)
        # ============================================================
        lookback = 20
        recent_high = high.rolling(lookback).max().shift(1)
        recent_low = low.rolling(lookback).min().shift(1)

        # Sweep = wick beyond level but close inside
        df['sweep_high'] = (high > recent_high) & (close < recent_high)
        df['sweep_low'] = (low < recent_low) & (close > recent_low)

        # ============================================================
        # 6. FIBONACCI LEVELS (Strategy 10)
        # ============================================================
        swing_high = high.rolling(50).max()
        swing_low = low.rolling(50).min()
        fib_range = swing_high - swing_low

        # Key Fib levels (for pullback entries)
        df['fib_382'] = swing_high - 0.382 * fib_range
        df['fib_500'] = swing_high - 0.500 * fib_range
        df['fib_618'] = swing_high - 0.618 * fib_range

        # At Fib level = within 0.5% of a key level
        tolerance = close * 0.005
        df['at_fib_level'] = (
            (abs(close - df['fib_382']) < tolerance) |
            (abs(close - df['fib_500']) < tolerance) |
            (abs(close - df['fib_618']) < tolerance)
        )

        # ============================================================
        # 7. HEIKIN ASHI (Strategy 7)
        # ============================================================
        ha_close = (open_ + high + low + close) / 4
        ha_open = pd.Series(index=df.index, dtype=float)
        ha_open.iloc[0] = (open_.iloc[0] + close.iloc[0]) / 2
        for i in range(1, len(df)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2

        df['ha_bullish'] = ha_close > ha_open
        df['ha_bearish'] = ha_close < ha_open

        # Strong trend = no opposite wick
        ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
        ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)
        df['ha_strong_bull'] = df['ha_bullish'] & (ha_low == pd.concat([ha_open, ha_close], axis=1).min(axis=1))
        df['ha_strong_bear'] = df['ha_bearish'] & (ha_high == pd.concat([ha_open, ha_close], axis=1).max(axis=1))

        # ============================================================
        # 8. CANDLESTICK CONTEXT (Strategy 3)
        # ============================================================
        body = abs(close - open_)
        upper_wick = high - pd.concat([close, open_], axis=1).max(axis=1)
        lower_wick = pd.concat([close, open_], axis=1).min(axis=1) - low
        total_range = high - low

        # Strength candle (large body, small wicks)
        df['strength_bull'] = (close > open_) & (body / total_range > 0.7)
        df['strength_bear'] = (close < open_) & (body / total_range > 0.7)

        # Control shift (reversal) candles
        df['reversal_bull'] = (lower_wick / total_range > 0.6)  # Long lower wick
        df['reversal_bear'] = (upper_wick / total_range > 0.6)  # Long upper wick

        # ============================================================
        # ATR for stops
        # ============================================================
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.atr_period).mean()

        return df

    def calculate_confluence_score(self, row: pd.Series) -> Tuple[int, int, list]:
        """
        Calculate confluence score for long and short.

        Returns:
            (long_score, short_score, reasons)
        """
        long_score = 0
        short_score = 0
        long_reasons = []
        short_reasons = []

        # 1. Williams Fractals (weight: 2 - primary signal)
        if row['bullish_fractal'] and row['bullish_ma_align'] and row['pullback_long'] and row['above_slow']:
            long_score += 2
            long_reasons.append("Fractals")
        if row['bearish_fractal'] and row['bearish_ma_align'] and row['pullback_short'] and row['below_slow']:
            short_score += 2
            short_reasons.append("Fractals")

        # 2. MACD trend alignment (weight: 1)
        if row['macd_bullish']:
            long_score += 1
            long_reasons.append("MACD")
        if row['macd_bearish']:
            short_score += 1
            short_reasons.append("MACD")

        # 3. Triple SuperTrend (weight: 1)
        if row['supertrend_bullish']:
            long_score += 1
            long_reasons.append("SuperTrend")
        if row['supertrend_bearish']:
            short_score += 1
            short_reasons.append("SuperTrend")

        # 4. Supply & Demand zones (weight: 1)
        if row['at_demand_zone'] and row['bullish_ma_align']:
            long_score += 1
            long_reasons.append("Demand Zone")
        if row['at_supply_zone'] and row['bearish_ma_align']:
            short_score += 1
            short_reasons.append("Supply Zone")

        # 5. Liquidity sweep (weight: 1)
        if row['sweep_low']:
            long_score += 1
            long_reasons.append("Liquidity Sweep")
        if row['sweep_high']:
            short_score += 1
            short_reasons.append("Liquidity Sweep")

        # 6. Fibonacci level (weight: 1)
        if row['at_fib_level']:
            # Fib adds to whichever direction has more confluence
            if long_score > short_score:
                long_score += 1
                long_reasons.append("Fib Level")
            elif short_score > long_score:
                short_score += 1
                short_reasons.append("Fib Level")

        # 7. Heikin Ashi trend (weight: 1)
        if row['ha_bullish']:
            long_score += 1
            long_reasons.append("Heikin Ashi")
        if row['ha_bearish']:
            short_score += 1
            short_reasons.append("Heikin Ashi")

        # 8. Candlestick context (weight: 1)
        if row['strength_bull'] or row['reversal_bull']:
            long_score += 1
            long_reasons.append("Candle Pattern")
        if row['strength_bear'] or row['reversal_bear']:
            short_score += 1
            short_reasons.append("Candle Pattern")

        return long_score, short_score, long_reasons, short_reasons

    def generate_signal(self, df: pd.DataFrame) -> ConfluenceSignal:
        """Generate trading signal based on confluence of all strategies."""
        if len(df) < 250:  # Need enough data for all indicators
            return ConfluenceSignal(0, 0, 0, 0, 0, 0, ["Insufficient data"])

        df = self.calculate_all_indicators(df)
        latest = df.iloc[-1]

        long_score, short_score, long_reasons, short_reasons = self.calculate_confluence_score(latest)

        atr = latest['atr']
        close = latest['close']

        # Determine direction based on confluence
        if long_score >= self.min_confluence and long_score > short_score:
            stop = close - self.atr_multiplier * atr
            target = close + self.target_rr * (close - stop)
            confidence = min(1.0, long_score / 8)  # Max 8 strategies

            return ConfluenceSignal(
                direction=1,
                entry_price=close,
                stop_loss=stop,
                take_profit=target,
                confidence=confidence,
                score=long_score,
                reasons=long_reasons
            )

        elif short_score >= self.min_confluence and short_score > long_score:
            stop = close + self.atr_multiplier * atr
            target = close - self.target_rr * (stop - close)
            confidence = min(1.0, short_score / 8)

            return ConfluenceSignal(
                direction=-1,
                entry_price=close,
                stop_loss=stop,
                take_profit=target,
                confidence=confidence,
                score=short_score,
                reasons=short_reasons
            )

        return ConfluenceSignal(0, close, 0, 0, 0, max(long_score, short_score),
                                [f"Insufficient confluence (L:{long_score} S:{short_score})"])

    def should_flatten(self, current_time: datetime) -> bool:
        """Check if we should flatten for EOD."""
        if hasattr(current_time, 'hour'):
            return current_time.hour >= self.flatten_hour
        return False


def backtest_confluence(df: pd.DataFrame, min_confluence: int = 3) -> dict:
    """Backtest the confluence strategy."""
    strategy = ConfluenceStrategy(min_confluence=min_confluence)
    df_ind = strategy.calculate_all_indicators(df)

    trades = []
    capital = 1000
    commission = 2.50

    for i in range(250, len(df) - 50):
        signal = strategy.generate_signal(df.iloc[:i+1])

        if signal.direction != 0 and signal.score >= min_confluence:
            entry = signal.entry_price
            stop = signal.stop_loss
            target = signal.take_profit

            future = df.iloc[i+1:i+51]

            if signal.direction == 1:  # Long
                stop_hit = (future['low'] <= stop).any()
                target_hit = (future['high'] >= target).any()

                if stop_hit and (not target_hit or future[future['low'] <= stop].index[0] < future[future['high'] >= target].index[0]):
                    pnl = (stop - entry) * 5 - commission
                elif target_hit:
                    pnl = (target - entry) * 5 - commission
                else:
                    pnl = (future['close'].iloc[-1] - entry) * 5 - commission
            else:  # Short
                stop_hit = (future['high'] >= stop).any()
                target_hit = (future['low'] <= target).any()

                if stop_hit and (not target_hit or future[future['high'] >= stop].index[0] < future[future['low'] <= target].index[0]):
                    pnl = (entry - stop) * 5 - commission
                elif target_hit:
                    pnl = (entry - target) * 5 - commission
                else:
                    pnl = (entry - future['close'].iloc[-1]) * 5 - commission

            trades.append({
                'direction': 'LONG' if signal.direction == 1 else 'SHORT',
                'pnl': pnl,
                'score': signal.score,
                'reasons': signal.reasons
            })
            capital += pnl

    if not trades:
        return {'total_return': 0, 'total_trades': 0}

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    longs = trades_df[trades_df['direction'] == 'LONG']
    shorts = trades_df[trades_df['direction'] == 'SHORT']

    return {
        'total_return': (capital - 1000) / 1000 * 100,
        'total_trades': len(trades_df),
        'win_rate': len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf'),
        'long_pnl': longs['pnl'].sum() if len(longs) > 0 else 0,
        'short_pnl': shorts['pnl'].sum() if len(shorts) > 0 else 0,
        'long_trades': len(longs),
        'short_trades': len(shorts),
        'final_capital': capital,
        'avg_score': trades_df['score'].mean() if len(trades_df) > 0 else 0,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("CONFLUENCE STRATEGY - Combining ALL Video Strategies")
    print("=" * 70)

    from pathlib import Path

    # Try to load data
    data_paths = [
        Path("/Users/leoneng/Downloads/topstep-trading-bot/data/historical/MES/MES_1s_2years.parquet"),
        Path("/Users/leoneng/Downloads/ninjatrader-bot/data/MES_1s_2years.parquet"),
    ]

    df = None
    for path in data_paths:
        if path.exists():
            print(f"Loading data from {path}...")
            df = pd.read_parquet(path)
            df = df.set_index('timestamp')
            break

    if df is None:
        print("No data found. Creating sample data for testing...")
        import numpy as np
        np.random.seed(42)
        n = 10000
        dates = pd.date_range('2024-01-01', periods=n, freq='5min')
        close = 5000 + np.cumsum(np.random.randn(n) * 2)
        df = pd.DataFrame({
            'open': close - np.random.rand(n),
            'high': close + np.abs(np.random.randn(n) * 2),
            'low': close - np.abs(np.random.randn(n) * 2),
            'close': close,
            'volume': np.random.randint(100, 1000, n)
        }, index=dates)
    else:
        # Aggregate to 5-minute
        df_5m = df.resample('5min').agg({
            'open': 'first', 'high': 'max',
            'low': 'min', 'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        df = df_5m.set_index('timestamp')

    print(f"Testing on {len(df):,} bars")

    # Test different confluence levels
    print("\n" + "=" * 70)
    print("TESTING DIFFERENT CONFLUENCE LEVELS")
    print("=" * 70)

    for min_conf in [2, 3, 4, 5]:
        print(f"\n--- Min Confluence: {min_conf} ---")
        results = backtest_confluence(df.reset_index(), min_confluence=min_conf)
        print(f"  Return: {results['total_return']:.1f}%")
        print(f"  Trades: {results['total_trades']}")
        print(f"  Win Rate: {results['win_rate']:.1f}%")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print(f"  Long P&L: ${results['long_pnl']:.2f}")
        print(f"  Short P&L: ${results['short_pnl']:.2f}")
        print(f"  Avg Score: {results['avg_score']:.1f}")
