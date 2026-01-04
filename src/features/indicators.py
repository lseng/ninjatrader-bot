"""
Technical Indicators from Video Strategies

Implements:
- Williams Fractals (Strategy 5)
- SuperTrend - single and triple (Strategies 8, 9)
- MACD with 200 MA filter (Strategy 6)
- DEMA (Strategy 9)
- ATR (Strategy 11)
- Fibonacci levels (Strategy 10)
- Moving Average alignment (Strategy 5)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional


def williams_fractals(
    df: pd.DataFrame,
    period: int = 2
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Williams Fractals (Strategy 5).

    A bullish fractal is a bar whose low is the lowest of surrounding bars.
    A bearish fractal is a bar whose high is the highest of surrounding bars.

    Args:
        df: DataFrame with 'high' and 'low' columns
        period: Number of bars on each side to check (default=2 means 5-bar pattern)

    Returns:
        bullish_fractal: Series of booleans (True at bullish fractal points)
        bearish_fractal: Series of booleans (True at bearish fractal points)
    """
    highs = df['high']
    lows = df['low']

    bullish = pd.Series(False, index=df.index)
    bearish = pd.Series(False, index=df.index)

    for i in range(period, len(df) - period):
        # Bullish fractal: low is lowest of surrounding bars
        window_lows = lows.iloc[i-period:i+period+1]
        if lows.iloc[i] == window_lows.min() and window_lows.iloc[period] < window_lows.iloc[:period].min():
            bullish.iloc[i] = True

        # Bearish fractal: high is highest of surrounding bars
        window_highs = highs.iloc[i-period:i+period+1]
        if highs.iloc[i] == window_highs.max() and window_highs.iloc[period] > window_highs.iloc[:period].max():
            bearish.iloc[i] = True

    return bullish, bearish


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (Strategy 11).

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period

    Returns:
        ATR series
    """
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def supertrend(
    df: pd.DataFrame,
    atr_period: int = 10,
    multiplier: float = 3.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate SuperTrend indicator (Strategy 8, 9).

    Args:
        df: DataFrame with OHLC data
        atr_period: ATR calculation period
        multiplier: ATR multiplier for bands

    Returns:
        trend: 1 for bullish, -1 for bearish
        upper_band: Upper band values
        lower_band: Lower band values
    """
    hl2 = (df['high'] + df['low']) / 2
    atr_val = atr(df, atr_period)

    upper_band = hl2 + (multiplier * atr_val)
    lower_band = hl2 - (multiplier * atr_val)

    close = df['close']
    trend = pd.Series(1, index=df.index)
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    for i in range(1, len(df)):
        # Update bands
        if lower_band.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]

        if upper_band.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]

        # Update trend
        if trend.iloc[i-1] == -1 and close.iloc[i] > final_upper.iloc[i-1]:
            trend.iloc[i] = 1
        elif trend.iloc[i-1] == 1 and close.iloc[i] < final_lower.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]

    return trend, final_upper, final_lower


def triple_supertrend(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate Triple SuperTrend (Strategy 8).

    Uses 3 SuperTrend indicators with different settings:
    1. ATR=12, Multiplier=3
    2. ATR=10, Multiplier=1
    3. ATR=11, Multiplier=2

    Returns:
        trend1, trend2, trend3: Individual trend signals
        combined: 1 if all bullish, -1 if all bearish, 0 if mixed
    """
    trend1, _, _ = supertrend(df, atr_period=12, multiplier=3.0)
    trend2, _, _ = supertrend(df, atr_period=10, multiplier=1.0)
    trend3, _, _ = supertrend(df, atr_period=11, multiplier=2.0)

    # Combined signal
    combined = pd.Series(0, index=df.index)
    all_bullish = (trend1 == 1) & (trend2 == 1) & (trend3 == 1)
    all_bearish = (trend1 == -1) & (trend2 == -1) & (trend3 == -1)

    combined[all_bullish] = 1
    combined[all_bearish] = -1

    return trend1, trend2, trend3, combined


def dema(series: pd.Series, period: int = 200) -> pd.Series:
    """
    Calculate Double Exponential Moving Average (Strategy 9).

    DEMA = 2 * EMA(period) - EMA(EMA(period), period)

    This reduces lag compared to regular EMA.

    Args:
        series: Price series
        period: DEMA period

    Returns:
        DEMA series
    """
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    return 2 * ema1 - ema2


def macd_signals(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    ma_period: int = 200
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    MACD with 200 MA filter (Strategy 6).

    Long signals only when price > 200 MA and MACD crosses up below zero.
    Short signals only when price < 200 MA and MACD crosses down above zero.

    Returns:
        macd_line: MACD line values
        signal_line: Signal line values
        histogram: MACD histogram
        long_signal: Boolean series for long entries
        short_signal: Boolean series for short entries
    """
    close = df['close']

    # Calculate MACD
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    # 200 MA trend filter
    ma_200 = close.rolling(ma_period).mean()

    # Long: MACD crosses above signal, below zero, price > 200 MA
    macd_cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    long_signal = macd_cross_up & (macd_line < 0) & (close > ma_200)

    # Short: MACD crosses below signal, above zero, price < 200 MA
    macd_cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
    short_signal = macd_cross_down & (macd_line > 0) & (close < ma_200)

    return macd_line, signal_line, histogram, long_signal, short_signal


def fibonacci_levels(
    swing_low: float,
    swing_high: float,
    direction: str = 'retracement'
) -> Dict[str, float]:
    """
    Calculate Fibonacci levels (Strategy 10).

    Args:
        swing_low: Swing low price
        swing_high: Swing high price
        direction: 'retracement' for pullback levels, 'extension' for targets

    Returns:
        Dictionary of Fibonacci levels
    """
    diff = swing_high - swing_low

    if direction == 'retracement':
        return {
            '0.0': swing_high,
            '23.6': swing_high - 0.236 * diff,
            '38.2': swing_high - 0.382 * diff,
            '50.0': swing_high - 0.500 * diff,
            '61.8': swing_high - 0.618 * diff,  # Golden ratio
            '78.6': swing_high - 0.786 * diff,
            '100.0': swing_low
        }
    else:  # extension
        return {
            '0.0': swing_high,
            '23.6': swing_high + 0.236 * diff,
            '38.2': swing_high + 0.382 * diff,
            '61.8': swing_high + 0.618 * diff,
            '100.0': swing_high + diff,
            '127.2': swing_high + 1.272 * diff,
            '161.8': swing_high + 1.618 * diff
        }


def ma_alignment(
    df: pd.DataFrame,
    fast: int = 20,
    medium: int = 50,
    slow: int = 100
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Check moving average alignment (Strategy 5).

    For Williams Fractals strategy, we need:
    - Bullish: fast > medium > slow (20 > 50 > 100)
    - Bearish: slow > medium > fast (100 > 50 > 20)

    Returns:
        bullish_aligned: Boolean series where MAs are bullish aligned
        bearish_aligned: Boolean series where MAs are bearish aligned
        tangled: Boolean series where MAs are not clearly aligned
        ma_fast: Fast MA values
        ma_medium: Medium MA values
        ma_slow: Slow MA values
    """
    close = df['close']

    ma_fast = close.rolling(fast).mean()
    ma_medium = close.rolling(medium).mean()
    ma_slow = close.rolling(slow).mean()

    bullish_aligned = (ma_fast > ma_medium) & (ma_medium > ma_slow)
    bearish_aligned = (ma_slow > ma_medium) & (ma_medium > ma_fast)
    tangled = ~bullish_aligned & ~bearish_aligned

    return bullish_aligned, bearish_aligned, tangled, ma_fast, ma_medium, ma_slow


def chandelier_exit(
    df: pd.DataFrame,
    period: int = 22,
    multiplier: float = 3.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Chandelier Exit for trailing stops (mentioned in Strategy 1).

    Args:
        df: DataFrame with OHLC data
        period: Lookback period for highest high/lowest low
        multiplier: ATR multiplier

    Returns:
        long_exit: Exit level for long positions
        short_exit: Exit level for short positions
    """
    atr_val = atr(df, period)

    highest_high = df['high'].rolling(period).max()
    lowest_low = df['low'].rolling(period).min()

    long_exit = highest_high - multiplier * atr_val
    short_exit = lowest_low + multiplier * atr_val

    return long_exit, short_exit


def atr_stop_loss(
    entry_price: float,
    atr_value: float,
    direction: str = 'long',
    multiplier: float = 2.0
) -> float:
    """
    Calculate ATR-based stop loss (Strategy 11).

    Args:
        entry_price: Entry price
        atr_value: Current ATR value
        direction: 'long' or 'short'
        multiplier: ATR multiplier (1.5=tight, 2.0=standard, 3.0=wide)

    Returns:
        Stop loss price
    """
    if direction == 'long':
        return entry_price - (atr_value * multiplier)
    else:
        return entry_price + (atr_value * multiplier)
