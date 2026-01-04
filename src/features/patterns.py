"""
Candlestick Patterns from Video Strategies

Implements:
- Heikin Ashi candles (Strategy 7)
- Doji detection (Strategies 7, 12)
- Candle type classification (Strategy 3)
- Momentum candles (Strategy 1)
- Strength and reversal candles (Strategy 3)
"""

import pandas as pd
import numpy as np
from typing import Tuple


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Heikin Ashi candles (Strategy 7).

    Heikin Ashi shows AVERAGE price, smoothing out noise.

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with Heikin Ashi OHLC
    """
    ha = pd.DataFrame(index=df.index)

    # HA Close = (Open + High + Low + Close) / 4
    ha['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    # HA Open = (Previous HA Open + Previous HA Close) / 2
    ha['open'] = ((df['open'].shift(1) + df['close'].shift(1)) / 2).fillna(df['open'].iloc[0])

    # For proper calculation, we need to recalculate open iteratively
    for i in range(1, len(ha)):
        ha['open'].iloc[i] = (ha['open'].iloc[i-1] + ha['close'].iloc[i-1]) / 2

    # HA High = Max(High, HA Open, HA Close)
    ha['high'] = pd.concat([df['high'], ha['open'], ha['close']], axis=1).max(axis=1)

    # HA Low = Min(Low, HA Open, HA Close)
    ha['low'] = pd.concat([df['low'], ha['open'], ha['close']], axis=1).min(axis=1)

    return ha


def detect_doji(
    df: pd.DataFrame,
    body_threshold: float = 0.3
) -> pd.Series:
    """
    Detect Doji candles (Strategies 7, 12).

    A Doji has a small body relative to its range and wicks on both sides.

    Key insight from Strategy 12: "A Doji by itself means NOTHING.
    Always wait for the next candle to confirm."

    Args:
        df: DataFrame with OHLC data
        body_threshold: Maximum body/range ratio to be considered doji

    Returns:
        Boolean series where True indicates doji
    """
    body = abs(df['close'] - df['open'])
    range_ = df['high'] - df['low']

    # Avoid division by zero
    range_safe = range_.replace(0, np.nan)
    body_ratio = body / range_safe

    # Doji: small body + wicks on both sides
    upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
    lower_wick = df[['close', 'open']].min(axis=1) - df['low']

    has_both_wicks = (upper_wick > 0) & (lower_wick > 0)

    return (body_ratio < body_threshold) & has_both_wicks


def candle_type(df: pd.DataFrame) -> pd.Series:
    """
    Classify candle types (Strategy 3).

    Returns integer codes:
    - 2: Strong bullish (strength candle)
    - 1: Mild bullish
    - 0: Indecision (doji-like)
    - -1: Mild bearish
    - -2: Strong bearish (strength candle)
    - 3: Bullish reversal (control shift)
    - -3: Bearish reversal (control shift)

    Args:
        df: DataFrame with OHLC data

    Returns:
        Series of candle type codes
    """
    body = abs(df['close'] - df['open'])
    range_ = df['high'] - df['low']
    range_safe = range_.replace(0, np.nan)

    body_ratio = body / range_safe
    is_bullish = df['close'] > df['open']

    upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
    lower_wick = df[['close', 'open']].min(axis=1) - df['low']

    upper_wick_ratio = upper_wick / range_safe
    lower_wick_ratio = lower_wick / range_safe

    result = pd.Series(0, index=df.index)

    # Strength candles: large body (>70% of range)
    strength_bull = (body_ratio > 0.7) & is_bullish
    strength_bear = (body_ratio > 0.7) & ~is_bullish
    result[strength_bull] = 2
    result[strength_bear] = -2

    # Mild directional candles
    mild_bull = (body_ratio > 0.3) & (body_ratio <= 0.7) & is_bullish
    mild_bear = (body_ratio > 0.3) & (body_ratio <= 0.7) & ~is_bullish
    result[mild_bull] = 1
    result[mild_bear] = -1

    # Reversal/control shift candles: long wick, small body
    # Bullish reversal: long lower wick (>60% of range)
    bullish_reversal = (lower_wick_ratio > 0.6) & (body_ratio < 0.3)
    bearish_reversal = (upper_wick_ratio > 0.6) & (body_ratio < 0.3)
    result[bullish_reversal] = 3
    result[bearish_reversal] = -3

    # Indecision: small body (<30%) with wicks on both sides
    indecision = (body_ratio < 0.3) & ~bullish_reversal & ~bearish_reversal
    result[indecision] = 0

    return result


def momentum_candles(
    df: pd.DataFrame,
    lookback: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect momentum candle patterns (Strategy 1).

    Two types of momentum confirmation:
    1. Single large candle (body > 1.5x average)
    2. Three consecutive candles in same direction

    Args:
        df: DataFrame with OHLC data
        lookback: Number of consecutive candles to check

    Returns:
        bullish_momentum: Boolean series for bullish momentum
        bearish_momentum: Boolean series for bearish momentum
    """
    close = df['close']
    open_ = df['open']

    body = abs(close - open_)
    avg_body = body.rolling(20).mean()

    is_bullish = close > open_
    is_bearish = close < open_

    # Single large bullish candle (body > 1.5x average)
    large_bullish = is_bullish & (body > 1.5 * avg_body)
    large_bearish = is_bearish & (body > 1.5 * avg_body)

    # Three consecutive candles in same direction
    consecutive_bull = is_bullish.copy()
    consecutive_bear = is_bearish.copy()

    for i in range(1, lookback):
        consecutive_bull = consecutive_bull & is_bullish.shift(i)
        consecutive_bear = consecutive_bear & is_bearish.shift(i)

    bullish_momentum = large_bullish | consecutive_bull
    bearish_momentum = large_bearish | consecutive_bear

    return bullish_momentum, bearish_momentum


def strength_candle(
    df: pd.DataFrame,
    body_threshold: float = 0.7
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect strength candles (Strategy 3).

    Strength candles have:
    - Large body (>70% of range)
    - Small or no wick on one side
    - Shows one side in complete control

    Returns:
        bullish_strength: Boolean for bullish strength candles
        bearish_strength: Boolean for bearish strength candles
    """
    body = abs(df['close'] - df['open'])
    range_ = df['high'] - df['low']
    range_safe = range_.replace(0, np.nan)

    body_ratio = body / range_safe

    is_bullish = df['close'] > df['open']
    is_bearish = df['close'] < df['open']

    bullish_strength = (body_ratio > body_threshold) & is_bullish
    bearish_strength = (body_ratio > body_threshold) & is_bearish

    return bullish_strength, bearish_strength


def reversal_candle(
    df: pd.DataFrame,
    wick_threshold: float = 0.6
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect reversal/control shift candles (Strategy 3).

    Reversal candles have:
    - Long wick (>60% of range)
    - Small body
    - Shows rejected price action

    Returns:
        bullish_reversal: Long lower wick (buyers taking over)
        bearish_reversal: Long upper wick (sellers taking over)
    """
    range_ = df['high'] - df['low']
    range_safe = range_.replace(0, np.nan)

    upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
    lower_wick = df[['close', 'open']].min(axis=1) - df['low']

    upper_wick_ratio = upper_wick / range_safe
    lower_wick_ratio = lower_wick / range_safe

    bullish_reversal = lower_wick_ratio > wick_threshold
    bearish_reversal = upper_wick_ratio > wick_threshold

    return bullish_reversal, bearish_reversal


def ha_trend_strength(ha_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Analyze Heikin Ashi trend strength (Strategy 7).

    Strong uptrend: Green candles, large bodies, NO lower wicks
    Strong downtrend: Red candles, large bodies, NO upper wicks

    Args:
        ha_df: Heikin Ashi DataFrame

    Returns:
        strong_uptrend: Boolean for strong uptrend bars
        strong_downtrend: Boolean for strong downtrend bars
    """
    is_green = ha_df['close'] > ha_df['open']
    is_red = ha_df['close'] < ha_df['open']

    body = abs(ha_df['close'] - ha_df['open'])
    range_ = ha_df['high'] - ha_df['low']
    range_safe = range_.replace(0, np.nan)

    body_ratio = body / range_safe

    # Lower wick for green candles (open is the bottom of body)
    lower_wick_green = ha_df['open'] - ha_df['low']
    # Upper wick for red candles (open is the top of body)
    upper_wick_red = ha_df['high'] - ha_df['open']

    # Strong uptrend: green, large body, no lower wick
    strong_uptrend = is_green & (body_ratio > 0.5) & (lower_wick_green < body * 0.1)

    # Strong downtrend: red, large body, no upper wick
    strong_downtrend = is_red & (body_ratio > 0.5) & (upper_wick_red < body * 0.1)

    return strong_uptrend, strong_downtrend


def doji_with_confirmation(
    df: pd.DataFrame,
    lookback: int = 5
) -> Tuple[pd.Series, pd.Series]:
    """
    Doji with confirmation (Strategy 12).

    The key insight: "A Doji by itself means NOTHING."
    We need to wait for the next candle to confirm direction.

    Args:
        df: DataFrame with OHLC data
        lookback: Bars to check for prior trend

    Returns:
        bullish_doji_reversal: Doji after downtrend with bullish confirmation
        bearish_doji_reversal: Doji after uptrend with bearish confirmation
    """
    doji = detect_doji(df)

    close = df['close']
    is_bullish = close > df['open']
    is_bearish = close < df['open']

    # Trend before doji
    prior_trend = close.diff(lookback)
    was_downtrend = prior_trend < 0
    was_uptrend = prior_trend > 0

    # Confirmation candle (next bar after doji)
    bullish_confirm = is_bullish.shift(-1) & (close.shift(-1) > close)
    bearish_confirm = is_bearish.shift(-1) & (close.shift(-1) < close)

    # Bullish doji reversal: downtrend + doji + bullish confirmation
    bullish_doji_reversal = doji & was_downtrend & bullish_confirm.shift(1)

    # Bearish doji reversal: uptrend + doji + bearish confirmation
    bearish_doji_reversal = doji & was_uptrend & bearish_confirm.shift(1)

    return bullish_doji_reversal, bearish_doji_reversal
