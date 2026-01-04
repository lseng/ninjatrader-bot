"""
Supply/Demand Zones and Liquidity Detection from Video Strategies

Implements:
- Consolidation detection (Strategy 1, 2)
- Supply/Demand zones (Strategy 4)
- Liquidity sweep detection (Strategy 2)
- Equal highs/lows detection (Strategy 2)
- Valid swing point detection (Strategy 4)
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def detect_consolidation(
    df: pd.DataFrame,
    window: int = 20,
    range_threshold: float = 0.015
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Detect consolidation periods (Strategies 1, 2).

    Consolidation is identified when:
    - Price range is tight relative to average
    - Multiple touches of support AND resistance

    Args:
        df: DataFrame with OHLC data
        window: Lookback window for range calculation
        range_threshold: Maximum range/price ratio for consolidation

    Returns:
        is_consolidation: Boolean series where consolidation is detected
        range_high: Upper boundary of consolidation
        range_low: Lower boundary of consolidation
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # Calculate rolling range
    rolling_high = high.rolling(window).max()
    rolling_low = low.rolling(window).min()
    rolling_range = (rolling_high - rolling_low) / close

    # Consolidation when range is below threshold
    is_consolidation = rolling_range < range_threshold

    return is_consolidation, rolling_high, rolling_low


def find_valid_swing_points(
    df: pd.DataFrame,
    lookback: int = 5
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Find valid swing highs and lows (Strategy 4).

    Key insight from video:
    - A low is only VALIDATED if it breaks the previous high
    - A high is only VALIDATED if it breaks the previous low
    - Ignore any swing that doesn't meet this criteria

    Args:
        df: DataFrame with OHLC data
        lookback: Bars to look back for swing detection

    Returns:
        valid_swing_high: Boolean where valid swing highs occur
        valid_swing_low: Boolean where valid swing lows occur
        last_valid_high: Series of last valid high values
        last_valid_low: Series of last valid low values
    """
    high = df['high']
    low = df['low']

    # Basic swing detection first
    swing_high = (high == high.rolling(2*lookback + 1, center=True).max())
    swing_low = (low == low.rolling(2*lookback + 1, center=True).min())

    valid_high = pd.Series(False, index=df.index)
    valid_low = pd.Series(False, index=df.index)
    last_valid_high = pd.Series(np.nan, index=df.index)
    last_valid_low = pd.Series(np.nan, index=df.index)

    prev_high = high.iloc[0]
    prev_low = low.iloc[0]
    last_h = high.iloc[0]
    last_l = low.iloc[0]

    for i in range(lookback, len(df)):
        # Update last valid values
        last_valid_high.iloc[i] = last_h
        last_valid_low.iloc[i] = last_l

        if swing_high.iloc[i]:
            # A high is valid if it broke the previous low
            if low.iloc[i-lookback:i].min() < prev_low:
                valid_high.iloc[i] = True
                last_h = high.iloc[i]
                prev_high = high.iloc[i]

        if swing_low.iloc[i]:
            # A low is valid if it broke the previous high
            if high.iloc[i-lookback:i].max() > prev_high:
                valid_low.iloc[i] = True
                last_l = low.iloc[i]
                prev_low = low.iloc[i]

    return valid_high, valid_low, last_valid_high, last_valid_low


def find_demand_zones(
    df: pd.DataFrame,
    impulse_threshold: float = 0.003,
    min_impulse_bars: int = 1
) -> List[dict]:
    """
    Find demand zones (Strategy 4).

    Demand zone = consolidation BEFORE a strong upward move.
    Mark the candle RIGHT BEFORE the impulse move.

    Args:
        df: DataFrame with OHLC data
        impulse_threshold: Minimum return for impulse move
        min_impulse_bars: Minimum bars of impulse

    Returns:
        List of demand zone dictionaries with:
        - zone_low, zone_high: Zone boundaries
        - impulse_start_idx: Index where impulse started
        - strength: Size of the impulse move
    """
    close = df['close']
    returns = close.pct_change()

    zones = []

    i = 20  # Start with some lookback
    while i < len(df) - min_impulse_bars:
        # Check for impulse move (strong up move)
        impulse_return = returns.iloc[i:i+min_impulse_bars+1].sum()

        if impulse_return > impulse_threshold:
            # Found impulse - mark candle before as demand zone
            zone_idx = i - 1
            if zone_idx >= 0:
                zone = {
                    'zone_low': df['low'].iloc[zone_idx],
                    'zone_high': df['high'].iloc[zone_idx],
                    'impulse_start_idx': i,
                    'strength': impulse_return,
                    'timestamp': df.index[zone_idx] if hasattr(df.index, 'to_pydatetime') else zone_idx
                }
                zones.append(zone)

            # Skip past this impulse
            i += min_impulse_bars + 1
        else:
            i += 1

    return zones


def find_supply_zones(
    df: pd.DataFrame,
    impulse_threshold: float = 0.003,
    min_impulse_bars: int = 1
) -> List[dict]:
    """
    Find supply zones (Strategy 4).

    Supply zone = consolidation BEFORE a strong downward move.
    Mark the candle RIGHT BEFORE the impulse move.

    Args:
        df: DataFrame with OHLC data
        impulse_threshold: Minimum return for impulse move (positive value)
        min_impulse_bars: Minimum bars of impulse

    Returns:
        List of supply zone dictionaries
    """
    close = df['close']
    returns = close.pct_change()

    zones = []

    i = 20
    while i < len(df) - min_impulse_bars:
        # Check for impulse move (strong down move)
        impulse_return = returns.iloc[i:i+min_impulse_bars+1].sum()

        if impulse_return < -impulse_threshold:
            # Found impulse - mark candle before as supply zone
            zone_idx = i - 1
            if zone_idx >= 0:
                zone = {
                    'zone_low': df['low'].iloc[zone_idx],
                    'zone_high': df['high'].iloc[zone_idx],
                    'impulse_start_idx': i,
                    'strength': abs(impulse_return),
                    'timestamp': df.index[zone_idx] if hasattr(df.index, 'to_pydatetime') else zone_idx
                }
                zones.append(zone)

            i += min_impulse_bars + 1
        else:
            i += 1

    return zones


def find_equal_highs_lows(
    df: pd.DataFrame,
    tolerance: float = 0.0005,
    lookback: int = 20
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect equal highs and equal lows (Strategy 2).

    Equal highs/lows are areas where multiple swing points align,
    creating obvious stop loss clusters that get hunted.

    Args:
        df: DataFrame with OHLC data
        tolerance: Price tolerance for "equal" (as % of price)
        lookback: Bars to look back

    Returns:
        equal_highs: Boolean series where equal highs detected
        equal_lows: Boolean series where equal lows detected
    """
    high = df['high']
    low = df['low']

    equal_highs = pd.Series(False, index=df.index)
    equal_lows = pd.Series(False, index=df.index)

    for i in range(lookback, len(df)):
        current_high = high.iloc[i]
        current_low = low.iloc[i]

        # Check for equal highs in lookback window
        window_highs = high.iloc[i-lookback:i]
        tol_amount = current_high * tolerance
        matches = abs(window_highs - current_high) < tol_amount
        if matches.sum() >= 2:  # At least 2 prior highs at same level
            equal_highs.iloc[i] = True

        # Check for equal lows
        window_lows = low.iloc[i-lookback:i]
        tol_amount = current_low * tolerance
        matches = abs(window_lows - current_low) < tol_amount
        if matches.sum() >= 2:
            equal_lows.iloc[i] = True

    return equal_highs, equal_lows


def detect_liquidity_sweep(
    df: pd.DataFrame,
    lookback: int = 20
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect liquidity sweeps (Strategy 2).

    A liquidity sweep occurs when:
    - Price breaks a level (triggering stops)
    - Then quickly reverses back into range

    This is identified by:
    - Wick beyond recent high/low
    - Close back inside the range

    Args:
        df: DataFrame with OHLC data
        lookback: Bars for range calculation

    Returns:
        sweep_high: Boolean where high was swept (bearish signal)
        sweep_low: Boolean where low was swept (bullish signal)
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # Recent range boundaries
    recent_high = high.rolling(lookback).max().shift(1)
    recent_low = low.rolling(lookback).min().shift(1)

    # Sweep high: wick breaks above recent high, but close below
    sweep_high = (high > recent_high) & (close < recent_high)

    # Sweep low: wick breaks below recent low, but close above
    sweep_low = (low < recent_low) & (close > recent_low)

    return sweep_high, sweep_low


def price_at_zone(
    df: pd.DataFrame,
    demand_zones: List[dict],
    supply_zones: List[dict],
    buffer: float = 0.001
) -> Tuple[pd.Series, pd.Series]:
    """
    Check if price is at a supply or demand zone.

    Args:
        df: DataFrame with OHLC data
        demand_zones: List of demand zone dicts
        supply_zones: List of supply zone dicts
        buffer: Buffer around zone boundaries

    Returns:
        at_demand: Boolean where price is at a demand zone
        at_supply: Boolean where price is at a supply zone
    """
    close = df['close']

    at_demand = pd.Series(False, index=df.index)
    at_supply = pd.Series(False, index=df.index)

    # Check demand zones
    for zone in demand_zones:
        zone_low = zone['zone_low'] * (1 - buffer)
        zone_high = zone['zone_high'] * (1 + buffer)
        at_zone = (close >= zone_low) & (close <= zone_high)
        at_demand = at_demand | at_zone

    # Check supply zones
    for zone in supply_zones:
        zone_low = zone['zone_low'] * (1 - buffer)
        zone_high = zone['zone_high'] * (1 + buffer)
        at_zone = (close >= zone_low) & (close <= zone_high)
        at_supply = at_supply | at_zone

    return at_demand, at_supply


def get_market_structure(
    df: pd.DataFrame,
    lookback: int = 5
) -> pd.Series:
    """
    Determine market structure based on valid swing points (Strategy 4).

    Returns:
        1 = Uptrend (higher highs and higher lows)
        -1 = Downtrend (lower highs and lower lows)
        0 = Ranging/Neutral
    """
    _, _, last_valid_high, last_valid_low = find_valid_swing_points(df, lookback)

    structure = pd.Series(0, index=df.index)

    # Compare consecutive valid swing points
    hh = last_valid_high > last_valid_high.shift(1)  # Higher high
    hl = last_valid_low > last_valid_low.shift(1)    # Higher low
    lh = last_valid_high < last_valid_high.shift(1)  # Lower high
    ll = last_valid_low < last_valid_low.shift(1)    # Lower low

    # Uptrend: HH + HL
    uptrend = hh & hl
    # Downtrend: LH + LL
    downtrend = lh & ll

    structure[uptrend] = 1
    structure[downtrend] = -1

    return structure


def calculate_risk_reward(
    entry: float,
    stop: float,
    target: float,
    min_rr: float = 2.5
) -> Tuple[bool, float]:
    """
    Calculate if trade meets minimum risk/reward (Strategy 4).

    The video emphasizes R:R >= 2.5:1 minimum.

    Args:
        entry: Entry price
        stop: Stop loss price
        target: Take profit price
        min_rr: Minimum required R:R ratio

    Returns:
        passes: Boolean if trade passes R:R filter
        rr_ratio: Actual R:R ratio
    """
    risk = abs(entry - stop)
    reward = abs(target - entry)

    if risk == 0:
        return False, 0.0

    rr_ratio = reward / risk
    passes = rr_ratio >= min_rr

    return passes, rr_ratio
