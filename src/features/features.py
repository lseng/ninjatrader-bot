"""
Main Feature Engineering Module

Combines all 12 video strategies into ML-ready features.
This module creates a comprehensive feature set for training
trading ML models.
"""

import pandas as pd
import numpy as np
from typing import Optional, List

from .indicators import (
    williams_fractals,
    supertrend,
    triple_supertrend,
    macd_signals,
    dema,
    atr,
    ma_alignment,
    chandelier_exit,
)

from .patterns import (
    heikin_ashi,
    detect_doji,
    candle_type,
    momentum_candles,
    strength_candle,
    reversal_candle,
    ha_trend_strength,
    doji_with_confirmation,
)

from .zones import (
    detect_consolidation,
    find_supply_zones,
    find_demand_zones,
    detect_liquidity_sweep,
    find_equal_highs_lows,
    find_valid_swing_points,
    get_market_structure,
)


class VideoStrategyFeatures:
    """
    Feature engineering class implementing all 12 video strategies.

    Strategies implemented:
    1. Breakout + Momentum Candles
    2. Liquidity Hunting (Smart Money)
    3. Candlestick Context Analysis
    4. Supply/Demand Zone Formula
    5. Williams Fractals Scalping
    6. MACD + 200 MA (86% Win Rate)
    7. Heikin Ashi Reversal
    8. Triple SuperTrend
    9. DEMA + SuperTrend (130% strategy)
    10. Fibonacci Retracement
    11. ATR-Based Stop Loss
    12. Doji Candle Rules
    """

    def __init__(
        self,
        consolidation_window: int = 20,
        ma_fast: int = 20,
        ma_medium: int = 50,
        ma_slow: int = 100,
        ma_trend: int = 200,
        atr_period: int = 14,
        fractal_period: int = 2,
    ):
        """
        Initialize feature generator.

        Args:
            consolidation_window: Bars for consolidation detection
            ma_fast: Fast MA period (for Williams Fractals)
            ma_medium: Medium MA period
            ma_slow: Slow MA period
            ma_trend: Trend filter MA period (200 for MACD strategy)
            atr_period: ATR calculation period
            fractal_period: Williams Fractal lookback
        """
        self.consolidation_window = consolidation_window
        self.ma_fast = ma_fast
        self.ma_medium = ma_medium
        self.ma_slow = ma_slow
        self.ma_trend = ma_trend
        self.atr_period = atr_period
        self.fractal_period = fractal_period

    def create_features(
        self,
        df: pd.DataFrame,
        include_zones: bool = True
    ) -> pd.DataFrame:
        """
        Create all features from video strategies.

        Args:
            df: DataFrame with OHLCV data
                Required columns: open, high, low, close
                Optional: volume

        Returns:
            DataFrame with all computed features
        """
        features = pd.DataFrame(index=df.index)

        # =================================================================
        # BASIC PRICE FEATURES
        # =================================================================
        features['returns'] = df['close'].pct_change()
        features['range'] = (df['high'] - df['low']) / df['close']
        features['body'] = (df['close'] - df['open']) / df['close']
        features['body_abs'] = abs(df['close'] - df['open']) / df['close']

        # =================================================================
        # STRATEGY 1: BREAKOUT + MOMENTUM CANDLES
        # =================================================================
        bull_momentum, bear_momentum = momentum_candles(df, lookback=3)
        features['bullish_momentum'] = bull_momentum.astype(int)
        features['bearish_momentum'] = bear_momentum.astype(int)

        is_consol, range_high, range_low = detect_consolidation(
            df, window=self.consolidation_window
        )
        features['in_consolidation'] = is_consol.astype(int)
        features['distance_to_range_high'] = (range_high - df['close']) / df['close']
        features['distance_to_range_low'] = (df['close'] - range_low) / df['close']

        # Breakout detection
        features['breakout_up'] = ((df['close'] > range_high.shift(1)) &
                                   (df['close'].shift(1) <= range_high.shift(2))).astype(int)
        features['breakout_down'] = ((df['close'] < range_low.shift(1)) &
                                     (df['close'].shift(1) >= range_low.shift(2))).astype(int)

        # =================================================================
        # STRATEGY 2: LIQUIDITY HUNTING
        # =================================================================
        sweep_high, sweep_low = detect_liquidity_sweep(df, lookback=self.consolidation_window)
        features['sweep_high'] = sweep_high.astype(int)
        features['sweep_low'] = sweep_low.astype(int)

        equal_highs, equal_lows = find_equal_highs_lows(df, lookback=self.consolidation_window)
        features['equal_highs'] = equal_highs.astype(int)
        features['equal_lows'] = equal_lows.astype(int)

        # =================================================================
        # STRATEGY 3: CANDLESTICK CONTEXT
        # =================================================================
        features['candle_type'] = candle_type(df)

        bull_strength, bear_strength = strength_candle(df)
        features['bullish_strength'] = bull_strength.astype(int)
        features['bearish_strength'] = bear_strength.astype(int)

        bull_reversal, bear_reversal = reversal_candle(df)
        features['bullish_reversal'] = bull_reversal.astype(int)
        features['bearish_reversal'] = bear_reversal.astype(int)

        # Relative body size
        avg_body = features['body_abs'].rolling(20).mean()
        features['relative_body_size'] = features['body_abs'] / avg_body.replace(0, np.nan)

        # =================================================================
        # STRATEGY 4: SUPPLY/DEMAND ZONES
        # =================================================================
        features['market_structure'] = get_market_structure(df)

        valid_high, valid_low, last_high, last_low = find_valid_swing_points(df)
        features['valid_swing_high'] = valid_high.astype(int)
        features['valid_swing_low'] = valid_low.astype(int)
        features['distance_to_last_high'] = (last_high - df['close']) / df['close']
        features['distance_to_last_low'] = (df['close'] - last_low) / df['close']

        if include_zones:
            demand_zones = find_demand_zones(df)
            supply_zones = find_supply_zones(df)
            features['demand_zone_count'] = len(demand_zones)
            features['supply_zone_count'] = len(supply_zones)

        # =================================================================
        # STRATEGY 5: WILLIAMS FRACTALS
        # =================================================================
        bull_frac, bear_frac = williams_fractals(df, period=self.fractal_period)
        features['bullish_fractal'] = bull_frac.astype(int)
        features['bearish_fractal'] = bear_frac.astype(int)

        bull_align, bear_align, tangled, ma20, ma50, ma100 = ma_alignment(
            df, fast=self.ma_fast, medium=self.ma_medium, slow=self.ma_slow
        )
        features['ma_bullish_aligned'] = bull_align.astype(int)
        features['ma_bearish_aligned'] = bear_align.astype(int)
        features['ma_tangled'] = tangled.astype(int)

        # Price position relative to MAs
        features['close_vs_ma20'] = (df['close'] - ma20) / ma20
        features['close_vs_ma50'] = (df['close'] - ma50) / ma50
        features['close_vs_ma100'] = (df['close'] - ma100) / ma100

        # Williams Fractal signal (per strategy rules)
        features['fractal_long_signal'] = (
            bull_frac & bull_align & ~tangled &
            (df['close'] < ma20) & (df['close'] > ma100)
        ).astype(int)
        features['fractal_short_signal'] = (
            bear_frac & bear_align & ~tangled &
            (df['close'] > ma20) & (df['close'] < ma100)
        ).astype(int)

        # =================================================================
        # STRATEGY 6: MACD + 200 MA
        # =================================================================
        macd_line, signal_line, histogram, macd_long, macd_short = macd_signals(
            df, ma_period=self.ma_trend
        )
        features['macd_line'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_histogram'] = histogram
        features['macd_long_signal'] = macd_long.astype(int)
        features['macd_short_signal'] = macd_short.astype(int)

        # Distance from 200 MA
        ma200 = df['close'].rolling(self.ma_trend).mean()
        features['close_vs_ma200'] = (df['close'] - ma200) / ma200

        # =================================================================
        # STRATEGY 7: HEIKIN ASHI
        # =================================================================
        ha_df = heikin_ashi(df)
        features['ha_is_green'] = (ha_df['close'] > ha_df['open']).astype(int)

        ha_strong_up, ha_strong_down = ha_trend_strength(ha_df)
        features['ha_strong_uptrend'] = ha_strong_up.astype(int)
        features['ha_strong_downtrend'] = ha_strong_down.astype(int)

        ha_doji = detect_doji(ha_df)
        features['ha_doji'] = ha_doji.astype(int)

        # =================================================================
        # STRATEGY 8: TRIPLE SUPERTREND
        # =================================================================
        st1, st2, st3, st_combined = triple_supertrend(df)
        features['supertrend_1'] = st1
        features['supertrend_2'] = st2
        features['supertrend_3'] = st3
        features['triple_supertrend'] = st_combined

        # =================================================================
        # STRATEGY 9: DEMA + SUPERTREND
        # =================================================================
        dema200 = dema(df['close'], period=self.ma_trend)
        features['dema_200'] = dema200
        features['close_vs_dema200'] = (df['close'] - dema200) / dema200

        st_main, st_upper, st_lower = supertrend(df, atr_period=12, multiplier=3.0)
        features['supertrend_main'] = st_main

        # DEMA + SuperTrend combined signal
        features['dema_st_long'] = ((df['close'] > dema200) & (st_main == 1) &
                                    (st_main.shift(1) == -1)).astype(int)
        features['dema_st_short'] = ((df['close'] < dema200) & (st_main == -1) &
                                     (st_main.shift(1) == 1)).astype(int)

        # =================================================================
        # STRATEGY 10: FIBONACCI (as features)
        # =================================================================
        # Find recent swing for Fib levels
        lookback = 50
        recent_high = df['high'].rolling(lookback).max()
        recent_low = df['low'].rolling(lookback).min()
        fib_range = recent_high - recent_low

        # Price position relative to Fib levels
        features['fib_236'] = (df['close'] - (recent_high - 0.236 * fib_range)) / df['close']
        features['fib_382'] = (df['close'] - (recent_high - 0.382 * fib_range)) / df['close']
        features['fib_500'] = (df['close'] - (recent_high - 0.500 * fib_range)) / df['close']
        features['fib_618'] = (df['close'] - (recent_high - 0.618 * fib_range)) / df['close']
        features['fib_786'] = (df['close'] - (recent_high - 0.786 * fib_range)) / df['close']

        # =================================================================
        # STRATEGY 11: ATR-BASED STOP LOSS
        # =================================================================
        atr_val = atr(df, period=self.atr_period)
        features['atr'] = atr_val
        features['atr_ratio'] = atr_val / df['close']

        # Chandelier exit levels
        long_exit, short_exit = chandelier_exit(df)
        features['chandelier_long_exit'] = (df['close'] - long_exit) / df['close']
        features['chandelier_short_exit'] = (short_exit - df['close']) / df['close']

        # =================================================================
        # STRATEGY 12: DOJI RULES
        # =================================================================
        doji = detect_doji(df)
        features['is_doji'] = doji.astype(int)

        bull_doji_rev, bear_doji_rev = doji_with_confirmation(df)
        features['bullish_doji_reversal'] = bull_doji_rev.astype(int)
        features['bearish_doji_reversal'] = bear_doji_rev.astype(int)

        # =================================================================
        # VOLUME FEATURES (if available)
        # =================================================================
        if 'volume' in df.columns:
            features['volume_sma_5'] = df['volume'].rolling(5).mean()
            features['volume_sma_20'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma_5'].replace(0, 1)
            features['volume_spike'] = (features['volume_ratio'] > 2).astype(int)

        # =================================================================
        # VOLATILITY FEATURES
        # =================================================================
        features['volatility_5'] = features['returns'].rolling(5).std()
        features['volatility_10'] = features['returns'].rolling(10).std()
        features['volatility_20'] = features['returns'].rolling(20).std()

        # =================================================================
        # RSI (for reference, though video said it's nearly useless)
        # =================================================================
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        features['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

        return features

    def get_feature_names(self) -> List[str]:
        """Return list of all feature names created."""
        # Create dummy data to get feature names
        dummy = pd.DataFrame({
            'open': [100] * 300,
            'high': [101] * 300,
            'low': [99] * 300,
            'close': [100.5] * 300,
            'volume': [1000] * 300
        })
        features = self.create_features(dummy, include_zones=False)
        return list(features.columns)


def create_all_features(
    df: pd.DataFrame,
    include_zones: bool = True
) -> pd.DataFrame:
    """
    Convenience function to create all features.

    Args:
        df: DataFrame with OHLCV data
        include_zones: Whether to include zone detection

    Returns:
        DataFrame with all computed features
    """
    generator = VideoStrategyFeatures()
    return generator.create_features(df, include_zones=include_zones)


def aggregate_to_timeframe(
    df: pd.DataFrame,
    timeframe_minutes: int
) -> pd.DataFrame:
    """
    Aggregate data to a higher timeframe.

    Args:
        df: DataFrame with 1-second or 1-minute OHLCV data
        timeframe_minutes: Target timeframe in minutes

    Returns:
        Aggregated DataFrame
    """
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    # Resample to target timeframe
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }

    if 'volume' in df.columns:
        agg_dict['volume'] = 'sum'

    resampled = df.resample(f'{timeframe_minutes}min').agg(agg_dict)
    return resampled.dropna()


if __name__ == '__main__':
    # Demo usage
    print("=" * 60)
    print("VIDEO STRATEGY FEATURES - DEMO")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n = 500
    prices = 100 + np.cumsum(np.random.randn(n) * 0.1)

    sample_df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 0.2),
        'low': prices - np.abs(np.random.randn(n) * 0.2),
        'close': prices + np.random.randn(n) * 0.1,
        'volume': np.random.randint(100, 1000, n)
    })

    # Generate features
    generator = VideoStrategyFeatures()
    features = generator.create_features(sample_df, include_zones=False)

    print(f"\nTotal features created: {len(features.columns)}")
    print("\nFeature categories:")

    # Group features by prefix
    prefixes = {}
    for col in features.columns:
        prefix = col.split('_')[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1

    for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1]):
        print(f"  {prefix}: {count} features")

    print(f"\nSample feature values (last row):")
    for col in ['market_structure', 'triple_supertrend', 'macd_long_signal',
                'fractal_long_signal', 'sweep_low', 'ha_strong_uptrend']:
        if col in features.columns:
            print(f"  {col}: {features[col].iloc[-1]}")
