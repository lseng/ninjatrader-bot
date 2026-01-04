"""
Video Strategy Features Package

Implements all 12 trading strategies extracted from TradingLab YouTube videos:
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

from .indicators import (
    williams_fractals,
    supertrend,
    triple_supertrend,
    macd_signals,
    dema,
    atr,
    fibonacci_levels,
    ma_alignment,
)

from .patterns import (
    heikin_ashi,
    detect_doji,
    candle_type,
    momentum_candles,
    strength_candle,
    reversal_candle,
)

from .zones import (
    detect_consolidation,
    find_supply_zones,
    find_demand_zones,
    detect_liquidity_sweep,
    find_equal_highs_lows,
    find_valid_swing_points,
)

from .features import (
    VideoStrategyFeatures,
    create_all_features,
)

__all__ = [
    # Indicators
    'williams_fractals',
    'supertrend',
    'triple_supertrend',
    'macd_signals',
    'dema',
    'atr',
    'fibonacci_levels',
    'ma_alignment',
    # Patterns
    'heikin_ashi',
    'detect_doji',
    'candle_type',
    'momentum_candles',
    'strength_candle',
    'reversal_candle',
    # Zones
    'detect_consolidation',
    'find_supply_zones',
    'find_demand_zones',
    'detect_liquidity_sweep',
    'find_equal_highs_lows',
    'find_valid_swing_points',
    # Features
    'VideoStrategyFeatures',
    'create_all_features',
]
