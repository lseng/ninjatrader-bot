#!/usr/bin/env python3
"""
Comprehensive Confluence Strategy Backtest for RunPod

Tests ALL video strategies combined with different confluence levels.
Uses 32 CPUs for parallel processing.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count
from itertools import product
import warnings
warnings.filterwarnings('ignore')


# Configuration
DATA_PATH = "/workspace/data/MES_1s_2years.parquet"
CONTRACT_VALUE = 5.0
COMMISSION = 2.50
INITIAL_CAPITAL = 1000


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data(timeframe: int) -> pd.DataFrame:
    """Load and aggregate data."""
    log(f"Loading data for {timeframe}m timeframe...")
    df = pd.read_parquet(DATA_PATH)
    df = df.set_index('timestamp')

    agg = df.resample(f'{timeframe}min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()

    log(f"  {len(agg):,} bars")
    return agg.reset_index()


# ============================================================
# ALL INDICATOR CALCULATIONS
# ============================================================

def calculate_all_indicators(df, params):
    """Calculate ALL strategy indicators."""
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']

    ma_fast = params.get('ma_fast', 20)
    ma_medium = params.get('ma_medium', 50)
    ma_slow = params.get('ma_slow', 100)
    fractal_period = params.get('fractal_period', 3)

    # 1. WILLIAMS FRACTALS + MA ALIGNMENT
    df['sma_fast'] = close.rolling(ma_fast).mean()
    df['sma_medium'] = close.rolling(ma_medium).mean()
    df['sma_slow'] = close.rolling(ma_slow).mean()

    df['bullish_ma_align'] = (df['sma_fast'] > df['sma_medium']) & (df['sma_medium'] > df['sma_slow'])
    df['bearish_ma_align'] = (df['sma_fast'] < df['sma_medium']) & (df['sma_medium'] < df['sma_slow'])

    window = 2 * fractal_period + 1
    df['rolling_low_min'] = low.rolling(window, center=True).min()
    df['rolling_high_max'] = high.rolling(window, center=True).max()
    df['bullish_fractal'] = low == df['rolling_low_min']
    df['bearish_fractal'] = high == df['rolling_high_max']

    df['pullback_long'] = close < df['sma_fast']
    df['pullback_short'] = close > df['sma_fast']
    df['above_slow'] = close > df['sma_slow']
    df['below_slow'] = close < df['sma_slow']

    # 2. MACD + 200 MA
    macd_fast = params.get('macd_fast', 12)
    macd_slow = params.get('macd_slow', 26)
    macd_signal_period = params.get('macd_signal', 9)
    ma_200_period = params.get('ma_200', 200)

    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=macd_signal_period, adjust=False).mean()
    df['ma_200'] = close.rolling(ma_200_period).mean()

    df['macd_bullish'] = (df['macd'] > df['macd_signal']) & (close > df['ma_200'])
    df['macd_bearish'] = (df['macd'] < df['macd_signal']) & (close < df['ma_200'])

    # 3. TRIPLE SUPERTREND
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

        trend = np.ones(len(close))
        for i in range(1, len(close)):
            if close.iloc[i] > upper.iloc[i-1]:
                trend[i] = 1
            elif close.iloc[i] < lower.iloc[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
        return pd.Series(trend, index=close.index)

    st1 = calc_supertrend(high, low, close, 12, 3.0)
    st2 = calc_supertrend(high, low, close, 10, 1.0)
    st3 = calc_supertrend(high, low, close, 11, 2.0)

    df['supertrend_bullish'] = (st1 == 1) & (st2 == 1) & (st3 == 1)
    df['supertrend_bearish'] = (st1 == -1) & (st2 == -1) & (st3 == -1)

    # 4. SUPPLY & DEMAND ZONES
    returns = close.pct_change()
    demand_low = low.rolling(20).min()
    demand_high = demand_low + (high.rolling(20).max() - low.rolling(20).min()) * 0.3
    supply_high = high.rolling(20).max()
    supply_low = supply_high - (high.rolling(20).max() - low.rolling(20).min()) * 0.3

    df['at_demand_zone'] = (close >= demand_low) & (close <= demand_high)
    df['at_supply_zone'] = (close >= supply_low) & (close <= supply_high)

    # 5. LIQUIDITY SWEEP
    lookback = 20
    recent_high = high.rolling(lookback).max().shift(1)
    recent_low = low.rolling(lookback).min().shift(1)
    df['sweep_high'] = (high > recent_high) & (close < recent_high)
    df['sweep_low'] = (low < recent_low) & (close > recent_low)

    # 6. FIBONACCI LEVELS
    swing_high = high.rolling(50).max()
    swing_low = low.rolling(50).min()
    fib_range = swing_high - swing_low
    df['fib_382'] = swing_high - 0.382 * fib_range
    df['fib_500'] = swing_high - 0.500 * fib_range
    df['fib_618'] = swing_high - 0.618 * fib_range
    tolerance = close * 0.005
    df['at_fib_level'] = (
        (abs(close - df['fib_382']) < tolerance) |
        (abs(close - df['fib_500']) < tolerance) |
        (abs(close - df['fib_618']) < tolerance)
    )

    # 7. HEIKIN ASHI
    ha_close = (open_ + high + low + close) / 4
    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = (open_.iloc[0] + close.iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    df['ha_bullish'] = ha_close > ha_open
    df['ha_bearish'] = ha_close < ha_open

    # 8. CANDLESTICK CONTEXT
    body = abs(close - open_)
    upper_wick = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_wick = pd.concat([close, open_], axis=1).min(axis=1) - low
    total_range = high - low
    df['strength_bull'] = (close > open_) & (body / total_range > 0.7)
    df['strength_bear'] = (close < open_) & (body / total_range > 0.7)
    df['reversal_bull'] = (lower_wick / total_range > 0.6)
    df['reversal_bear'] = (upper_wick / total_range > 0.6)

    # 9. DEMA + SUPERTREND (Strategy 9)
    dema_period = params.get('dema_period', 200)
    ema1 = close.ewm(span=dema_period, adjust=False).mean()
    ema2 = ema1.ewm(span=dema_period, adjust=False).mean()
    df['dema'] = 2 * ema1 - ema2
    df['above_dema'] = close > df['dema']
    df['below_dema'] = close < df['dema']

    # ATR for stops
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    return df


def calculate_confluence_score(row, weights):
    """Calculate weighted confluence score."""
    long_score = 0
    short_score = 0

    # 1. Williams Fractals (primary signal - higher weight)
    fractal_weight = weights.get('fractal', 2)
    if row['bullish_fractal'] and row['bullish_ma_align'] and row['pullback_long'] and row['above_slow']:
        long_score += fractal_weight
    if row['bearish_fractal'] and row['bearish_ma_align'] and row['pullback_short'] and row['below_slow']:
        short_score += fractal_weight

    # 2. MACD
    macd_weight = weights.get('macd', 1)
    if row['macd_bullish']:
        long_score += macd_weight
    if row['macd_bearish']:
        short_score += macd_weight

    # 3. Triple SuperTrend
    st_weight = weights.get('supertrend', 1)
    if row['supertrend_bullish']:
        long_score += st_weight
    if row['supertrend_bearish']:
        short_score += st_weight

    # 4. Supply & Demand
    sd_weight = weights.get('supply_demand', 1)
    if row['at_demand_zone'] and row['bullish_ma_align']:
        long_score += sd_weight
    if row['at_supply_zone'] and row['bearish_ma_align']:
        short_score += sd_weight

    # 5. Liquidity Sweep
    liq_weight = weights.get('liquidity', 1)
    if row['sweep_low']:
        long_score += liq_weight
    if row['sweep_high']:
        short_score += liq_weight

    # 6. Fibonacci
    fib_weight = weights.get('fib', 1)
    if row['at_fib_level']:
        if long_score > short_score:
            long_score += fib_weight
        elif short_score > long_score:
            short_score += fib_weight

    # 7. Heikin Ashi
    ha_weight = weights.get('heikin_ashi', 1)
    if row['ha_bullish']:
        long_score += ha_weight
    if row['ha_bearish']:
        short_score += ha_weight

    # 8. Candlestick pattern
    candle_weight = weights.get('candle', 1)
    if row['strength_bull'] or row['reversal_bull']:
        long_score += candle_weight
    if row['strength_bear'] or row['reversal_bear']:
        short_score += candle_weight

    # 9. DEMA
    dema_weight = weights.get('dema', 1)
    if row['above_dema']:
        long_score += dema_weight
    if row['below_dema']:
        short_score += dema_weight

    return long_score, short_score


def vectorized_confluence_backtest(df, min_confluence, mode, target_rr, atr_mult, weights):
    """Run confluence backtest."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    atr = df['atr'].values

    trades = []

    for i in range(250, len(df) - 50):
        row = df.iloc[i]
        long_score, short_score = calculate_confluence_score(row, weights)

        direction = 0
        if mode in ['both', 'long_only'] and long_score >= min_confluence and long_score > short_score:
            direction = 1
            score = long_score
        elif mode in ['both', 'short_only'] and short_score >= min_confluence and short_score > long_score:
            direction = -1
            score = short_score
        else:
            continue

        entry = close[i]
        if direction == 1:
            stop = entry - atr_mult * atr[i]
            target = entry + target_rr * (entry - stop)
        else:
            stop = entry + atr_mult * atr[i]
            target = entry - target_rr * (stop - entry)

        # Find exit
        future_high = high[i+1:min(i+51, len(df))]
        future_low = low[i+1:min(i+51, len(df))]

        if direction == 1:
            stop_idx = np.where(future_low <= stop)[0]
            target_idx = np.where(future_high >= target)[0]

            if len(stop_idx) > 0 and (len(target_idx) == 0 or stop_idx[0] <= target_idx[0]):
                pnl = (stop - entry) * CONTRACT_VALUE - COMMISSION
            elif len(target_idx) > 0:
                pnl = (target - entry) * CONTRACT_VALUE - COMMISSION
            else:
                pnl = (close[min(i+50, len(df)-1)] - entry) * CONTRACT_VALUE - COMMISSION
        else:
            stop_idx = np.where(future_high >= stop)[0]
            target_idx = np.where(future_low <= target)[0]

            if len(stop_idx) > 0 and (len(target_idx) == 0 or stop_idx[0] <= target_idx[0]):
                pnl = (entry - stop) * CONTRACT_VALUE - COMMISSION
            elif len(target_idx) > 0:
                pnl = (entry - target) * CONTRACT_VALUE - COMMISSION
            else:
                pnl = (entry - close[min(i+50, len(df)-1)]) * CONTRACT_VALUE - COMMISSION

        trades.append({'direction': 'LONG' if direction == 1 else 'SHORT', 'pnl': pnl, 'score': score})

    if not trades:
        return {'total_return': 0, 'total_trades': 0}

    trades_df = pd.DataFrame(trades)
    total_pnl = trades_df['pnl'].sum()
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    longs = trades_df[trades_df['direction'] == 'LONG']
    shorts = trades_df[trades_df['direction'] == 'SHORT']

    return {
        'total_return': total_pnl / INITIAL_CAPITAL * 100,
        'total_trades': len(trades_df),
        'win_rate': len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf'),
        'avg_pnl': trades_df['pnl'].mean(),
        'long_pnl': longs['pnl'].sum() if len(longs) > 0 else 0,
        'short_pnl': shorts['pnl'].sum() if len(shorts) > 0 else 0,
        'long_trades': len(longs),
        'short_trades': len(shorts),
        'total_pnl': total_pnl,
        'avg_score': trades_df['score'].mean(),
    }


def run_test(args):
    """Run a single test configuration."""
    df_ind, min_confluence, mode, target_rr, atr_mult, weights, timeframe, weight_name = args

    try:
        result = vectorized_confluence_backtest(df_ind, min_confluence, mode, target_rr, atr_mult, weights)

        result['min_confluence'] = min_confluence
        result['mode'] = mode
        result['target_rr'] = target_rr
        result['atr_mult'] = atr_mult
        result['timeframe'] = timeframe
        result['weights'] = weight_name

        return result
    except Exception as e:
        return {'error': str(e)}


# Weight configurations to test
WEIGHT_CONFIGS = {
    'equal': {
        'fractal': 1, 'macd': 1, 'supertrend': 1, 'supply_demand': 1,
        'liquidity': 1, 'fib': 1, 'heikin_ashi': 1, 'candle': 1, 'dema': 1
    },
    'fractal_heavy': {
        'fractal': 3, 'macd': 1, 'supertrend': 1, 'supply_demand': 1,
        'liquidity': 1, 'fib': 1, 'heikin_ashi': 1, 'candle': 1, 'dema': 1
    },
    'trend_focus': {
        'fractal': 2, 'macd': 2, 'supertrend': 2, 'supply_demand': 1,
        'liquidity': 1, 'fib': 1, 'heikin_ashi': 1, 'candle': 1, 'dema': 2
    },
    'smc_focus': {  # Smart Money Concepts
        'fractal': 2, 'macd': 1, 'supertrend': 1, 'supply_demand': 2,
        'liquidity': 2, 'fib': 2, 'heikin_ashi': 1, 'candle': 1, 'dema': 1
    },
}

INDICATOR_PARAMS = [
    {'ma_fast': 20, 'ma_medium': 50, 'ma_slow': 100, 'fractal_period': 3},
    {'ma_fast': 10, 'ma_medium': 30, 'ma_slow': 80, 'fractal_period': 2},
]

MIN_CONFLUENCES = [2, 3, 4, 5, 6]
MODES = ['long_only', 'short_only', 'both']
TARGET_RRS = [1.5, 2.0, 2.5]
ATR_MULTS = [1.5, 2.0, 2.5]


def main():
    log("=" * 70)
    log("CONFLUENCE STRATEGY COMPREHENSIVE BACKTEST")
    log(f"CPUs: {cpu_count()}")
    log("=" * 70)

    # Load data for both timeframes
    df_1m = load_data(1)
    df_5m = load_data(5)

    # Pre-calculate indicators for each param set
    log("Calculating indicators...")

    indicator_data = {}
    for tf, df in [(1, df_1m), (5, df_5m)]:
        for params in INDICATOR_PARAMS:
            key = f"{tf}m_{params['ma_fast']}_{params['ma_medium']}_{params['ma_slow']}"
            indicator_data[key] = calculate_all_indicators(df, params)
            log(f"  Calculated: {key}")

    # Generate all test configurations
    tasks = []

    for tf in [1, 5]:
        for params in INDICATOR_PARAMS:
            key = f"{tf}m_{params['ma_fast']}_{params['ma_medium']}_{params['ma_slow']}"
            df_ind = indicator_data[key]

            for weight_name, weights in WEIGHT_CONFIGS.items():
                for min_conf in MIN_CONFLUENCES:
                    for mode in MODES:
                        for target_rr in TARGET_RRS:
                            for atr_mult in ATR_MULTS:
                                tasks.append((
                                    df_ind, min_conf, mode, target_rr, atr_mult,
                                    weights, tf, weight_name
                                ))

    log(f"Total configurations: {len(tasks)}")
    log("Starting parallel backtests...")
    start = datetime.now()

    with Pool(cpu_count()) as pool:
        results = list(pool.imap_unordered(run_test, tasks, chunksize=5))

    elapsed = (datetime.now() - start).total_seconds()
    log(f"Completed in {elapsed:.1f}s ({len(results)/elapsed:.1f} tests/sec)")

    # Filter valid results
    valid = [r for r in results if 'error' not in r and r['total_trades'] >= 50]
    log(f"Valid strategies: {len(valid)}")

    # Convert to DataFrame and sort
    df_results = pd.DataFrame(valid)

    # Calculate score
    df_results['score'] = (
        df_results['profit_factor'].clip(0, 5) * 50 +
        df_results['win_rate'] +
        df_results['total_return'].clip(-100, 500) * 0.3
    )

    df_results = df_results.sort_values('score', ascending=False)

    log(f"\n{'='*70}")
    log("TOP 20 CONFLUENCE STRATEGIES")
    log(f"{'='*70}")

    for i, row in df_results.head(20).iterrows():
        log(f"\n#{df_results.index.get_loc(i)+1}: {row['timeframe']}m | {row['mode']} | conf>={row['min_confluence']}")
        log(f"   Weights: {row['weights']}")
        log(f"   Return: {row['total_return']:.1f}% | Win Rate: {row['win_rate']:.1f}%")
        log(f"   PF: {row['profit_factor']:.2f} | Trades: {row['total_trades']}")
        log(f"   Long P&L: ${row['long_pnl']:.0f} | Short P&L: ${row['short_pnl']:.0f}")
        log(f"   Avg Score: {row['avg_score']:.1f} | Target R:R: {row['target_rr']} | ATR: {row['atr_mult']}")

    # Best profitable
    profitable = df_results[df_results['total_return'] > 0]
    log(f"\n{'='*70}")
    log(f"PROFITABLE CONFIGURATIONS: {len(profitable)}")
    log(f"{'='*70}")

    if len(profitable) > 0:
        best = profitable.iloc[0]
        log(f"\nBEST CONFLUENCE STRATEGY:")
        log(f"  Timeframe: {best['timeframe']}m")
        log(f"  Mode: {best['mode']}")
        log(f"  Min Confluence: {best['min_confluence']}")
        log(f"  Weights: {best['weights']}")
        log(f"  Target R:R: {best['target_rr']}")
        log(f"  ATR Mult: {best['atr_mult']}")
        log(f"  Return: {best['total_return']:.1f}%")
        log(f"  Win Rate: {best['win_rate']:.1f}%")
        log(f"  Profit Factor: {best['profit_factor']:.2f}")
        log(f"  Total Trades: {best['total_trades']}")
        log(f"  Long P&L: ${best['long_pnl']:.2f}")
        log(f"  Short P&L: ${best['short_pnl']:.2f}")
        log(f"  Avg Confluence Score: {best['avg_score']:.1f}")

    # Compare with Williams Fractals only
    log(f"\n{'='*70}")
    log("COMPARISON: Best Confluence vs Williams Fractals Only")
    log(f"{'='*70}")

    # Save results
    output_dir = Path("/workspace/results")
    output_dir.mkdir(exist_ok=True)

    df_results.to_csv(output_dir / "confluence_all_results.csv", index=False)
    profitable.to_csv(output_dir / "confluence_profitable.csv", index=False)

    if len(profitable) > 0:
        best_config = profitable.head(20).to_dict('records')
        with open(output_dir / "confluence_top_20.json", 'w') as f:
            json.dump(best_config, f, indent=2, default=str)

    log(f"\nResults saved to {output_dir}/")
    log("=" * 70)


if __name__ == "__main__":
    main()
