#!/usr/bin/env python3
"""
Fast Vectorized Backtesting for RunPod

Uses numpy vectorization instead of loops for 100x speedup.
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
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


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
# VECTORIZED SIGNAL GENERATORS
# ============================================================

def williams_fractals_signals(df, period=2, ma_fast=20, ma_medium=50, ma_slow=100):
    """Vectorized Williams Fractals."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # Rolling MAs
    sma_fast = pd.Series(close).rolling(ma_fast).mean().values
    sma_medium = pd.Series(close).rolling(ma_medium).mean().values
    sma_slow = pd.Series(close).rolling(ma_slow).mean().values

    # Fractals using rolling min/max
    low_series = pd.Series(low)
    high_series = pd.Series(high)
    window = 2 * period + 1

    rolling_low_min = low_series.rolling(window, center=True).min().values
    rolling_high_max = high_series.rolling(window, center=True).max().values

    bullish_frac = low == rolling_low_min
    bearish_frac = high == rolling_high_max

    # Conditions
    bull_align = (sma_fast > sma_medium) & (sma_medium > sma_slow)
    bear_align = (sma_fast < sma_medium) & (sma_medium < sma_slow)

    long_sig = bullish_frac & bull_align & (close < sma_fast) & (close > sma_slow)
    short_sig = bearish_frac & bear_align & (close > sma_fast) & (close < sma_slow)

    return long_sig, short_sig


def macd_signals(df, fast=12, slow=26, signal=9, ma_period=200):
    """Vectorized MACD + 200 MA."""
    close = pd.Series(df['close'].values)

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    ma_200 = close.rolling(ma_period).mean()

    macd_arr = macd.values
    signal_arr = signal_line.values
    close_arr = close.values
    ma_200_arr = ma_200.values

    # Cross detection
    cross_up = (macd_arr > signal_arr) & (np.roll(macd_arr, 1) <= np.roll(signal_arr, 1))
    cross_down = (macd_arr < signal_arr) & (np.roll(macd_arr, 1) >= np.roll(signal_arr, 1))

    long_sig = cross_up & (macd_arr < 0) & (close_arr > ma_200_arr)
    short_sig = cross_down & (macd_arr > 0) & (close_arr < ma_200_arr)

    return long_sig, short_sig


def triple_supertrend_signals(df, atr1=12, mult1=3.0, atr2=10, mult2=1.0, atr3=11, mult3=2.0):
    """Vectorized Triple SuperTrend."""
    def supertrend(close, high, low, atr_period, mult):
        hl2 = (high + low) / 2
        tr = np.maximum(high - low, np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        ))
        atr = pd.Series(tr).rolling(atr_period).mean().values

        upper = hl2 + mult * atr
        lower = hl2 - mult * atr

        trend = np.ones(len(close))
        for i in range(1, len(close)):
            if close[i] > upper[i-1]:
                trend[i] = 1
            elif close[i] < lower[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
        return trend

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    st1 = supertrend(close, high, low, atr1, mult1)
    st2 = supertrend(close, high, low, atr2, mult2)
    st3 = supertrend(close, high, low, atr3, mult3)

    all_bull = (st1 == 1) & (st2 == 1) & (st3 == 1)
    all_bear = (st1 == -1) & (st2 == -1) & (st3 == -1)

    # Signal on change
    long_sig = all_bull & ~np.roll(all_bull, 1)
    short_sig = all_bear & ~np.roll(all_bear, 1)

    return long_sig, short_sig


def liquidity_sweep_signals(df, lookback=20, confirm=2):
    """Vectorized Liquidity Sweep."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values

    recent_high = pd.Series(high).rolling(lookback).max().shift(1).values
    recent_low = pd.Series(low).rolling(lookback).min().shift(1).values

    sweep_high = (high > recent_high) & (close < recent_high)
    sweep_low = (low < recent_low) & (close > recent_low)

    # Confirmation candles
    bullish = close > open_
    bearish = close < open_

    bull_confirm = pd.Series(bullish.astype(int)).rolling(confirm).sum().values == confirm
    bear_confirm = pd.Series(bearish.astype(int)).rolling(confirm).sum().values == confirm

    long_sig = np.roll(sweep_low, 1) & bull_confirm
    short_sig = np.roll(sweep_high, 1) & bear_confirm

    return long_sig, short_sig


# ============================================================
# VECTORIZED BACKTEST
# ============================================================

def vectorized_backtest(df, long_signals, short_signals, mode='both',
                        target_rr=1.5, atr_mult=2.0):
    """
    Super-fast vectorized backtest.

    Instead of simulating bar-by-bar, we:
    1. Find all signal bars
    2. Calculate potential P&L for each signal
    3. Apply filters and sum results
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # ATR for stops
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1))
    ))
    atr = pd.Series(tr).rolling(14).mean().values

    trades = []

    # Process long signals
    if mode in ['both', 'long_only']:
        long_idx = np.where(long_signals)[0]
        for i in long_idx:
            if i + 50 >= len(df):  # Need room for trade
                continue

            entry = close[i]
            stop = entry - atr_mult * atr[i]
            target = entry + target_rr * (entry - stop)

            # Find exit within next 50 bars
            future_high = high[i+1:min(i+51, len(df))]
            future_low = low[i+1:min(i+51, len(df))]

            # Check if stop hit first
            stop_idx = np.where(future_low <= stop)[0]
            target_idx = np.where(future_high >= target)[0]

            if len(stop_idx) > 0 and (len(target_idx) == 0 or stop_idx[0] <= target_idx[0]):
                pnl = (stop - entry) * CONTRACT_VALUE - COMMISSION
                trades.append({'direction': 'LONG', 'pnl': pnl})
            elif len(target_idx) > 0:
                pnl = (target - entry) * CONTRACT_VALUE - COMMISSION
                trades.append({'direction': 'LONG', 'pnl': pnl})
            else:
                # Exit at end of window
                exit_price = close[min(i+50, len(df)-1)]
                pnl = (exit_price - entry) * CONTRACT_VALUE - COMMISSION
                trades.append({'direction': 'LONG', 'pnl': pnl})

    # Process short signals
    if mode in ['both', 'short_only']:
        short_idx = np.where(short_signals)[0]
        for i in short_idx:
            if i + 50 >= len(df):
                continue

            entry = close[i]
            stop = entry + atr_mult * atr[i]
            target = entry - target_rr * (stop - entry)

            future_high = high[i+1:min(i+51, len(df))]
            future_low = low[i+1:min(i+51, len(df))]

            stop_idx = np.where(future_high >= stop)[0]
            target_idx = np.where(future_low <= target)[0]

            if len(stop_idx) > 0 and (len(target_idx) == 0 or stop_idx[0] <= target_idx[0]):
                pnl = (entry - stop) * CONTRACT_VALUE - COMMISSION
                trades.append({'direction': 'SHORT', 'pnl': pnl})
            elif len(target_idx) > 0:
                pnl = (entry - target) * CONTRACT_VALUE - COMMISSION
                trades.append({'direction': 'SHORT', 'pnl': pnl})
            else:
                exit_price = close[min(i+50, len(df)-1)]
                pnl = (entry - exit_price) * CONTRACT_VALUE - COMMISSION
                trades.append({'direction': 'SHORT', 'pnl': pnl})

    if not trades:
        return {'total_return': 0, 'total_trades': 0, 'win_rate': 0,
                'profit_factor': 0, 'long_pnl': 0, 'short_pnl': 0}

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
    }


# ============================================================
# TEST CONFIGURATIONS
# ============================================================

STRATEGIES = {
    'fractals': williams_fractals_signals,
    'macd': macd_signals,
    'triple_supertrend': triple_supertrend_signals,
    'liquidity_sweep': liquidity_sweep_signals,
}

PARAM_GRID = {
    'fractals': [
        {'period': 2, 'ma_fast': 20, 'ma_medium': 50, 'ma_slow': 100},
        {'period': 2, 'ma_fast': 10, 'ma_medium': 30, 'ma_slow': 80},
        {'period': 3, 'ma_fast': 20, 'ma_medium': 50, 'ma_slow': 100},
    ],
    'macd': [
        {'fast': 12, 'slow': 26, 'signal': 9, 'ma_period': 200},
        {'fast': 8, 'slow': 21, 'signal': 7, 'ma_period': 100},
        {'fast': 12, 'slow': 26, 'signal': 9, 'ma_period': 100},
    ],
    'triple_supertrend': [
        {'atr1': 12, 'mult1': 3.0, 'atr2': 10, 'mult2': 1.0, 'atr3': 11, 'mult3': 2.0},
        {'atr1': 10, 'mult1': 2.5, 'atr2': 8, 'mult2': 1.5, 'atr3': 9, 'mult3': 1.5},
    ],
    'liquidity_sweep': [
        {'lookback': 20, 'confirm': 2},
        {'lookback': 30, 'confirm': 2},
        {'lookback': 20, 'confirm': 3},
    ],
}

MODES = ['long_only', 'short_only', 'both']
TARGET_RRS = [1.5, 2.0, 2.5]
ATR_MULTS = [1.5, 2.0, 2.5]


def run_test(args):
    """Run a single test configuration."""
    df, strategy_name, params, mode, target_rr, atr_mult, timeframe = args

    try:
        strategy_fn = STRATEGIES[strategy_name]
        long_sig, short_sig = strategy_fn(df, **params)
        result = vectorized_backtest(df, long_sig, short_sig, mode, target_rr, atr_mult)

        result['strategy'] = strategy_name
        result['params'] = str(params)
        result['mode'] = mode
        result['target_rr'] = target_rr
        result['atr_mult'] = atr_mult
        result['timeframe'] = timeframe

        return result
    except Exception as e:
        return {'error': str(e), 'strategy': strategy_name}


def main():
    log("=" * 70)
    log("FAST VECTORIZED BACKTESTING")
    log(f"CPUs: {cpu_count()}")
    log("=" * 70)

    # Load data for both timeframes
    df_1m = load_data(1)
    df_5m = load_data(5)

    # Generate all test configurations
    tasks = []

    for strategy_name in STRATEGIES.keys():
        params_list = PARAM_GRID.get(strategy_name, [{}])
        for params in params_list:
            for mode in MODES:
                for target_rr in TARGET_RRS:
                    for atr_mult in ATR_MULTS:
                        tasks.append((df_1m, strategy_name, params, mode, target_rr, atr_mult, 1))
                        tasks.append((df_5m, strategy_name, params, mode, target_rr, atr_mult, 5))

    log(f"Total configurations: {len(tasks)}")

    # Run in parallel
    log("Starting parallel backtests...")
    start = datetime.now()

    with Pool(cpu_count()) as pool:
        results = list(pool.imap_unordered(run_test, tasks, chunksize=10))

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
        df_results['total_return'].clip(-100, 200) * 0.5
    )

    # Sort by score
    df_results = df_results.sort_values('score', ascending=False)

    log(f"\n{'='*70}")
    log("TOP 20 STRATEGIES")
    log(f"{'='*70}")

    for i, row in df_results.head(20).iterrows():
        log(f"\n#{df_results.index.get_loc(i)+1}: {row['strategy']} @ {row['timeframe']}m ({row['mode']})")
        log(f"   Return: {row['total_return']:.1f}% | Win Rate: {row['win_rate']:.1f}%")
        log(f"   PF: {row['profit_factor']:.2f} | Trades: {row['total_trades']}")
        log(f"   Long P&L: ${row['long_pnl']:.0f} | Short P&L: ${row['short_pnl']:.0f}")
        log(f"   Target R:R: {row['target_rr']} | ATR Mult: {row['atr_mult']}")

    # Find best PROFITABLE strategies
    profitable = df_results[df_results['total_return'] > 0]
    log(f"\n{'='*70}")
    log(f"PROFITABLE STRATEGIES: {len(profitable)}")
    log(f"{'='*70}")

    if len(profitable) > 0:
        best = profitable.iloc[0]
        log(f"\nBEST STRATEGY:")
        log(f"  Strategy: {best['strategy']}")
        log(f"  Timeframe: {best['timeframe']}m")
        log(f"  Mode: {best['mode']}")
        log(f"  Params: {best['params']}")
        log(f"  Target R:R: {best['target_rr']}")
        log(f"  ATR Mult: {best['atr_mult']}")
        log(f"  Return: {best['total_return']:.1f}%")
        log(f"  Win Rate: {best['win_rate']:.1f}%")
        log(f"  Profit Factor: {best['profit_factor']:.2f}")
        log(f"  Total Trades: {best['total_trades']}")
        log(f"  Long P&L: ${best['long_pnl']:.2f}")
        log(f"  Short P&L: ${best['short_pnl']:.2f}")

    # Save results
    output_dir = Path("/workspace/results")
    output_dir.mkdir(exist_ok=True)

    df_results.to_csv(output_dir / "all_results.csv", index=False)
    profitable.to_csv(output_dir / "profitable_strategies.csv", index=False)

    if len(profitable) > 0:
        best_config = profitable.head(10).to_dict('records')
        with open(output_dir / "top_10_strategies.json", 'w') as f:
            json.dump(best_config, f, indent=2, default=str)

    log(f"\nResults saved to {output_dir}/")
    log("=" * 70)


if __name__ == "__main__":
    main()
