#!/usr/bin/env python3
"""
FAST Vectorized Confluence Strategy Backtest

Uses numpy arrays for ALL operations - no per-bar loops.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')


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


def calculate_indicators_vectorized(df, params):
    """Calculate ALL indicators using pure vectorization."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values
    n = len(close)

    ma_fast = params.get('ma_fast', 20)
    ma_medium = params.get('ma_medium', 50)
    ma_slow = params.get('ma_slow', 100)
    fractal_period = params.get('fractal_period', 3)

    # Convert to pandas for rolling operations, then back to numpy
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)

    # 1. WILLIAMS FRACTALS + MA ALIGNMENT
    sma_fast = close_s.rolling(ma_fast).mean().values
    sma_medium = close_s.rolling(ma_medium).mean().values
    sma_slow = close_s.rolling(ma_slow).mean().values

    bullish_ma_align = (sma_fast > sma_medium) & (sma_medium > sma_slow)
    bearish_ma_align = (sma_fast < sma_medium) & (sma_medium < sma_slow)

    window = 2 * fractal_period + 1
    rolling_low_min = low_s.rolling(window, center=True).min().values
    rolling_high_max = high_s.rolling(window, center=True).max().values
    bullish_fractal = (low == rolling_low_min)
    bearish_fractal = (high == rolling_high_max)

    pullback_long = close < sma_fast
    pullback_short = close > sma_fast
    above_slow = close > sma_slow
    below_slow = close < sma_slow

    # 2. MACD + 200 MA
    ema_fast = close_s.ewm(span=12, adjust=False).mean().values
    ema_slow = close_s.ewm(span=26, adjust=False).mean().values
    macd = ema_fast - ema_slow
    macd_signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    ma_200 = close_s.rolling(200).mean().values

    macd_bullish = (macd > macd_signal) & (close > ma_200)
    macd_bearish = (macd < macd_signal) & (close < ma_200)

    # 3. TRIPLE SUPERTREND (simplified vectorized version)
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1))
    ))
    atr = pd.Series(tr).rolling(12).mean().values
    hl2 = (high + low) / 2

    # Simplified SuperTrend: just check if close > upper or < lower band
    upper = hl2 + 3.0 * atr
    lower = hl2 - 3.0 * atr
    supertrend_bullish = close > lower
    supertrend_bearish = close < upper

    # 4. SUPPLY & DEMAND (simplified)
    demand_low = low_s.rolling(20).min().values
    supply_high = high_s.rolling(20).max().values
    range_size = supply_high - demand_low
    demand_high = demand_low + range_size * 0.3
    supply_low = supply_high - range_size * 0.3

    at_demand_zone = (close >= demand_low) & (close <= demand_high)
    at_supply_zone = (close >= supply_low) & (close <= supply_high)

    # 5. LIQUIDITY SWEEP
    recent_high = high_s.rolling(20).max().shift(1).values
    recent_low = low_s.rolling(20).min().shift(1).values
    sweep_high = (high > recent_high) & (close < recent_high)
    sweep_low = (low < recent_low) & (close > recent_low)

    # 6. FIBONACCI (at 61.8% retracement)
    swing_high = high_s.rolling(50).max().values
    swing_low = low_s.rolling(50).min().values
    fib_618 = swing_high - 0.618 * (swing_high - swing_low)
    tolerance = close * 0.005
    at_fib_level = np.abs(close - fib_618) < tolerance

    # 7. HEIKIN ASHI trend
    ha_close = (open_ + high + low + close) / 4
    ha_open = np.zeros(n)
    ha_open[0] = (open_[0] + close[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_bullish = ha_close > ha_open
    ha_bearish = ha_close < ha_open

    # 8. CANDLESTICK PATTERNS
    body = np.abs(close - open_)
    total_range = high - low + 0.0001  # Avoid div by zero
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low

    strength_bull = (close > open_) & (body / total_range > 0.7)
    strength_bear = (close < open_) & (body / total_range > 0.7)
    reversal_bull = lower_wick / total_range > 0.6
    reversal_bear = upper_wick / total_range > 0.6

    # 9. DEMA
    ema1 = close_s.ewm(span=200, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=200, adjust=False).mean().values
    dema = 2 * ema1 - ema2
    above_dema = close > dema
    below_dema = close < dema

    # ATR for stops
    atr_14 = pd.Series(tr).rolling(14).mean().values

    return {
        # Fractals
        'bullish_fractal': bullish_fractal,
        'bearish_fractal': bearish_fractal,
        'bullish_ma_align': bullish_ma_align,
        'bearish_ma_align': bearish_ma_align,
        'pullback_long': pullback_long,
        'pullback_short': pullback_short,
        'above_slow': above_slow,
        'below_slow': below_slow,
        # MACD
        'macd_bullish': macd_bullish,
        'macd_bearish': macd_bearish,
        # SuperTrend
        'supertrend_bullish': supertrend_bullish,
        'supertrend_bearish': supertrend_bearish,
        # Supply/Demand
        'at_demand_zone': at_demand_zone,
        'at_supply_zone': at_supply_zone,
        # Liquidity
        'sweep_low': sweep_low,
        'sweep_high': sweep_high,
        # Fib
        'at_fib_level': at_fib_level,
        # Heikin Ashi
        'ha_bullish': ha_bullish,
        'ha_bearish': ha_bearish,
        # Candles
        'strength_bull': strength_bull,
        'strength_bear': strength_bear,
        'reversal_bull': reversal_bull,
        'reversal_bear': reversal_bear,
        # DEMA
        'above_dema': above_dema,
        'below_dema': below_dema,
        # Prices
        'close': close,
        'high': high,
        'low': low,
        'atr': atr_14,
    }


def vectorized_confluence_signals(ind, weights, min_confluence):
    """
    Calculate confluence scores for ALL bars at once using vectorized ops.
    """
    n = len(ind['close'])

    # Long score components (all vectorized)
    long_scores = np.zeros(n)
    short_scores = np.zeros(n)

    # 1. Williams Fractals (weight: fractal)
    fractal_w = weights.get('fractal', 2)
    fractal_long = (ind['bullish_fractal'] & ind['bullish_ma_align'] &
                    ind['pullback_long'] & ind['above_slow'])
    fractal_short = (ind['bearish_fractal'] & ind['bearish_ma_align'] &
                     ind['pullback_short'] & ind['below_slow'])
    long_scores += fractal_long.astype(float) * fractal_w
    short_scores += fractal_short.astype(float) * fractal_w

    # 2. MACD
    macd_w = weights.get('macd', 1)
    long_scores += ind['macd_bullish'].astype(float) * macd_w
    short_scores += ind['macd_bearish'].astype(float) * macd_w

    # 3. SuperTrend
    st_w = weights.get('supertrend', 1)
    long_scores += ind['supertrend_bullish'].astype(float) * st_w
    short_scores += ind['supertrend_bearish'].astype(float) * st_w

    # 4. Supply/Demand
    sd_w = weights.get('supply_demand', 1)
    long_scores += (ind['at_demand_zone'] & ind['bullish_ma_align']).astype(float) * sd_w
    short_scores += (ind['at_supply_zone'] & ind['bearish_ma_align']).astype(float) * sd_w

    # 5. Liquidity
    liq_w = weights.get('liquidity', 1)
    long_scores += ind['sweep_low'].astype(float) * liq_w
    short_scores += ind['sweep_high'].astype(float) * liq_w

    # 6. Fib (adds to whichever is stronger)
    fib_w = weights.get('fib', 1)
    fib_long = ind['at_fib_level'] & (long_scores > short_scores)
    fib_short = ind['at_fib_level'] & (short_scores > long_scores)
    long_scores += fib_long.astype(float) * fib_w
    short_scores += fib_short.astype(float) * fib_w

    # 7. Heikin Ashi
    ha_w = weights.get('heikin_ashi', 1)
    long_scores += ind['ha_bullish'].astype(float) * ha_w
    short_scores += ind['ha_bearish'].astype(float) * ha_w

    # 8. Candles
    candle_w = weights.get('candle', 1)
    long_scores += (ind['strength_bull'] | ind['reversal_bull']).astype(float) * candle_w
    short_scores += (ind['strength_bear'] | ind['reversal_bear']).astype(float) * candle_w

    # 9. DEMA
    dema_w = weights.get('dema', 1)
    long_scores += ind['above_dema'].astype(float) * dema_w
    short_scores += ind['below_dema'].astype(float) * dema_w

    # Determine signals based on min confluence
    long_signals = (long_scores >= min_confluence) & (long_scores > short_scores)
    short_signals = (short_scores >= min_confluence) & (short_scores > long_scores)

    return long_signals, short_signals, long_scores, short_scores


def vectorized_backtest(ind, long_signals, short_signals, mode, target_rr, atr_mult):
    """Fast vectorized backtest."""
    close = ind['close']
    high = ind['high']
    low = ind['low']
    atr = ind['atr']

    trades = []

    # Process long signals
    if mode in ['both', 'long_only']:
        long_idx = np.where(long_signals)[0]
        for i in long_idx:
            if i < 250 or i + 50 >= len(close):
                continue

            entry = close[i]
            stop = entry - atr_mult * atr[i]
            target = entry + target_rr * (entry - stop)

            future_high = high[i+1:min(i+51, len(close))]
            future_low = low[i+1:min(i+51, len(close))]

            stop_idx = np.where(future_low <= stop)[0]
            target_idx = np.where(future_high >= target)[0]

            if len(stop_idx) > 0 and (len(target_idx) == 0 or stop_idx[0] <= target_idx[0]):
                pnl = (stop - entry) * CONTRACT_VALUE - COMMISSION
            elif len(target_idx) > 0:
                pnl = (target - entry) * CONTRACT_VALUE - COMMISSION
            else:
                pnl = (close[min(i+50, len(close)-1)] - entry) * CONTRACT_VALUE - COMMISSION

            trades.append({'direction': 'LONG', 'pnl': pnl})

    # Process short signals
    if mode in ['both', 'short_only']:
        short_idx = np.where(short_signals)[0]
        for i in short_idx:
            if i < 250 or i + 50 >= len(close):
                continue

            entry = close[i]
            stop = entry + atr_mult * atr[i]
            target = entry - target_rr * (stop - entry)

            future_high = high[i+1:min(i+51, len(close))]
            future_low = low[i+1:min(i+51, len(close))]

            stop_idx = np.where(future_high >= stop)[0]
            target_idx = np.where(future_low <= target)[0]

            if len(stop_idx) > 0 and (len(target_idx) == 0 or stop_idx[0] <= target_idx[0]):
                pnl = (entry - stop) * CONTRACT_VALUE - COMMISSION
            elif len(target_idx) > 0:
                pnl = (entry - target) * CONTRACT_VALUE - COMMISSION
            else:
                pnl = (entry - close[min(i+50, len(close)-1)]) * CONTRACT_VALUE - COMMISSION

            trades.append({'direction': 'SHORT', 'pnl': pnl})

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
    }


def run_test(args):
    """Run a single test configuration."""
    ind, min_confluence, mode, target_rr, atr_mult, weights, timeframe, weight_name = args

    try:
        long_sig, short_sig, _, _ = vectorized_confluence_signals(ind, weights, min_confluence)
        result = vectorized_backtest(ind, long_sig, short_sig, mode, target_rr, atr_mult)

        result['min_confluence'] = min_confluence
        result['mode'] = mode
        result['target_rr'] = target_rr
        result['atr_mult'] = atr_mult
        result['timeframe'] = timeframe
        result['weights'] = weight_name

        return result
    except Exception as e:
        return {'error': str(e)}


# Weight configurations
WEIGHT_CONFIGS = {
    'equal': {'fractal': 1, 'macd': 1, 'supertrend': 1, 'supply_demand': 1,
              'liquidity': 1, 'fib': 1, 'heikin_ashi': 1, 'candle': 1, 'dema': 1},
    'fractal_heavy': {'fractal': 3, 'macd': 1, 'supertrend': 1, 'supply_demand': 1,
                      'liquidity': 1, 'fib': 1, 'heikin_ashi': 1, 'candle': 1, 'dema': 1},
    'trend_focus': {'fractal': 2, 'macd': 2, 'supertrend': 2, 'supply_demand': 1,
                    'liquidity': 1, 'fib': 1, 'heikin_ashi': 1, 'candle': 1, 'dema': 2},
    'smc_focus': {'fractal': 2, 'macd': 1, 'supertrend': 1, 'supply_demand': 2,
                  'liquidity': 2, 'fib': 2, 'heikin_ashi': 1, 'candle': 1, 'dema': 1},
    'minimal': {'fractal': 2, 'macd': 1, 'supertrend': 1, 'supply_demand': 0,
                'liquidity': 0, 'fib': 0, 'heikin_ashi': 1, 'candle': 0, 'dema': 0},
}

INDICATOR_PARAMS = [
    {'ma_fast': 20, 'ma_medium': 50, 'ma_slow': 100, 'fractal_period': 3},
    {'ma_fast': 10, 'ma_medium': 30, 'ma_slow': 80, 'fractal_period': 2},
]

MIN_CONFLUENCES = [2, 3, 4, 5, 6, 7]
MODES = ['long_only', 'short_only', 'both']
TARGET_RRS = [1.5, 2.0, 2.5]
ATR_MULTS = [1.5, 2.0, 2.5]


def main():
    log("=" * 70)
    log("FAST VECTORIZED CONFLUENCE BACKTEST")
    log(f"CPUs: {cpu_count()}")
    log("=" * 70)

    # Load data
    df_1m = load_data(1)
    df_5m = load_data(5)

    # Pre-calculate indicators
    log("Calculating indicators (vectorized)...")

    indicator_data = {}
    for tf, df in [(1, df_1m), (5, df_5m)]:
        for params in INDICATOR_PARAMS:
            key = f"{tf}m_{params['ma_fast']}_{params['ma_medium']}_{params['ma_slow']}"
            indicator_data[key] = calculate_indicators_vectorized(df, params)
            log(f"  Calculated: {key}")

    # Generate test configurations
    tasks = []

    for tf in [1, 5]:
        for params in INDICATOR_PARAMS:
            key = f"{tf}m_{params['ma_fast']}_{params['ma_medium']}_{params['ma_slow']}"
            ind = indicator_data[key]

            for weight_name, weights in WEIGHT_CONFIGS.items():
                for min_conf in MIN_CONFLUENCES:
                    for mode in MODES:
                        for target_rr in TARGET_RRS:
                            for atr_mult in ATR_MULTS:
                                tasks.append((
                                    ind, min_conf, mode, target_rr, atr_mult,
                                    weights, tf, weight_name
                                ))

    log(f"Total configurations: {len(tasks)}")
    log("Starting parallel backtests...")
    start = datetime.now()

    with Pool(cpu_count()) as pool:
        results = list(pool.imap_unordered(run_test, tasks, chunksize=10))

    elapsed = (datetime.now() - start).total_seconds()
    log(f"Completed in {elapsed:.1f}s ({len(results)/elapsed:.1f} tests/sec)")

    # Filter valid results
    valid = [r for r in results if 'error' not in r and r['total_trades'] >= 30]
    log(f"Valid strategies: {len(valid)}")

    # Sort by score
    df_results = pd.DataFrame(valid)

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
        log(f"   Target R:R: {row['target_rr']} | ATR: {row['atr_mult']}")

    # Profitable count
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
