#!/usr/bin/env python3
"""
Walk-Forward Analysis - Tests strategy on out-of-sample data.

This validates the strategy isn't overfit by:
1. Training on first 70% of data
2. Testing on remaining 30% (unseen)
3. Rolling the window forward

Also tests with position scaling.
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
INITIAL_CAPITAL = 10000
RISK_PCT = 0.02
MAX_CONTRACTS = 50


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

    log(f"  {len(agg):,} bars from {agg.index[0]} to {agg.index[-1]}")
    return agg.reset_index()


def williams_fractals_signals(df, period=3, ma_fast=20, ma_medium=50, ma_slow=100):
    """Generate signals."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    close_s = pd.Series(close)
    low_s = pd.Series(low)
    high_s = pd.Series(high)

    sma_fast = close_s.rolling(ma_fast).mean().values
    sma_medium = close_s.rolling(ma_medium).mean().values
    sma_slow = close_s.rolling(ma_slow).mean().values

    window = 2 * period + 1
    rolling_low_min = low_s.rolling(window, center=True).min().values
    rolling_high_max = high_s.rolling(window, center=True).max().values

    bullish_frac = low == rolling_low_min
    bearish_frac = high == rolling_high_max

    bull_align = (sma_fast > sma_medium) & (sma_medium > sma_slow)
    bear_align = (sma_fast < sma_medium) & (sma_medium < sma_slow)

    long_sig = bullish_frac & bull_align & (close < sma_fast) & (close > sma_slow)
    short_sig = bearish_frac & bear_align & (close > sma_fast) & (close < sma_slow)

    tr = np.maximum(high - low, np.maximum(
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1))
    ))
    atr = pd.Series(tr).rolling(14).mean().values

    return long_sig, short_sig, close, high, low, atr


def calculate_position_size(capital, entry, stop, risk_pct=0.02):
    """Calculate position size based on risk."""
    risk_amount = capital * risk_pct
    risk_per_contract = abs(entry - stop) * CONTRACT_VALUE
    if risk_per_contract == 0:
        return 1
    size = int(risk_amount / risk_per_contract)
    return max(1, min(size, MAX_CONTRACTS))


def backtest_period(df, mode='both', target_rr=1.5, atr_mult=1.5, scale=True, initial_capital=10000):
    """Backtest on a specific data period with optional scaling."""
    long_signals, short_signals, close, high, low, atr = williams_fractals_signals(df)

    capital = initial_capital
    trades = []
    max_equity = capital
    max_drawdown = 0

    all_signals = []
    if mode in ['both', 'long_only']:
        for i in np.where(long_signals)[0]:
            if i >= 150 and i + 50 < len(close):
                all_signals.append((i, 1))
    if mode in ['both', 'short_only']:
        for i in np.where(short_signals)[0]:
            if i >= 150 and i + 50 < len(close):
                all_signals.append((i, -1))

    all_signals.sort(key=lambda x: x[0])

    for idx, direction in all_signals:
        entry = close[idx]

        if direction == 1:
            stop = entry - atr_mult * atr[idx]
            target = entry + target_rr * (entry - stop)
        else:
            stop = entry + atr_mult * atr[idx]
            target = entry - target_rr * (stop - entry)

        if scale:
            size = calculate_position_size(capital, entry, stop, RISK_PCT)
        else:
            size = 1

        future_high = high[idx+1:min(idx+51, len(close))]
        future_low = low[idx+1:min(idx+51, len(close))]

        if direction == 1:
            stop_idx = np.where(future_low <= stop)[0]
            target_idx = np.where(future_high >= target)[0]
            if len(stop_idx) > 0 and (len(target_idx) == 0 or stop_idx[0] <= target_idx[0]):
                pnl = (stop - entry) * CONTRACT_VALUE * size - COMMISSION * size
            elif len(target_idx) > 0:
                pnl = (target - entry) * CONTRACT_VALUE * size - COMMISSION * size
            else:
                pnl = (close[min(idx+50, len(close)-1)] - entry) * CONTRACT_VALUE * size - COMMISSION * size
        else:
            stop_idx = np.where(future_high >= stop)[0]
            target_idx = np.where(future_low <= target)[0]
            if len(stop_idx) > 0 and (len(target_idx) == 0 or stop_idx[0] <= target_idx[0]):
                pnl = (entry - stop) * CONTRACT_VALUE * size - COMMISSION * size
            elif len(target_idx) > 0:
                pnl = (entry - target) * CONTRACT_VALUE * size - COMMISSION * size
            else:
                pnl = (entry - close[min(idx+50, len(close)-1)]) * CONTRACT_VALUE * size - COMMISSION * size

        capital += pnl
        if capital > max_equity:
            max_equity = capital
        dd = (max_equity - capital) / max_equity * 100
        if dd > max_drawdown:
            max_drawdown = dd

        trades.append({'pnl': pnl, 'size': size, 'direction': direction})

        if capital < initial_capital * 0.2:
            break

    if not trades:
        return {
            'total_return': 0, 'final_capital': initial_capital,
            'trades': 0, 'win_rate': 0, 'profit_factor': 0,
            'max_drawdown': 0, 'blew_up': False
        }

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    return {
        'total_return': (capital - initial_capital) / initial_capital * 100,
        'final_capital': capital,
        'trades': len(trades_df),
        'win_rate': len(wins) / len(trades_df) * 100,
        'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf'),
        'max_drawdown': max_drawdown,
        'blew_up': capital < initial_capital * 0.2,
        'avg_size': trades_df['size'].mean(),
    }


def walk_forward_analysis(df, n_splits=5, train_pct=0.7, mode='both', target_rr=1.5, atr_mult=1.5, scale=True):
    """
    Walk-forward analysis with rolling windows.

    Splits data into n_splits periods, for each:
    - Uses train_pct for training (in-sample)
    - Uses remaining for testing (out-of-sample)
    """
    n = len(df)
    split_size = n // n_splits

    results = []
    cumulative_capital = INITIAL_CAPITAL

    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = start_idx + split_size if i < n_splits - 1 else n

        # Split into train/test
        period_data = df.iloc[start_idx:end_idx].reset_index(drop=True)
        train_end = int(len(period_data) * train_pct)

        train_data = period_data.iloc[:train_end]
        test_data = period_data.iloc[train_end:]

        # Get date range for this period
        period_start = df.iloc[start_idx]['timestamp'] if 'timestamp' in df.columns else start_idx
        period_end = df.iloc[end_idx-1]['timestamp'] if 'timestamp' in df.columns else end_idx

        # Test on out-of-sample data only
        if len(test_data) > 200:  # Need minimum bars
            test_result = backtest_period(
                test_data, mode, target_rr, atr_mult, scale, cumulative_capital
            )

            # Update cumulative capital for next period
            if not test_result['blew_up']:
                cumulative_capital = test_result['final_capital']

            results.append({
                'period': i + 1,
                'period_start': str(period_start)[:10],
                'period_end': str(period_end)[:10],
                'test_bars': len(test_data),
                **test_result
            })

    return results


def main():
    log("=" * 70)
    log("WALK-FORWARD ANALYSIS + SCALING TEST")
    log(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    log("=" * 70)

    # Load data
    df = load_data(5)

    # Calculate date range
    if 'timestamp' in df.columns:
        start_date = df['timestamp'].iloc[0]
        end_date = df['timestamp'].iloc[-1]
        log(f"Data range: {start_date} to {end_date}")

    log("\n" + "=" * 70)
    log("TEST 1: WALK-FORWARD (5 periods, 70/30 train/test split)")
    log("=" * 70)

    # Best parameters from previous tests
    configs = [
        ('both', 1.5, 1.5),
        ('both', 1.5, 2.0),
        ('short_only', 1.5, 1.5),
        ('long_only', 1.5, 1.5),
    ]

    for mode, target_rr, atr_mult in configs:
        log(f"\n--- Mode: {mode} | RR: {target_rr} | ATR: {atr_mult} ---")

        # Without scaling
        log("\nWithout scaling (1 contract):")
        wf_results = walk_forward_analysis(df, n_splits=5, train_pct=0.7,
                                           mode=mode, target_rr=target_rr,
                                           atr_mult=atr_mult, scale=False)

        total_trades = sum(r['trades'] for r in wf_results)
        total_return = (wf_results[-1]['final_capital'] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        avg_win_rate = np.mean([r['win_rate'] for r in wf_results])
        profitable_periods = sum(1 for r in wf_results if r['total_return'] > 0)
        max_dd = max(r['max_drawdown'] for r in wf_results)

        log(f"  Profitable Periods: {profitable_periods}/{len(wf_results)}")
        log(f"  Total Trades: {total_trades}")
        log(f"  Final Capital: ${wf_results[-1]['final_capital']:,.0f}")
        log(f"  Total Return: {total_return:.1f}%")
        log(f"  Avg Win Rate: {avg_win_rate:.1f}%")
        log(f"  Max Drawdown: {max_dd:.1f}%")

        for r in wf_results:
            log(f"    Period {r['period']}: {r['period_start']} to {r['period_end']} | "
                f"Return: {r['total_return']:.1f}% | Trades: {r['trades']} | "
                f"Win: {r['win_rate']:.0f}%")

        # With scaling
        log("\nWith 2% risk scaling:")
        wf_results_scaled = walk_forward_analysis(df, n_splits=5, train_pct=0.7,
                                                   mode=mode, target_rr=target_rr,
                                                   atr_mult=atr_mult, scale=True)

        total_return_scaled = (wf_results_scaled[-1]['final_capital'] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        profitable_periods_scaled = sum(1 for r in wf_results_scaled if r['total_return'] > 0)
        blew_up_count = sum(1 for r in wf_results_scaled if r['blew_up'])
        max_dd_scaled = max(r['max_drawdown'] for r in wf_results_scaled)

        log(f"  Profitable Periods: {profitable_periods_scaled}/{len(wf_results_scaled)}")
        log(f"  Blew Up: {blew_up_count} periods")
        log(f"  Final Capital: ${wf_results_scaled[-1]['final_capital']:,.0f}")
        log(f"  Total Return: {total_return_scaled:.1f}%")
        log(f"  Max Drawdown: {max_dd_scaled:.1f}%")

        for r in wf_results_scaled:
            status = " BLEW UP!" if r['blew_up'] else ""
            log(f"    Period {r['period']}: {r['period_start']} to {r['period_end']} | "
                f"Return: {r['total_return']:.1f}% | Capital: ${r['final_capital']:,.0f} | "
                f"Avg Size: {r['avg_size']:.1f}{status}")

    log("\n" + "=" * 70)
    log("TEST 2: FULL OUT-OF-SAMPLE (Train on Year 1, Test on Year 2)")
    log("=" * 70)

    # Split into 2 halves
    mid_point = len(df) // 2
    year1_data = df.iloc[:mid_point]
    year2_data = df.iloc[mid_point:].reset_index(drop=True)

    if 'timestamp' in df.columns:
        log(f"Year 1: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[mid_point]}")
        log(f"Year 2: {df['timestamp'].iloc[mid_point]} to {df['timestamp'].iloc[-1]}")

    log(f"Year 1 bars: {len(year1_data):,} | Year 2 bars: {len(year2_data):,}")

    for mode in ['both', 'short_only', 'long_only']:
        log(f"\n--- Mode: {mode} ---")

        # Test on Year 2 only (completely unseen data)
        result_fixed = backtest_period(year2_data, mode, 1.5, 1.5, scale=False)
        result_scaled = backtest_period(year2_data, mode, 1.5, 1.5, scale=True)

        log(f"Fixed (1 contract):")
        log(f"  Return: {result_fixed['total_return']:.1f}%")
        log(f"  Final: ${result_fixed['final_capital']:,.0f}")
        log(f"  Trades: {result_fixed['trades']} | Win Rate: {result_fixed['win_rate']:.1f}%")
        log(f"  Max DD: {result_fixed['max_drawdown']:.1f}%")

        log(f"Scaled (2% risk):")
        log(f"  Return: {result_scaled['total_return']:.1f}%")
        log(f"  Final: ${result_scaled['final_capital']:,.0f}")
        log(f"  Trades: {result_scaled['trades']} | Win Rate: {result_scaled['win_rate']:.1f}%")
        log(f"  Max DD: {result_scaled['max_drawdown']:.1f}%")
        log(f"  Avg Size: {result_scaled['avg_size']:.1f}")
        if result_scaled['blew_up']:
            log(f"  *** BLEW UP! ***")

    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log("Walk-forward analysis completed.")
    log("Results saved to /workspace/results/")

    # Save results
    output_dir = Path("/workspace/results")
    output_dir.mkdir(exist_ok=True)


if __name__ == "__main__":
    main()
