#!/usr/bin/env python3
"""
Scaling Backtest - Tests strategies with DYNAMIC position sizing.

Instead of fixed 1 contract, we scale position size based on:
1. Account balance
2. Risk per trade (2% of account)
3. Stop loss distance

This tests if strategies remain profitable as we scale up.
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
CONTRACT_VALUE = 5.0  # $5 per point for MES
COMMISSION_PER_CONTRACT = 2.50
INITIAL_CAPITAL = 10000  # Start with $10k
RISK_PER_TRADE = 0.02  # Risk 2% per trade
MAX_CONTRACTS = 50  # Cap to prevent unrealistic sizes


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


def calculate_position_size(capital, entry, stop, risk_pct=0.02):
    """
    Calculate position size based on risk.

    Risk $ = Capital * risk_pct
    Risk per contract = |entry - stop| * CONTRACT_VALUE
    Position size = Risk $ / Risk per contract
    """
    risk_amount = capital * risk_pct
    risk_per_contract = abs(entry - stop) * CONTRACT_VALUE

    if risk_per_contract == 0:
        return 1

    size = int(risk_amount / risk_per_contract)
    return max(1, min(size, MAX_CONTRACTS))


def williams_fractals_signals(df, period=3, ma_fast=20, ma_medium=50, ma_slow=100):
    """Vectorized Williams Fractals."""
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

    # ATR for stops
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1))
    ))
    atr = pd.Series(tr).rolling(14).mean().values

    return long_sig, short_sig, close, high, low, atr


def scaling_backtest(df, mode='both', target_rr=1.5, atr_mult=1.5, scale_type='fixed'):
    """
    Backtest with dynamic position sizing.

    scale_type:
        'fixed' - Always 1 contract
        'risk_based' - Size based on 2% risk
        'aggressive' - Size based on 5% risk
    """
    long_signals, short_signals, close, high, low, atr = williams_fractals_signals(df)

    capital = INITIAL_CAPITAL
    trades = []
    equity_curve = [capital]
    max_equity = capital
    max_drawdown = 0

    # Risk percentage based on scale type
    if scale_type == 'fixed':
        risk_pct = 0  # Will use 1 contract
    elif scale_type == 'risk_based':
        risk_pct = 0.02  # 2% risk
    elif scale_type == 'aggressive':
        risk_pct = 0.05  # 5% risk
    else:
        risk_pct = 0.02

    # Process signals sequentially (needed for compounding)
    all_signals = []

    if mode in ['both', 'long_only']:
        for i in np.where(long_signals)[0]:
            if i >= 150 and i + 50 < len(close):
                all_signals.append((i, 1))  # (index, direction)

    if mode in ['both', 'short_only']:
        for i in np.where(short_signals)[0]:
            if i >= 150 and i + 50 < len(close):
                all_signals.append((i, -1))

    # Sort by index
    all_signals.sort(key=lambda x: x[0])

    for idx, direction in all_signals:
        entry = close[idx]

        if direction == 1:  # Long
            stop = entry - atr_mult * atr[idx]
            target = entry + target_rr * (entry - stop)
        else:  # Short
            stop = entry + atr_mult * atr[idx]
            target = entry - target_rr * (stop - entry)

        # Calculate position size
        if scale_type == 'fixed':
            size = 1
        else:
            size = calculate_position_size(capital, entry, stop, risk_pct)

        # Find exit
        future_high = high[idx+1:min(idx+51, len(close))]
        future_low = low[idx+1:min(idx+51, len(close))]

        if direction == 1:
            stop_idx = np.where(future_low <= stop)[0]
            target_idx = np.where(future_high >= target)[0]

            if len(stop_idx) > 0 and (len(target_idx) == 0 or stop_idx[0] <= target_idx[0]):
                pnl = (stop - entry) * CONTRACT_VALUE * size - COMMISSION_PER_CONTRACT * size
            elif len(target_idx) > 0:
                pnl = (target - entry) * CONTRACT_VALUE * size - COMMISSION_PER_CONTRACT * size
            else:
                pnl = (close[min(idx+50, len(close)-1)] - entry) * CONTRACT_VALUE * size - COMMISSION_PER_CONTRACT * size
        else:
            stop_idx = np.where(future_high >= stop)[0]
            target_idx = np.where(future_low <= target)[0]

            if len(stop_idx) > 0 and (len(target_idx) == 0 or stop_idx[0] <= target_idx[0]):
                pnl = (entry - stop) * CONTRACT_VALUE * size - COMMISSION_PER_CONTRACT * size
            elif len(target_idx) > 0:
                pnl = (entry - target) * CONTRACT_VALUE * size - COMMISSION_PER_CONTRACT * size
            else:
                pnl = (entry - close[min(idx+50, len(close)-1)]) * CONTRACT_VALUE * size - COMMISSION_PER_CONTRACT * size

        # Update capital
        capital += pnl
        equity_curve.append(capital)

        # Track drawdown
        if capital > max_equity:
            max_equity = capital
        drawdown = (max_equity - capital) / max_equity * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        trades.append({
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'pnl': pnl,
            'size': size,
            'capital_after': capital,
        })

        # Check for blowup (lose 80% of account)
        if capital < INITIAL_CAPITAL * 0.2:
            break

    if not trades:
        return {'total_return': 0, 'total_trades': 0, 'blew_up': False}

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    return {
        'total_return': (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        'final_capital': capital,
        'total_trades': len(trades_df),
        'win_rate': len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf'),
        'max_drawdown': max_drawdown,
        'avg_size': trades_df['size'].mean(),
        'max_size': trades_df['size'].max(),
        'blew_up': capital < INITIAL_CAPITAL * 0.2,
        'peak_capital': max_equity,
    }


def run_test(args):
    """Run a single test configuration."""
    df, mode, target_rr, atr_mult, scale_type = args

    try:
        result = scaling_backtest(df, mode, target_rr, atr_mult, scale_type)
        result['mode'] = mode
        result['target_rr'] = target_rr
        result['atr_mult'] = atr_mult
        result['scale_type'] = scale_type
        return result
    except Exception as e:
        return {'error': str(e)}


def main():
    log("=" * 70)
    log("SCALING BACKTEST - Dynamic Position Sizing")
    log(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    log(f"CPUs: {cpu_count()}")
    log("=" * 70)

    # Load 5-minute data (best performing timeframe)
    df = load_data(5)

    # Test configurations
    modes = ['long_only', 'short_only', 'both']
    target_rrs = [1.5, 2.0, 2.5]
    atr_mults = [1.5, 2.0, 2.5]
    scale_types = ['fixed', 'risk_based', 'aggressive']

    tasks = []
    for mode in modes:
        for target_rr in target_rrs:
            for atr_mult in atr_mults:
                for scale_type in scale_types:
                    tasks.append((df, mode, target_rr, atr_mult, scale_type))

    log(f"Total configurations: {len(tasks)}")
    log("Starting backtests...")
    start = datetime.now()

    with Pool(cpu_count()) as pool:
        results = list(pool.imap_unordered(run_test, tasks, chunksize=5))

    elapsed = (datetime.now() - start).total_seconds()
    log(f"Completed in {elapsed:.1f}s")

    # Filter valid results
    valid = [r for r in results if 'error' not in r]
    df_results = pd.DataFrame(valid)

    log(f"\n{'='*70}")
    log("RESULTS BY SCALING TYPE")
    log(f"{'='*70}")

    for scale_type in scale_types:
        subset = df_results[df_results['scale_type'] == scale_type]
        profitable = subset[subset['total_return'] > 0]
        blew_up = subset[subset['blew_up'] == True]

        log(f"\n=== {scale_type.upper()} ===")
        log(f"  Profitable: {len(profitable)}/{len(subset)}")
        log(f"  Blew up: {len(blew_up)}/{len(subset)}")

        if len(profitable) > 0:
            best = profitable.loc[profitable['total_return'].idxmax()]
            log(f"  Best Return: {best['total_return']:.1f}%")
            log(f"  Final Capital: ${best['final_capital']:,.0f}")
            log(f"  Max Drawdown: {best['max_drawdown']:.1f}%")
            log(f"  Avg Position Size: {best['avg_size']:.1f}")
            log(f"  Max Position Size: {best['max_size']:.0f}")

    log(f"\n{'='*70}")
    log("TOP 10 SCALING STRATEGIES")
    log(f"{'='*70}")

    # Sort by final capital (not just return %)
    df_results_sorted = df_results[~df_results['blew_up']].sort_values('final_capital', ascending=False)

    for i, row in df_results_sorted.head(10).iterrows():
        log(f"\n#{df_results_sorted.index.get_loc(i)+1}: {row['mode']} | {row['scale_type']}")
        log(f"   Return: {row['total_return']:.1f}%")
        log(f"   Final Capital: ${row['final_capital']:,.0f} (from ${INITIAL_CAPITAL:,})")
        log(f"   Peak Capital: ${row['peak_capital']:,.0f}")
        log(f"   Max Drawdown: {row['max_drawdown']:.1f}%")
        log(f"   Trades: {row['total_trades']}")
        log(f"   Win Rate: {row['win_rate']:.1f}%")
        log(f"   PF: {row['profit_factor']:.2f}")
        log(f"   Avg Size: {row['avg_size']:.1f} | Max Size: {row['max_size']:.0f}")
        log(f"   Target R:R: {row['target_rr']} | ATR: {row['atr_mult']}")

    # Compare scaling vs fixed
    log(f"\n{'='*70}")
    log("SCALING COMPARISON (Best in each category, mode=both)")
    log(f"{'='*70}")

    both_results = df_results[(df_results['mode'] == 'both') & (~df_results['blew_up'])]

    for scale_type in scale_types:
        subset = both_results[both_results['scale_type'] == scale_type]
        if len(subset) > 0:
            best = subset.loc[subset['final_capital'].idxmax()]
            log(f"\n{scale_type.upper()}:")
            log(f"  Final: ${best['final_capital']:,.0f}")
            log(f"  Return: {best['total_return']:.1f}%")
            log(f"  Max DD: {best['max_drawdown']:.1f}%")
            log(f"  Avg Size: {best['avg_size']:.1f}")

    # Save results
    output_dir = Path("/workspace/results")
    output_dir.mkdir(exist_ok=True)

    df_results.to_csv(output_dir / "scaling_results.csv", index=False)

    # Save best configs
    best_configs = df_results_sorted.head(10).to_dict('records')
    with open(output_dir / "scaling_top_10.json", 'w') as f:
        json.dump(best_configs, f, indent=2, default=str)

    log(f"\nResults saved to {output_dir}/")
    log("=" * 70)


if __name__ == "__main__":
    main()
