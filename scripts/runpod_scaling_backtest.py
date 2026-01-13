#!/usr/bin/env python3
"""
SCALING BACKTEST - Test position scaling as balance grows

Contract: MES (Micro E-mini S&P 500) = $5/point, $1.25/tick
Scaling Rule: Add 5 contracts every time balance doubles (2x)

Balance Thresholds (starting $1,000 with MES):
- $1,000    -> 1 contract
- $2,000    -> 6 contracts   (+5)
- $4,000    -> 11 contracts  (+5)
- $8,000    -> 16 contracts  (+5)
- $16,000   -> 21 contracts  (+5)
- $32,000   -> 26 contracts  (+5)
- $64,000   -> 31 contracts  (+5)

Usage on runpod:
    python3.11 runpod_scaling_backtest.py --data MES_1s_2years.parquet --timeframe 5min
"""

import argparse
import json
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd


def load_and_resample(filepath: str, timeframe: str = '5min') -> Dict:
    """Load parquet and resample to specified timeframe."""
    print(f"Loading {filepath}...")
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df):,} rows")

    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    df_resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    print(f"Resampled to {timeframe}: {len(df_resampled):,} bars")
    print(f"Data period: {df_resampled.index[0]} to {df_resampled.index[-1]}")

    return {
        'timestamp': df_resampled.index.values,
        'open': df_resampled['open'].values,
        'high': df_resampled['high'].values,
        'low': df_resampled['low'].values,
        'close': df_resampled['close'].values,
        'volume': df_resampled['volume'].values,
    }


# =============================================================================
# INDICATOR CALCULATIONS
# =============================================================================

def calc_sma(close: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average"""
    sma = np.full_like(close, np.nan)
    for i in range(period - 1, len(close)):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    return sma


def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range"""
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]

    for i in range(1, len(close)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

    atr = np.full_like(close, np.nan)
    for i in range(period - 1, len(close)):
        atr[i] = np.mean(tr[i - period + 1:i + 1])

    return atr


def detect_fractals(high: np.ndarray, low: np.ndarray, period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Detect Williams Fractals"""
    n = len(high)
    bullish = np.zeros(n, dtype=bool)
    bearish = np.zeros(n, dtype=bool)

    for i in range(period, n - period):
        # Bullish: low[i] is lowest of surrounding bars
        is_lowest = True
        for j in range(-period, period + 1):
            if j != 0 and low[i + j] <= low[i]:
                is_lowest = False
                break
        bullish[i] = is_lowest

        # Bearish: high[i] is highest of surrounding bars
        is_highest = True
        for j in range(-period, period + 1):
            if j != 0 and high[i + j] >= high[i]:
                is_highest = False
                break
        bearish[i] = is_highest

    return bullish, bearish


# =============================================================================
# WILLIAMS FRACTALS STRATEGY (Best performer)
# =============================================================================

def strategy_williams_fractals(data: Dict, params: Dict) -> List[Dict]:
    """Williams Fractals + MA Alignment Strategy"""
    target_rr = params.get('target_rr', 3.0)
    atr_mult = params.get('atr_mult', 1.5)
    fractal_period = params.get('fractal_period', 3)

    highs = data['high']
    lows = data['low']
    closes = data['close']
    n = len(closes)

    # Calculate indicators
    ma20 = calc_sma(closes, 20)
    ma50 = calc_sma(closes, 50)
    ma100 = calc_sma(closes, 100)
    atr = calc_atr(highs, lows, closes, 14)
    bullish_frac, bearish_frac = detect_fractals(highs, lows, fractal_period)

    trades = []
    min_bars_between = 10
    last_trade_idx = -min_bars_between

    for i in range(100, n - 50):
        if i - last_trade_idx < min_bars_between:
            continue

        if np.isnan(ma100[i]) or np.isnan(atr[i]):
            continue

        fractal_idx = i - fractal_period

        # Bullish setup: MAs aligned, price above 100 MA, bullish fractal
        bullish_align = ma20[i] > ma50[i] > ma100[i]
        pullback_long = closes[i] < ma20[i]
        above_slow = closes[i] > ma100[i]

        if bullish_frac[fractal_idx] and bullish_align and pullback_long and above_slow:
            entry = closes[i]
            stop = entry - atr_mult * atr[i]
            risk = entry - stop
            target = entry + target_rr * risk

            if risk > 0.5:
                trades.append({
                    'idx': i, 'direction': 'LONG',
                    'entry': entry, 'stop': stop, 'target': target
                })
                last_trade_idx = i
                continue

        # Bearish setup
        bearish_align = ma100[i] > ma50[i] > ma20[i]
        pullback_short = closes[i] > ma20[i]
        below_slow = closes[i] < ma100[i]

        if bearish_frac[fractal_idx] and bearish_align and pullback_short and below_slow:
            entry = closes[i]
            stop = entry + atr_mult * atr[i]
            risk = stop - entry
            target = entry - target_rr * risk

            if risk > 0.5:
                trades.append({
                    'idx': i, 'direction': 'SHORT',
                    'entry': entry, 'stop': stop, 'target': target
                })
                last_trade_idx = i

    return trades


# =============================================================================
# POSITION SIZING CALCULATOR
# =============================================================================

def calculate_contracts(balance: float, starting_balance: float = 1000,
                        scale_multiplier: float = 2.0, contracts_per_scale: int = 5) -> int:
    """
    Calculate number of contracts based on balance growth.

    Rule: Add 5 contracts every time balance doubles (2x).

    MES ($5/point) Thresholds at $1,000 start:
    - $1,000   -> 1 contract
    - $2,000   -> 6 contracts  (+5)
    - $4,000   -> 11 contracts (+5)
    - $8,000   -> 16 contracts (+5)
    - $16,000  -> 21 contracts (+5)

    Args:
        balance: Current account balance
        starting_balance: Original account balance ($1,000 for MES)
        scale_multiplier: Growth multiplier (default 2x = double)
        contracts_per_scale: Contracts to add per scale (default 5)

    Returns:
        Number of contracts to trade
    """
    if balance <= 0:
        return 0

    ratio = balance / starting_balance

    if ratio < 1:
        return 1  # Never go below 1 contract

    import math
    # How many times have we doubled?
    times_doubled = int(math.log(ratio) / math.log(scale_multiplier))

    # Contracts = 1 + (times_doubled * contracts_per_scale)
    contracts = 1 + (times_doubled * contracts_per_scale)

    return max(1, contracts)


# =============================================================================
# BACKTESTING ENGINES (With and Without Scaling)
# =============================================================================

def simulate_trades_no_scaling(data: Dict, trades: List[Dict],
                               starting_balance: float = 1000,
                               contract_value: float = 5) -> Dict:
    """Simulate trades WITHOUT scaling (1 contract always)."""
    highs = data['high']
    lows = data['low']
    closes = data['close']
    n = len(closes)

    balance = starting_balance
    peak_balance = starting_balance
    max_drawdown = 0
    wins = 0
    losses = 0
    total_pnl = 0
    gross_profit = 0
    gross_loss = 0
    trade_log = []

    for trade in trades:
        idx = trade['idx']
        direction = trade['direction']
        entry = trade['entry']
        stop = trade['stop']
        target = trade['target']

        contracts = 1  # Always 1 contract

        # Look forward max 100 bars for exit
        exit_price = None
        for j in range(idx + 1, min(idx + 100, n)):
            if direction == 'LONG':
                if lows[j] <= stop:
                    exit_price = stop
                    break
                if highs[j] >= target:
                    exit_price = target
                    break
            else:
                if highs[j] >= stop:
                    exit_price = stop
                    break
                if lows[j] <= target:
                    exit_price = target
                    break

        if exit_price is None:
            exit_price = closes[min(idx + 100, n - 1)]

        # Calculate P&L
        if direction == 'LONG':
            pnl_points = exit_price - entry
        else:
            pnl_points = entry - exit_price

        pnl_dollars = pnl_points * contract_value * contracts

        # Update stats
        total_pnl += pnl_dollars
        balance += pnl_dollars

        if pnl_dollars > 0:
            wins += 1
            gross_profit += pnl_dollars
        else:
            losses += 1
            gross_loss += abs(pnl_dollars)

        if balance > peak_balance:
            peak_balance = balance
        drawdown = peak_balance - balance
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        trade_log.append({
            'contracts': contracts,
            'pnl': pnl_dollars,
            'balance': balance
        })

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    return {
        'trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'pnl_dollars': total_pnl,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'final_balance': balance,
        'max_contracts_used': 1,
        'trade_log': trade_log
    }


def simulate_trades_with_scaling(data: Dict, trades: List[Dict],
                                  starting_balance: float = 1000,
                                  contract_value: float = 5,
                                  scale_multiplier: float = 2.0,
                                  contracts_per_scale: int = 5) -> Dict:
    """
    Simulate trades WITH scaling.

    Rule: Add 5 contracts every time balance doubles (2x).
    """
    highs = data['high']
    lows = data['low']
    closes = data['close']
    n = len(closes)

    balance = starting_balance
    peak_balance = starting_balance
    max_drawdown = 0
    wins = 0
    losses = 0
    total_pnl = 0
    gross_profit = 0
    gross_loss = 0
    max_contracts = 1
    trade_log = []

    # Track contract scaling events
    scaling_events = []

    for trade in trades:
        idx = trade['idx']
        direction = trade['direction']
        entry = trade['entry']
        stop = trade['stop']
        target = trade['target']

        # Calculate contracts based on current balance
        contracts = calculate_contracts(balance, starting_balance, scale_multiplier, contracts_per_scale)

        if contracts > max_contracts:
            scaling_events.append({
                'trade_num': len(trade_log) + 1,
                'balance': balance,
                'new_contracts': contracts
            })
            max_contracts = contracts

        # Look forward max 100 bars for exit
        exit_price = None
        for j in range(idx + 1, min(idx + 100, n)):
            if direction == 'LONG':
                if lows[j] <= stop:
                    exit_price = stop
                    break
                if highs[j] >= target:
                    exit_price = target
                    break
            else:
                if highs[j] >= stop:
                    exit_price = stop
                    break
                if lows[j] <= target:
                    exit_price = target
                    break

        if exit_price is None:
            exit_price = closes[min(idx + 100, n - 1)]

        # Calculate P&L (scaled by contracts)
        if direction == 'LONG':
            pnl_points = exit_price - entry
        else:
            pnl_points = entry - exit_price

        pnl_dollars = pnl_points * contract_value * contracts

        # Update stats
        total_pnl += pnl_dollars
        balance += pnl_dollars

        if pnl_dollars > 0:
            wins += 1
            gross_profit += pnl_dollars
        else:
            losses += 1
            gross_loss += abs(pnl_dollars)

        if balance > peak_balance:
            peak_balance = balance
        drawdown = peak_balance - balance
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        trade_log.append({
            'contracts': contracts,
            'pnl': pnl_dollars,
            'balance': balance
        })

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    return {
        'trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'pnl_dollars': total_pnl,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'final_balance': balance,
        'max_contracts_used': max_contracts,
        'scaling_events': scaling_events,
        'trade_log': trade_log
    }


def main():
    parser = argparse.ArgumentParser(description='Scaling Backtest - MES')
    parser.add_argument('--data', required=True, help='Path to parquet file')
    parser.add_argument('--timeframe', default='5min', help='Timeframe (default: 5min)')
    parser.add_argument('--starting-balance', type=float, default=1000, help='Starting balance (default: $1,000)')
    parser.add_argument('--scale-multiplier', type=float, default=2.0, help='Balance multiplier for scaling (default: 2x = double)')
    parser.add_argument('--contracts-per-scale', type=int, default=5, help='Contracts to add per scale (default: 5)')
    parser.add_argument('--contract-value', type=float, default=5.0, help='Dollar value per point (MES=5, ES=50)')
    args = parser.parse_args()

    print("=" * 80)
    print("SCALING BACKTEST - Williams Fractals Strategy (MES)")
    print("=" * 80)
    print(f"\nContract: MES (${args.contract_value}/point)")
    print(f"Starting Balance: ${args.starting_balance:,.0f}")
    print(f"Scaling Rule: Add {args.contracts_per_scale} contracts every time balance {args.scale_multiplier:.0f}x's")
    print(f"\nContract thresholds:")
    contracts = 1
    for i in range(8):
        threshold = args.starting_balance * (args.scale_multiplier ** i)
        print(f"  ${threshold:>12,.0f} -> {contracts:>2} contract(s)")
        contracts += args.contracts_per_scale
    print()

    # Load data
    data = load_and_resample(args.data, args.timeframe)

    # Best performing Williams Fractals params
    params = {
        'target_rr': 3.0,
        'atr_mult': 1.5,
        'fractal_period': 3
    }

    print("\nGenerating trades with Williams Fractals (RR=3.0, ATR=1.5)...")
    trades = strategy_williams_fractals(data, params)
    print(f"Generated {len(trades)} trades")

    # Run WITHOUT scaling
    print("\n" + "=" * 80)
    print("RESULTS WITHOUT SCALING (1 contract always)")
    print("=" * 80)

    results_no_scale = simulate_trades_no_scaling(
        data, trades,
        starting_balance=args.starting_balance,
        contract_value=args.contract_value
    )

    print(f"\nTrades: {results_no_scale['trades']}")
    print(f"Win Rate: {results_no_scale['win_rate']:.1f}%")
    print(f"Profit Factor: {results_no_scale['profit_factor']:.2f}")
    print(f"Max Drawdown: ${results_no_scale['max_drawdown']:,.0f}")
    print(f"Total P&L: ${results_no_scale['pnl_dollars']:,.0f}")
    print(f"Final Balance: ${results_no_scale['final_balance']:,.0f}")
    print(f"Return: {((results_no_scale['final_balance'] / args.starting_balance) - 1) * 100:.1f}%")

    # Run WITH scaling
    print("\n" + "=" * 80)
    print(f"RESULTS WITH SCALING ({args.scale_multiplier:.0f}x balance = +{args.contracts_per_scale} contracts)")
    print("=" * 80)

    results_scaled = simulate_trades_with_scaling(
        data, trades,
        starting_balance=args.starting_balance,
        contract_value=args.contract_value,
        scale_multiplier=args.scale_multiplier,
        contracts_per_scale=args.contracts_per_scale
    )

    print(f"\nTrades: {results_scaled['trades']}")
    print(f"Win Rate: {results_scaled['win_rate']:.1f}%")
    print(f"Profit Factor: {results_scaled['profit_factor']:.2f}")
    print(f"Max Drawdown: ${results_scaled['max_drawdown']:,.0f}")
    print(f"Total P&L: ${results_scaled['pnl_dollars']:,.0f}")
    print(f"Final Balance: ${results_scaled['final_balance']:,.0f}")
    print(f"Return: {((results_scaled['final_balance'] / args.starting_balance) - 1) * 100:.1f}%")
    print(f"Max Contracts Used: {results_scaled['max_contracts_used']}")

    if results_scaled['scaling_events']:
        print(f"\nScaling Events:")
        for event in results_scaled['scaling_events']:
            print(f"  Trade #{event['trade_num']}: Balance ${event['balance']:,.0f} -> {event['new_contracts']} contracts")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    pnl_increase = results_scaled['pnl_dollars'] - results_no_scale['pnl_dollars']
    pnl_increase_pct = (pnl_increase / results_no_scale['pnl_dollars']) * 100 if results_no_scale['pnl_dollars'] > 0 else 0

    dd_increase = results_scaled['max_drawdown'] - results_no_scale['max_drawdown']
    dd_increase_pct = (dd_increase / results_no_scale['max_drawdown']) * 100 if results_no_scale['max_drawdown'] > 0 else 0

    print(f"\nP&L Increase with Scaling: +${pnl_increase:,.0f} ({pnl_increase_pct:+.1f}%)")
    print(f"Drawdown Increase: +${dd_increase:,.0f} ({dd_increase_pct:+.1f}%)")
    print(f"Risk-Adjusted Improvement: {pnl_increase_pct / max(dd_increase_pct, 1):.2f}x")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'strategy_params': params,
        'contract': 'MES',
        'contract_value': args.contract_value,
        'starting_balance': args.starting_balance,
        'scale_multiplier': args.scale_multiplier,
        'contracts_per_scale': args.contracts_per_scale,
        'no_scaling': {
            'trades': results_no_scale['trades'],
            'win_rate': results_no_scale['win_rate'],
            'profit_factor': results_no_scale['profit_factor'],
            'pnl_dollars': results_no_scale['pnl_dollars'],
            'max_drawdown': results_no_scale['max_drawdown'],
            'final_balance': results_no_scale['final_balance'],
        },
        'with_scaling': {
            'trades': results_scaled['trades'],
            'win_rate': results_scaled['win_rate'],
            'profit_factor': results_scaled['profit_factor'],
            'pnl_dollars': results_scaled['pnl_dollars'],
            'max_drawdown': results_scaled['max_drawdown'],
            'final_balance': results_scaled['final_balance'],
            'max_contracts': results_scaled['max_contracts_used'],
            'scaling_events': results_scaled['scaling_events'],
        },
        'comparison': {
            'pnl_increase_dollars': pnl_increase,
            'pnl_increase_percent': pnl_increase_pct,
            'drawdown_increase_dollars': dd_increase,
            'drawdown_increase_percent': dd_increase_pct,
        }
    }

    with open('scaling_backtest_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to scaling_backtest_results.json")


if __name__ == '__main__':
    main()
