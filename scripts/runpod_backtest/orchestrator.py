#!/usr/bin/env python3
"""
RunPod Backtesting Orchestrator

Distributes backtests across multiple workers and aggregates results.
Designed for 32 vCPU pod with aggressive parallelization.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from config import *
from worker import (
    WilliamsFractalsStrategy,
    MACDStrategy,
    TripleSuperTrendStrategy,
    LiquiditySweepStrategy,
    run_single_backtest,
)

# Strategy registry
STRATEGIES = {
    'fractals': WilliamsFractalsStrategy,
    'macd': MACDStrategy,
    'triple_supertrend': TripleSuperTrendStrategy,
    'liquidity_sweep': LiquiditySweepStrategy,
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_and_aggregate_data(timeframe_minutes: int) -> pd.DataFrame:
    """Load 1-second data and aggregate to target timeframe."""
    data_path = Path(DATA_PATH) if Path(DATA_PATH).exists() else Path(LOCAL_DATA_PATH)

    log(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    log(f"Loaded {len(df):,} 1-second bars")

    # Set timestamp as index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    # Aggregate
    log(f"Aggregating to {timeframe_minutes}-minute bars...")
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in df.columns:
        agg_dict['volume'] = 'sum'

    df_agg = df.resample(f'{timeframe_minutes}min').agg(agg_dict).dropna()
    log(f"Aggregated to {len(df_agg):,} bars")

    return df_agg.reset_index()


def generate_param_combinations(strategy_name: str) -> list:
    """Generate all parameter combinations for a strategy."""
    if strategy_name not in STRATEGY_PARAMS:
        return [{}]

    params = STRATEGY_PARAMS[strategy_name]
    keys = list(params.keys())
    values = list(params.values())

    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def generate_all_tasks(df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> list:
    """Generate all backtest task configurations."""
    tasks = []

    # Single strategy backtests
    for strategy_name, strategy_class in STRATEGIES.items():
        param_combos = generate_param_combinations(strategy_name)

        for params in param_combos:
            for mode in TRADING_MODES:
                for exit_strat in EXIT_STRATEGIES:
                    # 1-minute timeframe
                    tasks.append({
                        'df': df_1m,
                        'strategy': strategy_class,
                        'params': params,
                        'mode': mode,
                        'exit_strategy': exit_strat,
                        'timeframe': 1,
                        'strategy_name': strategy_name,
                    })

                    # 5-minute timeframe
                    tasks.append({
                        'df': df_5m,
                        'strategy': strategy_class,
                        'params': params,
                        'mode': mode,
                        'exit_strategy': exit_strat,
                        'timeframe': 5,
                        'strategy_name': strategy_name,
                    })

    log(f"Generated {len(tasks):,} total backtest configurations")
    return tasks


def run_backtest_wrapper(args):
    """Wrapper for multiprocessing."""
    task_idx, config = args
    try:
        result = run_single_backtest(config)
        result['task_idx'] = task_idx
        result['strategy_name'] = config.get('strategy_name', '')
        return result
    except Exception as e:
        return {
            'error': str(e),
            'task_idx': task_idx,
            'strategy_name': config.get('strategy_name', ''),
            'params': str(config.get('params', {})),
        }


def run_parallel_backtests(tasks: list, num_workers: int = None) -> list:
    """Run backtests in parallel across workers."""
    if num_workers is None:
        num_workers = cpu_count()

    log(f"Running {len(tasks):,} backtests across {num_workers} workers...")

    # Prepare tasks with indices
    indexed_tasks = [(i, task) for i, task in enumerate(tasks)]

    results = []
    completed = 0

    with Pool(num_workers) as pool:
        for result in pool.imap_unordered(run_backtest_wrapper, indexed_tasks, chunksize=10):
            results.append(result)
            completed += 1

            if completed % 100 == 0:
                log(f"Completed {completed:,}/{len(tasks):,} backtests ({completed/len(tasks)*100:.1f}%)")

    log(f"All backtests complete!")
    return results


def analyze_results(results: list) -> pd.DataFrame:
    """Analyze and rank backtest results."""
    # Filter out errors
    valid_results = [r for r in results if 'error' not in r]
    errors = [r for r in results if 'error' in r]

    if errors:
        log(f"Warning: {len(errors)} backtests failed")

    # Convert to DataFrame
    df = pd.DataFrame(valid_results)

    # Calculate composite score
    # Prioritize: profit factor > win rate > return > low drawdown
    df['score'] = (
        df['profit_factor'].clip(0, 10) * 20 +  # Max 200 points
        df['win_rate'] * 1 +                     # Max 100 points
        df['total_return'].clip(-100, 500) * 0.2 +  # Max 100 points
        (100 - df['max_drawdown'].clip(0, 100)) * 0.5  # Max 50 points
    )

    # Filter minimum viable strategies
    viable = df[
        (df['total_trades'] >= 50) &  # Enough trades for significance
        (df['profit_factor'] > 1.0) &  # Must be profitable
        (df['win_rate'] > 45)  # Reasonable win rate
    ].copy()

    log(f"Found {len(viable):,} viable strategies out of {len(df):,} tested")

    return viable.sort_values('score', ascending=False)


def main():
    log("=" * 70)
    log("RUNPOD DISTRIBUTED BACKTESTING")
    log(f"CPUs available: {cpu_count()}")
    log("=" * 70)

    # Load data
    df_1m = load_and_aggregate_data(1)
    df_5m = load_and_aggregate_data(5)

    log(f"1m data: {len(df_1m):,} bars")
    log(f"5m data: {len(df_5m):,} bars")

    # Generate tasks
    tasks = generate_all_tasks(df_1m, df_5m)

    # Run backtests
    start_time = datetime.now()
    results = run_parallel_backtests(tasks, num_workers=cpu_count())
    elapsed = (datetime.now() - start_time).total_seconds()

    log(f"Total time: {elapsed:.1f} seconds ({len(results)/elapsed:.1f} backtests/sec)")

    # Analyze results
    viable = analyze_results(results)

    # Save all results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Save top strategies
    top_n = 50
    top_strategies = viable.head(top_n)

    log(f"\n{'='*70}")
    log(f"TOP {min(top_n, len(top_strategies))} STRATEGIES")
    log(f"{'='*70}")

    for i, row in top_strategies.head(20).iterrows():
        log(f"\n#{viable.index.get_loc(i)+1}: {row['strategy_name']} @ {row['timeframe']}m ({row['mode']})")
        log(f"   Return: {row['total_return']:.1f}% | Win Rate: {row['win_rate']:.1f}%")
        log(f"   Profit Factor: {row['profit_factor']:.2f} | Max DD: {row['max_drawdown']:.1f}%")
        log(f"   Trades: {row['total_trades']} (L: {row['long_trades']}, S: {row['short_trades']})")
        log(f"   Long P&L: ${row['long_pnl']:.2f} | Short P&L: ${row['short_pnl']:.2f}")
        log(f"   Score: {row['score']:.1f}")

    # Save to files
    top_strategies.to_csv(output_dir / "top_strategies.csv", index=False)

    # Save full results
    all_results_df = pd.DataFrame([r for r in results if 'error' not in r])
    all_results_df.to_csv(output_dir / "all_results.csv", index=False)

    # Save best config
    if len(top_strategies) > 0:
        best = top_strategies.iloc[0]
        best_config = {
            'strategy': best['strategy_name'],
            'timeframe': int(best['timeframe']),
            'mode': best['mode'],
            'params': best.get('params', {}),
            'exit_strategy': best.get('exit_strategy', {}),
            'performance': {
                'return': float(best['total_return']),
                'win_rate': float(best['win_rate']),
                'profit_factor': float(best['profit_factor']),
                'max_drawdown': float(best['max_drawdown']),
                'total_trades': int(best['total_trades']),
                'long_pnl': float(best['long_pnl']),
                'short_pnl': float(best['short_pnl']),
            }
        }
        with open(output_dir / "best_strategy.json", 'w') as f:
            json.dump(best_config, f, indent=2, default=str)

    log(f"\nResults saved to {output_dir}/")
    log("=" * 70)


if __name__ == "__main__":
    main()
