#!/usr/bin/env python3
"""
Backtest CRT (Candle Range Theory) and CISD (Change in State of Delivery)
- Standalone strategies
- Combined with each other
- Combined with existing SMC strategies
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.smc_analysis import parse_bars, find_fvgs, find_swing_points, get_high_confidence_signals, Bar
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# CRT DETECTION
# ============================================================================

def detect_crt_setup(bars: List[Bar], trend: str = None) -> Optional[Dict]:
    """
    Detect Candle Range Theory setup.

    Rules:
    1. Identify a range candle (CR high, CRL low)
    2. Next candle sweeps beyond range
    3. Next candle CLOSES back inside range
    4. Entry in direction opposite to sweep

    Args:
        bars: List of bars
        trend: 'BULLISH' or 'BEARISH' - only trade CRT setups with trend
    """
    if len(bars) < 10:
        return None

    # Look at last 3-5 candles for CRT pattern
    for lookback in range(3, 6):
        if len(bars) < lookback + 2:
            continue

        # Range candle (candle 1)
        range_candle = bars[-(lookback)]
        cr_high = range_candle.H
        cr_low = range_candle.L

        # Check subsequent candles for sweep + close inside
        for i in range(1, lookback):
            sweep_candle = bars[-(lookback - i)]

            # Bullish CRT: Sweep below CRL, close back inside
            if sweep_candle.L < cr_low and sweep_candle.C > cr_low and sweep_candle.C < cr_high:
                # Trend filter
                if trend == 'BEARISH':
                    continue  # Don't take longs in bearish trend

                current = bars[-1]
                entry = current.C
                stop = sweep_candle.L - 0.5
                target = cr_high

                risk = entry - stop
                reward = target - entry

                if risk > 0 and reward > 0 and reward / risk >= 1.5:
                    return {
                        'direction': 'LONG',
                        'strategy': 'CRT',
                        'entry': entry,
                        'stop': stop,
                        'target': target,
                        'rr': reward / risk,
                        'cr_high': cr_high,
                        'cr_low': cr_low,
                        'reasoning': f"CRT Long: Swept {cr_low:.2f}, closed inside, target {cr_high:.2f}"
                    }

            # Bearish CRT: Sweep above CR high, close back inside
            if sweep_candle.H > cr_high and sweep_candle.C < cr_high and sweep_candle.C > cr_low:
                if trend == 'BULLISH':
                    continue  # Don't take shorts in bullish trend

                current = bars[-1]
                entry = current.C
                stop = sweep_candle.H + 0.5
                target = cr_low

                risk = stop - entry
                reward = entry - target

                if risk > 0 and reward > 0 and reward / risk >= 1.5:
                    return {
                        'direction': 'SHORT',
                        'strategy': 'CRT',
                        'entry': entry,
                        'stop': stop,
                        'target': target,
                        'rr': reward / risk,
                        'cr_high': cr_high,
                        'cr_low': cr_low,
                        'reasoning': f"CRT Short: Swept {cr_high:.2f}, closed inside, target {cr_low:.2f}"
                    }

    return None


# ============================================================================
# CISD DETECTION
# ============================================================================

def get_trend(bars: List[Bar], lookback: int = 50) -> str:
    """Determine trend based on recent price action."""
    if len(bars) < lookback:
        return 'NEUTRAL'

    recent = bars[-lookback:]
    start = recent[0].C
    end = recent[-1].C

    # Also check higher highs / lower lows
    highs = [b.H for b in recent]
    lows = [b.L for b in recent]

    higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
    lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])

    if end > start and higher_highs > lower_lows:
        return 'BULLISH'
    elif end < start and lower_lows > higher_highs:
        return 'BEARISH'
    return 'NEUTRAL'


def find_last_series_candles(bars: List[Bar], candle_type: str, lookback: int = 10) -> Optional[Dict]:
    """
    Find the last series of up-close or down-close candles.

    Args:
        candle_type: 'UP' for up-close candles, 'DOWN' for down-close

    Returns:
        Dict with 'level' (bottom for up-close, top for down-close) and 'idx'
    """
    if len(bars) < lookback:
        return None

    recent = bars[-lookback:]
    series_start = None
    series_end = None

    for i in range(len(recent) - 1, 0, -1):
        candle = recent[i]
        is_up = candle.C > candle.O
        is_down = candle.C < candle.O

        if candle_type == 'UP' and is_up:
            if series_end is None:
                series_end = i
            series_start = i
        elif candle_type == 'DOWN' and is_down:
            if series_end is None:
                series_end = i
            series_start = i
        elif series_start is not None:
            break

    if series_start is None or series_end is None:
        return None

    series_candles = recent[series_start:series_end + 1]

    if candle_type == 'UP':
        # For up-close candles, CISD level is the bottom (candle lows, not wicks)
        level = min(min(c.O, c.C) for c in series_candles)
    else:
        # For down-close candles, CISD level is the top (candle highs, not wicks)
        level = max(max(c.O, c.C) for c in series_candles)

    return {
        'level': level,
        'start_idx': len(bars) - lookback + series_start,
        'end_idx': len(bars) - lookback + series_end,
        'candle_type': candle_type
    }


def detect_cisd(bars: List[Bar], require_displacement: bool = True) -> Optional[Dict]:
    """
    Detect Change in State of Delivery.

    Rules:
    1. Find last series of up-close or down-close candles
    2. Wait for price to CLOSE through that level (not just wick)
    3. Displacement = strong follow-through
    """
    if len(bars) < 20:
        return None

    current = bars[-1]
    prev = bars[-2]

    # Check for bullish CISD (was bearish, now bullish)
    down_series = find_last_series_candles(bars[:-2], 'DOWN', lookback=15)
    if down_series:
        cisd_level = down_series['level']
        # Current candle closes above the series tops
        if current.C > cisd_level and prev.C <= cisd_level:
            # Check displacement (strong close, not just barely through)
            displacement = current.C - cisd_level
            if not require_displacement or displacement > 1.0:
                # Find target (next swing high / buy-side liquidity)
                recent_highs = [b.H for b in bars[-50:]]
                target = max(recent_highs)

                entry = current.C
                stop = current.L - 0.5

                risk = entry - stop
                reward = target - entry

                if risk > 0 and reward > 0 and reward / risk >= 1.5:
                    return {
                        'direction': 'LONG',
                        'strategy': 'CISD',
                        'entry': entry,
                        'stop': stop,
                        'target': target,
                        'rr': reward / risk,
                        'cisd_level': cisd_level,
                        'displacement': displacement,
                        'reasoning': f"Bullish CISD: Closed above {cisd_level:.2f}, displacement {displacement:.1f}"
                    }

    # Check for bearish CISD (was bullish, now bearish)
    up_series = find_last_series_candles(bars[:-2], 'UP', lookback=15)
    if up_series:
        cisd_level = up_series['level']
        # Current candle closes below the series bottoms
        if current.C < cisd_level and prev.C >= cisd_level:
            displacement = cisd_level - current.C
            if not require_displacement or displacement > 1.0:
                recent_lows = [b.L for b in bars[-50:]]
                target = min(recent_lows)

                entry = current.C
                stop = current.H + 0.5

                risk = stop - entry
                reward = entry - target

                if risk > 0 and reward > 0 and reward / risk >= 1.5:
                    return {
                        'direction': 'SHORT',
                        'strategy': 'CISD',
                        'entry': entry,
                        'stop': stop,
                        'target': target,
                        'rr': reward / risk,
                        'cisd_level': cisd_level,
                        'displacement': displacement,
                        'reasoning': f"Bearish CISD: Closed below {cisd_level:.2f}, displacement {displacement:.1f}"
                    }

    return None


# ============================================================================
# COMBINED STRATEGIES
# ============================================================================

def detect_crt_with_cisd(bars: List[Bar]) -> Optional[Dict]:
    """
    CRT + CISD combined: Look for CRT setup confirmed by CISD.
    Higher probability when both align.
    """
    crt = detect_crt_setup(bars)
    cisd = detect_cisd(bars, require_displacement=False)

    if crt and cisd and crt['direction'] == cisd['direction']:
        return {
            **crt,
            'strategy': 'CRT+CISD',
            'reasoning': f"{crt['reasoning']} | CISD confirmed"
        }

    return None


def detect_crt_with_smc(bars: List[Bar]) -> Optional[Dict]:
    """
    CRT + SMC: CRT setup in line with SMC signals.
    """
    crt = detect_crt_setup(bars)
    smc_signals = get_high_confidence_signals(bars, min_confidence=60)

    if crt and smc_signals:
        for smc in smc_signals:
            if smc['direction'] == crt['direction']:
                return {
                    **crt,
                    'strategy': 'CRT+SMC',
                    'reasoning': f"{crt['reasoning']} | SMC {smc['strategy']} aligned"
                }

    return None


def detect_cisd_with_smc(bars: List[Bar]) -> Optional[Dict]:
    """
    CISD + SMC: CISD confirmation with SMC signals.
    """
    cisd = detect_cisd(bars)
    smc_signals = get_high_confidence_signals(bars, min_confidence=60)

    if cisd and smc_signals:
        for smc in smc_signals:
            if smc['direction'] == cisd['direction']:
                return {
                    **cisd,
                    'strategy': 'CISD+SMC',
                    'reasoning': f"{cisd['reasoning']} | SMC {smc['strategy']} aligned"
                }

    return None


def detect_all_combined(bars: List[Bar]) -> Optional[Dict]:
    """
    CRT + CISD + SMC: All three must align.
    Highest conviction setup.
    """
    crt = detect_crt_setup(bars)
    cisd = detect_cisd(bars, require_displacement=False)
    smc_signals = get_high_confidence_signals(bars, min_confidence=60)

    if crt and cisd and smc_signals:
        if crt['direction'] == cisd['direction']:
            for smc in smc_signals:
                if smc['direction'] == crt['direction']:
                    return {
                        **crt,
                        'strategy': 'CRT+CISD+SMC',
                        'confidence': 95,
                        'reasoning': f"Triple confluence: CRT + CISD + SMC ({smc['strategy']})"
                    }

    return None


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

def backtest_strategy(bars: List[Bar], strategy_func, strategy_name: str) -> Dict:
    """
    Backtest a strategy function on historical bars.
    """
    results = {
        'name': strategy_name,
        'signals': 0,
        'trades': [],
        'wins': 0,
        'losses': 0,
        'total_pnl': 0
    }

    if len(bars) < 200:
        return results

    in_trade = False
    trade = None
    last_signal_bar = -50

    for i in range(150, len(bars)):
        window = bars[:i+1]
        current = bars[i]

        # Check if in trade
        if in_trade and trade:
            if trade['direction'] == 'LONG':
                if current.L <= trade['stop']:
                    pnl = trade['stop'] - trade['entry']
                    results['trades'].append({**trade, 'exit': trade['stop'], 'pnl': pnl, 'outcome': 'LOSS'})
                    results['losses'] += 1
                    results['total_pnl'] += pnl
                    in_trade = False
                    trade = None
                elif current.H >= trade['target']:
                    pnl = trade['target'] - trade['entry']
                    results['trades'].append({**trade, 'exit': trade['target'], 'pnl': pnl, 'outcome': 'WIN'})
                    results['wins'] += 1
                    results['total_pnl'] += pnl
                    in_trade = False
                    trade = None
            else:  # SHORT
                if current.H >= trade['stop']:
                    pnl = trade['entry'] - trade['stop']
                    results['trades'].append({**trade, 'exit': trade['stop'], 'pnl': pnl, 'outcome': 'LOSS'})
                    results['losses'] += 1
                    results['total_pnl'] += pnl
                    in_trade = False
                    trade = None
                elif current.L <= trade['target']:
                    pnl = trade['entry'] - trade['target']
                    results['trades'].append({**trade, 'exit': trade['target'], 'pnl': pnl, 'outcome': 'WIN'})
                    results['wins'] += 1
                    results['total_pnl'] += pnl
                    in_trade = False
                    trade = None
            continue

        # Look for new signals (min 20 bars between signals)
        if i - last_signal_bar < 20:
            continue

        signal = strategy_func(window)

        if signal:
            results['signals'] += 1
            # Handle both 'stop' and 'stop_loss' key formats
            stop = signal.get('stop') or signal.get('stop_loss')
            target = signal.get('target') or signal.get('take_profit')

            if stop is None or target is None:
                continue  # Skip malformed signals

            trade = {
                'bar_idx': i,
                'direction': signal['direction'],
                'entry': signal['entry'],
                'stop': stop,
                'target': target,
                'reasoning': signal.get('reasoning', '')
            }
            in_trade = True
            last_signal_bar = i

    return results


def _wrap_smc_signal(signals):
    """Wrap SMC signal to match expected format."""
    if not signals:
        return None
    sig = signals[0]
    return {
        'direction': sig['direction'],
        'strategy': sig['strategy'],
        'entry': sig['entry'],
        'stop': sig['stop_loss'],
        'target': sig['take_profit'],
        'rr': sig.get('rr', 2.0),
        'reasoning': sig.get('reasoning', '')
    }


def run_all_backtests():
    """Run all strategy combinations and compare."""

    log_dir = r'C:\Users\leone\topstep-trading-bot\logs'

    log_files = [
        ('Jan 4-5', os.path.join(log_dir, 'fractal_bot_multi.2026-01-04_00-49-00_551914.log')),
        ('Jan 5-6', os.path.join(log_dir, 'fractal_bot_multi.2026-01-05_00-49-02_993653.log')),
        ('Jan 6-7', os.path.join(log_dir, 'fractal_bot_multi.2026-01-06_00-49-14_063595.log')),
        ('Current', os.path.join(log_dir, 'fractal_bot_multi.log')),
    ]

    strategies = [
        ('CRT Standalone', lambda bars: detect_crt_setup(bars)),
        ('CISD Standalone', lambda bars: detect_cisd(bars)),
        ('CRT + CISD', lambda bars: detect_crt_with_cisd(bars)),
        ('CRT + SMC', lambda bars: detect_crt_with_smc(bars)),
        ('CISD + SMC', lambda bars: detect_cisd_with_smc(bars)),
        ('CRT + CISD + SMC', lambda bars: detect_all_combined(bars)),
        ('SMC Only (baseline)', lambda bars: _wrap_smc_signal(get_high_confidence_signals(bars, min_confidence=70))),
    ]

    print("=" * 80)
    print("CRT & CISD STRATEGY BACKTEST")
    print("=" * 80)
    print("\nTesting strategies across this week + last week's data...")
    print()

    # Aggregate results by strategy
    strategy_totals = {name: {'signals': 0, 'wins': 0, 'losses': 0, 'pnl': 0, 'trades': []} for name, _ in strategies}

    for file_name, log_file in log_files:
        if not os.path.exists(log_file):
            continue

        bars = parse_bars(log_file)
        if len(bars) < 200:
            continue

        print(f"\n{'='*60}")
        print(f"Data: {file_name} ({len(bars)} bars)")
        print(f"{'='*60}")

        for strat_name, strat_func in strategies:
            results = backtest_strategy(bars, strat_func, strat_name)

            strategy_totals[strat_name]['signals'] += results['signals']
            strategy_totals[strat_name]['wins'] += results['wins']
            strategy_totals[strat_name]['losses'] += results['losses']
            strategy_totals[strat_name]['pnl'] += results['total_pnl']
            strategy_totals[strat_name]['trades'].extend(results['trades'])

            if results['signals'] > 0:
                wr = results['wins'] / (results['wins'] + results['losses']) * 100 if (results['wins'] + results['losses']) > 0 else 0
                print(f"  {strat_name:<20} | Signals: {results['signals']:>2} | W/L: {results['wins']}/{results['losses']} | WR: {wr:>5.1f}% | P&L: {results['total_pnl']:>+7.2f}")

    # Final summary
    print("\n")
    print("=" * 80)
    print("OVERALL SUMMARY - ALL DATA COMBINED")
    print("=" * 80)
    print()
    print(f"{'Strategy':<25} {'Signals':>8} {'Wins':>6} {'Losses':>7} {'Win%':>7} {'P&L pts':>10} {'ES $':>10}")
    print("-" * 80)

    # Sort by P&L
    sorted_strats = sorted(strategy_totals.items(), key=lambda x: x[1]['pnl'], reverse=True)

    for strat_name, totals in sorted_strats:
        signals = totals['signals']
        wins = totals['wins']
        losses = totals['losses']
        pnl = totals['pnl']

        if wins + losses > 0:
            wr = wins / (wins + losses) * 100
        else:
            wr = 0

        es_dollars = pnl * 50

        marker = " <-- BEST" if strat_name == sorted_strats[0][0] and pnl > 0 else ""
        print(f"{strat_name:<25} {signals:>8} {wins:>6} {losses:>7} {wr:>6.1f}% {pnl:>+10.2f} {es_dollars:>+10,.0f}{marker}")

    # Detailed trade analysis for top strategies
    print("\n")
    print("=" * 80)
    print("DETAILED TRADE LOG - TOP 3 STRATEGIES")
    print("=" * 80)

    for strat_name, totals in sorted_strats[:3]:
        if not totals['trades']:
            continue
        print(f"\n{strat_name}:")
        print(f"{'Dir':<6} {'Entry':>9} {'Stop':>9} {'Target':>9} {'Exit':>9} {'P&L':>8} {'Result'}")
        print("-" * 65)
        for t in totals['trades'][:10]:  # Show first 10 trades
            print(f"{t['direction']:<6} {t['entry']:>9.2f} {t['stop']:>9.2f} {t['target']:>9.2f} {t['exit']:>9.2f} {t['pnl']:>+8.2f} {t['outcome']}")
        if len(totals['trades']) > 10:
            print(f"  ... and {len(totals['trades']) - 10} more trades")

    # Recommendation
    print("\n")
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    best_strat = sorted_strats[0][0]
    best_pnl = sorted_strats[0][1]['pnl']
    best_wr = sorted_strats[0][1]['wins'] / max(sorted_strats[0][1]['wins'] + sorted_strats[0][1]['losses'], 1) * 100

    print(f"\nBest performing: {best_strat}")
    print(f"  P&L: {best_pnl:+.2f} pts (${best_pnl * 50:+,.0f} ES)")
    print(f"  Win Rate: {best_wr:.1f}%")

    # Check if combining improves over standalone SMC
    smc_pnl = strategy_totals['SMC Only (baseline)']['pnl']
    if best_pnl > smc_pnl:
        improvement = best_pnl - smc_pnl
        print(f"\n  {best_strat} outperforms SMC baseline by {improvement:+.2f} pts (${improvement * 50:+,.0f})")
    else:
        print(f"\n  SMC baseline still performs best. Keep existing strategy.")


if __name__ == '__main__':
    run_all_backtests()
