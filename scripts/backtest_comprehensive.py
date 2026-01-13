#!/usr/bin/env python3
"""
Comprehensive backtest of all strategy combinations.
Tests CISD, SMC, and combinations with various parameters.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.smc_analysis import (
    parse_bars, detect_cisd, find_cisd_series,
    get_high_confidence_signals, get_4hour_range_signal,
    find_order_blocks, find_fvgs, find_swing_points,
    find_breaker_blocks, find_inducement_traps, Bar
)
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass


# ============================================================================
# TREND DETECTION
# ============================================================================

def get_trend(bars: List[Bar], lookback: int = 50) -> str:
    """Determine trend based on MA and price action."""
    if len(bars) < lookback:
        return 'NEUTRAL'

    recent = bars[-lookback:]

    # Simple: compare start vs end
    start_price = recent[0].C
    end_price = recent[-1].C

    # Also check higher highs / lower lows
    highs = [b.H for b in recent]
    lows = [b.L for b in recent]

    higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
    lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])

    price_change = (end_price - start_price) / start_price * 100

    if price_change > 0.1 and higher_highs > lower_lows:
        return 'BULLISH'
    elif price_change < -0.1 and lower_lows > higher_highs:
        return 'BEARISH'
    return 'NEUTRAL'


def get_trend_strength(bars: List[Bar], lookback: int = 20) -> float:
    """Get trend strength as a value from -1 (bearish) to +1 (bullish)."""
    if len(bars) < lookback:
        return 0.0

    recent = bars[-lookback:]
    up_moves = sum(1 for i in range(1, len(recent)) if recent[i].C > recent[i-1].C)
    down_moves = lookback - 1 - up_moves

    return (up_moves - down_moves) / (lookback - 1)


# ============================================================================
# STRATEGY FUNCTIONS
# ============================================================================

def strategy_cisd_only(bars: List[Bar], params: Dict) -> Optional[Dict]:
    """CISD standalone with configurable parameters."""
    min_disp = params.get('min_displacement', 1.0)
    min_rr = params.get('min_rr', 1.5)

    cisd = detect_cisd(bars, min_displacement=min_disp)
    if cisd and cisd['rr'] >= min_rr:
        return cisd
    return None


def strategy_cisd_with_trend(bars: List[Bar], params: Dict) -> Optional[Dict]:
    """CISD only when aligned with trend."""
    min_disp = params.get('min_displacement', 1.0)
    trend_lookback = params.get('trend_lookback', 50)

    trend = get_trend(bars, lookback=trend_lookback)
    cisd = detect_cisd(bars, min_displacement=min_disp)

    if cisd:
        # Only take signals aligned with trend
        if trend == 'BULLISH' and cisd['direction'] == 'LONG':
            return cisd
        elif trend == 'BEARISH' and cisd['direction'] == 'SHORT':
            return cisd
        # In neutral, take either direction
        elif trend == 'NEUTRAL':
            return cisd
    return None


def strategy_cisd_counter_trend(bars: List[Bar], params: Dict) -> Optional[Dict]:
    """CISD as reversal - only counter-trend."""
    min_disp = params.get('min_displacement', 2.0)  # Require stronger displacement for reversals
    trend_lookback = params.get('trend_lookback', 50)

    trend = get_trend(bars, lookback=trend_lookback)
    cisd = detect_cisd(bars, min_displacement=min_disp)

    if cisd:
        # Only take signals AGAINST trend (reversals)
        if trend == 'BULLISH' and cisd['direction'] == 'SHORT':
            return cisd
        elif trend == 'BEARISH' and cisd['direction'] == 'LONG':
            return cisd
    return None


def strategy_smc_only(bars: List[Bar], params: Dict) -> Optional[Dict]:
    """SMC strategies only (4H Range, OB, Breaker, Inducement)."""
    min_conf = params.get('min_confidence', 60)

    signals = get_high_confidence_signals(bars, min_confidence=min_conf)
    # Filter out CISD signals - only want pure SMC
    smc_signals = [s for s in signals if s['strategy'] != 'CISD']

    if smc_signals:
        return smc_signals[0]
    return None


def strategy_cisd_plus_smc(bars: List[Bar], params: Dict) -> Optional[Dict]:
    """CISD confirmed by any SMC signal."""
    min_disp = params.get('min_displacement', 1.0)
    min_conf = params.get('min_confidence', 50)

    cisd = detect_cisd(bars, min_displacement=min_disp)
    if not cisd:
        return None

    # Check for SMC confirmation
    signals = get_high_confidence_signals(bars, min_confidence=min_conf)
    smc_signals = [s for s in signals if s['strategy'] != 'CISD']

    for smc in smc_signals:
        if smc['direction'] == cisd['direction']:
            # Combine - use CISD entry but boost confidence
            return {
                **cisd,
                'strategy': f"CISD+{smc['strategy']}",
                'confidence': min(95, cisd['confidence'] + 15),
                'reasoning': f"{cisd['reasoning']} | SMC: {smc['strategy']} aligned"
            }
    return None


def strategy_cisd_trend_smc(bars: List[Bar], params: Dict) -> Optional[Dict]:
    """CISD + Trend + SMC - triple confirmation."""
    min_disp = params.get('min_displacement', 1.0)
    trend_lookback = params.get('trend_lookback', 50)
    min_conf = params.get('min_confidence', 50)

    trend = get_trend(bars, lookback=trend_lookback)
    cisd = detect_cisd(bars, min_displacement=min_disp)

    if not cisd:
        return None

    # Must be with trend
    if trend == 'BULLISH' and cisd['direction'] != 'LONG':
        return None
    if trend == 'BEARISH' and cisd['direction'] != 'SHORT':
        return None

    # Must have SMC confirmation
    signals = get_high_confidence_signals(bars, min_confidence=min_conf)
    smc_signals = [s for s in signals if s['strategy'] != 'CISD']

    for smc in smc_signals:
        if smc['direction'] == cisd['direction']:
            return {
                **cisd,
                'strategy': f"CISD+TREND+{smc['strategy']}",
                'confidence': min(98, cisd['confidence'] + 20),
                'reasoning': f"{cisd['reasoning']} | Trend: {trend} | SMC: {smc['strategy']}"
            }
    return None


def strategy_4h_range_only(bars: List[Bar], params: Dict) -> Optional[Dict]:
    """4H Range strategy only."""
    signal = get_4hour_range_signal(bars, four_hour_range=None)
    if signal and signal.get('status') == 'SIGNAL_ACTIVE':
        setup = signal['setup']
        return {
            'direction': setup['signal'],
            'strategy': '4H_RANGE',
            'entry': setup['entry'],
            'stop_loss': setup['stop_loss'],
            'take_profit': setup['take_profit'],
            'confidence': signal['confidence'],
            'rr': setup['rr'],
            'reasoning': f"4H Range {setup['signal']}"
        }
    return None


def strategy_high_rr_only(bars: List[Bar], params: Dict) -> Optional[Dict]:
    """Any signal with R:R >= 3."""
    min_rr = params.get('min_rr', 3.0)

    signals = get_high_confidence_signals(bars, min_confidence=40)
    for s in signals:
        if s.get('rr', 0) >= min_rr:
            return s
    return None


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def backtest_strategy(bars: List[Bar], strategy_func: Callable, params: Dict,
                      min_bars_between: int = 20) -> Dict:
    """Run backtest on a strategy function."""
    results = {
        'signals': 0,
        'wins': 0,
        'losses': 0,
        'total_pnl': 0.0,
        'trades': [],
        'max_drawdown': 0.0,
        'peak_pnl': 0.0
    }

    if len(bars) < 150:
        return results

    in_trade = False
    trade = None
    last_signal_bar = -50
    running_pnl = 0.0

    for i in range(100, len(bars)):
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
                    running_pnl += pnl
                    in_trade = False
                    trade = None
                elif current.H >= trade['target']:
                    pnl = trade['target'] - trade['entry']
                    results['trades'].append({**trade, 'exit': trade['target'], 'pnl': pnl, 'outcome': 'WIN'})
                    results['wins'] += 1
                    results['total_pnl'] += pnl
                    running_pnl += pnl
                    if running_pnl > results['peak_pnl']:
                        results['peak_pnl'] = running_pnl
                    in_trade = False
                    trade = None
            else:  # SHORT
                if current.H >= trade['stop']:
                    pnl = trade['entry'] - trade['stop']
                    results['trades'].append({**trade, 'exit': trade['stop'], 'pnl': pnl, 'outcome': 'LOSS'})
                    results['losses'] += 1
                    results['total_pnl'] += pnl
                    running_pnl += pnl
                    in_trade = False
                    trade = None
                elif current.L <= trade['target']:
                    pnl = trade['entry'] - trade['target']
                    results['trades'].append({**trade, 'exit': trade['target'], 'pnl': pnl, 'outcome': 'WIN'})
                    results['wins'] += 1
                    results['total_pnl'] += pnl
                    running_pnl += pnl
                    if running_pnl > results['peak_pnl']:
                        results['peak_pnl'] = running_pnl
                    in_trade = False
                    trade = None

            # Track drawdown
            if results['peak_pnl'] - running_pnl > results['max_drawdown']:
                results['max_drawdown'] = results['peak_pnl'] - running_pnl
            continue

        # Look for new signals
        if i - last_signal_bar < min_bars_between:
            continue

        try:
            signal = strategy_func(window, params)
        except Exception as e:
            continue

        if signal:
            stop = signal.get('stop_loss') or signal.get('stop')
            target = signal.get('take_profit') or signal.get('target')

            if stop is None or target is None:
                continue

            results['signals'] += 1
            trade = {
                'bar_idx': i,
                'direction': signal['direction'],
                'entry': signal['entry'],
                'stop': stop,
                'target': target,
                'strategy': signal.get('strategy', 'unknown'),
                'rr': signal.get('rr', 0)
            }
            in_trade = True
            last_signal_bar = i

    return results


def run_comprehensive_backtest():
    """Run all strategy combinations and find the best."""

    log_dir = r'C:\Users\leone\topstep-trading-bot\logs'
    log_files = [
        ('Jan 4-5', 'fractal_bot_multi.2026-01-04_00-49-00_551914.log'),
        ('Jan 5-6', 'fractal_bot_multi.2026-01-05_00-49-02_993653.log'),
        ('Jan 6-7', 'fractal_bot_multi.2026-01-06_00-49-14_063595.log'),
        ('Jan 7-8', 'fractal_bot_multi.2026-01-07_00-49-15_048380.log'),
        ('Current', 'fractal_bot_multi.log'),
    ]

    # Load all bars
    all_bars = []
    for name, filename in log_files:
        filepath = os.path.join(log_dir, filename)
        if os.path.exists(filepath):
            bars = parse_bars(filepath)
            if bars:
                all_bars.extend(bars)
                print(f"Loaded {name}: {len(bars)} bars")

    print(f"\nTotal bars: {len(all_bars)}")
    print("=" * 90)

    # Define strategies to test
    strategies = [
        # CISD variations
        ('CISD (disp=1.0)', strategy_cisd_only, {'min_displacement': 1.0}),
        ('CISD (disp=1.5)', strategy_cisd_only, {'min_displacement': 1.5}),
        ('CISD (disp=2.0)', strategy_cisd_only, {'min_displacement': 2.0}),
        ('CISD (disp=3.0)', strategy_cisd_only, {'min_displacement': 3.0}),

        # CISD + Trend
        ('CISD+Trend (50bar)', strategy_cisd_with_trend, {'min_displacement': 1.0, 'trend_lookback': 50}),
        ('CISD+Trend (100bar)', strategy_cisd_with_trend, {'min_displacement': 1.0, 'trend_lookback': 100}),
        ('CISD CounterTrend', strategy_cisd_counter_trend, {'min_displacement': 2.0}),

        # SMC only
        ('SMC Only (conf>=60)', strategy_smc_only, {'min_confidence': 60}),
        ('SMC Only (conf>=70)', strategy_smc_only, {'min_confidence': 70}),
        ('SMC Only (conf>=80)', strategy_smc_only, {'min_confidence': 80}),

        # CISD + SMC
        ('CISD+SMC (conf>=50)', strategy_cisd_plus_smc, {'min_displacement': 1.0, 'min_confidence': 50}),
        ('CISD+SMC (conf>=60)', strategy_cisd_plus_smc, {'min_displacement': 1.0, 'min_confidence': 60}),
        ('CISD+SMC (disp>=2)', strategy_cisd_plus_smc, {'min_displacement': 2.0, 'min_confidence': 50}),

        # Triple confirmation
        ('CISD+Trend+SMC', strategy_cisd_trend_smc, {'min_displacement': 1.0, 'trend_lookback': 50, 'min_confidence': 50}),

        # 4H Range only
        ('4H Range Only', strategy_4h_range_only, {}),

        # High R:R
        ('High R:R (>=3)', strategy_high_rr_only, {'min_rr': 3.0}),
        ('High R:R (>=4)', strategy_high_rr_only, {'min_rr': 4.0}),
    ]

    results_table = []

    print(f"\n{'Strategy':<25} {'Signals':>8} {'W/L':>10} {'WR%':>7} {'P&L pts':>10} {'Avg Win':>9} {'Avg Loss':>9} {'R:R':>6}")
    print("-" * 90)

    for name, func, params in strategies:
        result = backtest_strategy(all_bars, func, params)

        signals = result['signals']
        wins = result['wins']
        losses = result['losses']
        pnl = result['total_pnl']

        if wins + losses > 0:
            wr = wins / (wins + losses) * 100
        else:
            wr = 0

        wins_list = [t['pnl'] for t in result['trades'] if t['outcome'] == 'WIN']
        losses_list = [t['pnl'] for t in result['trades'] if t['outcome'] == 'LOSS']

        avg_win = sum(wins_list) / len(wins_list) if wins_list else 0
        avg_loss = sum(losses_list) / len(losses_list) if losses_list else 0
        rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        results_table.append({
            'name': name,
            'signals': signals,
            'wins': wins,
            'losses': losses,
            'wr': wr,
            'pnl': pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'rr': rr,
            'max_dd': result['max_drawdown']
        })

        print(f"{name:<25} {signals:>8} {f'{wins}/{losses}':>10} {wr:>6.1f}% {pnl:>+10.2f} {avg_win:>+9.2f} {avg_loss:>+9.2f} {rr:>6.2f}")

    # Sort by P&L and show top 5
    print("\n" + "=" * 90)
    print("TOP 5 STRATEGIES BY P&L")
    print("=" * 90)

    sorted_results = sorted(results_table, key=lambda x: x['pnl'], reverse=True)

    for i, r in enumerate(sorted_results[:5], 1):
        es_pnl = r['pnl'] * 50
        print(f"\n{i}. {r['name']}")
        print(f"   Signals: {r['signals']} | W/L: {r['wins']}/{r['losses']} | WR: {r['wr']:.1f}%")
        print(f"   P&L: {r['pnl']:+.2f} pts (${es_pnl:+,.0f} ES)")
        print(f"   Avg Win: {r['avg_win']:+.2f} | Avg Loss: {r['avg_loss']:+.2f} | R:R: {r['rr']:.2f}:1")
        print(f"   Max Drawdown: {r['max_dd']:.2f} pts")

    # Recommendation
    print("\n" + "=" * 90)
    print("RECOMMENDATION")
    print("=" * 90)

    best = sorted_results[0]
    print(f"\nBest strategy: {best['name']}")
    print(f"Expected P&L: {best['pnl']:+.2f} pts/week (${best['pnl']*50:+,.0f} ES)")

    # Check which strategies are actually profitable
    profitable = [r for r in sorted_results if r['pnl'] > 0]
    print(f"\nProfitable strategies: {len(profitable)} out of {len(sorted_results)}")

    if profitable:
        print("\nProfitable strategies:")
        for r in profitable:
            print(f"  - {r['name']}: {r['pnl']:+.2f} pts, {r['wr']:.1f}% WR")


if __name__ == '__main__':
    run_comprehensive_backtest()
