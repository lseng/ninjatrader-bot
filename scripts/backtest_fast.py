#!/usr/bin/env python3
"""
Fast backtest - samples data and uses efficient strategies.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.smc_analysis import parse_bars, detect_cisd, Bar
from typing import List, Dict, Optional


def get_trend(bars: List[Bar], lookback: int = 50) -> str:
    """Fast trend detection."""
    if len(bars) < lookback:
        return 'NEUTRAL'
    start = bars[-lookback].C
    end = bars[-1].C
    change = (end - start) / start * 100
    if change > 0.15:
        return 'BULLISH'
    elif change < -0.15:
        return 'BEARISH'
    return 'NEUTRAL'


def backtest(bars: List[Bar], strategy_name: str, strategy_func, min_bars_between: int = 15) -> Dict:
    """Run backtest."""
    results = {'signals': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0, 'trades': []}

    if len(bars) < 100:
        return results

    in_trade = False
    trade = None
    last_signal_bar = -30

    for i in range(60, len(bars)):
        current = bars[i]

        if in_trade and trade:
            if trade['dir'] == 'LONG':
                if current.L <= trade['stop']:
                    pnl = trade['stop'] - trade['entry']
                    results['losses'] += 1
                    results['pnl'] += pnl
                    results['trades'].append({'pnl': pnl, 'outcome': 'LOSS'})
                    in_trade = False
                elif current.H >= trade['target']:
                    pnl = trade['target'] - trade['entry']
                    results['wins'] += 1
                    results['pnl'] += pnl
                    results['trades'].append({'pnl': pnl, 'outcome': 'WIN'})
                    in_trade = False
            else:
                if current.H >= trade['stop']:
                    pnl = trade['entry'] - trade['stop']
                    results['losses'] += 1
                    results['pnl'] += pnl
                    results['trades'].append({'pnl': pnl, 'outcome': 'LOSS'})
                    in_trade = False
                elif current.L <= trade['target']:
                    pnl = trade['entry'] - trade['target']
                    results['wins'] += 1
                    results['pnl'] += pnl
                    results['trades'].append({'pnl': pnl, 'outcome': 'WIN'})
                    in_trade = False
            continue

        if i - last_signal_bar < min_bars_between:
            continue

        signal = strategy_func(bars[:i+1])
        if signal:
            results['signals'] += 1
            trade = signal
            in_trade = True
            last_signal_bar = i

    return results


# Strategy functions
def cisd_basic(bars):
    s = detect_cisd(bars, min_displacement=1.0)
    if s:
        return {'dir': s['direction'], 'entry': s['entry'], 'stop': s['stop_loss'], 'target': s['take_profit']}
    return None

def cisd_disp2(bars):
    s = detect_cisd(bars, min_displacement=2.0)
    if s:
        return {'dir': s['direction'], 'entry': s['entry'], 'stop': s['stop_loss'], 'target': s['take_profit']}
    return None

def cisd_disp3(bars):
    s = detect_cisd(bars, min_displacement=3.0)
    if s:
        return {'dir': s['direction'], 'entry': s['entry'], 'stop': s['stop_loss'], 'target': s['take_profit']}
    return None

def cisd_with_trend(bars):
    trend = get_trend(bars, 50)
    s = detect_cisd(bars, min_displacement=1.0)
    if s:
        if (trend == 'BULLISH' and s['direction'] == 'LONG') or \
           (trend == 'BEARISH' and s['direction'] == 'SHORT') or \
           trend == 'NEUTRAL':
            return {'dir': s['direction'], 'entry': s['entry'], 'stop': s['stop_loss'], 'target': s['take_profit']}
    return None

def cisd_with_trend_disp2(bars):
    trend = get_trend(bars, 50)
    s = detect_cisd(bars, min_displacement=2.0)
    if s:
        if (trend == 'BULLISH' and s['direction'] == 'LONG') or \
           (trend == 'BEARISH' and s['direction'] == 'SHORT') or \
           trend == 'NEUTRAL':
            return {'dir': s['direction'], 'entry': s['entry'], 'stop': s['stop_loss'], 'target': s['take_profit']}
    return None

def cisd_counter_trend(bars):
    trend = get_trend(bars, 50)
    s = detect_cisd(bars, min_displacement=2.0)
    if s:
        if (trend == 'BULLISH' and s['direction'] == 'SHORT') or \
           (trend == 'BEARISH' and s['direction'] == 'LONG'):
            return {'dir': s['direction'], 'entry': s['entry'], 'stop': s['stop_loss'], 'target': s['take_profit']}
    return None

def cisd_high_rr(bars):
    s = detect_cisd(bars, min_displacement=1.0)
    if s and s['rr'] >= 3.0:
        return {'dir': s['direction'], 'entry': s['entry'], 'stop': s['stop_loss'], 'target': s['take_profit']}
    return None

def cisd_very_high_rr(bars):
    s = detect_cisd(bars, min_displacement=1.0)
    if s and s['rr'] >= 5.0:
        return {'dir': s['direction'], 'entry': s['entry'], 'stop': s['stop_loss'], 'target': s['take_profit']}
    return None

def cisd_trend_high_rr(bars):
    trend = get_trend(bars, 50)
    s = detect_cisd(bars, min_displacement=1.5)
    if s and s['rr'] >= 3.0:
        if (trend == 'BULLISH' and s['direction'] == 'LONG') or \
           (trend == 'BEARISH' and s['direction'] == 'SHORT'):
            return {'dir': s['direction'], 'entry': s['entry'], 'stop': s['stop_loss'], 'target': s['take_profit']}
    return None


def main():
    log_dir = r'C:\Users\leone\topstep-trading-bot\logs'
    log_files = [
        'fractal_bot_multi.2026-01-04_00-49-00_551914.log',
        'fractal_bot_multi.2026-01-05_00-49-02_993653.log',
        'fractal_bot_multi.2026-01-06_00-49-14_063595.log',
        'fractal_bot_multi.2026-01-07_00-49-15_048380.log',
        'fractal_bot_multi.log',
    ]

    all_bars = []
    for f in log_files:
        path = os.path.join(log_dir, f)
        if os.path.exists(path):
            bars = parse_bars(path)
            all_bars.extend(bars)

    print(f"Total bars: {len(all_bars)}")
    print("=" * 95)

    strategies = [
        ('CISD Basic (disp=1)', cisd_basic),
        ('CISD (disp=2)', cisd_disp2),
        ('CISD (disp=3)', cisd_disp3),
        ('CISD + Trend', cisd_with_trend),
        ('CISD + Trend (disp=2)', cisd_with_trend_disp2),
        ('CISD Counter-Trend', cisd_counter_trend),
        ('CISD High R:R (>=3)', cisd_high_rr),
        ('CISD Very High R:R (>=5)', cisd_very_high_rr),
        ('CISD+Trend+HighRR', cisd_trend_high_rr),
    ]

    results_list = []

    print(f"\n{'Strategy':<28} {'Sigs':>6} {'W/L':>10} {'WR%':>7} {'P&L pts':>10} {'AvgWin':>8} {'AvgLoss':>8} {'R:R':>6}")
    print("-" * 95)

    for name, func in strategies:
        r = backtest(all_bars, name, func)

        wins_pnl = [t['pnl'] for t in r['trades'] if t['outcome'] == 'WIN']
        loss_pnl = [t['pnl'] for t in r['trades'] if t['outcome'] == 'LOSS']
        avg_win = sum(wins_pnl) / len(wins_pnl) if wins_pnl else 0
        avg_loss = sum(loss_pnl) / len(loss_pnl) if loss_pnl else 0
        rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        wr = r['wins'] / (r['wins'] + r['losses']) * 100 if (r['wins'] + r['losses']) > 0 else 0

        results_list.append({
            'name': name, 'signals': r['signals'], 'wins': r['wins'], 'losses': r['losses'],
            'wr': wr, 'pnl': r['pnl'], 'avg_win': avg_win, 'avg_loss': avg_loss, 'rr': rr
        })

        wl = f"{r['wins']}/{r['losses']}"
        print(f"{name:<28} {r['signals']:>6} {wl:>10} {wr:>6.1f}% {r['pnl']:>+10.2f} {avg_win:>+8.2f} {avg_loss:>+8.2f} {rr:>6.2f}")

    # Sort and show best
    print("\n" + "=" * 95)
    print("RANKED BY P&L")
    print("=" * 95)

    sorted_r = sorted(results_list, key=lambda x: x['pnl'], reverse=True)
    for i, r in enumerate(sorted_r, 1):
        marker = " <-- BEST" if i == 1 and r['pnl'] > 0 else ""
        marker = " <-- LOSING" if r['pnl'] < 0 else marker
        print(f"{i}. {r['name']:<25} P&L: {r['pnl']:>+8.2f} pts (${r['pnl']*50:>+8,.0f} ES) | WR: {r['wr']:.1f}% | R:R: {r['rr']:.2f}{marker}")


if __name__ == '__main__':
    main()
