#!/usr/bin/env python3
"""Test CISD on all available data."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.smc_analysis import parse_bars, detect_cisd

log_dir = r'C:\Users\leone\topstep-trading-bot\logs'
log_files = [
    ('Jan 4-5', 'fractal_bot_multi.2026-01-04_00-49-00_551914.log'),
    ('Jan 5-6', 'fractal_bot_multi.2026-01-05_00-49-02_993653.log'),
    ('Jan 6-7', 'fractal_bot_multi.2026-01-06_00-49-14_063595.log'),
    ('Jan 7-8', 'fractal_bot_multi.2026-01-07_00-49-15_048380.log'),
    ('Current', 'fractal_bot_multi.log'),
]

total_signals = 0
total_wins = 0
total_losses = 0
total_pnl = 0.0
all_trades = []

print("=" * 80)
print("CISD BACKTEST - ALL AVAILABLE DATA")
print("=" * 80)

for name, filename in log_files:
    filepath = os.path.join(log_dir, filename)
    if not os.path.exists(filepath):
        continue

    bars = parse_bars(filepath)
    if len(bars) < 100:
        continue

    signals = 0
    wins = 0
    losses = 0
    pnl = 0.0

    in_trade = False
    trade = None
    last_signal_bar = -20

    for i in range(50, len(bars)):
        window = bars[:i+1]
        current = bars[i]

        # Check if in trade
        if in_trade and trade:
            if trade['direction'] == 'LONG':
                if current.L <= trade['stop']:
                    trade_pnl = trade['stop'] - trade['entry']
                    losses += 1
                    pnl += trade_pnl
                    all_trades.append({**trade, 'pnl': trade_pnl, 'outcome': 'LOSS'})
                    in_trade = False
                    trade = None
                elif current.H >= trade['target']:
                    trade_pnl = trade['target'] - trade['entry']
                    wins += 1
                    pnl += trade_pnl
                    all_trades.append({**trade, 'pnl': trade_pnl, 'outcome': 'WIN'})
                    in_trade = False
                    trade = None
            else:  # SHORT
                if current.H >= trade['stop']:
                    trade_pnl = trade['entry'] - trade['stop']
                    losses += 1
                    pnl += trade_pnl
                    all_trades.append({**trade, 'pnl': trade_pnl, 'outcome': 'LOSS'})
                    in_trade = False
                    trade = None
                elif current.L <= trade['target']:
                    trade_pnl = trade['entry'] - trade['target']
                    wins += 1
                    pnl += trade_pnl
                    all_trades.append({**trade, 'pnl': trade_pnl, 'outcome': 'WIN'})
                    in_trade = False
                    trade = None
            continue

        # Look for new signals (min 20 bars between)
        if i - last_signal_bar < 20:
            continue

        cisd = detect_cisd(window)
        if cisd:
            signals += 1
            trade = {
                'direction': cisd['direction'],
                'entry': cisd['entry'],
                'stop': cisd['stop_loss'],
                'target': cisd['take_profit'],
            }
            in_trade = True
            last_signal_bar = i

    total_signals += signals
    total_wins += wins
    total_losses += losses
    total_pnl += pnl

    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    print(f"{name:12} | Bars: {len(bars):5} | Signals: {signals:3} | W/L: {wins:2}/{losses:2} | WR: {wr:5.1f}% | P&L: {pnl:+8.2f} pts")

print("-" * 80)
total_wr = total_wins / (total_wins + total_losses) * 100 if (total_wins + total_losses) > 0 else 0
print(f"{'TOTAL':12} |       {'':5} | Signals: {total_signals:3} | W/L: {total_wins:2}/{total_losses:2} | WR: {total_wr:5.1f}% | P&L: {total_pnl:+8.2f} pts")
print()
print(f"ES Dollar P&L (1 contract): ${total_pnl * 50:+,.0f}")
print(f"MES Dollar P&L (1 contract): ${total_pnl * 5:+,.0f}")

# Calculate avg win/loss
if all_trades:
    wins_list = [t['pnl'] for t in all_trades if t['outcome'] == 'WIN']
    losses_list = [t['pnl'] for t in all_trades if t['outcome'] == 'LOSS']
    avg_win = sum(wins_list) / len(wins_list) if wins_list else 0
    avg_loss = sum(losses_list) / len(losses_list) if losses_list else 0
    print()
    print(f"Avg Win:  {avg_win:+.2f} pts (${avg_win*50:+,.0f} ES)")
    print(f"Avg Loss: {avg_loss:+.2f} pts (${avg_loss*50:+,.0f} ES)")
    if avg_loss != 0:
        print(f"R:R Ratio: {abs(avg_win/avg_loss):.2f}:1")
