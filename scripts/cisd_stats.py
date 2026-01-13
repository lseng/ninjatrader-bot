#!/usr/bin/env python3
"""Quick CISD trade stats."""
import sys
sys.path.insert(0, '.')
from scripts.backtest_crt_cisd import parse_bars, backtest_strategy, detect_cisd

log_file = r'C:\Users\leone\topstep-trading-bot\logs\fractal_bot_multi.2026-01-04_00-49-00_551914.log'
bars = parse_bars(log_file)

result = backtest_strategy(bars, lambda b: detect_cisd(b), 'CISD')

wins = [t for t in result['trades'] if t['outcome'] == 'WIN']
losses = [t for t in result['trades'] if t['outcome'] == 'LOSS']

avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0

print(f'CISD Standalone Results (1 contract):')
print(f'  Signals: {result["signals"]}')
print(f'  Wins: {len(wins)} | Losses: {len(losses)}')
print(f'  Win Rate: {len(wins)/(len(wins)+len(losses))*100:.1f}%')
print(f'  Avg Win: {avg_win:+.2f} pts (ES: ${avg_win*50:+,.0f})')
print(f'  Avg Loss: {avg_loss:+.2f} pts (ES: ${avg_loss*50:+,.0f})')
print(f'  Total P&L: {result["total_pnl"]:+.2f} pts (ES: ${result["total_pnl"]*50:+,.0f})')
if avg_loss != 0:
    print(f'  R:R Ratio: {abs(avg_win/avg_loss):.2f}:1')
print()
print('Sample trades:')
for t in result['trades'][:10]:
    print(f'  {t["direction"]:5} Entry:{t["entry"]:.2f} Stop:{t["stop"]:.2f} Target:{t["target"]:.2f} Exit:{t["exit"]:.2f} PnL:{t["pnl"]:+.2f} {t["outcome"]}')
