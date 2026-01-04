#!/usr/bin/env python3
"""
Detailed Trade Log - Exports EVERY trade with full context.

Outputs CSV with:
- Trade ID, timestamp, direction
- Entry/exit price, stop, target
- P&L, running balance
- Position size
- Reason (which indicators triggered)
- Duration (bars held)
- All market data (OHLC, volume, indicators)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
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

    log(f"  {len(agg):,} bars")
    return agg.reset_index()


def calculate_all_indicators(df):
    """Calculate all indicators and return with reasons."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values

    close_s = pd.Series(close)
    low_s = pd.Series(low)
    high_s = pd.Series(high)

    # Moving Averages
    sma_20 = close_s.rolling(20).mean().values
    sma_50 = close_s.rolling(50).mean().values
    sma_100 = close_s.rolling(100).mean().values

    # Williams Fractals
    period = 3
    window = 2 * period + 1
    rolling_low_min = low_s.rolling(window, center=True).min().values
    rolling_high_max = high_s.rolling(window, center=True).max().values
    bullish_fractal = low == rolling_low_min
    bearish_fractal = high == rolling_high_max

    # MA alignment
    bull_align = (sma_20 > sma_50) & (sma_50 > sma_100)
    bear_align = (sma_20 < sma_50) & (sma_50 < sma_100)

    # Pullback
    pullback_long = close < sma_20
    pullback_short = close > sma_20
    above_slow = close > sma_100
    below_slow = close < sma_100

    # ATR
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1))
    ))
    atr = pd.Series(tr).rolling(14).mean().values

    # MACD
    ema_12 = close_s.ewm(span=12, adjust=False).mean().values
    ema_26 = close_s.ewm(span=26, adjust=False).mean().values
    macd = ema_12 - ema_26
    macd_signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    ma_200 = close_s.rolling(200).mean().values

    # RSI
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).values

    return {
        'close': close,
        'high': high,
        'low': low,
        'open': open_,
        'sma_20': sma_20,
        'sma_50': sma_50,
        'sma_100': sma_100,
        'atr': atr,
        'macd': macd,
        'macd_signal': macd_signal,
        'ma_200': ma_200,
        'rsi': rsi,
        'bullish_fractal': bullish_fractal,
        'bearish_fractal': bearish_fractal,
        'bull_align': bull_align,
        'bear_align': bear_align,
        'pullback_long': pullback_long,
        'pullback_short': pullback_short,
        'above_slow': above_slow,
        'below_slow': below_slow,
    }


def get_trade_reason(ind, idx, direction):
    """Generate human-readable reason for the trade."""
    reasons = []

    if direction == 1:  # Long
        if ind['bullish_fractal'][idx]:
            reasons.append("Bullish Fractal")
        if ind['bull_align'][idx]:
            reasons.append("MA Alignment (20>50>100)")
        if ind['pullback_long'][idx]:
            reasons.append("Pullback below 20 MA")
        if ind['above_slow'][idx]:
            reasons.append("Above 100 MA")
        if ind['macd'][idx] > ind['macd_signal'][idx]:
            reasons.append("MACD Bullish")
        if ind['close'][idx] > ind['ma_200'][idx]:
            reasons.append("Above 200 MA")
    else:  # Short
        if ind['bearish_fractal'][idx]:
            reasons.append("Bearish Fractal")
        if ind['bear_align'][idx]:
            reasons.append("MA Alignment (20<50<100)")
        if ind['pullback_short'][idx]:
            reasons.append("Rally above 20 MA")
        if ind['below_slow'][idx]:
            reasons.append("Below 100 MA")
        if ind['macd'][idx] < ind['macd_signal'][idx]:
            reasons.append("MACD Bearish")
        if ind['close'][idx] < ind['ma_200'][idx]:
            reasons.append("Below 200 MA")

    return " | ".join(reasons) if reasons else "Signal triggered"


def calculate_position_size(capital, entry, stop, risk_pct=0.02):
    """Calculate position size based on risk."""
    risk_amount = capital * risk_pct
    risk_per_contract = abs(entry - stop) * CONTRACT_VALUE
    if risk_per_contract == 0:
        return 1
    size = int(risk_amount / risk_per_contract)
    return max(1, min(size, MAX_CONTRACTS))


def generate_detailed_trade_log(df, mode='both', target_rr=1.5, atr_mult=1.5, scale=True):
    """Generate detailed trade log with all data."""
    ind = calculate_all_indicators(df)

    # Generate signals
    long_signals = (ind['bullish_fractal'] & ind['bull_align'] &
                    ind['pullback_long'] & ind['above_slow'])
    short_signals = (ind['bearish_fractal'] & ind['bear_align'] &
                     ind['pullback_short'] & ind['below_slow'])

    capital = INITIAL_CAPITAL
    trade_id = 0
    trades = []
    max_equity = capital

    all_signals = []
    if mode in ['both', 'long_only']:
        for i in np.where(long_signals)[0]:
            if i >= 200 and i + 50 < len(ind['close']):
                all_signals.append((i, 1))
    if mode in ['both', 'short_only']:
        for i in np.where(short_signals)[0]:
            if i >= 200 and i + 50 < len(ind['close']):
                all_signals.append((i, -1))

    all_signals.sort(key=lambda x: x[0])

    for idx, direction in all_signals:
        trade_id += 1

        entry_price = ind['close'][idx]
        entry_time = df['timestamp'].iloc[idx] if 'timestamp' in df.columns else idx

        if direction == 1:
            stop_loss = entry_price - atr_mult * ind['atr'][idx]
            take_profit = entry_price + target_rr * (entry_price - stop_loss)
        else:
            stop_loss = entry_price + atr_mult * ind['atr'][idx]
            take_profit = entry_price - target_rr * (stop_loss - entry_price)

        # Position size
        if scale:
            size = calculate_position_size(capital, entry_price, stop_loss, RISK_PCT)
        else:
            size = 1

        # Find exit
        future_high = ind['high'][idx+1:min(idx+51, len(ind['close']))]
        future_low = ind['low'][idx+1:min(idx+51, len(ind['close']))]
        future_close = ind['close'][idx+1:min(idx+51, len(ind['close']))]

        exit_idx = None
        exit_price = None
        exit_reason = None

        if direction == 1:
            stop_bars = np.where(future_low <= stop_loss)[0]
            target_bars = np.where(future_high >= take_profit)[0]

            if len(stop_bars) > 0 and (len(target_bars) == 0 or stop_bars[0] <= target_bars[0]):
                exit_idx = idx + 1 + stop_bars[0]
                exit_price = stop_loss
                exit_reason = "Stop Loss Hit"
                pnl = (stop_loss - entry_price) * CONTRACT_VALUE * size - COMMISSION * size
            elif len(target_bars) > 0:
                exit_idx = idx + 1 + target_bars[0]
                exit_price = take_profit
                exit_reason = "Take Profit Hit"
                pnl = (take_profit - entry_price) * CONTRACT_VALUE * size - COMMISSION * size
            else:
                exit_idx = min(idx + 50, len(ind['close']) - 1)
                exit_price = ind['close'][exit_idx]
                exit_reason = "Time Exit (50 bars)"
                pnl = (exit_price - entry_price) * CONTRACT_VALUE * size - COMMISSION * size
        else:
            stop_bars = np.where(future_high >= stop_loss)[0]
            target_bars = np.where(future_low <= take_profit)[0]

            if len(stop_bars) > 0 and (len(target_bars) == 0 or stop_bars[0] <= target_bars[0]):
                exit_idx = idx + 1 + stop_bars[0]
                exit_price = stop_loss
                exit_reason = "Stop Loss Hit"
                pnl = (entry_price - stop_loss) * CONTRACT_VALUE * size - COMMISSION * size
            elif len(target_bars) > 0:
                exit_idx = idx + 1 + target_bars[0]
                exit_price = take_profit
                exit_reason = "Take Profit Hit"
                pnl = (entry_price - take_profit) * CONTRACT_VALUE * size - COMMISSION * size
            else:
                exit_idx = min(idx + 50, len(ind['close']) - 1)
                exit_price = ind['close'][exit_idx]
                exit_reason = "Time Exit (50 bars)"
                pnl = (entry_price - exit_price) * CONTRACT_VALUE * size - COMMISSION * size

        exit_time = df['timestamp'].iloc[exit_idx] if 'timestamp' in df.columns else exit_idx
        duration_bars = exit_idx - idx

        # Update capital
        capital += pnl
        if capital > max_equity:
            max_equity = capital
        drawdown = (max_equity - capital) / max_equity * 100

        # Get trade reason
        trade_reason = get_trade_reason(ind, idx, direction)

        trades.append({
            'trade_id': trade_id,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'position_size': size,
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl / (capital - pnl) * 100, 2) if (capital - pnl) > 0 else 0,
            'running_balance': round(capital, 2),
            'drawdown_pct': round(drawdown, 2),
            'duration_bars': duration_bars,
            'exit_reason': exit_reason,
            'trade_reason': trade_reason,
            'result': 'WIN' if pnl > 0 else 'LOSS',
            # Market data at entry
            'entry_open': round(ind['open'][idx], 2),
            'entry_high': round(ind['high'][idx], 2),
            'entry_low': round(ind['low'][idx], 2),
            'entry_close': round(ind['close'][idx], 2),
            'entry_sma_20': round(ind['sma_20'][idx], 2),
            'entry_sma_50': round(ind['sma_50'][idx], 2),
            'entry_sma_100': round(ind['sma_100'][idx], 2),
            'entry_atr': round(ind['atr'][idx], 2),
            'entry_macd': round(ind['macd'][idx], 4),
            'entry_rsi': round(ind['rsi'][idx], 2) if not np.isnan(ind['rsi'][idx]) else 50,
        })

        if capital < INITIAL_CAPITAL * 0.2:
            log(f"Account blew up at trade {trade_id}")
            break

    return pd.DataFrame(trades)


def main():
    log("=" * 70)
    log("DETAILED TRADE LOG GENERATOR")
    log("=" * 70)

    df = load_data(5)

    log("\nGenerating detailed trade logs...")

    # Generate for different modes
    configs = [
        ('both', True, 'both_scaled'),
        ('both', False, 'both_fixed'),
        ('short_only', True, 'short_scaled'),
        ('long_only', True, 'long_scaled'),
    ]

    output_dir = Path("/workspace/results")
    output_dir.mkdir(exist_ok=True)

    for mode, scale, name in configs:
        log(f"\nProcessing: {name}...")

        trades_df = generate_detailed_trade_log(
            df, mode=mode, target_rr=1.5, atr_mult=1.5, scale=scale
        )

        if len(trades_df) > 0:
            # Save CSV
            csv_path = output_dir / f"trades_{name}.csv"
            trades_df.to_csv(csv_path, index=False)
            log(f"  Saved: {csv_path}")

            # Summary stats
            wins = trades_df[trades_df['pnl'] > 0]
            losses = trades_df[trades_df['pnl'] <= 0]

            log(f"  Total Trades: {len(trades_df)}")
            log(f"  Winners: {len(wins)} ({len(wins)/len(trades_df)*100:.1f}%)")
            log(f"  Losers: {len(losses)} ({len(losses)/len(trades_df)*100:.1f}%)")
            log(f"  Final Balance: ${trades_df['running_balance'].iloc[-1]:,.2f}")
            log(f"  Total P&L: ${trades_df['pnl'].sum():,.2f}")
            log(f"  Max Drawdown: {trades_df['drawdown_pct'].max():.1f}%")
            log(f"  Avg Trade Duration: {trades_df['duration_bars'].mean():.1f} bars")

            if scale:
                log(f"  Avg Position Size: {trades_df['position_size'].mean():.1f}")
                log(f"  Max Position Size: {trades_df['position_size'].max()}")

    log("\n" + "=" * 70)
    log("DONE - Trade logs saved to /workspace/results/")
    log("=" * 70)


if __name__ == "__main__":
    main()
