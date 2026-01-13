#!/usr/bin/env python3
"""
DISCOVERED STRATEGIES - Live Trading Implementation

Based on 2-year backtest pattern discovery:
1. Williams Fractals (+$61,650) - Best overall profit
2. Gap Continuation (+$25,787, 2.49 PF) - Best profit factor
3. Gap Fade (+$17,117, 62.7% WR) - Best win rate
4. London Mean Reversion (+$42,044) - Best session strategy
5. Consecutive Bar Reversal (5-7% edge) - Simple pattern

Usage:
    from discovered_strategies import (
        detect_williams_fractals,
        detect_gap_signal,
        detect_london_mean_reversion,
        detect_consecutive_reversal
    )
"""

from typing import List, Dict, Optional, Tuple
from collections import namedtuple
from datetime import datetime
import math

Bar = namedtuple('Bar', ['idx', 'O', 'H', 'L', 'C', 'color', 'body'])


# =============================================================================
# POSITION SIZING WITH SCALING
# =============================================================================

def calculate_contracts(balance: float, starting_balance: float = 1000,
                        scale_multiplier: float = 2.0, contracts_per_scale: int = 5,
                        contract: str = 'MES') -> int:
    """
    Calculate number of contracts based on balance growth.

    Rule: Add 5 contracts every time balance doubles (2x) from original.

    MES ($5/point) Thresholds at $1,000 start:
    - $1,000   -> 1 contract
    - $2,000   -> 6 contracts  (+5)
    - $4,000   -> 11 contracts (+5)
    - $8,000   -> 16 contracts (+5)
    - $16,000  -> 21 contracts (+5)
    - $32,000  -> 26 contracts (+5)

    Args:
        balance: Current account balance
        starting_balance: Original account balance ($1,000 for MES)
        scale_multiplier: Growth multiplier for adding contracts (default 2x = double)
        contracts_per_scale: How many contracts to add per scale (default 5)
        contract: 'MES' or 'ES'

    Returns:
        Number of contracts to trade
    """
    if balance <= 0:
        return 0

    ratio = balance / starting_balance

    if ratio < 1:
        return 1  # Never go below 1 contract

    # How many times have we doubled?
    # At 2x -> 1 double, at 4x -> 2 doubles, at 8x -> 3 doubles
    times_doubled = int(math.log(ratio) / math.log(scale_multiplier))

    # Contracts = 1 + (times_doubled * contracts_per_scale)
    contracts = 1 + (times_doubled * contracts_per_scale)

    return max(1, contracts)


def get_contract_value(contract: str = 'MES') -> float:
    """Get dollar value per point for a contract."""
    return {'MES': 5.0, 'ES': 50.0}.get(contract.upper(), 5.0)


def calculate_pnl(entry: float, exit: float, direction: str,
                  contracts: int = 1, contract: str = 'MES') -> float:
    """
    Calculate P&L in dollars.

    Args:
        entry: Entry price
        exit: Exit price
        direction: 'LONG' or 'SHORT'
        contracts: Number of contracts
        contract: 'MES' or 'ES'

    Returns:
        P&L in dollars
    """
    contract_value = get_contract_value(contract)

    if direction == 'LONG':
        pnl_points = exit - entry
    else:
        pnl_points = entry - exit

    return pnl_points * contract_value * contracts


def get_position_info(balance: float, starting_balance: float = 1000,
                      contract: str = 'MES') -> Dict:
    """
    Get current position sizing information.

    Scaling: +5 contracts every time balance doubles

    Returns dict with:
    - contracts: Current number to trade
    - contract_value: Dollar value per point
    - next_threshold: Balance needed for next scale (+5 contracts)
    - times_doubled: How many times we've doubled
    """
    contracts = calculate_contracts(balance, starting_balance, 2.0, 5, contract)
    contract_value = get_contract_value(contract)

    # Calculate how many times we've doubled
    ratio = balance / starting_balance
    times_doubled = int(math.log(max(1, ratio)) / math.log(2.0)) if ratio >= 1 else 0

    # Next threshold is next doubling
    next_threshold = starting_balance * (2.0 ** (times_doubled + 1))

    return {
        'contracts': contracts,
        'contract_value': contract_value,
        'contract_type': contract.upper(),
        'balance': balance,
        'times_doubled': times_doubled,
        'next_threshold': next_threshold,
        'until_next': next_threshold - balance,
        'next_contracts': contracts + 5,
        'total_exposure_per_point': contracts * contract_value
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calc_sma(bars: List[Bar], period: int) -> List[float]:
    """Calculate Simple Moving Average from bars."""
    closes = [b.C for b in bars]
    sma = []
    for i in range(len(closes)):
        if i < period - 1:
            sma.append(None)
        else:
            sma.append(sum(closes[i - period + 1:i + 1]) / period)
    return sma


def calc_atr(bars: List[Bar], period: int = 14) -> List[float]:
    """Calculate Average True Range from bars."""
    if len(bars) < 2:
        return [None] * len(bars)

    tr = []
    for i, bar in enumerate(bars):
        if i == 0:
            tr.append(bar.H - bar.L)
        else:
            prev_close = bars[i - 1].C
            tr.append(max(
                bar.H - bar.L,
                abs(bar.H - prev_close),
                abs(bar.L - prev_close)
            ))

    atr = []
    for i in range(len(tr)):
        if i < period - 1:
            atr.append(None)
        else:
            atr.append(sum(tr[i - period + 1:i + 1]) / period)
    return atr


def get_current_session_et() -> str:
    """Get current trading session in ET."""
    try:
        from zoneinfo import ZoneInfo
        et = datetime.now(ZoneInfo('America/New_York'))
    except:
        et = datetime.now()
        hour = (et.hour + 3) % 24  # PST + 3 = ET approximation
        et = et.replace(hour=hour)

    hour = et.hour
    minute = et.minute
    time_val = hour * 60 + minute

    if time_val >= 18 * 60 or time_val < 3 * 60:
        return 'ASIA'
    elif time_val < 9 * 60 + 30:
        return 'LONDON'
    elif time_val < 12 * 60:
        return 'NY_AM'
    elif time_val < 16 * 60:
        return 'NY_PM'
    return 'CLOSED'


# =============================================================================
# WILLIAMS FRACTALS STRATEGY (+$61,650 backtest)
# =============================================================================

def find_williams_fractals(bars: List[Bar], period: int = 3) -> Tuple[List[bool], List[bool]]:
    """
    Find Williams Fractals in price data.

    A bullish fractal is a bar where the low is the lowest of surrounding bars.
    A bearish fractal is a bar where the high is the highest of surrounding bars.

    Returns: (bullish_fractals, bearish_fractals) - lists of booleans
    """
    n = len(bars)
    bullish = [False] * n
    bearish = [False] * n

    for i in range(period, n - period):
        # Get window of bars
        window_lows = [bars[j].L for j in range(i - period, i + period + 1)]
        window_highs = [bars[j].H for j in range(i - period, i + period + 1)]

        # Bullish fractal: this bar's low is lowest
        if bars[i].L == min(window_lows):
            bullish[i] = True

        # Bearish fractal: this bar's high is highest
        if bars[i].H == max(window_highs):
            bearish[i] = True

    return bullish, bearish


def detect_williams_fractals(bars: List[Bar], target_rr: float = 3.0,
                             atr_mult: float = 1.5, fractal_period: int = 3) -> Optional[Dict]:
    """
    WILLIAMS FRACTALS + MA ALIGNMENT STRATEGY

    Backtest: +$61,650 over 2 years (3,725 trades, 28% win rate, 1.10 PF)

    Long Setup:
    - MA20 > MA50 > MA100 (bullish alignment)
    - Price pulled back below MA20
    - Price still above MA100 (critical filter)
    - Bullish fractal confirmed

    Short Setup:
    - MA100 > MA50 > MA20 (bearish alignment)
    - Price rallied above MA20
    - Price still below MA100 (critical filter)
    - Bearish fractal confirmed

    Returns signal dict or None
    """
    if len(bars) < 150:
        return None

    # Calculate indicators
    ma20 = calc_sma(bars, 20)
    ma50 = calc_sma(bars, 50)
    ma100 = calc_sma(bars, 100)
    atr = calc_atr(bars, 14)
    bullish_frac, bearish_frac = find_williams_fractals(bars, fractal_period)

    # Current bar
    i = len(bars) - 1
    current_price = bars[i].C

    if ma100[i] is None or atr[i] is None or atr[i] < 0.5:
        return None

    # Check for confirmed fractal (fractal_period bars back due to centered detection)
    fractal_idx = i - fractal_period - 1
    if fractal_idx < 0:
        return None

    # LONG SETUP
    bullish_align = ma20[i] > ma50[i] > ma100[i]
    pullback_long = current_price < ma20[i]
    above_slow = current_price > ma100[i]

    if bullish_frac[fractal_idx] and bullish_align and pullback_long and above_slow:
        entry = current_price
        stop = entry - atr_mult * atr[i]
        risk = entry - stop
        target = entry + target_rr * risk

        return {
            'strategy': 'WILLIAMS_FRACTALS',
            'direction': 'LONG',
            'confidence': 75,
            'entry': round(entry, 2),
            'stop_loss': round(stop, 2),
            'take_profit': round(target, 2),
            'risk': round(risk, 2),
            'rr': target_rr,
            'reasoning': f'Bullish fractal + MA alignment (20>{int(ma50[i])}>{int(ma100[i])}), pullback entry'
        }

    # SHORT SETUP
    bearish_align = ma100[i] > ma50[i] > ma20[i]
    pullback_short = current_price > ma20[i]
    below_slow = current_price < ma100[i]

    if bearish_frac[fractal_idx] and bearish_align and pullback_short and below_slow:
        entry = current_price
        stop = entry + atr_mult * atr[i]
        risk = stop - entry
        target = entry - target_rr * risk

        return {
            'strategy': 'WILLIAMS_FRACTALS',
            'direction': 'SHORT',
            'confidence': 75,
            'entry': round(entry, 2),
            'stop_loss': round(stop, 2),
            'take_profit': round(target, 2),
            'risk': round(risk, 2),
            'rr': target_rr,
            'reasoning': f'Bearish fractal + MA alignment ({int(ma100[i])}>{int(ma50[i])}>20), pullback entry'
        }

    return None


# =============================================================================
# GAP STRATEGIES (+$25,787 continuation, +$17,117 fade)
# =============================================================================

def detect_gap_signal(bars: List[Bar], prev_session_close: float = None,
                      target_rr: float = 2.5) -> Optional[Dict]:
    """
    GAP CONTINUATION / FADE STRATEGY

    Gap Continuation Backtest: +$25,787 (72 trades, 55.6% WR, 2.49 PF)
    Gap Fade Backtest: +$17,117 (303 trades, 62.7% WR, 1.43 PF)

    Rules:
    - Large gaps (8+ pts): Trade continuation (don't fill)
    - Small gaps (1-5 pts): Fade toward fill (81% fill rate)

    Args:
        prev_session_close: Previous session's closing price
        target_rr: Risk:Reward ratio

    Returns signal dict or None
    """
    if len(bars) < 20 or prev_session_close is None:
        return None

    current_price = bars[-1].C
    current_open = bars[-1].O
    atr = calc_atr(bars, 14)

    if atr[-1] is None or atr[-1] < 0.5:
        return None

    # Calculate gap from previous session close to current open
    gap = current_open - prev_session_close
    gap_size = abs(gap)

    # LARGE GAP - Trade continuation (8+ points don't fill)
    if gap_size >= 8:
        if gap > 0:  # Gap up - go LONG
            entry = current_price
            stop = entry - atr[-1] * 2
            risk = entry - stop
            target = entry + target_rr * risk

            return {
                'strategy': 'GAP_CONTINUATION',
                'direction': 'LONG',
                'confidence': 80,
                'entry': round(entry, 2),
                'stop_loss': round(stop, 2),
                'take_profit': round(target, 2),
                'risk': round(risk, 2),
                'rr': target_rr,
                'reasoning': f'Large gap up ({gap_size:.1f}pts) - continuation (20% fill rate for large gaps)'
            }
        else:  # Gap down - go SHORT
            entry = current_price
            stop = entry + atr[-1] * 2
            risk = stop - entry
            target = entry - target_rr * risk

            return {
                'strategy': 'GAP_CONTINUATION',
                'direction': 'SHORT',
                'confidence': 80,
                'entry': round(entry, 2),
                'stop_loss': round(stop, 2),
                'take_profit': round(target, 2),
                'risk': round(risk, 2),
                'rr': target_rr,
                'reasoning': f'Large gap down ({gap_size:.1f}pts) - continuation (20% fill rate for large gaps)'
            }

    # SMALL GAP - Fade toward fill (1-5 points, 81% fill rate)
    elif 1 <= gap_size < 5:
        if gap > 0:  # Gap up - SHORT to fade
            entry = current_price
            stop = entry + atr[-1] * 1.5
            target = prev_session_close  # Gap fill target

            return {
                'strategy': 'GAP_FADE',
                'direction': 'SHORT',
                'confidence': 85,
                'entry': round(entry, 2),
                'stop_loss': round(stop, 2),
                'take_profit': round(target, 2),
                'risk': round(stop - entry, 2),
                'rr': round(abs(entry - target) / (stop - entry), 1),
                'reasoning': f'Small gap up ({gap_size:.1f}pts) - fade to fill (81% fill rate)'
            }
        else:  # Gap down - LONG to fade
            entry = current_price
            stop = entry - atr[-1] * 1.5
            target = prev_session_close  # Gap fill target

            return {
                'strategy': 'GAP_FADE',
                'direction': 'LONG',
                'confidence': 85,
                'entry': round(entry, 2),
                'stop_loss': round(stop, 2),
                'take_profit': round(target, 2),
                'risk': round(entry - stop, 2),
                'rr': round(abs(target - entry) / (entry - stop), 1),
                'reasoning': f'Small gap down ({gap_size:.1f}pts) - fade to fill (81% fill rate)'
            }

    return None


# =============================================================================
# LONDON MEAN REVERSION (+$42,044 backtest)
# =============================================================================

def detect_london_mean_reversion(bars: List[Bar], target_rr: float = 1.5,
                                  atr_mult: float = 1.5,
                                  move_threshold: float = 1.0) -> Optional[Dict]:
    """
    LONDON SESSION MEAN REVERSION STRATEGY

    Backtest: +$42,044 over 2 years (4,325 trades, 42% WR, 1.08 PF)

    Rules:
    - Only trade during LONDON session (3AM-9:30AM ET)
    - If price moved up significantly in last 3 bars, go SHORT (mean revert)
    - If price moved down significantly in last 3 bars, go LONG (mean revert)
    - Use ATR-based stops and targets

    Returns signal dict or None
    """
    if len(bars) < 20:
        return None

    # Check session
    session = get_current_session_et()
    if session != 'LONDON':
        return None

    current_price = bars[-1].C
    atr = calc_atr(bars, 14)

    if atr[-1] is None or atr[-1] < 0.5:
        return None

    # Recent move (last 3 bars)
    if len(bars) < 4:
        return None
    recent_move = current_price - bars[-4].C

    # Mean reversion: fade the move
    if recent_move > move_threshold * atr[-1]:
        # Price moved up, go SHORT
        entry = current_price
        stop = entry + atr_mult * atr[-1]
        risk = stop - entry
        target = entry - target_rr * risk

        return {
            'strategy': 'LONDON_MEAN_REV',
            'direction': 'SHORT',
            'confidence': 70,
            'entry': round(entry, 2),
            'stop_loss': round(stop, 2),
            'take_profit': round(target, 2),
            'risk': round(risk, 2),
            'rr': target_rr,
            'reasoning': f'London session mean reversion - price up {recent_move:.1f}pts, fading'
        }

    elif recent_move < -move_threshold * atr[-1]:
        # Price moved down, go LONG
        entry = current_price
        stop = entry - atr_mult * atr[-1]
        risk = entry - stop
        target = entry + target_rr * risk

        return {
            'strategy': 'LONDON_MEAN_REV',
            'direction': 'LONG',
            'confidence': 70,
            'entry': round(entry, 2),
            'stop_loss': round(stop, 2),
            'take_profit': round(target, 2),
            'risk': round(risk, 2),
            'rr': target_rr,
            'reasoning': f'London session mean reversion - price down {abs(recent_move):.1f}pts, fading'
        }

    return None


# =============================================================================
# CONSECUTIVE BAR REVERSAL (5-7% edge)
# =============================================================================

def detect_consecutive_reversal(bars: List[Bar], min_consecutive: int = 4,
                                 target_rr: float = 1.5,
                                 atr_mult: float = 1.5) -> Optional[Dict]:
    """
    CONSECUTIVE BAR REVERSAL STRATEGY

    Backtest: 5-7% edge (after 4+ consecutive bars, 54-57% reversal rate)

    Rules:
    - After 4+ consecutive green bars, go SHORT (57% reversal rate)
    - After 4+ consecutive red bars, go LONG (57% reversal rate)

    Returns signal dict or None
    """
    if len(bars) < min_consecutive + 10:
        return None

    atr = calc_atr(bars, 14)
    if atr[-1] is None or atr[-1] < 0.5:
        return None

    current_price = bars[-1].C

    # Count consecutive up/down bars ending at current bar
    consec_up = 0
    consec_down = 0

    for i in range(len(bars) - 1, -1, -1):
        bar = bars[i]
        if bar.C > bar.O:  # Green bar
            if consec_down > 0:
                break
            consec_up += 1
        elif bar.C < bar.O:  # Red bar
            if consec_up > 0:
                break
            consec_down += 1
        else:
            break

    # After consecutive UP bars - go SHORT (fade)
    if consec_up >= min_consecutive:
        entry = current_price
        stop = entry + atr_mult * atr[-1]
        risk = stop - entry
        target = entry - target_rr * risk

        return {
            'strategy': 'CONSEC_REVERSAL',
            'direction': 'SHORT',
            'confidence': 65,
            'entry': round(entry, 2),
            'stop_loss': round(stop, 2),
            'take_profit': round(target, 2),
            'risk': round(risk, 2),
            'rr': target_rr,
            'reasoning': f'{consec_up} consecutive green bars - fade for reversal (57% rate)'
        }

    # After consecutive DOWN bars - go LONG (fade)
    if consec_down >= min_consecutive:
        entry = current_price
        stop = entry - atr_mult * atr[-1]
        risk = entry - stop
        target = entry + target_rr * risk

        return {
            'strategy': 'CONSEC_REVERSAL',
            'direction': 'LONG',
            'confidence': 65,
            'entry': round(entry, 2),
            'stop_loss': round(stop, 2),
            'take_profit': round(target, 2),
            'risk': round(risk, 2),
            'rr': target_rr,
            'reasoning': f'{consec_down} consecutive red bars - fade for reversal (57% rate)'
        }

    return None


# =============================================================================
# AGGREGATE ALL DISCOVERED SIGNALS
# =============================================================================

def get_all_discovered_signals(bars: List[Bar],
                                prev_session_close: float = None,
                                live_price: float = None) -> List[Dict]:
    """
    Get all signals from discovered strategies.

    Args:
        bars: List of Bar objects with OHLC data
        prev_session_close: Previous session close for gap detection
        live_price: Current live price (optional, uses bar close if not provided)

    Returns:
        List of signal dicts sorted by confidence
    """
    signals = []

    # Use live price for entry if provided
    if live_price and bars:
        # Create a modified last bar with live price
        last_bar = bars[-1]
        bars = bars[:-1] + [Bar(last_bar.idx, last_bar.O, last_bar.H, last_bar.L,
                               live_price, last_bar.color, last_bar.body)]

    # 1. Williams Fractals (best overall profit)
    williams = detect_williams_fractals(bars)
    if williams:
        signals.append(williams)

    # 2. Gap signals (best profit factor)
    gap = detect_gap_signal(bars, prev_session_close)
    if gap:
        signals.append(gap)

    # 3. London Mean Reversion (session-based)
    london = detect_london_mean_reversion(bars)
    if london:
        signals.append(london)

    # 4. Consecutive Bar Reversal (simple pattern)
    consec = detect_consecutive_reversal(bars)
    if consec:
        signals.append(consec)

    # Sort by confidence
    signals.sort(key=lambda x: x['confidence'], reverse=True)

    return signals


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("DISCOVERED STRATEGIES - Live Trading Implementation")
    print("=" * 60)
    print("\nStrategies implemented:")
    print("  1. Williams Fractals (+$61,650, 28% WR, 1.10 PF)")
    print("  2. Gap Continuation (+$25,787, 55.6% WR, 2.49 PF)")
    print("  3. Gap Fade (+$17,117, 62.7% WR, 1.43 PF)")
    print("  4. London Mean Reversion (+$42,044, 42% WR, 1.08 PF)")
    print("  5. Consecutive Reversal (5-7% edge, 57% reversal rate)")
    print("\nCurrent session:", get_current_session_et())
    print("\nTo use in live_signal_watcher.py:")
    print("  from discovered_strategies import get_all_discovered_signals")
