#!/usr/bin/env python3
"""
SMC (Smart Money Concepts) Confluence Analysis
Integrates with fractal_bot for signal confirmation

Usage:
    python scripts/smc_analysis.py [--log /tmp/fractal_bot.log]
    python scripts/smc_analysis.py --asia-high 6920 --asia-low 6900

================================================================================
LIVE BOT CONFIGURATION (as of 2026-01-12)
================================================================================

ACTIVE STRATEGIES:
  1. CISD (Change in State of Delivery) - PRIMARY
     - Filter: R:R >= 5.0 only
     - Backtest: +$16,512 ES over 226 trades (27.4% WR, 6:1 R:R)
     - Best during ETH (overnight), loses money during RTH

  2. 4H Range Strategy
     - Trades false breakouts of first 4H candle range

  3. Order Blocks / Breaker Blocks / Inducement Traps
     - SMC confluence signals (fewer signals, higher confidence)

DISABLED/NOT RECOMMENDED:
  - CRT (Candle Range Theory) - Loses money standalone (-$2,887 backtest)
  - Basic CISD without R:R filter - Loses money
  - CISD during RTH - Loses money (-$438)

BACKTEST RESULTS (Jan 4-12, 2026):
  Strategy                    | P&L        | WR    | Notes
  ----------------------------|------------|-------|------------------
  CISD R:R >= 5 (ETH only)    | +$16,950   | 28.2% | BEST - Use this
  CISD R:R >= 5 (all hours)   | +$16,512   | 27.4% | Good
  CISD R:R >= 3               | +$12,601   | 28.4% | Decent
  CISD + Trend + High RR      | +$7,375    | 58.5% | Highest WR
  CISD Basic (no filter)      | -$1,162    | 23.1% | LOSING

Run live bot: python scripts/live_signal_watcher.py --realtime
================================================================================
"""

import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import re
import argparse
from datetime import datetime
from collections import namedtuple
from typing import List, Dict, Tuple, Optional


# Session times in Eastern Time (24-hour format)
SESSIONS = {
    'ASIA': (18, 0, 3, 0),      # 6:00 PM - 3:00 AM ET
    'LONDON': (3, 0, 9, 30),    # 3:00 AM - 9:30 AM ET
    'NY_AM': (9, 30, 12, 0),    # 9:30 AM - 12:00 PM ET
    'NY_PM': (12, 0, 16, 0),    # 12:00 PM - 4:00 PM ET
    'CLOSED': (16, 0, 18, 0),   # 4:00 PM - 6:00 PM ET
}


def get_current_session() -> str:
    """Determine current trading session based on Eastern Time"""
    try:
        from zoneinfo import ZoneInfo
        et = datetime.now(ZoneInfo('America/New_York'))
    except:
        # Fallback: assume system is PT, add 3 hours for ET
        et = datetime.now()
        # Rough approximation
        hour = (et.hour + 3) % 24
        minute = et.minute
        et = et.replace(hour=hour)

    hour = et.hour
    minute = et.minute
    time_val = hour * 60 + minute

    # Check each session
    if time_val >= 18 * 60 or time_val < 3 * 60:  # 6PM - 3AM
        return 'ASIA'
    elif time_val < 9 * 60 + 30:  # 3AM - 9:30AM
        return 'LONDON'
    elif time_val < 12 * 60:  # 9:30AM - 12PM
        return 'NY_AM'
    elif time_val < 16 * 60:  # 12PM - 4PM
        return 'NY_PM'
    else:  # 4PM - 6PM
        return 'CLOSED'


def get_session_info() -> Dict:
    """Get current session and relevant info"""
    session = get_current_session()

    info = {
        'current': session,
        'description': {
            'ASIA': 'Asia Session - Range forming, liquidity building',
            'LONDON': 'London Session - Watch for Judas swing',
            'NY_AM': 'NY AM Session - High volume, real moves',
            'NY_PM': 'NY PM Session - Continuation or reversal',
            'CLOSED': 'Market Closed - Wait for Asia open',
        }.get(session, 'Unknown'),
        'liquidity_targets': {
            'ASIA': 'Building session range',
            'LONDON': 'Target Asia H/L, watch for fake move',
            'NY_AM': 'Target London H/L or Asia H/L',
            'NY_PM': 'Target unfilled levels from AM',
            'CLOSED': 'N/A',
        }.get(session, 'Unknown'),
    }

    return info

Bar = namedtuple('Bar', ['idx', 'O', 'H', 'L', 'C', 'color', 'body'])


# ============================================================================
# DISPLACEMENT DETECTION
# ============================================================================

def detect_displacement(bars: List[Bar], min_consecutive: int = 2,
                        min_body_ratio: float = 0.6) -> List[Dict]:
    """
    Detect displacement (strong directional moves).

    Displacement = consecutive candles with:
    - Same direction (all green or all red)
    - Large bodies (body > 60% of candle range)
    - Small wicks relative to body

    Returns list of displacement events with direction and strength.
    """
    displacements = []

    if len(bars) < min_consecutive:
        return displacements

    i = 0
    while i < len(bars) - min_consecutive + 1:
        # Check for bullish displacement
        bullish_run = []
        j = i
        while j < len(bars) and bars[j].color == 'GREEN':
            candle_range = bars[j].H - bars[j].L
            if candle_range > 0:
                body_ratio = bars[j].body / candle_range
                if body_ratio >= min_body_ratio:
                    bullish_run.append(bars[j])
                else:
                    break
            j += 1

        if len(bullish_run) >= min_consecutive:
            total_move = bullish_run[-1].C - bullish_run[0].O
            avg_body = sum(b.body for b in bullish_run) / len(bullish_run)
            displacements.append({
                'type': 'BULLISH',
                'start_idx': bullish_run[0].idx,
                'end_idx': bullish_run[-1].idx,
                'candles': len(bullish_run),
                'move': total_move,
                'avg_body': avg_body,
                'start_price': bullish_run[0].O,
                'end_price': bullish_run[-1].C,
                'strength': 'STRONG' if len(bullish_run) >= 3 or total_move > 5 else 'MODERATE'
            })
            i = j
            continue

        # Check for bearish displacement
        bearish_run = []
        j = i
        while j < len(bars) and bars[j].color == 'RED':
            candle_range = bars[j].H - bars[j].L
            if candle_range > 0:
                body_ratio = bars[j].body / candle_range
                if body_ratio >= min_body_ratio:
                    bearish_run.append(bars[j])
                else:
                    break
            j += 1

        if len(bearish_run) >= min_consecutive:
            total_move = bearish_run[0].O - bearish_run[-1].C
            avg_body = sum(b.body for b in bearish_run) / len(bearish_run)
            displacements.append({
                'type': 'BEARISH',
                'start_idx': bearish_run[0].idx,
                'end_idx': bearish_run[-1].idx,
                'candles': len(bearish_run),
                'move': total_move,
                'avg_body': avg_body,
                'start_price': bearish_run[0].O,
                'end_price': bearish_run[-1].C,
                'strength': 'STRONG' if len(bearish_run) >= 3 or total_move > 5 else 'MODERATE'
            })
            i = j
            continue

        i += 1

    return displacements


# ============================================================================
# BREAK OF STRUCTURE (BOS) DETECTION
# ============================================================================

def detect_bos(bars: List[Bar], swings: Dict, lookback: int = 20) -> List[Dict]:
    """
    Detect Break of Structure events.

    BOS = candle CLOSES through a swing point:
    - Bullish BOS: Close above swing high
    - Bearish BOS: Close below swing low

    Returns list of BOS events with details.
    """
    bos_events = []

    if len(bars) < 5:
        return bos_events

    # Get recent swing points
    recent_swing_highs = sorted(swings.get('highs', []), key=lambda x: x['idx'], reverse=True)[:lookback]
    recent_swing_lows = sorted(swings.get('lows', []), key=lambda x: x['idx'], reverse=True)[:lookback]

    # Check last N bars for BOS
    for bar in bars[-lookback:]:
        # Bullish BOS: Close above swing high
        for swing in recent_swing_highs:
            if swing['idx'] < bar.idx:  # Swing must be before the bar
                if bar.C > swing['price'] and bars[bar.idx - 1].C <= swing['price'] if bar.idx > 0 else True:
                    bos_events.append({
                        'type': 'BULLISH',
                        'bar_idx': bar.idx,
                        'swing_price': swing['price'],
                        'close_price': bar.C,
                        'break_size': bar.C - swing['price'],
                        'description': f"Bullish BOS: Close {bar.C:.2f} > Swing High {swing['price']:.2f}"
                    })
                    break

        # Bearish BOS: Close below swing low
        for swing in recent_swing_lows:
            if swing['idx'] < bar.idx:
                if bar.C < swing['price'] and bars[bar.idx - 1].C >= swing['price'] if bar.idx > 0 else True:
                    bos_events.append({
                        'type': 'BEARISH',
                        'bar_idx': bar.idx,
                        'swing_price': swing['price'],
                        'close_price': bar.C,
                        'break_size': swing['price'] - bar.C,
                        'description': f"Bearish BOS: Close {bar.C:.2f} < Swing Low {swing['price']:.2f}"
                    })
                    break

    return bos_events


def detect_choch(bars: List[Bar], swings: Dict, lookback: int = 30) -> Optional[Dict]:
    """
    Detect Change of Character (CHoCH).

    CHoCH = First BOS against the prevailing trend.
    More significant than regular BOS - signals potential reversal.
    """
    if len(bars) < 10:
        return None

    # Determine prevailing trend from recent price action
    recent_bars = bars[-lookback:] if len(bars) >= lookback else bars
    first_price = recent_bars[0].C
    last_price = recent_bars[-1].C

    if last_price > first_price + 5:
        trend = 'BULLISH'
    elif last_price < first_price - 5:
        trend = 'BEARISH'
    else:
        trend = 'NEUTRAL'

    # Look for BOS against the trend
    bos_events = detect_bos(bars, swings, lookback)

    for bos in bos_events:
        if trend == 'BULLISH' and bos['type'] == 'BEARISH':
            return {
                'type': 'BEARISH_CHOCH',
                'prev_trend': trend,
                'bos': bos,
                'description': f"CHoCH: Bearish break in bullish trend at {bos['swing_price']:.2f}"
            }
        elif trend == 'BEARISH' and bos['type'] == 'BULLISH':
            return {
                'type': 'BULLISH_CHOCH',
                'prev_trend': trend,
                'bos': bos,
                'description': f"CHoCH: Bullish break in bearish trend at {bos['swing_price']:.2f}"
            }

    return None


# ============================================================================
# LIQUIDITY RAID DETECTION
# Per knowledge base: "Price sweeps highs/lows (stop hunts)" before reversal
# ============================================================================

def detect_liquidity_raid(bars: List[Bar], swings: Dict, direction: str,
                          lookback: int = 30) -> Optional[Dict]:
    """
    Detect if a liquidity raid occurred before the current setup.

    Per SMC framework Step 1:
    - For LONG: Price must have swept BELOW a swing low (sell-side liquidity raid)
    - For SHORT: Price must have swept ABOVE a swing high (buy-side liquidity raid)

    A "sweep" means price wicked through the level but closed back above/below it.

    Returns raid info if found, None otherwise.
    """
    if len(bars) < 10:
        return None

    recent_bars = bars[-lookback:] if len(bars) >= lookback else bars

    if direction == 'LONG':
        # Looking for sell-side liquidity raid (sweep below swing low)
        swing_lows = sorted(swings.get('lows', []), key=lambda x: x['idx'], reverse=True)

        for swing in swing_lows[:10]:  # Check recent swing lows
            swing_price = swing['price']
            swing_idx = swing['idx']

            # Look for bars AFTER the swing that swept below it
            for i, bar in enumerate(recent_bars):
                bar_idx = bar.idx if hasattr(bar, 'idx') else i
                if bar_idx > swing_idx:
                    # Check if bar wicked below swing but closed above
                    if bar.L < swing_price and bar.C > swing_price:
                        return {
                            'type': 'SELL_SIDE_RAID',
                            'swing_price': swing_price,
                            'raid_low': bar.L,
                            'raid_bar_idx': bar_idx,
                            'sweep_depth': swing_price - bar.L,
                            'description': f"Swept SSL at {swing_price:.2f}, low {bar.L:.2f}"
                        }
                    # Also check if price went below then recovered in subsequent bars
                    elif bar.L < swing_price:
                        # Check if a later bar closed back above
                        for recovery_bar in recent_bars[i+1:]:
                            if recovery_bar.C > swing_price:
                                return {
                                    'type': 'SELL_SIDE_RAID',
                                    'swing_price': swing_price,
                                    'raid_low': bar.L,
                                    'raid_bar_idx': bar_idx,
                                    'sweep_depth': swing_price - bar.L,
                                    'description': f"Swept SSL at {swing_price:.2f}, recovered"
                                }
                            break

    elif direction == 'SHORT':
        # Looking for buy-side liquidity raid (sweep above swing high)
        swing_highs = sorted(swings.get('highs', []), key=lambda x: x['idx'], reverse=True)

        for swing in swing_highs[:10]:  # Check recent swing highs
            swing_price = swing['price']
            swing_idx = swing['idx']

            # Look for bars AFTER the swing that swept above it
            for i, bar in enumerate(recent_bars):
                bar_idx = bar.idx if hasattr(bar, 'idx') else i
                if bar_idx > swing_idx:
                    # Check if bar wicked above swing but closed below
                    if bar.H > swing_price and bar.C < swing_price:
                        return {
                            'type': 'BUY_SIDE_RAID',
                            'swing_price': swing_price,
                            'raid_high': bar.H,
                            'raid_bar_idx': bar_idx,
                            'sweep_depth': bar.H - swing_price,
                            'description': f"Swept BSL at {swing_price:.2f}, high {bar.H:.2f}"
                        }
                    # Also check if price went above then dropped in subsequent bars
                    elif bar.H > swing_price:
                        for recovery_bar in recent_bars[i+1:]:
                            if recovery_bar.C < swing_price:
                                return {
                                    'type': 'BUY_SIDE_RAID',
                                    'swing_price': swing_price,
                                    'raid_high': bar.H,
                                    'raid_bar_idx': bar_idx,
                                    'sweep_depth': bar.H - swing_price,
                                    'description': f"Swept BSL at {swing_price:.2f}, dropped"
                                }
                            break

    return None


# ============================================================================
# RETRACEMENT VERIFICATION
# Per knowledge base: "DO NOT CHASE - Wait for price to retrace"
# ============================================================================

def verify_retracement(bars: List[Bar], displacement: Dict, direction: str,
                       entry_zone_low: float, entry_zone_high: float) -> Dict:
    """
    Verify that price RETRACED to the entry zone AFTER displacement.

    Per SMC framework Step 4:
    - After displacement + BOS, price should move FURTHER in that direction
    - Then RETRACE back to the FVG/OB zone
    - Entry is on the retracement, NOT during the displacement

    Returns:
    - 'status': 'VALID_RETRACEMENT', 'STILL_IN_DISPLACEMENT', 'NO_DISPLACEMENT'
    - Additional context about the retracement
    """
    if not displacement:
        return {'status': 'NO_DISPLACEMENT', 'valid': False}

    disp_end_idx = displacement.get('end_idx', 0)
    disp_end_price = displacement.get('end_price', 0)

    # Get bars AFTER displacement
    post_disp_bars = [b for b in bars if hasattr(b, 'idx') and b.idx > disp_end_idx]

    if len(post_disp_bars) < 2:
        return {'status': 'STILL_IN_DISPLACEMENT', 'valid': False,
                'reason': 'Displacement just completed - wait for retracement'}

    current_price = bars[-1].C

    if direction == 'LONG':
        # After bullish displacement, price should:
        # 1. Have moved higher (continuation of displacement)
        # 2. Then pulled back DOWN to the entry zone

        # Find highest point after displacement
        post_disp_high = max(b.H for b in post_disp_bars)

        # Check if price extended beyond displacement end
        extended = post_disp_high > disp_end_price + 1.0  # At least 1 pt extension

        # Check if current price is back in entry zone (retracement)
        in_zone = entry_zone_low <= current_price <= entry_zone_high

        # Check if price pulled back from the high
        pullback = post_disp_high - current_price
        meaningful_pullback = pullback >= 1.5  # At least 1.5 pts pullback

        if extended and in_zone and meaningful_pullback:
            return {
                'status': 'VALID_RETRACEMENT',
                'valid': True,
                'extension_high': post_disp_high,
                'pullback_amount': pullback,
                'reason': f"Extended to {post_disp_high:.2f}, pulled back {pullback:.1f} pts to zone"
            }
        elif not extended:
            return {
                'status': 'STILL_IN_DISPLACEMENT',
                'valid': False,
                'reason': f"Price hasn't extended beyond displacement yet"
            }
        elif not meaningful_pullback:
            return {
                'status': 'STILL_IN_DISPLACEMENT',
                'valid': False,
                'reason': f"No meaningful pullback yet (only {pullback:.1f} pts)"
            }
        else:
            return {
                'status': 'PRICE_NOT_IN_ZONE',
                'valid': False,
                'reason': f"Price {current_price:.2f} not in zone {entry_zone_low:.2f}-{entry_zone_high:.2f}"
            }

    elif direction == 'SHORT':
        # After bearish displacement, price should:
        # 1. Have moved lower (continuation of displacement)
        # 2. Then pulled back UP to the entry zone

        # Find lowest point after displacement
        post_disp_low = min(b.L for b in post_disp_bars)

        # Check if price extended beyond displacement end
        extended = post_disp_low < disp_end_price - 1.0  # At least 1 pt extension

        # Check if current price is back in entry zone (retracement)
        in_zone = entry_zone_low <= current_price <= entry_zone_high

        # Check if price pulled back from the low
        pullback = current_price - post_disp_low
        meaningful_pullback = pullback >= 1.5  # At least 1.5 pts pullback

        if extended and in_zone and meaningful_pullback:
            return {
                'status': 'VALID_RETRACEMENT',
                'valid': True,
                'extension_low': post_disp_low,
                'pullback_amount': pullback,
                'reason': f"Extended to {post_disp_low:.2f}, pulled back {pullback:.1f} pts to zone"
            }
        elif not extended:
            return {
                'status': 'STILL_IN_DISPLACEMENT',
                'valid': False,
                'reason': f"Price hasn't extended beyond displacement yet"
            }
        elif not meaningful_pullback:
            return {
                'status': 'STILL_IN_DISPLACEMENT',
                'valid': False,
                'reason': f"No meaningful pullback yet (only {pullback:.1f} pts)"
            }
        else:
            return {
                'status': 'PRICE_NOT_IN_ZONE',
                'valid': False,
                'reason': f"Price {current_price:.2f} not in zone {entry_zone_low:.2f}-{entry_zone_high:.2f}"
            }

    return {'status': 'UNKNOWN', 'valid': False}


# ============================================================================
# MANIPULATION COMPLETION DETECTION
# ============================================================================

def check_manipulation_complete(bars: List[Bar], daily_open: float = None,
                                 session_extreme: str = 'low') -> Dict:
    """
    Check if manipulation phase is likely complete.

    For LONG entries after manipulation:
    1. Price swept below daily open (if bullish PO3)
    2. Session low has formed (not still making new lows)
    3. Displacement occurred (bullish candles showing reversal)
    4. BOS confirmed (close above a swing high)

    Returns status dict with confirmation flags.
    """
    result = {
        'complete': False,
        'flags': [],
        'missing': [],
        'confidence': 0
    }

    if len(bars) < 10:
        result['missing'].append('Insufficient data')
        return result

    current_price = bars[-1].C
    session_high = max(b.H for b in bars)
    session_low = min(b.L for b in bars)

    # Find when session extreme occurred
    if session_extreme == 'low':
        extreme_bar = min(bars, key=lambda b: b.L)
        extreme_price = extreme_bar.L
        bars_since_extreme = len(bars) - extreme_bar.idx - 1

        # Check 1: Session low formed (not in last 3 bars)
        if bars_since_extreme >= 3:
            result['flags'].append(f'Session low formed {bars_since_extreme} bars ago at {extreme_price:.2f}')
            result['confidence'] += 20
        else:
            result['missing'].append(f'Session low too recent ({bars_since_extreme} bars ago) - may go lower')

        # Check 2: Price recovered from low
        recovery = current_price - extreme_price
        if recovery > 3:
            result['flags'].append(f'Price recovered {recovery:.2f} pts from low')
            result['confidence'] += 15
        else:
            result['missing'].append(f'Weak recovery ({recovery:.2f} pts) - reversal not confirmed')

        # Check 3: Bullish displacement after low
        swings = find_swing_points(bars, lookback=3)
        displacements = detect_displacement(bars[extreme_bar.idx:])
        bullish_displacements = [d for d in displacements if d['type'] == 'BULLISH']

        if bullish_displacements:
            latest = bullish_displacements[-1]
            result['flags'].append(f"Bullish displacement: {latest['candles']} candles, {latest['move']:.2f} pts")
            result['confidence'] += 25
        else:
            result['missing'].append('No bullish displacement detected after low')

        # Check 4: BOS above swing high
        bos_events = detect_bos(bars, swings)
        bullish_bos = [b for b in bos_events if b['type'] == 'BULLISH']

        if bullish_bos:
            latest = bullish_bos[-1]
            result['flags'].append(f"Bullish BOS confirmed at {latest['swing_price']:.2f}")
            result['confidence'] += 25
        else:
            result['missing'].append('No bullish BOS - structure not broken')

        # Check 5: Daily open relationship (if provided)
        if daily_open:
            if current_price > daily_open:
                result['flags'].append(f'Price above daily open ({daily_open:.2f}) - expansion started')
                result['confidence'] += 15
            elif extreme_price < daily_open < current_price:
                result['flags'].append(f'Swept below open, now recovering')
                result['confidence'] += 10

    elif session_extreme == 'high':
        extreme_bar = max(bars, key=lambda b: b.H)
        extreme_price = extreme_bar.H
        bars_since_extreme = len(bars) - extreme_bar.idx - 1

        # Similar logic for bearish setup
        if bars_since_extreme >= 3:
            result['flags'].append(f'Session high formed {bars_since_extreme} bars ago at {extreme_price:.2f}')
            result['confidence'] += 20
        else:
            result['missing'].append(f'Session high too recent ({bars_since_extreme} bars ago) - may go higher')

        recovery = extreme_price - current_price
        if recovery > 3:
            result['flags'].append(f'Price dropped {recovery:.2f} pts from high')
            result['confidence'] += 15
        else:
            result['missing'].append(f'Weak drop ({recovery:.2f} pts) - reversal not confirmed')

        swings = find_swing_points(bars, lookback=3)
        displacements = detect_displacement(bars[extreme_bar.idx:])
        bearish_displacements = [d for d in displacements if d['type'] == 'BEARISH']

        if bearish_displacements:
            latest = bearish_displacements[-1]
            result['flags'].append(f"Bearish displacement: {latest['candles']} candles, {latest['move']:.2f} pts")
            result['confidence'] += 25
        else:
            result['missing'].append('No bearish displacement detected after high')

        bos_events = detect_bos(bars, swings)
        bearish_bos = [b for b in bos_events if b['type'] == 'BEARISH']

        if bearish_bos:
            latest = bearish_bos[-1]
            result['flags'].append(f"Bearish BOS confirmed at {latest['swing_price']:.2f}")
            result['confidence'] += 25
        else:
            result['missing'].append('No bearish BOS - structure not broken')

        if daily_open:
            if current_price < daily_open:
                result['flags'].append(f'Price below daily open ({daily_open:.2f}) - expansion started')
                result['confidence'] += 15
            elif extreme_price > daily_open > current_price:
                result['flags'].append(f'Swept above open, now dropping')
                result['confidence'] += 10

    # Determine if manipulation is complete
    result['confidence'] = min(100, result['confidence'])
    result['complete'] = result['confidence'] >= 60 and len(result['missing']) <= 1

    return result


def parse_bars(filepath: str = '/tmp/fractal_bot.log') -> List[Bar]:
    """Parse OHLC bars from fractal_bot log"""
    bars = []
    idx = 0
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if 'BAR' in line and 'O:' in line:
                    match = re.search(r'O:([\d.]+)\s+H:([\d.]+)\s+L:([\d.]+)\s+C:([\d.]+)', line)
                    if match:
                        o, h, l, c = float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))
                        color = 'GREEN' if c > o else 'RED' if c < o else 'DOJI'
                        body = abs(c - o)
                        bars.append(Bar(idx, o, h, l, c, color, body))
                        idx += 1
    except FileNotFoundError:
        print(f"Log file not found: {filepath}")
    return bars


def find_ny_midnight_bar_idx(filepath: str = '/tmp/fractal_bot.log') -> Optional[int]:
    """
    Find the bar index corresponding to NY midnight (00:00 EST) for the current trading day.

    The log timestamps are in PST, so NY midnight = 21:00 PST the previous day.
    Returns the bar index, or None if not found.
    """
    from datetime import datetime, timedelta

    # Get current NY date
    try:
        from zoneinfo import ZoneInfo
        ny_now = datetime.now(ZoneInfo('America/New_York'))
    except ImportError:
        # Fallback: assume PST + 3 = NY
        ny_now = datetime.now() + timedelta(hours=3)

    # NY midnight is 21:00 PST the day before
    # Format: 2026-01-07 21:00 for Jan 8 NY midnight
    pst_midnight = ny_now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(hours=3)
    target_str = pst_midnight.strftime('%Y-%m-%d %H:0')  # e.g., "2026-01-07 21:0"

    bar_idx = 0
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if 'BAR' in line and 'O:' in line:
                    if target_str in line:
                        return bar_idx
                    bar_idx += 1
    except FileNotFoundError:
        pass
    return None


def find_swing_points(bars: List[Bar], lookback: int = 5) -> Dict:
    """Find swing highs and lows (liquidity pools)"""
    swings = {'highs': [], 'lows': []}
    for i in range(lookback, len(bars) - lookback):
        is_swing_high = all(bars[i].H >= bars[i+j].H for j in range(-lookback, lookback+1) if j != 0)
        if is_swing_high:
            swings['highs'].append({'idx': i, 'price': bars[i].H})

        is_swing_low = all(bars[i].L <= bars[i+j].L for j in range(-lookback, lookback+1) if j != 0)
        if is_swing_low:
            swings['lows'].append({'idx': i, 'price': bars[i].L})
    return swings


def find_fvgs(bars: List[Bar], min_gap: float = 0.5) -> List[Dict]:
    """Find Fair Value Gaps (imbalances)"""
    fvgs = []
    for i in range(2, len(bars)):
        # Bullish FVG: Gap between candle 1 high and candle 3 low
        if bars[i].L > bars[i-2].H:
            gap = bars[i].L - bars[i-2].H
            if gap >= min_gap:
                fvgs.append({
                    'type': 'BULLISH',
                    'top': bars[i].L,
                    'bottom': bars[i-2].H,
                    'ce': (bars[i].L + bars[i-2].H) / 2,
                    'idx': i-1,
                    'filled': False
                })

        # Bearish FVG: Gap between candle 1 low and candle 3 high
        if bars[i].H < bars[i-2].L:
            gap = bars[i-2].L - bars[i].H
            if gap >= min_gap:
                fvgs.append({
                    'type': 'BEARISH',
                    'top': bars[i-2].L,
                    'bottom': bars[i].H,
                    'ce': (bars[i-2].L + bars[i].H) / 2,
                    'idx': i-1,
                    'filled': False
                })

    # Check if FVGs are filled
    for fvg in fvgs:
        for bar in bars[fvg['idx']+2:]:
            if fvg['type'] == 'BULLISH' and bar.L <= fvg['bottom']:
                fvg['filled'] = True
                break
            if fvg['type'] == 'BEARISH' and bar.H >= fvg['top']:
                fvg['filled'] = True
                break

    return fvgs


def find_balanced_price_ranges(fvgs: List[Dict]) -> List[Dict]:
    """
    Find Balanced Price Ranges (overlapping bullish + bearish FVGs)
    BPR = buy-side delivery + sell-side delivery = balanced zone
    Acts as strong support/resistance when price returns
    """
    bprs = []
    bullish_fvgs = [f for f in fvgs if f['type'] == 'BULLISH']
    bearish_fvgs = [f for f in fvgs if f['type'] == 'BEARISH']

    for bull in bullish_fvgs:
        for bear in bearish_fvgs:
            # Check if they overlap
            overlap_top = min(bull['top'], bear['top'])
            overlap_bottom = max(bull['bottom'], bear['bottom'])

            if overlap_bottom < overlap_top:  # They overlap
                # Determine direction price moved after BPR formed
                # If bearish FVG formed after bullish, price went down -> BPR is resistance
                # If bullish FVG formed after bearish, price went up -> BPR is support
                if bear['idx'] > bull['idx']:
                    bpr_type = 'RESISTANCE'  # Price moved down after
                else:
                    bpr_type = 'SUPPORT'  # Price moved up after

                bprs.append({
                    'type': bpr_type,
                    'top': overlap_top,
                    'bottom': overlap_bottom,
                    'mid': (overlap_top + overlap_bottom) / 2,
                    'bull_fvg': bull,
                    'bear_fvg': bear
                })

    # Dedupe overlapping BPRs
    unique_bprs = []
    for bpr in bprs:
        is_dup = any(abs(bpr['mid'] - u['mid']) < 2 for u in unique_bprs)
        if not is_dup:
            unique_bprs.append(bpr)

    return unique_bprs


def find_order_blocks(bars: List[Bar], lookback: int = 100,
                       fvgs: List[Dict] = None, swings: Dict = None) -> List[Dict]:
    """
    Find Order Blocks with full 3-rule validation:
    1. Must create inefficiency/gap (FVG)
    2. Must be unmitigated (untested)
    3. Must lead to BOS/CHoCH

    Returns validated OBs with confidence scores.
    """
    obs = []

    # Pre-calculate FVGs and swings if not provided
    if fvgs is None:
        fvgs = find_fvgs(bars, min_gap=0.25)
    if swings is None:
        swings = find_swing_points(bars, lookback=3)

    for i in range(4, min(lookback, len(bars))):
        idx = len(bars) - i
        if idx < 4:
            break

        # Check for bullish displacement (2+ green candles with decent body)
        if (bars[idx].color == 'GREEN' and bars[idx-1].color == 'GREEN' and
            bars[idx].body > 0.5 and bars[idx-1].body > 0.5):
            for j in range(idx-2, max(0, idx-8), -1):
                if bars[j].color == 'RED':
                    ob_candidate = {
                        'type': 'BULLISH',
                        'high': bars[j].H,
                        'low': bars[j].L,
                        'idx': j,
                        'displacement_start': idx - 1,
                        'displacement_end': idx,
                        'valid': False,
                        'rules': {'has_fvg': False, 'unmitigated': True, 'has_bos': False},
                        'confidence': 0,
                        'mitigated_at': None
                    }

                    # RULE 1: Check for FVG after this candle
                    for fvg in fvgs:
                        if fvg['type'] == 'BULLISH' and fvg['idx'] >= j and fvg['idx'] <= idx + 2:
                            ob_candidate['rules']['has_fvg'] = True
                            ob_candidate['fvg'] = fvg
                            break

                    # RULE 2: Check if unmitigated (no price touch after formation)
                    for k in range(idx + 1, len(bars)):
                        if bars[k].L <= ob_candidate['high']:
                            ob_candidate['rules']['unmitigated'] = False
                            ob_candidate['mitigated_at'] = k
                            break

                    # RULE 3: Check for BOS (higher high after the move)
                    if idx + 5 < len(bars):
                        pre_high = max(bars[m].H for m in range(max(0, j-10), j))
                        post_high = max(bars[m].H for m in range(idx, min(len(bars), idx + 20)))
                        if post_high > pre_high:
                            ob_candidate['rules']['has_bos'] = True
                            ob_candidate['bos_price'] = post_high

                    # Calculate confidence score
                    score = 0
                    if ob_candidate['rules']['has_fvg']:
                        score += 35
                    if ob_candidate['rules']['unmitigated']:
                        score += 35
                    if ob_candidate['rules']['has_bos']:
                        score += 30
                    ob_candidate['confidence'] = score

                    # Valid if all 3 rules pass
                    ob_candidate['valid'] = all(ob_candidate['rules'].values())

                    obs.append(ob_candidate)
                    break

        # Check for bearish displacement
        if (bars[idx].color == 'RED' and bars[idx-1].color == 'RED' and
            bars[idx].body > 0.5 and bars[idx-1].body > 0.5):
            for j in range(idx-2, max(0, idx-8), -1):
                if bars[j].color == 'GREEN':
                    ob_candidate = {
                        'type': 'BEARISH',
                        'high': bars[j].H,
                        'low': bars[j].L,
                        'idx': j,
                        'displacement_start': idx - 1,
                        'displacement_end': idx,
                        'valid': False,
                        'rules': {'has_fvg': False, 'unmitigated': True, 'has_bos': False},
                        'confidence': 0,
                        'mitigated_at': None
                    }

                    # RULE 1: Check for FVG after this candle
                    for fvg in fvgs:
                        if fvg['type'] == 'BEARISH' and fvg['idx'] >= j and fvg['idx'] <= idx + 2:
                            ob_candidate['rules']['has_fvg'] = True
                            ob_candidate['fvg'] = fvg
                            break

                    # RULE 2: Check if unmitigated (no price touch after formation)
                    for k in range(idx + 1, len(bars)):
                        if bars[k].H >= ob_candidate['low']:
                            ob_candidate['rules']['unmitigated'] = False
                            ob_candidate['mitigated_at'] = k
                            break

                    # RULE 3: Check for BOS (lower low after the move)
                    if idx + 5 < len(bars):
                        pre_low = min(bars[m].L for m in range(max(0, j-10), j))
                        post_low = min(bars[m].L for m in range(idx, min(len(bars), idx + 20)))
                        if post_low < pre_low:
                            ob_candidate['rules']['has_bos'] = True
                            ob_candidate['bos_price'] = post_low

                    # Calculate confidence score
                    score = 0
                    if ob_candidate['rules']['has_fvg']:
                        score += 35
                    if ob_candidate['rules']['unmitigated']:
                        score += 35
                    if ob_candidate['rules']['has_bos']:
                        score += 30
                    ob_candidate['confidence'] = score

                    # Valid if all 3 rules pass
                    ob_candidate['valid'] = all(ob_candidate['rules'].values())

                    obs.append(ob_candidate)
                    break

    # Dedupe by price zone
    unique_obs = []
    for ob in obs:
        is_dup = any(abs(ob['low'] - u['low']) < 1 and ob['type'] == u['type'] for u in unique_obs)
        if not is_dup:
            unique_obs.append(ob)

    return unique_obs


def find_breaker_blocks(bars: List[Bar], order_blocks: List[Dict]) -> List[Dict]:
    """
    Find Breaker Blocks - previously valid OBs that have been broken through.
    A broken bullish OB becomes bearish resistance.
    A broken bearish OB becomes bullish support.
    """
    breakers = []

    for ob in order_blocks:
        # Only consider OBs that were mitigated (broken)
        if ob['rules']['unmitigated']:
            continue

        mitigated_idx = ob.get('mitigated_at')
        if mitigated_idx is None:
            continue

        # Check if price broke THROUGH the OB (not just touched)
        if ob['type'] == 'BULLISH':
            # Bullish OB broken = price closed below the OB low
            broke_through = False
            for k in range(mitigated_idx, min(len(bars), mitigated_idx + 5)):
                if bars[k].C < ob['low']:
                    broke_through = True
                    break

            if broke_through:
                # Check if OB is now unmitigated from above (for short entry)
                retested = False
                for k in range(mitigated_idx + 1, len(bars)):
                    if bars[k].H >= ob['low']:
                        retested = True
                        break

                breakers.append({
                    'type': 'BEARISH',  # Flipped - now resistance
                    'original_type': 'BULLISH',
                    'high': ob['high'],
                    'low': ob['low'],
                    'idx': ob['idx'],
                    'broke_at': mitigated_idx,
                    'unmitigated': not retested,
                    'description': f"Broken bullish OB at {ob['low']:.2f}-{ob['high']:.2f} now resistance"
                })

        elif ob['type'] == 'BEARISH':
            # Bearish OB broken = price closed above the OB high
            broke_through = False
            for k in range(mitigated_idx, min(len(bars), mitigated_idx + 5)):
                if bars[k].C > ob['high']:
                    broke_through = True
                    break

            if broke_through:
                # Check if OB is now unmitigated from below (for long entry)
                retested = False
                for k in range(mitigated_idx + 1, len(bars)):
                    if bars[k].L <= ob['high']:
                        retested = True
                        break

                breakers.append({
                    'type': 'BULLISH',  # Flipped - now support
                    'original_type': 'BEARISH',
                    'high': ob['high'],
                    'low': ob['low'],
                    'idx': ob['idx'],
                    'broke_at': mitigated_idx,
                    'unmitigated': not retested,
                    'description': f"Broken bearish OB at {ob['low']:.2f}-{ob['high']:.2f} now support"
                })

    return breakers


def find_inducement_traps(bars: List[Bar], order_blocks: List[Dict],
                          lookback: int = 50, max_ob_age: int = 100) -> List[Dict]:
    """
    Find Inducement Trap setups - minor key level near major OB.

    Pattern: Minor S/R level forms above/below a major OB,
    price breaks the minor level to trap traders, then reverses from OB.

    Args:
        max_ob_age: Maximum bars since OB formation (default 100 = ~8 hours on 5min)
    """
    traps = []

    if len(bars) < lookback:
        return traps

    current_bar_idx = len(bars) - 1

    # Only consider valid, unmitigated, FRESH OBs
    valid_obs = [ob for ob in order_blocks
                 if ob['valid'] and ob['rules']['unmitigated']
                 and (current_bar_idx - ob['idx']) <= max_ob_age]

    for ob in valid_obs:
        # Look for minor key levels near this OB
        ob_mid = (ob['high'] + ob['low']) / 2

        if ob['type'] == 'BULLISH':
            # Look for minor resistance ABOVE the bullish OB
            # (inducement level that might get swept before OB entry)

            # Find swing highs between OB and current price
            recent_highs = []
            for i in range(ob['idx'] + 5, len(bars) - 2):
                if (bars[i].H > bars[i-1].H and bars[i].H > bars[i+1].H and
                    bars[i].H > ob['high'] and bars[i].H < ob['high'] + 10):
                    recent_highs.append({'idx': i, 'price': bars[i].H})

            # Check for multiple rejections (2+) at similar level = inducement
            if len(recent_highs) >= 2:
                # Group by price level
                level_price = recent_highs[0]['price']
                cluster = [h for h in recent_highs if abs(h['price'] - level_price) < 1.5]

                if len(cluster) >= 2:
                    inducement_level = sum(h['price'] for h in cluster) / len(cluster)
                    current_price = bars[-1].C
                    last_rejection_idx = max(h['idx'] for h in cluster)

                    # Check if inducement was already swept (price closed above it after rejections)
                    inducement_swept = any(
                        bars[i].C > inducement_level + 0.5
                        for i in range(last_rejection_idx + 1, len(bars))
                    )

                    # Check if price is between inducement and OB, and inducement NOT swept
                    if ob['high'] < current_price < inducement_level and not inducement_swept:
                        traps.append({
                            'type': 'LONG',
                            'ob': ob,
                            'inducement_level': inducement_level,
                            'ob_entry': ob_mid,
                            'rejections': len(cluster),
                            'description': f"Inducement at {inducement_level:.2f}, OB entry at {ob_mid:.2f}",
                            'setup': 'ACTIVE' if current_price > ob['high'] else 'WAITING'
                        })

        elif ob['type'] == 'BEARISH':
            # Look for minor support BELOW the bearish OB
            recent_lows = []
            for i in range(ob['idx'] + 5, len(bars) - 2):
                if (bars[i].L < bars[i-1].L and bars[i].L < bars[i+1].L and
                    bars[i].L < ob['low'] and bars[i].L > ob['low'] - 10):
                    recent_lows.append({'idx': i, 'price': bars[i].L})

            if len(recent_lows) >= 2:
                level_price = recent_lows[0]['price']
                cluster = [l for l in recent_lows if abs(l['price'] - level_price) < 1.5]

                if len(cluster) >= 2:
                    inducement_level = sum(l['price'] for l in cluster) / len(cluster)
                    current_price = bars[-1].C
                    last_rejection_idx = max(l['idx'] for l in cluster)

                    # Check if inducement was already swept (price closed below it after rejections)
                    inducement_swept = any(
                        bars[i].C < inducement_level - 0.5
                        for i in range(last_rejection_idx + 1, len(bars))
                    )

                    if inducement_level < current_price < ob['low'] and not inducement_swept:
                        traps.append({
                            'type': 'SHORT',
                            'ob': ob,
                            'inducement_level': inducement_level,
                            'ob_entry': ob_mid,
                            'rejections': len(cluster),
                            'description': f"Inducement at {inducement_level:.2f}, OB entry at {ob_mid:.2f}",
                            'setup': 'ACTIVE' if current_price < ob['low'] else 'WAITING'
                        })

    return traps


# ============================================================================
# 4-HOUR RANGE SCALPING STRATEGY
# ============================================================================

def calculate_4hour_range(bars: List[Bar], ny_midnight_idx: int = None,
                          filepath: str = None) -> Optional[Dict]:
    """
    Calculate the 4-hour range from the first 4H candle of the day (NY time).

    For intraday 5-min bars, the first 4H candle spans bars 0-47 (48 x 5min = 4 hours).
    If ny_midnight_idx is provided, use that as the start.
    If filepath is provided and ny_midnight_idx is None, auto-detect the midnight bar.

    Returns: {'high': float, 'low': float, 'range': float, 'bars_used': int}
    """
    if len(bars) < 48:
        return None

    # Auto-detect NY midnight bar if not provided
    if ny_midnight_idx is None and filepath:
        ny_midnight_idx = find_ny_midnight_bar_idx(filepath)

    # Use detected midnight or fall back to first bar
    start_idx = ny_midnight_idx if ny_midnight_idx is not None else 0
    end_idx = min(start_idx + 48, len(bars))

    if end_idx - start_idx < 48:
        # Not enough bars for full 4H range
        return None

    range_bars = bars[start_idx:end_idx]
    range_high = max(b.H for b in range_bars)
    range_low = min(b.L for b in range_bars)

    return {
        'high': range_high,
        'low': range_low,
        'range': range_high - range_low,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'bars_used': len(range_bars)
    }


def analyze_4hour_range_strategy(bars: List[Bar], four_hour_range: Dict = None,
                                  filepath: str = None) -> Dict:
    """
    Analyze 4-Hour Range Scalping Strategy setups.

    Strategy Rules:
    1. Mark high/low of first 4H candle (NY time)
    2. Wait for 5-min candle to CLOSE outside range
    3. Wait for candle to CLOSE back inside range
    4. Enter opposite direction: broke high -> SHORT, broke low -> LONG
    5. Stop at breakout extreme, TP at 2R

    Args:
        bars: List of Bar objects
        four_hour_range: Optional manual range override
        filepath: Log file path for auto-detecting NY midnight

    Returns analysis with signals and trade setups.
    """
    result = {
        'range': four_hour_range,
        'status': 'NO_RANGE',
        'signals': [],
        'current_setup': None,
        'historical_setups': []
    }

    # Check if we're past 4:00 AM NY time (first 4H candle must be closed)
    from datetime import datetime, timedelta
    try:
        from zoneinfo import ZoneInfo
        ny_now = datetime.now(ZoneInfo('America/New_York'))
    except ImportError:
        ny_now = datetime.now() + timedelta(hours=3)  # Fallback PST+3

    ny_hour = ny_now.hour
    # Only allow 4H Range signals after 4:00 AM NY (first 4H candle closed)
    # and before midnight (same trading day)
    if ny_hour < 4:
        result['status'] = 'WAITING_4H_CLOSE'
        return result

    if four_hour_range is None:
        four_hour_range = calculate_4hour_range(bars, filepath=filepath)

    if four_hour_range is None:
        return result

    result['range'] = four_hour_range
    result['status'] = 'RANGE_SET'

    range_high = four_hour_range['high']
    range_low = four_hour_range['low']
    range_end_idx = four_hour_range['end_idx']

    # Track breakout states
    broke_high = False
    broke_low = False
    breakout_extreme = None
    breakout_idx = None

    # Analyze bars after the 4H range formed
    for i in range(range_end_idx, len(bars)):
        bar = bars[i]
        prev_bar = bars[i-1] if i > 0 else bar

        # Check for breakout CLOSE above range high
        if bar.C > range_high and not broke_high:
            broke_high = True
            broke_low = False
            breakout_extreme = bar.H
            breakout_idx = i
            result['signals'].append({
                'idx': i,
                'type': 'BREAKOUT_HIGH',
                'price': bar.C,
                'description': f"Closed above range high {range_high:.2f}"
            })

        # Check for breakout CLOSE below range low
        elif bar.C < range_low and not broke_low:
            broke_low = True
            broke_high = False
            breakout_extreme = bar.L
            breakout_idx = i
            result['signals'].append({
                'idx': i,
                'type': 'BREAKOUT_LOW',
                'price': bar.C,
                'description': f"Closed below range low {range_low:.2f}"
            })

        # Update breakout extreme while outside range
        if broke_high and bar.C > range_high:
            if bar.H > breakout_extreme:
                breakout_extreme = bar.H

        if broke_low and bar.C < range_low:
            if bar.L < breakout_extreme:
                breakout_extreme = bar.L

        # Check for re-entry CLOSE back inside range
        if broke_high and bar.C <= range_high and bar.C >= range_low:
            # SHORT signal - price broke above, came back inside
            stop_loss = breakout_extreme + 0.25  # Just above the high
            risk = stop_loss - bar.C
            take_profit = bar.C - (risk * 2)  # 2R target

            setup = {
                'signal': 'SHORT',
                'idx': i,
                'entry': bar.C,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk': risk,
                'reward': risk * 2,
                'rr': 2.0,
                'breakout_extreme': breakout_extreme,
                'description': f"4H Range SHORT: Entry {bar.C:.2f}, Stop {stop_loss:.2f}, TP {take_profit:.2f}"
            }
            result['historical_setups'].append(setup)
            result['signals'].append({
                'idx': i,
                'type': 'REENTRY_SHORT',
                'price': bar.C,
                'description': f"Re-entered range from above - SHORT signal"
            })

            # Reset state
            broke_high = False
            breakout_extreme = None

        elif broke_low and bar.C >= range_low and bar.C <= range_high:
            # LONG signal - price broke below, came back inside
            stop_loss = breakout_extreme - 0.25  # Just below the low
            risk = bar.C - stop_loss
            take_profit = bar.C + (risk * 2)  # 2R target

            setup = {
                'signal': 'LONG',
                'idx': i,
                'entry': bar.C,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk': risk,
                'reward': risk * 2,
                'rr': 2.0,
                'breakout_extreme': breakout_extreme,
                'description': f"4H Range LONG: Entry {bar.C:.2f}, Stop {stop_loss:.2f}, TP {take_profit:.2f}"
            }
            result['historical_setups'].append(setup)
            result['signals'].append({
                'idx': i,
                'type': 'REENTRY_LONG',
                'price': bar.C,
                'description': f"Re-entered range from below - LONG signal"
            })

            # Reset state
            broke_low = False
            breakout_extreme = None

    # Determine current status
    current_price = bars[-1].C if bars else 0

    if broke_high:
        result['status'] = 'WAITING_REENTRY_SHORT'
        result['current_setup'] = {
            'pending_signal': 'SHORT',
            'condition': f"Wait for close <= {range_high:.2f}",
            'stop_will_be': breakout_extreme + 0.25,
            'current_price': current_price
        }
    elif broke_low:
        result['status'] = 'WAITING_REENTRY_LONG'
        result['current_setup'] = {
            'pending_signal': 'LONG',
            'condition': f"Wait for close >= {range_low:.2f}",
            'stop_will_be': breakout_extreme - 0.25,
            'current_price': current_price
        }
    elif range_low <= current_price <= range_high:
        result['status'] = 'INSIDE_RANGE'
    else:
        result['status'] = 'OUTSIDE_RANGE_NO_CLOSE'

    return result


def get_4hour_range_signal(bars: List[Bar], four_hour_range: Dict = None,
                           filepath: str = None) -> Optional[Dict]:
    """
    Get current actionable signal from 4-hour range strategy.

    Returns signal dict if there's an active setup, None otherwise.
    """
    analysis = analyze_4hour_range_strategy(bars, four_hour_range, filepath=filepath)

    if analysis['status'] in ['WAITING_REENTRY_SHORT', 'WAITING_REENTRY_LONG']:
        return {
            'strategy': '4H_RANGE',
            'status': analysis['status'],
            'setup': analysis['current_setup'],
            'range': analysis['range'],
            'confidence': 70  # Base confidence for this strategy
        }

    # Check if we have a recent signal (within last 10 bars) that's still valid
    if analysis['historical_setups']:
        last_setup = analysis['historical_setups'][-1]
        bars_since_signal = len(bars) - 1 - last_setup['idx']

        # Signal is active if it triggered recently (within 10 bars)
        # and hasn't been invalidated (stop/target not yet hit)
        if bars_since_signal <= 10:
            current_price = bars[-1].C
            stop = last_setup['stop_loss']
            target = last_setup['take_profit']

            # Check if trade would still be valid (not stopped or targeted)
            if last_setup['signal'] == 'LONG':
                stopped = current_price <= stop
                targeted = current_price >= target
            else:  # SHORT
                stopped = current_price >= stop
                targeted = current_price <= target

            if not stopped and not targeted:
                return {
                    'strategy': '4H_RANGE',
                    'status': 'SIGNAL_ACTIVE',
                    'setup': last_setup,
                    'range': analysis['range'],
                    'confidence': 75
                }

    return None


def calculate_drt(high: float, low: float) -> Dict:
    """Calculate Dealing Range levels"""
    range_size = high - low
    return {
        'high': high,
        'low': low,
        'range': range_size,
        '25': high - (range_size * 0.25),
        '50': high - (range_size * 0.50),
        '75': high - (range_size * 0.75),
    }


def analyze_confluence(current_price: float, drt: Dict, obs: List[Dict],
                       fvgs: List[Dict], swings: Dict, bprs: List[Dict] = None) -> Tuple[List, str, int, int]:
    """Analyze confluence of SMC factors at current price"""
    signals = []

    # Check DRT zone
    price_pct = ((drt['high'] - current_price) / drt['range']) * 100 if drt['range'] > 0 else 50

    if price_pct >= 75:
        signals.append(('DRT', 'BULLISH', f"Price in DISCOUNT zone ({price_pct:.0f}% DRT) - favor LONGS"))
    elif price_pct >= 50:
        signals.append(('DRT', 'NEUTRAL', f"Price BELOW equilibrium ({price_pct:.0f}% DRT)"))
    elif price_pct >= 25:
        signals.append(('DRT', 'NEUTRAL', f"Price ABOVE equilibrium ({price_pct:.0f}% DRT)"))
    else:
        signals.append(('DRT', 'BEARISH', f"Price in PREMIUM zone ({price_pct:.0f}% DRT) - favor SHORTS"))

    # Check Order Blocks
    for ob in obs:
        if ob['low'] <= current_price <= ob['high']:
            signals.append(('OB', ob['type'], f"IN {ob['type']} OB: {ob['low']:.2f}-{ob['high']:.2f}"))
        elif ob['type'] == 'BULLISH' and current_price < ob['low'] and current_price > ob['low'] - 3:
            signals.append(('OB', 'BULLISH', f"Near BULLISH OB: {ob['low']:.2f}-{ob['high']:.2f}"))
        elif ob['type'] == 'BEARISH' and current_price > ob['high'] and current_price < ob['high'] + 3:
            signals.append(('OB', 'BEARISH', f"Near BEARISH OB: {ob['low']:.2f}-{ob['high']:.2f}"))

    # Check unfilled FVGs
    for fvg in fvgs:
        if not fvg['filled']:
            if fvg['bottom'] <= current_price <= fvg['top']:
                signals.append(('FVG', fvg['type'], f"IN {fvg['type']} FVG: {fvg['bottom']:.2f}-{fvg['top']:.2f} (CE: {fvg['ce']:.2f})"))
            elif fvg['type'] == 'BULLISH' and current_price < fvg['bottom'] and current_price > fvg['bottom'] - 5:
                signals.append(('FVG', 'BULLISH', f"Unfilled BULLISH FVG above: {fvg['bottom']:.2f}-{fvg['top']:.2f}"))
            elif fvg['type'] == 'BEARISH' and current_price > fvg['top'] and current_price < fvg['top'] + 5:
                signals.append(('FVG', 'BEARISH', f"Unfilled BEARISH FVG below: {fvg['bottom']:.2f}-{fvg['top']:.2f}"))

    # Check Balanced Price Ranges
    if bprs:
        for bpr in bprs:
            if bpr['bottom'] <= current_price <= bpr['top']:
                signals.append(('BPR', 'BEARISH' if bpr['type'] == 'RESISTANCE' else 'BULLISH',
                               f"IN BPR ({bpr['type']}): {bpr['bottom']:.2f}-{bpr['top']:.2f}"))
            elif bpr['type'] == 'RESISTANCE' and current_price < bpr['bottom'] and current_price > bpr['bottom'] - 5:
                signals.append(('BPR', 'BEARISH', f"BPR RESISTANCE above: {bpr['bottom']:.2f}-{bpr['top']:.2f}"))
            elif bpr['type'] == 'SUPPORT' and current_price > bpr['top'] and current_price < bpr['top'] + 5:
                signals.append(('BPR', 'BULLISH', f"BPR SUPPORT below: {bpr['bottom']:.2f}-{bpr['top']:.2f}"))

    # Check swing points for liquidity targets
    recent_swing_highs = [s for s in swings['highs'] if s['price'] > current_price]
    recent_swing_lows = [s for s in swings['lows'] if s['price'] < current_price]

    if recent_swing_highs:
        nearest_high = min(recent_swing_highs, key=lambda x: x['price'])
        signals.append(('LIQ', 'BSL', f"Buy-side liquidity target: {nearest_high['price']:.2f}"))

    if recent_swing_lows:
        nearest_low = max(recent_swing_lows, key=lambda x: x['price'])
        signals.append(('LIQ', 'SSL', f"Sell-side liquidity target: {nearest_low['price']:.2f}"))

    # Check for CE breaks (FVG fill prediction)
    for fvg in fvgs:
        if not fvg['filled']:
            ce = fvg['ce']
            if fvg['type'] == 'BEARISH':
                # If close is above CE of bearish FVG, expect fill to top
                if current_price > ce and current_price < fvg['top']:
                    signals.append(('CE_BREAK', 'BULLISH',
                                   f"CE BREAK: Close above CE ({ce:.2f}) of bearish FVG - expect fill to {fvg['top']:.2f}"))
            elif fvg['type'] == 'BULLISH':
                # If close is below CE of bullish FVG, expect fill to bottom
                if current_price < ce and current_price > fvg['bottom']:
                    signals.append(('CE_BREAK', 'BEARISH',
                                   f"CE BREAK: Close below CE ({ce:.2f}) of bullish FVG - expect fill to {fvg['bottom']:.2f}"))

    # Check for Turtle Soup setups
    # Bearish: Old high with bearish FVG above it
    for swing in swings['highs']:
        for fvg in fvgs:
            if fvg['type'] == 'BEARISH' and not fvg['filled']:
                if fvg['bottom'] > swing['price'] and fvg['bottom'] - swing['price'] < 5:
                    # Bearish FVG sits just above old high
                    if current_price > swing['price'] and current_price <= fvg['top']:
                        signals.append(('TURTLE', 'BEARISH',
                                       f"TURTLE SOUP SHORT: Swept {swing['price']:.2f}, FVG {fvg['bottom']:.2f}-{fvg['top']:.2f}"))
                        break

    # Bullish: Old low with bullish FVG below it
    for swing in swings['lows']:
        for fvg in fvgs:
            if fvg['type'] == 'BULLISH' and not fvg['filled']:
                if fvg['top'] < swing['price'] and swing['price'] - fvg['top'] < 5:
                    # Bullish FVG sits just below old low
                    if current_price < swing['price'] and current_price >= fvg['bottom']:
                        signals.append(('TURTLE', 'BULLISH',
                                       f"TURTLE SOUP LONG: Swept {swing['price']:.2f}, FVG {fvg['bottom']:.2f}-{fvg['top']:.2f}"))
                        break

    # Calculate bias
    bullish_count = sum(1 for s in signals if s[1] in ['BULLISH', 'SSL'])
    bearish_count = sum(1 for s in signals if s[1] in ['BEARISH', 'BSL'])

    if bullish_count > bearish_count + 1:
        bias = 'BULLISH'
    elif bearish_count > bullish_count + 1:
        bias = 'BEARISH'
    else:
        bias = 'NEUTRAL'

    return signals, bias, bullish_count, bearish_count


def calculate_trade_probability(signal_type: str, current_price: float, drt: Dict,
                                 obs: List[Dict], fvgs: List[Dict], swings: Dict,
                                 bprs: List[Dict] = None, bars: List[Bar] = None,
                                 daily_open: float = None) -> Dict:
    """
    Calculate trade probability using weighted scoring.
    Each SMC element contributes to overall probability.

    NEW: Includes displacement, BOS, and manipulation completion checks.

    Returns probability, entry, stop, target, and reasoning.
    """
    score = 0  # Start at zero - must earn confidence from SMC factors
    factors = []
    entry = current_price
    stop = None
    target = None

    price_pct = ((drt['high'] - current_price) / drt['range']) * 100 if drt['range'] > 0 else 50

    # ========================================================================
    # NEW: CRITICAL CONFIRMATION FACTORS (Displacement, BOS, Manipulation)
    # These are the factors that were missing and caused the failed trade
    # ========================================================================

    has_displacement = False
    has_bos = False
    manipulation_complete = False
    has_liquidity_raid = False
    has_retracement = False

    if bars and len(bars) >= 10:
        # Detect displacement
        displacements = detect_displacement(bars)

        # Detect BOS
        bos_events = detect_bos(bars, swings)

        # Check manipulation completion
        if signal_type == 'LONG':
            manip_check = check_manipulation_complete(bars, daily_open, 'low')

            # Check for bullish displacement in RECENT bars (last 10 only)
            # DISPLACEMENT IS CRITICAL - without it, no proof of reversal
            # Per knowledge base: "Displacement is NOT an entry - it's confirmation"
            recent_bullish_disp = [d for d in displacements
                                   if d['type'] == 'BULLISH' and d['end_idx'] >= len(bars) - 10]
            if recent_bullish_disp:
                has_displacement = True
                latest = recent_bullish_disp[-1]
                if latest['strength'] == 'STRONG':
                    score += 25
                    factors.append(('+25', f"STRONG displacement: {latest['candles']} candles, {latest['move']:.2f} pts ★"))
                else:
                    score += 15
                    factors.append(('+15', f"Displacement detected: {latest['move']:.2f} pts"))
            else:
                # Heavy penalty - displacement is required for confirmation
                score -= 35
                factors.append(('-35', 'NO RECENT DISPLACEMENT (last 10 bars) - reversal NOT confirmed ⚠'))

            # Check for bullish BOS in RECENT bars (last 12 only)
            # Per knowledge base: "Wait for candle to CLOSE through structure"
            recent_bullish_bos = [b for b in bos_events
                                  if b['type'] == 'BULLISH' and b['bar_idx'] >= len(bars) - 12]
            if recent_bullish_bos:
                has_bos = True
                latest = recent_bullish_bos[-1]
                score += 15
                factors.append(('+15', f"Bullish BOS at {latest['swing_price']:.2f} ★"))
            else:
                score -= 15
                factors.append(('-15', 'No recent BOS - structure NOT broken'))

            # Check manipulation completion
            if manip_check['complete']:
                manipulation_complete = True
                score += 15
                factors.append(('+15', f"Manipulation COMPLETE ({manip_check['confidence']}% confidence)"))
            else:
                if manip_check['missing']:
                    missing_str = '; '.join(manip_check['missing'][:2])
                    score -= 10
                    factors.append(('-10', f"Manipulation incomplete: {missing_str}"))

            # ================================================================
            # NEW: LIQUIDITY RAID CHECK (Step 1 of SMC Framework)
            # Per knowledge base: "Price sweeps highs/lows (stop hunts)"
            # ================================================================
            liquidity_raid = detect_liquidity_raid(bars, swings, 'LONG')
            if liquidity_raid:
                has_liquidity_raid = True
                score += 20
                factors.append(('+20', f"LIQUIDITY RAID: {liquidity_raid['description']} ★"))
            else:
                has_liquidity_raid = False
                score -= 25
                factors.append(('-25', 'NO LIQUIDITY RAID - Step 1 NOT met ⚠'))

            # ================================================================
            # NEW: RETRACEMENT CHECK (Step 4 of SMC Framework)
            # Per knowledge base: "DO NOT CHASE - Wait for price to retrace"
            # ================================================================
            if recent_bullish_disp:
                # Find entry zone from FVG or OB
                entry_zone_low, entry_zone_high = None, None
                for fvg in fvgs:
                    if fvg['type'] == 'BULLISH' and not fvg['filled']:
                        entry_zone_low = fvg['bottom']
                        entry_zone_high = fvg['top']
                        break
                if entry_zone_low is None:
                    for ob in obs:
                        if ob['type'] == 'BULLISH':
                            entry_zone_low = ob['low']
                            entry_zone_high = ob['high']
                            break

                if entry_zone_low is not None:
                    retracement = verify_retracement(bars, recent_bullish_disp[-1], 'LONG',
                                                     entry_zone_low, entry_zone_high)
                    if retracement['valid']:
                        has_retracement = True
                        score += 25
                        factors.append(('+25', f"VALID RETRACEMENT: {retracement['reason']} ★"))
                    else:
                        has_retracement = False
                        score -= 30
                        factors.append(('-30', f"NO RETRACEMENT: {retracement.get('reason', 'Chasing displacement')} ⚠"))
                else:
                    has_retracement = False
                    score -= 15
                    factors.append(('-15', 'No FVG/OB zone found for retracement'))
            else:
                has_retracement = False

        elif signal_type == 'SHORT':
            manip_check = check_manipulation_complete(bars, daily_open, 'high')

            # Check for bearish displacement in RECENT bars (last 10 only)
            # DISPLACEMENT IS CRITICAL - without it, no proof of reversal
            # Per knowledge base: "Displacement is NOT an entry - it's confirmation"
            recent_bearish_disp = [d for d in displacements
                                   if d['type'] == 'BEARISH' and d['end_idx'] >= len(bars) - 10]
            if recent_bearish_disp:
                has_displacement = True
                latest = recent_bearish_disp[-1]
                if latest['strength'] == 'STRONG':
                    score += 25
                    factors.append(('+25', f"STRONG displacement: {latest['candles']} candles, {latest['move']:.2f} pts ★"))
                else:
                    score += 15
                    factors.append(('+15', f"Displacement detected: {latest['move']:.2f} pts"))
            else:
                # Heavy penalty - displacement is required for confirmation
                score -= 35
                factors.append(('-35', 'NO RECENT DISPLACEMENT (last 10 bars) - reversal NOT confirmed ⚠'))

            # Check for bearish BOS in RECENT bars (last 12 only)
            # Per knowledge base: "Wait for candle to CLOSE through structure"
            recent_bearish_bos = [b for b in bos_events
                                  if b['type'] == 'BEARISH' and b['bar_idx'] >= len(bars) - 12]
            if recent_bearish_bos:
                has_bos = True
                latest = recent_bearish_bos[-1]
                score += 15
                factors.append(('+15', f"Bearish BOS at {latest['swing_price']:.2f} ★"))
            else:
                score -= 15
                factors.append(('-15', 'No recent BOS - structure NOT broken'))

            # Check manipulation completion
            if manip_check['complete']:
                manipulation_complete = True
                score += 15
                factors.append(('+15', f"Manipulation COMPLETE ({manip_check['confidence']}% confidence)"))
            else:
                if manip_check['missing']:
                    missing_str = '; '.join(manip_check['missing'][:2])
                    score -= 10
                    factors.append(('-10', f"Manipulation incomplete: {missing_str}"))

            # ================================================================
            # NEW: LIQUIDITY RAID CHECK (Step 1 of SMC Framework)
            # Per knowledge base: "Price sweeps highs/lows (stop hunts)"
            # ================================================================
            liquidity_raid = detect_liquidity_raid(bars, swings, 'SHORT')
            if liquidity_raid:
                has_liquidity_raid = True
                score += 20
                factors.append(('+20', f"LIQUIDITY RAID: {liquidity_raid['description']} ★"))
            else:
                has_liquidity_raid = False
                score -= 25
                factors.append(('-25', 'NO LIQUIDITY RAID - Step 1 NOT met ⚠'))

            # ================================================================
            # NEW: RETRACEMENT CHECK (Step 4 of SMC Framework)
            # Per knowledge base: "DO NOT CHASE - Wait for price to retrace"
            # ================================================================
            if recent_bearish_disp:
                # Find entry zone from FVG or OB
                entry_zone_low, entry_zone_high = None, None
                for fvg in fvgs:
                    if fvg['type'] == 'BEARISH' and not fvg['filled']:
                        entry_zone_low = fvg['bottom']
                        entry_zone_high = fvg['top']
                        break
                if entry_zone_low is None:
                    for ob in obs:
                        if ob['type'] == 'BEARISH':
                            entry_zone_low = ob['low']
                            entry_zone_high = ob['high']
                            break

                if entry_zone_low is not None:
                    retracement = verify_retracement(bars, recent_bearish_disp[-1], 'SHORT',
                                                     entry_zone_low, entry_zone_high)
                    if retracement['valid']:
                        has_retracement = True
                        score += 25
                        factors.append(('+25', f"VALID RETRACEMENT: {retracement['reason']} ★"))
                    else:
                        has_retracement = False
                        score -= 30
                        factors.append(('-30', f"NO RETRACEMENT: {retracement.get('reason', 'Chasing displacement')} ⚠"))
                else:
                    has_retracement = False
                    score -= 15
                    factors.append(('-15', 'No FVG/OB zone found for retracement'))
            else:
                has_retracement = False

    # ========================================================================
    # ORIGINAL FACTORS (DRT, OB, FVG, etc.)
    # ========================================================================

    if signal_type == 'LONG':
        # === HIGH WEIGHT FACTORS (+15-25) ===

        # DRT Zone (most important)
        if price_pct >= 80:
            score += 20
            factors.append(('+20', 'DEEP DISCOUNT zone (>80% DRT)'))
        elif price_pct >= 70:
            score += 15
            factors.append(('+15', f'DISCOUNT zone ({price_pct:.0f}% DRT)'))
        elif price_pct >= 50:
            score += 5
            factors.append(('+5', f'Below equilibrium ({price_pct:.0f}% DRT)'))
        elif price_pct <= 25:
            score -= 20
            factors.append(('-20', f'PREMIUM zone ({price_pct:.0f}% DRT) - wrong zone for longs'))

        # Turtle Soup (high probability setup) - CAPPED at 2 patterns max
        turtle_soup_count = 0
        turtle_soup_max = 2  # Max patterns to count
        # Sort swings by recency (most recent first)
        sorted_lows = sorted(swings.get('lows', []), key=lambda x: x.get('idx', 0), reverse=True)
        for swing in sorted_lows:
            if turtle_soup_count >= turtle_soup_max:
                break
            for fvg in fvgs:
                if fvg['type'] == 'BULLISH' and not fvg['filled']:
                    if fvg['top'] < swing['price'] and swing['price'] - fvg['top'] < 5:
                        if current_price < swing['price'] and current_price >= fvg['bottom']:
                            score += 25
                            factors.append(('+25', f'TURTLE SOUP: Swept {swing["price"]:.2f}, in FVG'))
                            stop = fvg['bottom'] - 1
                            turtle_soup_count += 1
                            break

        # BPR Support
        if bprs:
            for bpr in bprs:
                if bpr['type'] == 'SUPPORT' and bpr['bottom'] <= current_price <= bpr['top']:
                    score += 20
                    factors.append(('+20', f'IN BPR SUPPORT: {bpr["bottom"]:.2f}-{bpr["top"]:.2f}'))
                    stop = bpr['bottom'] - 1
                elif bpr['type'] == 'SUPPORT' and current_price > bpr['top'] and current_price < bpr['top'] + 3:
                    score += 10
                    factors.append(('+10', f'Near BPR SUPPORT below'))

        # === MEDIUM WEIGHT FACTORS (+10-15) ===

        # Bullish Order Block
        for ob in obs:
            if ob['type'] == 'BULLISH':
                if ob['low'] <= current_price <= ob['high']:
                    score += 15
                    factors.append(('+15', f'IN Bullish OB: {ob["low"]:.2f}-{ob["high"]:.2f}'))
                    if not stop:
                        stop = ob['low'] - 1
                elif current_price < ob['low'] and current_price > ob['low'] - 3:
                    score += 8
                    factors.append(('+8', f'Near Bullish OB below'))

        # Bullish FVG
        for fvg in fvgs:
            if fvg['type'] == 'BULLISH' and not fvg['filled']:
                if fvg['bottom'] <= current_price <= fvg['top']:
                    score += 12
                    factors.append(('+12', f'IN Bullish FVG: {fvg["bottom"]:.2f}-{fvg["top"]:.2f}'))
                    if not stop:
                        stop = fvg['bottom'] - 1

        # Clear liquidity target
        bsl_targets = [s['price'] for s in swings.get('highs', []) if s['price'] > current_price]
        if bsl_targets:
            target = min(bsl_targets)
            score += 10
            factors.append(('+10', f'BSL target at {target:.2f}'))

        # === NEGATIVE FACTORS ===

        # In Bearish OB (resistance)
        for ob in obs:
            if ob['type'] == 'BEARISH' and ob['low'] <= current_price <= ob['high']:
                score -= 15
                factors.append(('-15', f'IN Bearish OB (resistance)'))

        # In Bearish FVG
        for fvg in fvgs:
            if fvg['type'] == 'BEARISH' and not fvg['filled']:
                if fvg['bottom'] <= current_price <= fvg['top']:
                    score -= 10
                    factors.append(('-10', f'IN Bearish FVG (resistance)'))

    elif signal_type == 'SHORT':
        # === HIGH WEIGHT FACTORS ===

        # DRT Zone
        if price_pct <= 20:
            score += 20
            factors.append(('+20', 'DEEP PREMIUM zone (<20% DRT)'))
        elif price_pct <= 30:
            score += 15
            factors.append(('+15', f'PREMIUM zone ({price_pct:.0f}% DRT)'))
        elif price_pct <= 50:
            score += 5
            factors.append(('+5', f'Above equilibrium ({price_pct:.0f}% DRT)'))
        elif price_pct >= 75:
            score -= 20
            factors.append(('-20', f'DISCOUNT zone ({price_pct:.0f}% DRT) - wrong zone for shorts'))

        # Turtle Soup (high probability setup) - CAPPED at 2 patterns max
        turtle_soup_count = 0
        turtle_soup_max = 2  # Max patterns to count
        # Sort swings by recency (most recent first)
        sorted_highs = sorted(swings.get('highs', []), key=lambda x: x.get('idx', 0), reverse=True)
        for swing in sorted_highs:
            if turtle_soup_count >= turtle_soup_max:
                break
            for fvg in fvgs:
                if fvg['type'] == 'BEARISH' and not fvg['filled']:
                    if fvg['bottom'] > swing['price'] and fvg['bottom'] - swing['price'] < 5:
                        if current_price > swing['price'] and current_price <= fvg['top']:
                            score += 25
                            factors.append(('+25', f'TURTLE SOUP: Swept {swing["price"]:.2f}, in FVG'))
                            stop = fvg['top'] + 1
                            turtle_soup_count += 1
                            break

        # BPR Resistance
        if bprs:
            for bpr in bprs:
                if bpr['type'] == 'RESISTANCE' and bpr['bottom'] <= current_price <= bpr['top']:
                    score += 20
                    factors.append(('+20', f'IN BPR RESISTANCE: {bpr["bottom"]:.2f}-{bpr["top"]:.2f}'))
                    stop = bpr['top'] + 1
                elif bpr['type'] == 'RESISTANCE' and current_price < bpr['bottom'] and current_price > bpr['bottom'] - 3:
                    score += 10
                    factors.append(('+10', f'Near BPR RESISTANCE above'))

        # === MEDIUM WEIGHT FACTORS ===

        # Bearish Order Block
        for ob in obs:
            if ob['type'] == 'BEARISH':
                if ob['low'] <= current_price <= ob['high']:
                    score += 15
                    factors.append(('+15', f'IN Bearish OB: {ob["low"]:.2f}-{ob["high"]:.2f}'))
                    if not stop:
                        stop = ob['high'] + 1
                elif current_price > ob['high'] and current_price < ob['high'] + 3:
                    score += 8
                    factors.append(('+8', f'Near Bearish OB above'))

        # Bearish FVG
        for fvg in fvgs:
            if fvg['type'] == 'BEARISH' and not fvg['filled']:
                if fvg['bottom'] <= current_price <= fvg['top']:
                    score += 12
                    factors.append(('+12', f'IN Bearish FVG: {fvg["bottom"]:.2f}-{fvg["top"]:.2f}'))
                    if not stop:
                        stop = fvg['top'] + 1

        # Clear liquidity target
        ssl_targets = [s['price'] for s in swings.get('lows', []) if s['price'] < current_price]
        if ssl_targets:
            target = max(ssl_targets)
            score += 10
            factors.append(('+10', f'SSL target at {target:.2f}'))

        # === NEGATIVE FACTORS ===

        # In Bullish OB (support)
        for ob in obs:
            if ob['type'] == 'BULLISH' and ob['low'] <= current_price <= ob['high']:
                score -= 15
                factors.append(('-15', f'IN Bullish OB (support)'))

        # In Bullish FVG
        for fvg in fvgs:
            if fvg['type'] == 'BULLISH' and not fvg['filled']:
                if fvg['bottom'] <= current_price <= fvg['top']:
                    score -= 10
                    factors.append(('-10', f'IN Bullish FVG (support)'))

    # Fallback stops using DRT levels if no structure-based stop found
    if not stop:
        if signal_type == 'LONG':
            # Stop below 100 DRT (session low) or 75 DRT
            stop = drt['low'] - 0.5
            factors.append(('+0', f'Stop at session low: {stop:.2f}'))
        else:
            # Stop above 0 DRT (session high) or 25 DRT
            stop = drt['high'] + 0.5
            factors.append(('+0', f'Stop at session high: {stop:.2f}'))

    # Fallback target using DRT levels
    if not target:
        if signal_type == 'LONG':
            target = drt['50']  # Target equilibrium
            factors.append(('+5', f'Target 50 DRT: {target:.2f}'))
            score += 5
        else:
            target = drt['50']  # Target equilibrium
            factors.append(('+5', f'Target 50 DRT: {target:.2f}'))
            score += 5

    # Clamp score to 0-100
    score = max(0, min(100, score))

    # Determine grade
    if score >= 80:
        grade = 'A'
        recommendation = 'HIGH PROBABILITY - Take trade'
    elif score >= 65:
        grade = 'B'
        recommendation = 'GOOD SETUP - Consider with tight stop'
    elif score >= 50:
        grade = 'C'
        recommendation = 'MARGINAL - Wait for better entry'
    else:
        grade = 'D'
        recommendation = 'LOW PROBABILITY - Avoid'

    # Calculate R:R if we have stop and target
    risk_reward = None
    if stop and target:
        risk = abs(current_price - stop)
        reward = abs(target - current_price)
        if risk > 0:
            risk_reward = reward / risk

    return {
        'signal': signal_type,
        'probability': score,
        'grade': grade,
        'recommendation': recommendation,
        'entry': entry,
        'stop': stop,
        'target': target,
        'risk_reward': risk_reward,
        'factors': factors,
        'has_displacement': has_displacement,
        'has_bos': has_bos,
        'has_liquidity_raid': has_liquidity_raid,
        'has_retracement': has_retracement
    }


def check_entry_confirmation(signal_type: str, current_price: float,
                             drt: Dict, obs: List[Dict], fvgs: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Check if a fractal signal has SMC confluence confirmation
    Returns (is_confirmed, list of reasons)
    """
    confirmations = []
    rejections = []

    price_pct = ((drt['high'] - current_price) / drt['range']) * 100 if drt['range'] > 0 else 50

    if signal_type == 'LONG':
        # For LONG signals, we want:
        # 1. Price in discount zone (75-100% DRT)
        if price_pct >= 70:
            confirmations.append(f"Price in discount zone ({price_pct:.0f}% DRT)")
        elif price_pct <= 30:
            rejections.append(f"Price in premium zone ({price_pct:.0f}% DRT) - avoid longs")

        # 2. Near or in a bullish OB
        for ob in obs:
            if ob['type'] == 'BULLISH' and ob['low'] - 2 <= current_price <= ob['high'] + 1:
                confirmations.append(f"At BULLISH OB: {ob['low']:.2f}-{ob['high']:.2f}")

        # 3. In or near a bullish FVG
        for fvg in fvgs:
            if not fvg['filled'] and fvg['type'] == 'BULLISH':
                if fvg['bottom'] - 2 <= current_price <= fvg['top'] + 1:
                    confirmations.append(f"At BULLISH FVG: {fvg['bottom']:.2f}-{fvg['top']:.2f}")

        # Check for bearish rejection zones
        for ob in obs:
            if ob['type'] == 'BEARISH' and ob['low'] <= current_price <= ob['high']:
                rejections.append(f"In BEARISH OB - resistance zone")

    elif signal_type == 'SHORT':
        # For SHORT signals, we want:
        # 1. Price in premium zone (0-25% DRT)
        if price_pct <= 30:
            confirmations.append(f"Price in premium zone ({price_pct:.0f}% DRT)")
        elif price_pct >= 70:
            rejections.append(f"Price in discount zone ({price_pct:.0f}% DRT) - avoid shorts")

        # 2. Near or in a bearish OB
        for ob in obs:
            if ob['type'] == 'BEARISH' and ob['low'] - 1 <= current_price <= ob['high'] + 2:
                confirmations.append(f"At BEARISH OB: {ob['low']:.2f}-{ob['high']:.2f}")

        # 3. In or near a bearish FVG
        for fvg in fvgs:
            if not fvg['filled'] and fvg['type'] == 'BEARISH':
                if fvg['bottom'] - 1 <= current_price <= fvg['top'] + 2:
                    confirmations.append(f"At BEARISH FVG: {fvg['bottom']:.2f}-{fvg['top']:.2f}")

        # Check for bullish rejection zones
        for ob in obs:
            if ob['type'] == 'BULLISH' and ob['low'] <= current_price <= ob['high']:
                rejections.append(f"In BULLISH OB - support zone")

    is_confirmed = len(confirmations) >= 2 and len(rejections) == 0
    reasons = confirmations if is_confirmed else rejections if rejections else ["No strong confluence"]

    return is_confirmed, reasons


def print_analysis(bars: List[Bar], show_all: bool = False, daily_open: float = None,
                   four_hour_range: Dict = None):
    """Print full SMC analysis with enhanced OB validation, breakers, inducements, and 4H range"""
    if not bars:
        print("No bar data available")
        return

    current_price = bars[-1].C
    session_high = max(b.H for b in bars)
    session_low = min(b.L for b in bars)

    drt = calculate_drt(session_high, session_low)
    fvgs = find_fvgs(bars)
    swings = find_swing_points(bars)
    bprs = find_balanced_price_ranges(fvgs)

    # Enhanced OB detection with 3-rule validation
    obs = find_order_blocks(bars, fvgs=fvgs, swings=swings)

    # NEW: Breaker blocks and inducement traps
    breakers = find_breaker_blocks(bars, obs)
    inducements = find_inducement_traps(bars, obs)

    # NEW: 4-Hour Range Strategy
    four_hour_analysis = analyze_4hour_range_strategy(bars, four_hour_range, filepath=filepath)

    # Detect displacement and BOS
    displacements = detect_displacement(bars)
    bos_events = detect_bos(bars, swings)
    choch = detect_choch(bars, swings)

    signals, bias, bullish_count, bearish_count = analyze_confluence(
        current_price, drt, obs, fvgs, swings, bprs
    )

    # Get session info
    session_info = get_session_info()

    print("=" * 60)
    print("           SMC CONFLUENCE ANALYSIS")
    print("=" * 60)
    print(f"\n  CURRENT PRICE: {current_price:.2f}")
    print(f"  SESSION: {session_info['current']} - {session_info['description']}")
    print(f"  TARGETS: {session_info['liquidity_targets']}")
    print(f"  DATA RANGE: {session_low:.2f} - {session_high:.2f} ({drt['range']:.2f} pts)")

    # Daily open relationship (if provided)
    if daily_open:
        rel = "ABOVE" if current_price > daily_open else "BELOW"
        diff = abs(current_price - daily_open)
        print(f"  DAILY OPEN: {daily_open:.2f} (price is {rel} by {diff:.2f} pts)")

    print(f"\n  DRT LEVELS:")
    print(f"    0 DRT (HIGH):  {drt['high']:.2f}  [BSL Target]")
    print(f"   25 DRT:         {drt['25']:.2f}  [Short zone]")
    print(f"   50 DRT:         {drt['50']:.2f}  [Equilibrium]")
    print(f"   75 DRT:         {drt['75']:.2f}  [Long zone]")
    print(f"  100 DRT (LOW):   {drt['low']:.2f}  [SSL Target]")

    # ========================================================================
    # 4-HOUR RANGE STRATEGY
    # ========================================================================
    print("\n" + "=" * 60)
    print("           4-HOUR RANGE SCALPING STRATEGY")
    print("=" * 60)

    if four_hour_analysis['range']:
        fr = four_hour_analysis['range']
        print(f"\n  4H RANGE: {fr['low']:.2f} - {fr['high']:.2f} ({fr['range']:.2f} pts)")

        # Show where current price is relative to range
        if current_price > fr['high']:
            print(f"  POSITION: ABOVE range by {current_price - fr['high']:.2f} pts")
        elif current_price < fr['low']:
            print(f"  POSITION: BELOW range by {fr['low'] - current_price:.2f} pts")
        else:
            print(f"  POSITION: INSIDE range")

        print(f"  STATUS: {four_hour_analysis['status']}")

        if four_hour_analysis['current_setup']:
            setup = four_hour_analysis['current_setup']
            print(f"\n  [!] PENDING SETUP:")
            print(f"      Signal: {setup['pending_signal']}")
            print(f"      Condition: {setup['condition']}")
            print(f"      Stop will be: {setup['stop_will_be']:.2f}")

        if four_hour_analysis['historical_setups']:
            print(f"\n  RECENT 4H RANGE SIGNALS ({len(four_hour_analysis['historical_setups'])} today):")
            for setup in four_hour_analysis['historical_setups'][-3:]:
                print(f"    * {setup['signal']}: Entry {setup['entry']:.2f}, Stop {setup['stop_loss']:.2f}, TP {setup['take_profit']:.2f} (2R)")
    else:
        print("\n  4H Range not yet established (need 48+ bars)")

    # ========================================================================
    # VALIDATED ORDER BLOCKS (3-Rule)
    # ========================================================================
    print("\n" + "=" * 60)
    print("           ORDER BLOCKS (3-Rule Validated)")
    print("=" * 60)

    valid_obs = [ob for ob in obs if ob['valid']]
    invalid_obs = [ob for ob in obs if not ob['valid']]

    print(f"\n  VALID ORDER BLOCKS ({len(valid_obs)} found):")
    if valid_obs:
        for ob in valid_obs[:5]:
            marker = ">>>" if ob['low'] <= current_price <= ob['high'] else "   "
            rules = ob['rules']
            rule_str = f"[FVG:{'Y' if rules['has_fvg'] else 'N'}|UNMIT:{'Y' if rules['unmitigated'] else 'N'}|BOS:{'Y' if rules['has_bos'] else 'N'}]"
            print(f"  {marker} {ob['type']:8} OB: {ob['low']:.2f} - {ob['high']:.2f} {rule_str} ({ob['confidence']}%)")
    else:
        print("      None - no OBs pass all 3 rules")

    if show_all and invalid_obs:
        print(f"\n  INVALID ORDER BLOCKS ({len(invalid_obs)} found):")
        for ob in invalid_obs[:3]:
            rules = ob['rules']
            missing = []
            if not rules['has_fvg']:
                missing.append('NO FVG')
            if not rules['unmitigated']:
                missing.append('MITIGATED')
            if not rules['has_bos']:
                missing.append('NO BOS')
            print(f"      {ob['type']:8} OB: {ob['low']:.2f} - {ob['high']:.2f} - Missing: {', '.join(missing)}")

    # ========================================================================
    # BREAKER BLOCKS
    # ========================================================================
    print(f"\n  BREAKER BLOCKS ({len(breakers)} found):")
    if breakers:
        for bb in breakers[:3]:
            marker = ">>>" if bb['low'] <= current_price <= bb['high'] else "   "
            unmit = "FRESH" if bb['unmitigated'] else "RETESTED"
            print(f"  {marker} {bb['type']:8} BREAKER: {bb['low']:.2f} - {bb['high']:.2f} [{unmit}]")
            print(f"        (was {bb['original_type']} OB, now flipped)")
    else:
        print("      None detected")

    # ========================================================================
    # INDUCEMENT TRAPS
    # ========================================================================
    print(f"\n  INDUCEMENT TRAPS ({len(inducements)} found):")
    if inducements:
        for trap in inducements:
            status_icon = "[!]" if trap['setup'] == 'ACTIVE' else "[ ]"
            print(f"  {status_icon} {trap['type']} TRAP:")
            print(f"        Inducement level: {trap['inducement_level']:.2f} ({trap['rejections']} rejections)")
            print(f"        OB entry zone: {trap['ob_entry']:.2f}")
            print(f"        Status: {trap['setup']}")
    else:
        print("      None detected")

    unfilled_fvgs = [f for f in fvgs if not f['filled']]
    print(f"\n  UNFILLED FVGs ({len(unfilled_fvgs)} active):")
    for fvg in unfilled_fvgs[-5:]:
        marker = ">>>" if fvg['bottom'] <= current_price <= fvg['top'] else "   "
        print(f"  {marker} {fvg['type']:8} FVG: {fvg['bottom']:.2f} - {fvg['top']:.2f} (CE: {fvg['ce']:.2f})")

    print(f"\n  BALANCED PRICE RANGES ({len(bprs)} found):")
    if bprs:
        for bpr in bprs[:3]:
            marker = ">>>" if bpr['bottom'] <= current_price <= bpr['top'] else "   "
            print(f"  {marker} BPR ({bpr['type']:10}): {bpr['bottom']:.2f} - {bpr['top']:.2f}")
    else:
        print("      None detected")

    print(f"\n  LIQUIDITY POOLS:")
    bsl_targets = sorted(set(s['price'] for s in swings['highs'] if s['price'] > current_price))[:3]
    ssl_targets = sorted(set(s['price'] for s in swings['lows'] if s['price'] < current_price), reverse=True)[:3]
    print(f"    BSL (above): {', '.join(f'{p:.2f}' for p in bsl_targets) if bsl_targets else 'None nearby'}")
    print(f"    SSL (below): {', '.join(f'{p:.2f}' for p in ssl_targets) if ssl_targets else 'None nearby'}")

    # NEW: Displacement detection
    print(f"\n  DISPLACEMENT ({len(displacements)} detected):")
    if displacements:
        recent_disp = displacements[-5:]  # Last 5
        for d in recent_disp:
            strength_icon = "***" if d['strength'] == 'STRONG' else "  *"
            print(f"  {strength_icon} {d['type']:8} | {d['candles']} candles | {d['move']:.2f} pts | {d['start_price']:.2f} -> {d['end_price']:.2f}")
    else:
        print("      None detected - no strong directional moves")

    # NEW: Break of Structure
    print(f"\n  BREAK OF STRUCTURE ({len(bos_events)} detected):")
    if bos_events:
        recent_bos = bos_events[-3:]  # Last 3
        for b in recent_bos:
            print(f"    * {b['description']}")
    else:
        print("      None detected - structure intact")

    # NEW: Change of Character
    if choch:
        print(f"\n  [!]  CHANGE OF CHARACTER DETECTED:")
        print(f"      {choch['description']}")

    print("\n" + "=" * 60)
    print("           CONFLUENCE SIGNALS")
    print("=" * 60)

    for source, direction, desc in signals:
        if direction in ['BULLISH', 'SSL']:
            icon = "+"
        elif direction in ['BEARISH', 'BSL']:
            icon = "-"
        else:
            icon = "~"
        print(f"  [{icon}] [{source}] {desc}")

    print("\n" + "-" * 60)
    print(f"  CONFLUENCE BIAS: {bias} ({bullish_count} bullish / {bearish_count} bearish)")
    print("=" * 60)

    # Trade probability analysis
    print("\n" + "=" * 60)
    print("           TRADE PROBABILITY ANALYSIS")
    print("=" * 60)
    print("  (Now includes Displacement, BOS, and Manipulation checks)")

    for sig_type in ['LONG', 'SHORT']:
        result = calculate_trade_probability(
            sig_type, current_price, drt, obs, fvgs, swings, bprs,
            bars=bars, daily_open=daily_open
        )

        print(f"\n  {sig_type} @ {current_price:.2f}")
        print(f"  {'-' * 40}")
        print(f"  Probability: {result['probability']}% (Grade {result['grade']})")
        print(f"  {result['recommendation']}")

        if result['factors']:
            print(f"\n  Scoring Breakdown:")
            for weight, reason in result['factors'][:6]:
                print(f"    {weight:>4}  {reason}")

        if result['stop'] or result['target']:
            print(f"\n  Trade Setup:")
            print(f"    Entry:  {result['entry']:.2f}")
            if result['stop']:
                stop_dist = abs(result['entry'] - result['stop'])
                print(f"    Stop:   {result['stop']:.2f} ({stop_dist:.2f} pts = ${stop_dist * 50:.0f} ES)")
            if result['target']:
                target_dist = abs(result['target'] - result['entry'])
                print(f"    Target: {result['target']:.2f} ({target_dist:.2f} pts = ${target_dist * 50:.0f} ES)")
            if result['risk_reward']:
                rr = result['risk_reward']
                if rr >= 2.0:
                    rr_quality = "(EXCELLENT)"
                elif rr >= 1.5:
                    rr_quality = "(GOOD)"
                elif rr >= 1.0:
                    rr_quality = "(ACCEPTABLE)"
                else:
                    rr_quality = "(POOR - consider wider target)"
                print(f"    R:R:    1:{rr:.1f} {rr_quality}")

    print("\n" + "=" * 60)


# ============================================================================
# CISD (Change in State of Delivery) DETECTION
# ============================================================================

def find_cisd_series(bars: List[Bar], candle_type: str, lookback: int = 15) -> Optional[Dict]:
    """
    Find the last series of consecutive up-close or down-close candles.

    CISD is measured by candle BODIES, not wicks.

    Args:
        candle_type: 'UP' for up-close candles, 'DOWN' for down-close
        lookback: how many bars to search back

    Returns:
        Dict with 'level' (bottom of up-series or top of down-series) and indices
    """
    if len(bars) < lookback:
        return None

    recent = bars[-lookback:]
    series_start = None
    series_end = None

    # Walk backwards to find series
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

    if len(series_candles) < 2:
        return None  # Need at least 2 consecutive same-direction candles

    if candle_type == 'UP':
        # For up-close candles, CISD level is the bottom of the bodies
        level = min(min(c.O, c.C) for c in series_candles)
    else:
        # For down-close candles, CISD level is the top of the bodies
        level = max(max(c.O, c.C) for c in series_candles)

    return {
        'level': level,
        'start_idx': len(bars) - lookback + series_start,
        'end_idx': len(bars) - lookback + series_end,
        'candle_type': candle_type,
        'candle_count': len(series_candles)
    }


def detect_cisd(bars: List[Bar], min_displacement: float = 1.0) -> Optional[Dict]:
    """
    Detect Change in State of Delivery.

    CISD = When price CLOSES through the last series of same-direction candle bodies.
    This indicates a shift in order flow (who's in control of delivery).

    Rules:
    1. Find last series of up-close or down-close candles
    2. Wait for price to CLOSE through that level (bodies, not wicks)
    3. Displacement = strong follow-through (not just barely through)

    Returns signal dict or None
    """
    if len(bars) < 20:
        return None

    current = bars[-1]
    prev = bars[-2]

    # Check for BULLISH CISD (was bearish, now bullish)
    # Look for series of down-close candles, then price closes above their tops
    down_series = find_cisd_series(bars[:-2], 'DOWN', lookback=15)
    if down_series:
        cisd_level = down_series['level']
        # Current candle closes above the series tops (bodies)
        if current.C > cisd_level and prev.C <= cisd_level:
            displacement = current.C - cisd_level
            if displacement >= min_displacement:
                # Find target (next liquidity - recent swing high)
                recent_highs = [b.H for b in bars[-50:]]
                target = max(recent_highs)

                entry = current.C
                stop = current.L - 0.5  # Tight stop below entry candle

                risk = entry - stop
                reward = target - entry

                if risk > 0 and reward > 0 and reward / risk >= 1.5:
                    return {
                        'direction': 'LONG',
                        'strategy': 'CISD',
                        'entry': entry,
                        'stop_loss': stop,
                        'take_profit': target,
                        'risk': risk,
                        'rr': reward / risk,
                        'cisd_level': cisd_level,
                        'displacement': displacement,
                        'confidence': min(85, 60 + int(displacement * 5)),
                        'reasoning': f"Bullish CISD: Closed above {cisd_level:.2f} with {displacement:.1f}pt displacement"
                    }

    # Check for BEARISH CISD (was bullish, now bearish)
    # Look for series of up-close candles, then price closes below their bottoms
    up_series = find_cisd_series(bars[:-2], 'UP', lookback=15)
    if up_series:
        cisd_level = up_series['level']
        # Current candle closes below the series bottoms (bodies)
        if current.C < cisd_level and prev.C >= cisd_level:
            displacement = cisd_level - current.C
            if displacement >= min_displacement:
                recent_lows = [b.L for b in bars[-50:]]
                target = min(recent_lows)

                entry = current.C
                stop = current.H + 0.5  # Tight stop above entry candle

                risk = stop - entry
                reward = entry - target

                if risk > 0 and reward > 0 and reward / risk >= 1.5:
                    return {
                        'direction': 'SHORT',
                        'strategy': 'CISD',
                        'entry': entry,
                        'stop_loss': stop,
                        'take_profit': target,
                        'risk': risk,
                        'rr': reward / risk,
                        'cisd_level': cisd_level,
                        'displacement': displacement,
                        'confidence': min(85, 60 + int(displacement * 5)),
                        'reasoning': f"Bearish CISD: Closed below {cisd_level:.2f} with {displacement:.1f}pt displacement"
                    }

    return None


# ============================================================================
# HIGH CONFIDENCE SIGNAL AGGREGATOR
# ============================================================================

def get_high_confidence_signals(bars: List[Bar], daily_open: float = None,
                                 four_hour_range: Dict = None,
                                 min_confidence: int = 70,
                                 filepath: str = None,
                                 live_price: float = None) -> List[Dict]:
    """
    Aggregate all high-confidence signals from multiple strategies.

    Checks:
    1. 4-Hour Range strategy signals
    2. Valid Order Blocks with price at entry zone
    3. Breaker blocks (fresh, unmitigated)
    4. Inducement trap setups
    5. SMC confluence score

    Args:
        live_price: If provided, use this as entry price instead of bar close.
                    This ensures entry prices reflect current market price.

    Returns list of signals sorted by confidence, each with:
    - strategy: source of signal
    - direction: LONG/SHORT
    - confidence: 0-100
    - entry, stop_loss, take_profit
    - reasoning: why this signal
    """
    signals = []
    bar_close_price = bars[-1].C if bars else 0
    # Use live price for entry if provided, but bar close for pattern analysis
    current_price = bar_close_price
    entry_price = live_price if live_price else bar_close_price

    # Calculate all components
    fvgs = find_fvgs(bars)
    swings = find_swing_points(bars)
    obs = find_order_blocks(bars, fvgs=fvgs, swings=swings)
    breakers = find_breaker_blocks(bars, obs)
    inducements = find_inducement_traps(bars, obs)
    drt = calculate_drt(max(b.H for b in bars), min(b.L for b in bars))
    bprs = find_balanced_price_ranges(fvgs)

    # SMC STRATEGIES ONLY (CISD and 4H Range removed - now using discovered strategies)
    # See live_signal_watcher.py for Williams Fractals, Gap Continuation, London Mean Reversion

    # 1. Check Valid Order Blocks at entry zone
    valid_obs = [ob for ob in obs if ob['valid'] and ob['rules']['unmitigated']]
    for ob in valid_obs:
        # Check if price is at the OB
        if ob['low'] <= current_price <= ob['high']:
            direction = 'LONG' if ob['type'] == 'BULLISH' else 'SHORT'
            if direction == 'LONG':
                stop = ob['low'] - 0.5
                target = entry_price + 2 * (entry_price - stop)
            else:
                stop = ob['high'] + 0.5
                target = entry_price - 2 * (stop - entry_price)

            signals.append({
                'strategy': 'ORDER_BLOCK',
                'direction': direction,
                'confidence': ob['confidence'],
                'entry': entry_price,
                'stop_loss': stop,
                'take_profit': target,
                'risk': abs(entry_price - stop),
                'rr': 2.0,
                'reasoning': f"Price at validated {ob['type']} OB ({ob['low']:.2f}-{ob['high']:.2f})"
            })

    # 3. Check Fresh Breaker Blocks
    fresh_breakers = [bb for bb in breakers if bb['unmitigated']]
    for bb in fresh_breakers:
        if bb['low'] <= current_price <= bb['high']:
            direction = bb['type']  # Already flipped direction
            if direction == 'LONG':
                stop = bb['low'] - 0.5
                target = entry_price + 2 * (entry_price - stop)
            else:
                stop = bb['high'] + 0.5
                target = entry_price - 2 * (stop - entry_price)

            signals.append({
                'strategy': 'BREAKER_BLOCK',
                'direction': direction,
                'confidence': 72,
                'entry': entry_price,
                'stop_loss': stop,
                'take_profit': target,
                'risk': abs(entry_price - stop),
                'rr': 2.0,
                'reasoning': f"Fresh breaker block: {bb['description']}"
            })

    # 4. Check Active Inducement Traps (only if price is near entry zone)
    current_price = bars[-1].C
    for trap in inducements:
        if trap['setup'] == 'ACTIVE':
            ob = trap['ob']
            direction = trap['type']
            entry = trap['ob_entry']

            # Only fire if current price is within 3 points of entry (actionable market order)
            if abs(current_price - entry) > 3.0:
                continue  # Skip - entry too far from current price

            if direction == 'LONG':
                stop = ob['low'] - 0.5
                target = trap['inducement_level']
                # Also skip if target is below current price (already hit)
                if target <= current_price:
                    continue
            else:
                stop = ob['high'] + 0.5
                target = trap['inducement_level']
                # Also skip if target is above current price (already hit)
                if target >= current_price:
                    continue

            signals.append({
                'strategy': 'INDUCEMENT_TRAP',
                'direction': direction,
                'confidence': 75,
                'entry': current_price,  # Use current price, not OB level
                'stop_loss': stop,
                'take_profit': target,
                'risk': abs(current_price - stop),
                'rr': abs(target - current_price) / abs(current_price - stop) if abs(current_price - stop) > 0 else 0,
                'reasoning': trap['description']
            })

    # 5. Calculate SMC confluence probability with RECENT TREND BIAS
    # Check recent price direction (last 20 bars) to filter conflicting signals
    recent_bars = bars[-20:] if len(bars) >= 20 else bars
    recent_change = recent_bars[-1].C - recent_bars[0].O
    recent_trend = 'BULLISH' if recent_change > 0 else 'BEARISH'
    trend_strength = abs(recent_change)

    # STOP DISTANCE LIMITS - Per risk management principles
    # Risk max 1-2% per trade. Wide stops = too much risk for most accounts
    # Minimum stop prevents untradeable micro-scalps that get stopped by noise
    MAX_STOP_DISTANCE = 8.0  # points - max risk per trade (allows more room)
    MIN_STOP_DISTANCE = 2.0  # points - minimum to avoid noise stops (1-1.8 pt stops got hunted)

    smc_candidates = []
    for direction in ['LONG', 'SHORT']:
        prob_result = calculate_trade_probability(
            direction, current_price, drt, obs, fvgs, swings, bprs,
            bars=bars, daily_open=daily_open
        )

        raw_confidence = prob_result['probability']
        stop = prob_result['stop']
        has_displacement = prob_result.get('has_displacement', False)
        has_bos = prob_result.get('has_bos', False)
        has_liquidity_raid = prob_result.get('has_liquidity_raid', False)
        has_retracement = prob_result.get('has_retracement', False)

        # Skip if no valid stop or low base confidence
        if not stop or raw_confidence < 50:
            continue

        # ================================================================
        # HARD REQUIREMENT: Must have liquidity raid AND retracement
        # Per knowledge base 5-step framework - no exceptions
        # ================================================================
        if not has_liquidity_raid:
            continue  # Step 1 not met - no signal
        if not has_retracement:
            continue  # Step 4 not met - still chasing, wait for pullback

        # CRITICAL: Validate entry is on correct side of stop
        if direction == 'LONG' and entry_price <= stop:
            continue
        if direction == 'SHORT' and entry_price >= stop:
            continue

        # Calculate R:R with LIVE entry price
        if direction == 'LONG':
            target = entry_price + 2 * (entry_price - stop)
        else:
            target = entry_price - 2 * (stop - entry_price)

        risk = abs(entry_price - stop)
        reward = abs(target - entry_price)
        live_rr = reward / risk if risk > 0 else 0

        # STOP DISTANCE FILTER - Skip if stop is too wide OR too tight
        # Per knowledge base: "Position size appropriate (max 2% risk)"
        if risk > MAX_STOP_DISTANCE:
            continue  # Stop too wide - skip this signal entirely
        if risk < MIN_STOP_DISTANCE:
            continue  # Stop too tight - would get stopped by noise

        # COUNTER-TREND REJECTION
        # Per knowledge base: "No BOS = no trade" and "Displacement is confirmation"
        is_counter_trend = (direction == 'LONG' and recent_trend == 'BEARISH') or \
                          (direction == 'SHORT' and recent_trend == 'BULLISH')

        if is_counter_trend:
            # Counter-trend trades REQUIRE both displacement AND BOS
            # Per checklist: "Before Entry - ALL Must Be True"
            if not has_displacement and not has_bos:
                continue  # No confirmation at all - reject entirely
            elif not has_displacement:
                # Have BOS but no displacement - heavy penalty
                raw_confidence = max(0, raw_confidence - 30)
            elif not has_bos:
                # Have displacement but no BOS - moderate penalty
                raw_confidence = max(0, raw_confidence - 20)

        # TREND BIAS: Additional penalty for counter-trend
        adjusted_confidence = raw_confidence
        if is_counter_trend:
            # Stronger penalty: trend_strength * 5 (was * 3)
            penalty = min(50, int(trend_strength * 5))  # Up to -50 for strong trend
            adjusted_confidence = max(0, raw_confidence - penalty)

        # R:R penalty
        if live_rr < 1.0:
            adjusted_confidence = max(0, adjusted_confidence - 30)
        elif live_rr < 1.5:
            adjusted_confidence = max(0, adjusted_confidence - 10)

        # Skip if adjusted confidence too low or R:R below 1.5
        if adjusted_confidence < min_confidence or live_rr < 1.5:
            continue

        # Build reasoning with confirmation status
        # All 4 should be ✓ since we require liquidity_raid and retracement now
        confirmation_status = []
        confirmation_status.append("raid:✓" if has_liquidity_raid else "raid:✗")
        confirmation_status.append("disp:✓" if has_displacement else "disp:✗")
        confirmation_status.append("BOS:✓" if has_bos else "BOS:✗")
        confirmation_status.append("retrace:✓" if has_retracement else "retrace:✗")
        conf_str = " ".join(confirmation_status)

        # Build clearer trend description
        if is_counter_trend:
            if direction == 'LONG':
                trend_desc = "REVERSAL: bearish→bullish"
            else:
                trend_desc = "REVERSAL: bullish→bearish"
        else:
            trend_desc = f"WITH TREND: {recent_trend}"

        smc_candidates.append({
            'strategy': 'SMC_CONFLUENCE',
            'direction': direction,
            'confidence': adjusted_confidence,
            'raw_score': raw_confidence,
            'entry': entry_price,
            'stop_loss': stop,
            'take_profit': target,
            'risk': risk,
            'rr': live_rr,
            'reasoning': f"SMC confluence (R:R {live_rr:.1f}, {trend_desc}, {conf_str})"
        })

    # Only add the BEST SMC confluence signal
    if smc_candidates:
        smc_candidates.sort(key=lambda x: (x['confidence'], x['raw_score']), reverse=True)
        best = smc_candidates[0]
        del best['raw_score']
        signals.append(best)

    # Filter by minimum confidence and sort
    signals = [s for s in signals if s['confidence'] >= min_confidence]
    signals.sort(key=lambda x: x['confidence'], reverse=True)

    return signals


def print_high_confidence_summary(bars: List[Bar], daily_open: float = None,
                                   four_hour_range: Dict = None, filepath: str = None):
    """Print a summary of all high-confidence signals for quick decision making."""
    signals = get_high_confidence_signals(bars, daily_open, four_hour_range, filepath=filepath)

    print("\n" + "=" * 60)
    print("        HIGH CONFIDENCE SIGNALS SUMMARY")
    print("=" * 60)

    if not signals:
        print("\n  No high-confidence signals at this time.")
        print("  Wait for better setup or lower confidence threshold.")
        return

    for i, sig in enumerate(signals[:5], 1):
        print(f"\n  #{i} [{sig['strategy']}] {sig['direction']} - {sig['confidence']}% confidence")
        print(f"      Entry:  {sig['entry']:.2f}")
        print(f"      Stop:   {sig['stop_loss']:.2f} (risk: {sig['risk']:.2f} pts = ${sig['risk'] * 50:.0f} ES)")
        print(f"      Target: {sig['take_profit']:.2f} (R:R = 1:{sig['rr']:.1f})")
        print(f"      Reason: {sig['reasoning']}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMC Confluence Analysis')
    parser.add_argument('--log', default='/tmp/fractal_bot.log', help='Path to fractal_bot log')
    parser.add_argument('--all', action='store_true', help='Show all details')
    parser.add_argument('--signals-only', action='store_true', help='Show only high-confidence signals')
    parser.add_argument('--daily-open', type=float, help='Daily open price (midnight ET)')
    parser.add_argument('--4h-high', type=float, dest='four_h_high', help='4-hour range high')
    parser.add_argument('--4h-low', type=float, dest='four_h_low', help='4-hour range low')
    parser.add_argument('--asia-high', type=float, help='Asia session high')
    parser.add_argument('--asia-low', type=float, help='Asia session low')
    parser.add_argument('--london-high', type=float, help='London session high')
    parser.add_argument('--london-low', type=float, help='London session low')
    parser.add_argument('--pdh', type=float, help='Previous day high')
    parser.add_argument('--pdl', type=float, help='Previous day low')
    args = parser.parse_args()

    bars = parse_bars(args.log)

    # Build 4-hour range from args if provided
    four_hour_range = None
    if args.four_h_high and args.four_h_low:
        four_hour_range = {
            'high': args.four_h_high,
            'low': args.four_h_low,
            'range': args.four_h_high - args.four_h_low,
            'start_idx': 0,
            'end_idx': 48,
            'bars_used': 48
        }

    # Display session levels if provided
    session_levels = {}
    if args.daily_open:
        session_levels['Daily Open'] = args.daily_open
    if args.four_h_high:
        session_levels['4H High'] = args.four_h_high
    if args.four_h_low:
        session_levels['4H Low'] = args.four_h_low
    if args.asia_high:
        session_levels['Asia High'] = args.asia_high
    if args.asia_low:
        session_levels['Asia Low'] = args.asia_low
    if args.london_high:
        session_levels['London High'] = args.london_high
    if args.london_low:
        session_levels['London Low'] = args.london_low
    if args.pdh:
        session_levels['PDH'] = args.pdh
    if args.pdl:
        session_levels['PDL'] = args.pdl

    if session_levels:
        print("\n  SESSION LEVELS (manual input):")
        for name, level in session_levels.items():
            print(f"    {name}: {level:.2f}")
        print()

    if args.signals_only:
        print_high_confidence_summary(bars, args.daily_open, four_hour_range, filepath=args.log)
    else:
        print_analysis(bars, args.all, daily_open=args.daily_open, four_hour_range=four_hour_range)
        print_high_confidence_summary(bars, args.daily_open, four_hour_range, filepath=args.log)
