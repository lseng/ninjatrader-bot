#!/usr/bin/env python3
"""
SMC (Smart Money Concepts) Confluence Analysis
Integrates with fractal_bot for signal confirmation

Usage:
    python scripts/smc_analysis.py [--log /tmp/fractal_bot.log]
    python scripts/smc_analysis.py --asia-high 6920 --asia-low 6900
"""

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
                # If bearish FVG formed after bullish, price went down → BPR is resistance
                # If bullish FVG formed after bearish, price went up → BPR is support
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


def find_order_blocks(bars: List[Bar], lookback: int = 100) -> List[Dict]:
    """Find Order Blocks (last opposing candle before displacement)"""
    obs = []

    for i in range(4, min(lookback, len(bars))):
        idx = len(bars) - i
        if idx < 4:
            break

        # Check for bullish displacement (2+ green candles with decent body)
        if (bars[idx].color == 'GREEN' and bars[idx-1].color == 'GREEN' and
            bars[idx].body > 0.5 and bars[idx-1].body > 0.5):
            for j in range(idx-2, max(0, idx-8), -1):
                if bars[j].color == 'RED':
                    obs.append({
                        'type': 'BULLISH',
                        'high': bars[j].H,
                        'low': bars[j].L,
                        'idx': j
                    })
                    break

        # Check for bearish displacement
        if (bars[idx].color == 'RED' and bars[idx-1].color == 'RED' and
            bars[idx].body > 0.5 and bars[idx-1].body > 0.5):
            for j in range(idx-2, max(0, idx-8), -1):
                if bars[j].color == 'GREEN':
                    obs.append({
                        'type': 'BEARISH',
                        'high': bars[j].H,
                        'low': bars[j].L,
                        'idx': j
                    })
                    break

    # Dedupe by price zone
    unique_obs = []
    for ob in obs:
        is_dup = any(abs(ob['low'] - u['low']) < 1 and ob['type'] == u['type'] for u in unique_obs)
        if not is_dup:
            unique_obs.append(ob)

    return unique_obs


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
    score = 50  # Base score (neutral)
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

    if bars and len(bars) >= 10:
        # Detect displacement
        displacements = detect_displacement(bars)

        # Detect BOS
        bos_events = detect_bos(bars, swings)

        # Check manipulation completion
        if signal_type == 'LONG':
            manip_check = check_manipulation_complete(bars, daily_open, 'low')

            # Check for bullish displacement in recent bars
            # DISPLACEMENT IS CRITICAL - without it, no proof of reversal
            recent_bullish_disp = [d for d in displacements
                                   if d['type'] == 'BULLISH' and d['end_idx'] >= len(bars) - 20]
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
                score -= 25
                factors.append(('-25', 'NO DISPLACEMENT - reversal NOT confirmed ⚠'))

            # Check for bullish BOS (helpful but not as critical as displacement)
            recent_bullish_bos = [b for b in bos_events
                                  if b['type'] == 'BULLISH' and b['bar_idx'] >= len(bars) - 15]
            if recent_bullish_bos:
                has_bos = True
                latest = recent_bullish_bos[-1]
                score += 12
                factors.append(('+12', f"Bullish BOS at {latest['swing_price']:.2f}"))
            else:
                score -= 8
                factors.append(('-8', 'No BOS yet - structure intact'))

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

        elif signal_type == 'SHORT':
            manip_check = check_manipulation_complete(bars, daily_open, 'high')

            # Check for bearish displacement
            # DISPLACEMENT IS CRITICAL - without it, no proof of reversal
            recent_bearish_disp = [d for d in displacements
                                   if d['type'] == 'BEARISH' and d['end_idx'] >= len(bars) - 20]
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
                score -= 25
                factors.append(('-25', 'NO DISPLACEMENT - reversal NOT confirmed ⚠'))

            # Check for bearish BOS (helpful but not as critical as displacement)
            recent_bearish_bos = [b for b in bos_events
                                  if b['type'] == 'BEARISH' and b['bar_idx'] >= len(bars) - 15]
            if recent_bearish_bos:
                has_bos = True
                latest = recent_bearish_bos[-1]
                score += 12
                factors.append(('+12', f"Bearish BOS at {latest['swing_price']:.2f}"))
            else:
                score -= 8
                factors.append(('-8', 'No BOS yet - structure intact'))

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

        # Turtle Soup (high probability setup)
        for swing in swings.get('lows', []):
            for fvg in fvgs:
                if fvg['type'] == 'BULLISH' and not fvg['filled']:
                    if fvg['top'] < swing['price'] and swing['price'] - fvg['top'] < 5:
                        if current_price < swing['price'] and current_price >= fvg['bottom']:
                            score += 25
                            factors.append(('+25', f'TURTLE SOUP: Swept {swing["price"]:.2f}, in FVG'))
                            stop = fvg['bottom'] - 1
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

        # Turtle Soup (high probability setup)
        for swing in swings.get('highs', []):
            for fvg in fvgs:
                if fvg['type'] == 'BEARISH' and not fvg['filled']:
                    if fvg['bottom'] > swing['price'] and fvg['bottom'] - swing['price'] < 5:
                        if current_price > swing['price'] and current_price <= fvg['top']:
                            score += 25
                            factors.append(('+25', f'TURTLE SOUP: Swept {swing["price"]:.2f}, in FVG'))
                            stop = fvg['top'] + 1
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
        'factors': factors
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


def print_analysis(bars: List[Bar], show_all: bool = False, daily_open: float = None):
    """Print full SMC analysis"""
    if not bars:
        print("No bar data available")
        return

    current_price = bars[-1].C
    session_high = max(b.H for b in bars)
    session_low = min(b.L for b in bars)

    drt = calculate_drt(session_high, session_low)
    obs = find_order_blocks(bars)
    fvgs = find_fvgs(bars)
    swings = find_swing_points(bars)
    bprs = find_balanced_price_ranges(fvgs)

    # NEW: Detect displacement and BOS
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

    print(f"\n  ORDER BLOCKS ({len(obs)} found):")
    for ob in obs[:5]:
        marker = ">>>" if ob['low'] <= current_price <= ob['high'] else "   "
        print(f"  {marker} {ob['type']:8} OB: {ob['low']:.2f} - {ob['high']:.2f}")

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
            print(f"  {strength_icon} {d['type']:8} | {d['candles']} candles | {d['move']:.2f} pts | {d['start_price']:.2f} → {d['end_price']:.2f}")
    else:
        print("      None detected - no strong directional moves")

    # NEW: Break of Structure
    print(f"\n  BREAK OF STRUCTURE ({len(bos_events)} detected):")
    if bos_events:
        recent_bos = bos_events[-3:]  # Last 3
        for b in recent_bos:
            print(f"    • {b['description']}")
    else:
        print("      None detected - structure intact")

    # NEW: Change of Character
    if choch:
        print(f"\n  ⚠️  CHANGE OF CHARACTER DETECTED:")
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
        print(f"  {'─' * 40}")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMC Confluence Analysis')
    parser.add_argument('--log', default='/tmp/fractal_bot.log', help='Path to fractal_bot log')
    parser.add_argument('--all', action='store_true', help='Show all details')
    parser.add_argument('--daily-open', type=float, help='Daily open price (midnight ET)')
    parser.add_argument('--asia-high', type=float, help='Asia session high')
    parser.add_argument('--asia-low', type=float, help='Asia session low')
    parser.add_argument('--london-high', type=float, help='London session high')
    parser.add_argument('--london-low', type=float, help='London session low')
    parser.add_argument('--pdh', type=float, help='Previous day high')
    parser.add_argument('--pdl', type=float, help='Previous day low')
    args = parser.parse_args()

    bars = parse_bars(args.log)

    # Display session levels if provided
    session_levels = {}
    if args.daily_open:
        session_levels['Daily Open'] = args.daily_open
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

    print_analysis(bars, args.all, daily_open=args.daily_open)
