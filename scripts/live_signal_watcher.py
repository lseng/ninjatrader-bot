#!/usr/bin/env python3
"""
Live Signal Watcher - Monitors for high-confidence trading signals

================================================================================
CURRENT LIVE CONFIGURATION (as of 2026-01-12)
================================================================================

ACTIVE SIGNALS:
  - CISD with R:R >= 5.0 (primary signal, best during ETH/overnight)
  - 4H Range false breakouts
  - SMC confluence (Order Blocks, Breakers, Inducement Traps)

AUDIO ALERTS:
  - Airport chime + voice announcement
  - Says: direction, strategy, entry price, stop, target

USAGE:
    python scripts/live_signal_watcher.py --realtime

================================================================================

Runs continuously, checks every N seconds, and alerts when:
- New high-confidence signal appears
- 4H Range setup triggers
- Price enters key zones (OB, breaker, etc.)

TIMEZONE HANDLING:
- Bot logs are in PST (Pacific)
- 4H Range strategy uses NY time (Eastern)
- NY = PST + 3 hours

Usage:
    python scripts/live_signal_watcher.py
    python scripts/live_signal_watcher.py --interval 30
    python scripts/live_signal_watcher.py --min-confidence 75
"""

import sys
import io
import os
import time
import argparse
import logging
from datetime import datetime, timedelta

# Non-blocking keyboard input
if sys.platform == 'win32':
    import msvcrt
    def get_key():
        """Get key press (non-blocking). Returns None if no key."""
        if msvcrt.kbhit():
            return msvcrt.getch().decode('utf-8', errors='ignore').upper()
        return None
else:
    import select
    def get_key():
        """Get key press (non-blocking). Returns None if no key."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).upper()
        return None

# Fix Windows console encoding and enable colors
if sys.platform == 'win32':
    try:
        os.system('chcp 65001 >nul 2>&1')
    except:
        pass

# Colors
from colorama import init, Fore, Back, Style
init()  # Enable ANSI colors on Windows

# Color shortcuts
GREEN = Fore.GREEN
RED = Fore.RED
YELLOW = Fore.YELLOW
CYAN = Fore.CYAN
MAGENTA = Fore.MAGENTA
WHITE = Fore.WHITE
BRIGHT = Style.BRIGHT
RESET = Style.RESET_ALL

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.smc_analysis import (
    parse_bars, get_high_confidence_signals, get_4hour_range_signal,
    calculate_4hour_range, analyze_4hour_range_strategy,
    get_session_info
)

from scripts.audio_alerts import speak_signal, VOICES
import re


def get_live_price(filepath):
    """Get the latest price from Price: lines in log file.

    The log contains tick-by-tick prices like:
    Price: 6953.25 | Ticks: +5 ...

    This returns the most recent price for accurate entry calculations.
    """
    price = None
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if 'Price:' in line and 'Ticks:' in line:
                    match = re.search(r'Price:\s*([\d.]+)', line)
                    if match:
                        price = float(match.group(1))
    except:
        pass
    return price


# ============================================================================
# LOGGING SETUP
# ============================================================================

ALERT_LOG_DIR = r'C:\Users\leone\ninjatrader-bot\logs'
ALERT_LOG_FILE = os.path.join(ALERT_LOG_DIR, 'signal_alerts.log')

def setup_logging():
    """Setup logging to file"""
    os.makedirs(ALERT_LOG_DIR, exist_ok=True)

    # Create logger
    logger = logging.getLogger('signal_watcher')
    logger.setLevel(logging.INFO)

    # File handler - append mode
    fh = logging.FileHandler(ALERT_LOG_FILE, encoding='utf-8')
    fh.setLevel(logging.INFO)

    # Format: timestamp | level | message
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger

alert_logger = setup_logging()


# ============================================================================
# TIMEZONE HANDLING
# ============================================================================

def get_ny_time() -> datetime:
    """Get current time in New York (Eastern)"""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo('America/New_York'))
    except ImportError:
        # Fallback: PST + 3 hours = ET
        return datetime.now() + timedelta(hours=3)


def get_pst_time() -> datetime:
    """Get current time in Pacific"""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo('America/Los_Angeles'))
    except ImportError:
        return datetime.now()


def get_session_for_ny_time() -> dict:
    """Get trading session based on NY time"""
    ny = get_ny_time()
    hour = ny.hour
    minute = ny.minute
    time_val = hour * 60 + minute

    # Session times in NY
    if time_val >= 18 * 60 or time_val < 3 * 60:  # 6PM - 3AM NY
        session = 'ASIA'
        desc = 'Asia Session - Range forming, liquidity building'
        targets = 'Building session range'
    elif time_val < 9 * 60 + 30:  # 3AM - 9:30AM NY
        session = 'LONDON'
        desc = 'London Session - Watch for Judas swing'
        targets = 'Target Asia H/L, watch for fake move'
    elif time_val < 12 * 60:  # 9:30AM - 12PM NY
        session = 'NY_AM'
        desc = 'NY AM Session - High volume, real moves'
        targets = 'Target London H/L or Asia H/L'
    elif time_val < 16 * 60:  # 12PM - 4PM NY
        session = 'NY_PM'
        desc = 'NY PM Session - Continuation or reversal'
        targets = 'Target unfilled levels from AM'
    else:  # 4PM - 6PM NY
        session = 'CLOSED'
        desc = 'Market Closed - Wait for Asia open'
        targets = 'N/A'

    return {
        'current': session,
        'description': desc,
        'liquidity_targets': targets,
        'ny_time': ny.strftime('%H:%M:%S'),
        'pst_time': get_pst_time().strftime('%H:%M:%S')
    }


def get_4h_candle_info() -> dict:
    """
    Get info about current 4H candle in NY time.

    4H candles start at:
    - 00:00 NY (Candle 1 - USE THIS FOR RANGE)
    - 04:00 NY
    - 08:00 NY
    - 12:00 NY
    - 16:00 NY
    - 20:00 NY
    """
    ny = get_ny_time()
    hour = ny.hour

    # Which 4H candle are we in?
    candle_num = hour // 4 + 1
    candle_start_hour = (candle_num - 1) * 4
    candle_end_hour = candle_num * 4

    # Time until next 4H candle
    mins_into_candle = (hour - candle_start_hour) * 60 + ny.minute
    mins_remaining = 240 - mins_into_candle

    # Is the first 4H candle (00:00-04:00 NY) complete?
    first_candle_complete = hour >= 4

    return {
        'candle_number': candle_num,
        'candle_start': f"{candle_start_hour:02d}:00 NY",
        'candle_end': f"{candle_end_hour:02d}:00 NY",
        'mins_into_candle': mins_into_candle,
        'mins_remaining': mins_remaining,
        'first_candle_complete': first_candle_complete,
        'can_trade_4h_range': first_candle_complete
    }


def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def beep():
    """Make alert sound"""
    print('\a', end='', flush=True)  # Terminal bell


def format_signal(sig: dict) -> str:
    """Format a signal for display"""
    arrow = ">>>" if sig['direction'] == 'LONG' else "<<<"
    return f"""
{'='*50}
  {arrow} [{sig['strategy']}] {sig['direction']} - {sig['confidence']}% confidence
{'='*50}
  Entry:  {sig['entry']:.2f}
  Stop:   {sig['stop_loss']:.2f} ({sig['risk']:.2f} pts = ${sig['risk'] * 50:.0f} ES / ${sig['risk'] * 5:.0f} MES)
  Target: {sig['take_profit']:.2f} (R:R = 1:{sig['rr']:.1f})
  Reason: {sig['reasoning']}
"""


def watch_signals(log_path: str, interval: int = 60, min_confidence: int = 70,
                  four_hour_range: dict = None, voice: str = 'aria',
                  silent: bool = False):
    """
    Continuously watch for trading signals.

    Args:
        log_path: Path to fractal_bot log
        interval: Seconds between checks
        min_confidence: Minimum confidence to alert
        four_hour_range: Manual 4H range override
        voice: TTS voice (aria, guy, jenny, davis)
        silent: Disable audio alerts
    """
    last_signal_hash = None
    alert_count = 0
    first_run = True  # Skip alert on first run
    alert_history = []  # Store recent alerts to display
    check_count = 0  # For periodic status logging

    print(f"\n{CYAN}{'='*60}{RESET}", flush=True)
    print(f"  {BRIGHT}{WHITE}LIVE SIGNAL WATCHER{RESET}", flush=True)
    print(f"  {CYAN}Log:{RESET} {log_path}", flush=True)
    print(f"  {CYAN}Alert Log:{RESET} {ALERT_LOG_FILE}", flush=True)
    print(f"  {CYAN}Interval:{RESET} {interval}s | {CYAN}Min Confidence:{RESET} {min_confidence}%", flush=True)
    print(f"  {CYAN}Timezone:{RESET} Bot=PST, Strategy=NY (PST+3)", flush=True)
    print(f"  {CYAN}Voice:{RESET} {voice.upper()} {YELLOW}{'(muted)' if silent else ''}{RESET}", flush=True)
    print(f"  {YELLOW}Press Ctrl+C to stop{RESET}", flush=True)
    print(f"{CYAN}{'='*60}{RESET}\n", flush=True)
    print(f"  {YELLOW}Starting... (first update in a few seconds){RESET}", flush=True)
    print(f"  {CYAN}Stream alerts:{RESET} tail -f \"{ALERT_LOG_FILE}\"", flush=True)

    # Log startup
    alert_logger.info("="*60)
    alert_logger.info("SIGNAL WATCHER STARTED")
    alert_logger.info(f"  Min confidence: {min_confidence}%")
    alert_logger.info(f"  Interval: {interval}s")
    alert_logger.info("="*60)

    while True:
        try:
            bars = parse_bars(log_path)

            if not bars:
                pst = get_pst_time().strftime('%H:%M:%S')
                print(f"[{pst} PST] Waiting for bar data...")
                time.sleep(interval)
                continue

            # Get live price for accurate entry calculations
            live_price = get_live_price(log_path)
            current_price = live_price or bars[-1].C

            # Use NY-based session info
            session = get_session_for_ny_time()
            candle_info = get_4h_candle_info()

            # Get signals with live price for accurate entry
            signals = get_high_confidence_signals(
                bars,
                four_hour_range=four_hour_range,
                min_confidence=min_confidence,
                filepath=log_path,
                live_price=current_price
            )

            # Get 4H range status
            four_hour = analyze_4hour_range_strategy(bars, four_hour_range, filepath=log_path)

            # Create hash of current signals to detect changes
            signal_hash = str([(s['strategy'], s['direction'], s['confidence']) for s in signals])

            # Check if signals changed (but NOT on first run)
            new_signal = signal_hash != last_signal_hash and signals and not first_run

            # Display status
            pst_time = get_pst_time().strftime('%H:%M:%S')
            ny_time = get_ny_time().strftime('%H:%M:%S')

            if new_signal:
                alert_count += 1

                # Voice alert for top signal
                if not silent and signals:
                    try:
                        speak_signal(signals[0], voice=voice, chime=True,
                                    chime_style='airport', overlap_seconds=3.0,
                                    speech_rate='+20%')
                    except Exception as e:
                        print(f"\n{RED}[Audio Error] {e}{RESET}")
                        beep()  # Fallback to beep

                # Log alert to file (flush immediately)
                top = signals[0]
                risk_pts = top['risk']
                alert_logger.info(f"NEW ALERT | {top['direction']} {top['confidence']}% | {top['strategy']}")
                alert_logger.info(f"  Entry: {top['entry']:.2f} | Stop: {top['stop_loss']:.2f} ({risk_pts:.1f}pts/${risk_pts*50:.0f}) | Target: {top['take_profit']:.2f} | R:R {top['rr']:.1f}")
                alert_logger.info(f"  Reason: {top['reasoning']}")
                for handler in alert_logger.handlers:
                    handler.flush()

                # Store alert in history (keep last 5)
                alert_history.append({
                    'time': pst_time,
                    'direction': top['direction'],
                    'confidence': top['confidence'],
                    'entry': top['entry'],
                    'stop': top['stop_loss'],
                    'target': top['take_profit'],
                    'risk': top['risk'],
                    'rr': top['rr'],
                    'strategy': top['strategy']
                })
                # Keep all alerts (no limit)

            # Update hash (including first run)
            last_signal_hash = signal_hash
            first_run = False
            check_count += 1

            # Log status every 10 checks (~10 minutes at 60s interval)
            if check_count % 10 == 0:
                if signals:
                    top = signals[0]
                    alert_logger.info(f"STATUS | Price: {current_price:.2f} | {session['current']} | Current: {top['direction']} {top['confidence']}%")
                else:
                    alert_logger.info(f"STATUS | Price: {current_price:.2f} | {session['current']} | No active signal")

            # Always update display (whether new signal or not)
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"{CYAN}{'='*80}{RESET}")
            print(f"  {BRIGHT}LIVE SIGNAL WATCHER{RESET} - Press Ctrl+C to stop")
            print(f"{CYAN}{'='*80}{RESET}")

            # Current status
            print(f"\n{CYAN}[{pst_time} PST]{RESET} {YELLOW}{session['current']}{RESET} | Price: {WHITE}{current_price:.2f}{RESET}")

            if signals:
                top = signals[0]
                sig_color = GREEN if top['direction'] == 'LONG' else RED
                risk_pts = top['risk']
                print(f"\n{BRIGHT}CURRENT SIGNAL:{RESET}")
                print(f"  {sig_color}{BRIGHT}{top['direction']}{RESET} {top['confidence']}% [{top['strategy']}]")
                print(f"  Entry:  {WHITE}{top['entry']:.2f}{RESET}")
                print(f"  Stop:   {RED}{top['stop_loss']:.2f}{RESET} ({risk_pts:.1f} pts = ${risk_pts*50:.0f} ES / ${risk_pts*5:.0f} MES)")
                print(f"  Target: {GREEN}{top['take_profit']:.2f}{RESET}")
                print(f"  R:R:    {top['rr']:.1f}")
            else:
                print(f"\n{WHITE}No active signal - waiting for setup{RESET}")

            # 4H Range status
            if four_hour['status'].startswith('WAITING'):
                print(f"\n{MAGENTA}4H RANGE PENDING:{RESET} {four_hour['current_setup']['pending_signal']}")
                print(f"  {four_hour['current_setup']['condition']}")

            # Alert history
            if alert_history:
                print(f"\n{CYAN}{'─'*80}{RESET}")
                print(f"{BRIGHT}ALERT HISTORY (this session):{RESET}")
                for i, alert in enumerate(reversed(alert_history), 1):
                    a_color = GREEN if alert['direction'] == 'LONG' else RED
                    print(f"  {alert['time']} | {a_color}{alert['direction']}{RESET} {alert['confidence']}% | E:{alert['entry']:.2f} S:{alert['stop']:.2f} T:{alert['target']:.2f} | R:R {alert['rr']:.1f}")

            print(f"\n{CYAN}{'─'*80}{RESET}")
            print(f"Alerts fired: {alert_count} | Interval: {interval}s", flush=True)

            time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n\nWatcher stopped. Total alerts: {alert_count}")
            alert_logger.info(f"WATCHER STOPPED | Total alerts: {alert_count}")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(interval)


def tail_file(filepath, interval=0.02):
    """Tail a file and yield new lines as they appear.
    Yields None when no new line (for keyboard polling).
    interval=0.02 means ~50 keyboard checks per second."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        f.seek(0, 2)  # Go to end
        while True:
            line = f.readline()
            if line:
                yield line.strip()
            else:
                yield None  # Allow keyboard check
                time.sleep(interval)


def watch_realtime(log_path: str, min_confidence: int = 70,
                   voice: str = 'aria', silent: bool = False,
                   four_hour_range: dict = None):
    """
    Real-time signal watcher - tails log file for instant updates.
    Tracks active trades until stop/target is hit.

    KEYBOARD CONTROLS:
    - T = Took trade (starts tracking P&L)
    - S = Skipped trade (ignores signal, waits for next)
    - X = Exit trade early (manual close, stops tracking)
    """
    last_signal_hash = None
    alert_count = 0
    alert_history = []

    # Trade tracking - stays locked until stop/target hit
    active_trade = None  # {'direction', 'entry', 'stop', 'target', 'time'}

    # Pending signal - waiting for user to press T or S
    pending_signal = None  # Same structure as active_trade

    # Track skipped signals so they don't keep showing
    skipped_signal_hash = None

    print(f"\n{CYAN}{'='*70}{RESET}")
    print(f"  {BRIGHT}REAL-TIME SIGNAL WATCHER{RESET}")
    print(f"  Tailing: {log_path}")
    print(f"  Alert Log: {ALERT_LOG_FILE}")
    print(f"  Voice: {voice.upper()} {'(muted)' if silent else ''}")
    print(f"{CYAN}{'='*70}{RESET}")
    print(f"  {YELLOW}KEYS: [T]=Took trade  [S]=Skip  [X]=Exit early{RESET}")

    alert_logger.info("="*60)
    alert_logger.info("REAL-TIME WATCHER STARTED")
    alert_logger.info("="*60)

    # Initial load
    bars = parse_bars(log_path)
    if bars:
        print(f"  Loaded {len(bars)} bars, price: {bars[-1].C:.2f}")

    print(f"  {YELLOW}Streaming... (Ctrl+C to stop){RESET}\n")

    try:
        for line in tail_file(log_path):
            # Check for keyboard input FIRST (runs every 0.1s even with no log data)
            key = get_key()
            if key:
                if key == 'T':
                    if pending_signal and not active_trade:
                        # User took the trade - start tracking
                        active_trade = pending_signal
                        pending_signal = None
                        print(f"\n{GREEN}{BRIGHT}>>> TRADE TAKEN{RESET} - Tracking {active_trade['direction']} @ {active_trade['entry']:.2f}")
                        alert_logger.info(f"USER TOOK TRADE | {active_trade['direction']} @ {active_trade['entry']:.2f}")
                    else:
                        print(f"\n{YELLOW}[T pressed - no pending signal]{RESET}")

                elif key == 'S':
                    if pending_signal:
                        # User skipped the trade - track the signal hash so it doesn't show again
                        print(f"\n{YELLOW}>>> SKIPPED{RESET} - Waiting for next signal")
                        alert_logger.info(f"USER SKIPPED | {pending_signal['direction']} @ {pending_signal['entry']:.2f}")
                        skipped_signal_hash = last_signal_hash  # Remember which signal was skipped
                        pending_signal = None
                    else:
                        print(f"\n{YELLOW}[S pressed - no pending signal]{RESET}")

                elif key == 'X':
                    if active_trade:
                        # User manually exiting trade
                        live_price = get_live_price(log_path)
                        bars_temp = parse_bars(log_path)
                        exit_price = live_price or (bars_temp[-1].C if bars_temp else active_trade['entry'])
                        if active_trade['direction'] == 'LONG':
                            pnl = exit_price - active_trade['entry']
                        else:
                            pnl = active_trade['entry'] - exit_price
                        pnl_color = GREEN if pnl >= 0 else RED
                        print(f"\n{MAGENTA}{BRIGHT}>>> MANUAL EXIT{RESET} @ {exit_price:.2f} | P&L: {pnl_color}{pnl:+.2f}{RESET} pts")
                        alert_logger.info(f"USER EXITED | {active_trade['direction']} @ {exit_price:.2f} | P&L: {pnl:+.2f} pts")
                        active_trade = None
                    else:
                        print(f"\n{YELLOW}[X pressed - no active trade]{RESET}")

                elif key in ('T', 'S', 'X'):
                    pass  # Already handled above
                else:
                    # Show any other key press for debugging
                    print(f"\n{CYAN}[Key: {repr(key)}]{RESET}")

            # Skip if no new log line (but keyboard was still checked above)
            if line is None:
                continue

            # Check for price update or bar
            is_price = 'Price:' in line and 'Ticks:' in line
            is_bar = 'BAR' in line and 'O:' in line

            if not is_price and not is_bar:
                continue

            # Re-parse bars
            bars = parse_bars(log_path)
            if not bars:
                continue

            # Get live price for accurate entry calculations
            live_price = get_live_price(log_path)
            current_price = live_price or bars[-1].C
            pst_time = get_pst_time().strftime('%H:%M:%S')
            session = get_session_for_ny_time()

            # Check if we have a pending signal waiting for user input
            if pending_signal and not active_trade:
                # Show pending signal with prompt
                sig_color = GREEN if pending_signal['direction'] == 'LONG' else RED
                if pending_signal['direction'] == 'LONG':
                    pnl = current_price - pending_signal['entry']
                else:
                    pnl = pending_signal['entry'] - current_price
                pnl_color = GREEN if pnl >= 0 else RED
                print(f"\r{CYAN}[{pst_time}]{RESET} {WHITE}{current_price:.2f}{RESET} | {sig_color}{BRIGHT}{pending_signal['direction']}{RESET} @ {pending_signal['entry']:.2f} | {pnl_color}{pnl:+.2f}{RESET} pts | {YELLOW}{BRIGHT}[T]ake [S]kip?{RESET}    ", end='', flush=True)
                continue

            # Check active trade status first
            if active_trade:
                pnl = 0
                trade_status = 'ACTIVE'
                if active_trade['direction'] == 'LONG':
                    pnl = current_price - active_trade['entry']
                    if current_price <= active_trade['stop']:
                        trade_status = 'STOPPED'
                        pnl = active_trade['stop'] - active_trade['entry']
                    elif current_price >= active_trade['target']:
                        trade_status = 'TARGET'
                        pnl = active_trade['target'] - active_trade['entry']
                else:  # SHORT
                    pnl = active_trade['entry'] - current_price
                    if current_price >= active_trade['stop']:
                        trade_status = 'STOPPED'
                        pnl = active_trade['entry'] - active_trade['stop']
                    elif current_price <= active_trade['target']:
                        trade_status = 'TARGET'
                        pnl = active_trade['entry'] - active_trade['target']

                if trade_status in ['STOPPED', 'TARGET']:
                    status_color = RED if trade_status == 'STOPPED' else GREEN
                    print(f"\n{BRIGHT}{status_color}{'='*60}{RESET}")
                    print(f"  TRADE {trade_status}! P&L: {pnl:+.2f} pts (${pnl*50:+.0f} ES)")
                    print(f"  Entry: {active_trade['entry']:.2f} -> Exit: {current_price:.2f}")
                    print(f"{BRIGHT}{status_color}{'='*60}{RESET}\n")
                    alert_logger.info(f"TRADE {trade_status} | {active_trade['direction']} | P&L: {pnl:+.2f} pts")
                    active_trade = None  # Clear trade, ready for new signals
                    last_signal_hash = None  # Reset so next signal triggers new alert
                    continue  # Don't immediately show new signal - give user a moment
                else:
                    # Show active trade status
                    pnl_color = GREEN if pnl >= 0 else RED
                    sig_color = GREEN if active_trade['direction'] == 'LONG' else RED
                    print(f"\r{CYAN}[{pst_time}]{RESET} {session['current']} | {WHITE}{current_price:.2f}{RESET} | {sig_color}{active_trade['direction']}{RESET} @ {active_trade['entry']:.2f} | P&L: {pnl_color}{pnl:+.2f}{RESET} pts {YELLOW}[X=exit]{RESET}    ", end='', flush=True)
                    last_signal_hash = None  # Don't check for new signals while trade active
                    continue  # Skip signal generation while trade is active

            signals = get_high_confidence_signals(
                bars, four_hour_range=four_hour_range, min_confidence=min_confidence,
                filepath=log_path, live_price=current_price
            )
            four_hour = analyze_4hour_range_strategy(bars, four_hour_range, filepath=log_path)

            signal_hash = str([(s['strategy'], s['direction'], s['confidence']) for s in signals])

            if signal_hash != last_signal_hash and signals:
                # NEW SIGNAL - clear any previous skipped signal
                skipped_signal_hash = None
                alert_count += 1
                top = signals[0]
                sig_color = GREEN if top['direction'] == 'LONG' else RED
                risk_pts = top['risk']

                # Print FIRST (immediate visual feedback)
                print(f"\n{BRIGHT}{YELLOW}{'!'*60}{RESET}")
                print(f"  {BRIGHT}NEW ALERT #{alert_count}{RESET} @ {pst_time}")
                print(f"{BRIGHT}{YELLOW}{'!'*60}{RESET}")
                print(f"  {sig_color}{BRIGHT}{top['direction']}{RESET} {top['confidence']}% [{top['strategy']}]")
                print(f"  Entry:  {WHITE}{top['entry']:.2f}{RESET}")
                print(f"  Stop:   {RED}{top['stop_loss']:.2f}{RESET} ({risk_pts:.1f} pts = ${risk_pts*50:.0f} ES)")
                print(f"  Target: {GREEN}{top['take_profit']:.2f}{RESET}")
                print(f"  R:R:    {top['rr']:.1f}")
                print(f"  Reason: {top['reasoning']}")
                print(f"\n  {YELLOW}{BRIGHT}>>> Press [T] to TAKE or [S] to SKIP <<<{RESET}\n", flush=True)

                # Audio AFTER text (so you see it while hearing it)
                if not silent:
                    try:
                        speak_signal(top, voice=voice, chime=True,
                                    chime_style='airport', overlap_seconds=3.0,
                                    speech_rate='+25%')
                    except Exception as e:
                        print(f"\n{RED}[Audio Error] {e}{RESET}")
                        print('\a', end='', flush=True)

                # Log
                alert_logger.info(f"NEW ALERT | {top['direction']} {top['confidence']}% | {top['strategy']}")
                alert_logger.info(f"  Entry: {top['entry']:.2f} | Stop: {top['stop_loss']:.2f} ({risk_pts:.1f}pts/${risk_pts*50:.0f}) | Target: {top['take_profit']:.2f} | R:R {top['rr']:.1f}")
                alert_logger.info(f"  Reason: {top['reasoning']}")
                for handler in alert_logger.handlers:
                    handler.flush()

                # History
                alert_history.append({
                    'time': pst_time,
                    'direction': top['direction'],
                    'confidence': top['confidence'],
                    'entry': top['entry'],
                    'stop': top['stop_loss'],
                    'target': top['take_profit'],
                    'rr': top['rr']
                })

                # Set as pending - wait for user to confirm with T or skip with S
                pending_signal = {
                    'direction': top['direction'],
                    'entry': top['entry'],
                    'stop': top['stop_loss'],
                    'target': top['take_profit'],
                    'time': pst_time
                }

            elif signals:
                # Check if this signal was already skipped
                if signal_hash == skipped_signal_hash:
                    print(f"\r{CYAN}[{pst_time}]{RESET} {session['current']} | {WHITE}{current_price:.2f}{RESET} | {YELLOW}Skipped - waiting for new signal{RESET}    ", end='', flush=True)
                else:
                    top = signals[0]
                    sig_color = GREEN if top['direction'] == 'LONG' else RED
                    print(f"\r{CYAN}[{pst_time}]{RESET} {session['current']} | {WHITE}{current_price:.2f}{RESET} | {sig_color}{top['direction']}{RESET} {top['confidence']}% E:{top['entry']:.2f} S:{top['stop_loss']:.2f} T:{top['take_profit']:.2f}    ", end='', flush=True)
            else:
                print(f"\r{CYAN}[{pst_time}]{RESET} {session['current']} | {WHITE}{current_price:.2f}{RESET} | No signal    ", end='', flush=True)

            last_signal_hash = signal_hash

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Stopped. Alerts: {alert_count}{RESET}")
        alert_logger.info(f"WATCHER STOPPED | Alerts: {alert_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live Signal Watcher')
    parser.add_argument('--log',
                        default=r'C:\Users\leone\topstep-trading-bot\logs\fractal_bot_multi.log',
                        help='Path to fractal_bot log')
    parser.add_argument('--interval', type=int, default=60,
                        help='Seconds between checks (default: 60)')
    parser.add_argument('--min-confidence', type=int, default=70,
                        help='Minimum confidence to alert (default: 70)')
    parser.add_argument('--voice', choices=list(VOICES.keys()), default='aria',
                        help='TTS voice: aria (warm female), guy (friendly male), jenny (pro female), davis (calm male)')
    parser.add_argument('--silent', action='store_true',
                        help='Disable audio alerts')
    parser.add_argument('--realtime', action='store_true',
                        help='Real-time mode: tail log file instead of polling')
    parser.add_argument('--4h-high', type=float, dest='four_h_high',
                        help='Manual 4H range high')
    parser.add_argument('--4h-low', type=float, dest='four_h_low',
                        help='Manual 4H range low')

    args = parser.parse_args()

    # Build 4H range if provided
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

    if args.realtime:
        watch_realtime(
            args.log,
            min_confidence=args.min_confidence,
            voice=args.voice,
            silent=args.silent,
            four_hour_range=four_hour_range
        )
    else:
        watch_signals(
            args.log,
            interval=args.interval,
            min_confidence=args.min_confidence,
            four_hour_range=four_hour_range,
            voice=args.voice,
            silent=args.silent
        )
