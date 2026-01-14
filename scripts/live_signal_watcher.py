#!/usr/bin/env python3
"""
Live Signal Watcher - Monitors for high-confidence trading signals

================================================================================
CURRENT LIVE CONFIGURATION (as of 2026-01-13)
================================================================================

ACTIVE STRATEGIES:
  1. Williams Fractals (+$61,650, 28% WR) - Best overall profit
  2. Gap Continuation (+$25,787, 2.49 PF) - Best profit factor
  3. London Mean Reversion (+$42,044) - Best session strategy
  4. SMC (Order Blocks, Breakers, Inducement) - Confluence signals

REMOVED:
  - CISD (moved to discovered_strategies if needed)
  - 4H Range (moved to discovered_strategies if needed)
  - Position scaling (disabled for now)

AUDIO ALERTS:
  - Airport chime + voice announcement
  - Says: direction, strategy, entry price, stop, target

DATA SOURCES:
  - --api mode: Direct TopstepX API WebSocket connection (recommended)
  - --realtime mode: Tails fractal_bot log file
  - default mode: Polls fractal_bot log file at interval

USAGE:
    python scripts/live_signal_watcher.py --api                    # Direct API (recommended)
    python scripts/live_signal_watcher.py --realtime               # Tail log file

================================================================================

Runs continuously, checks every N seconds, and alerts when:
- New high-confidence signal appears
- Williams Fractal forms
- Gap continuation/fade setup
- London session mean reversion
- SMC zones (OB, breaker, etc.)

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
import asyncio
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
from typing import Optional

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

# Add parent to path for imports (AFTER topstep-trading-bot for correct module resolution)
NINJATRADER_BOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add topstep-trading-bot to path FIRST for API access
TOPSTEP_BOT_PATH = r'C:\Users\leone\topstep-trading-bot'
if TOPSTEP_BOT_PATH not in sys.path:
    sys.path.insert(0, TOPSTEP_BOT_PATH)

# Suppress noisy API logging (loguru) BEFORE imports
try:
    from loguru import logger as loguru_logger
    loguru_logger.disable('src.api')
    loguru_logger.disable('src.api.client')
    loguru_logger.disable('src.api.streaming')
except ImportError:
    pass

# TopstepX API imports (optional - only needed for --api mode)
TOPSTEP_API_AVAILABLE = False
TOPSTEP_API_KEY = None
TOPSTEP_USERNAME = None
TopstepXClient = None
MarketDataStream = None
try:
    import os as _os
    from dotenv import load_dotenv
    _env_path = _os.path.join(TOPSTEP_BOT_PATH, '.env')
    load_dotenv(_env_path, override=True)
    TOPSTEP_API_KEY = _os.getenv('TOPSTEPX_API_KEY') or _os.getenv('API_KEY')
    TOPSTEP_USERNAME = _os.getenv('TOPSTEPX_USERNAME') or _os.getenv('USERNAME')

    from src.api.client import TopstepXClient
    from src.api.streaming import MarketDataStream
    TOPSTEP_API_AVAILABLE = True
except ImportError:
    pass  # API not available, will use log file mode

# Now add ninjatrader-bot path for local scripts
if NINJATRADER_BOT_PATH not in sys.path:
    sys.path.insert(0, NINJATRADER_BOT_PATH)

from scripts.smc_analysis import (
    parse_bars, get_high_confidence_signals,
    get_session_info
)

from scripts.audio_alerts import speak_signal, VOICES
from scripts.discovered_strategies import (
    detect_williams_fractals,
    detect_gap_signal,
    detect_london_mean_reversion
)
import re

# ===========================================================================
# ACTIVE STRATEGIES (as of 2026-01-13)
# ===========================================================================
# 1. Williams Fractals (+$61,650, 28% WR) - Best overall profit
# 2. Gap Continuation (+$25,787, 2.49 PF) - Best profit factor
# 3. London Mean Reversion (+$42,044) - Best session strategy
# 4. SMC (Order Blocks, Breakers, Inducement) - Confluence signals
# ===========================================================================

# ===========================================================================
# LOGGING PATHS (must be defined before classes that use them)
# ===========================================================================
ALERT_LOG_DIR = r'C:\Users\leone\ninjatrader-bot\logs'
os.makedirs(ALERT_LOG_DIR, exist_ok=True)

# ===========================================================================
# BAR CLASS FOR API MODE
# ===========================================================================
@dataclass
class APIBar:
    """Bar data structure matching smc_analysis expectations.

    Must match: Bar = namedtuple('Bar', ['idx', 'O', 'H', 'L', 'C', 'color', 'body'])
    """
    timestamp: datetime
    O: float  # Open
    H: float  # High
    L: float  # Low
    C: float  # Close
    volume: int = 0
    idx: int = 0  # Bar index - required by smc_analysis

    @property
    def color(self) -> str:
        """Return bar color based on close vs open."""
        return 'GREEN' if self.C >= self.O else 'RED'

    @property
    def body(self) -> float:
        """Return the absolute body size."""
        return abs(self.C - self.O)

    @property
    def body_size(self) -> float:
        """Return the absolute body size (alias for body)."""
        return abs(self.C - self.O)

    @property
    def upper_wick(self) -> float:
        """Return upper wick size."""
        return self.H - max(self.O, self.C)

    @property
    def lower_wick(self) -> float:
        """Return lower wick size."""
        return min(self.O, self.C) - self.L

    @property
    def range(self) -> float:
        """Return total bar range."""
        return self.H - self.L


class TopstepXSignalWatcher:
    """
    Signal watcher that connects directly to TopstepX API.

    Receives real-time market data via WebSocket and generates signals
    using the same strategies as the log-based watcher.

    Bar data is saved to disk so restarts don't lose history.
    """

    # File to persist bar data
    BAR_CACHE_FILE = os.path.join(ALERT_LOG_DIR, 'bar_cache.json')

    # Tick data file - appends forever, never deleted
    TICK_DATA_FILE = os.path.join(ALERT_LOG_DIR, 'tick_data.csv')

    def __init__(
        self,
        min_confidence: int = 70,
        voice: str = 'aria',
        silent: bool = False,
        bar_seconds: int = 300,  # 5-minute bars
    ):
        self.min_confidence = min_confidence
        self.voice = voice
        self.silent = silent
        self.bar_seconds = bar_seconds

        # API connections
        self._client: Optional[TopstepXClient] = None
        self._market_stream: Optional[MarketDataStream] = None
        self.contract_id: Optional[str] = None

        # Bar building - load from cache if available
        self._bars: deque = deque(maxlen=250)
        self._current_bar: Optional[APIBar] = None
        self._bar_start_time: Optional[datetime] = None
        self._current_price: float = 0.0
        self._last_bar_save: Optional[datetime] = None

        # Load cached bars on init
        self._load_bar_cache()

        # Initialize tick data file
        self._init_tick_file()
        self._tick_file_handle = None

        # Signal tracking
        self._last_signal_hash: Optional[str] = None
        self._alert_count: int = 0
        self._alert_history: list = []
        self._pending_signal: Optional[dict] = None
        self._active_trade: Optional[dict] = None
        self._skipped_signal_hash: Optional[str] = None

        # State
        self._is_running: bool = False

    def _load_bar_cache(self):
        """Load bar history from disk cache."""
        try:
            if os.path.exists(self.BAR_CACHE_FILE):
                import json
                with open(self.BAR_CACHE_FILE, 'r') as f:
                    data = json.load(f)

                for i, bar_data in enumerate(data.get('bars', [])):
                    bar = APIBar(
                        timestamp=datetime.fromisoformat(bar_data['timestamp']),
                        O=bar_data['O'],
                        H=bar_data['H'],
                        L=bar_data['L'],
                        C=bar_data['C'],
                        volume=bar_data.get('volume', 0),
                        idx=bar_data.get('idx', i),
                    )
                    self._bars.append(bar)
        except Exception:
            pass

    def _save_bar_cache(self):
        """Save bar history to disk cache."""
        try:
            import json
            bars_data = []
            for bar in self._bars:
                bars_data.append({
                    'timestamp': bar.timestamp.isoformat(),
                    'O': bar.O,
                    'H': bar.H,
                    'L': bar.L,
                    'C': bar.C,
                    'volume': bar.volume,
                    'idx': bar.idx,
                })

            with open(self.BAR_CACHE_FILE, 'w') as f:
                json.dump({'bars': bars_data, 'saved_at': datetime.now().isoformat()}, f)

        except Exception as e:
            pass  # Silent fail on save

    def _init_tick_file(self):
        """Initialize tick data CSV file with headers if needed."""
        try:
            if not os.path.exists(self.TICK_DATA_FILE) or os.path.getsize(self.TICK_DATA_FILE) == 0:
                with open(self.TICK_DATA_FILE, 'w') as f:
                    f.write('timestamp,price,volume,type\n')
        except Exception:
            pass

    def _log_tick(self, price: float, volume: int = 0, tick_type: str = 'Q'):
        """Append a tick to the CSV file.

        Args:
            price: Tick price
            volume: Trade volume (0 for quotes)
            tick_type: 'Q' for quote, 'T' for trade
        """
        try:
            timestamp = datetime.now().isoformat(timespec='milliseconds')
            with open(self.TICK_DATA_FILE, 'a') as f:
                f.write(f'{timestamp},{price},{volume},{tick_type}\n')
        except Exception:
            pass  # Silent fail - don't interrupt trading

    async def start(self):
        """Start the signal watcher with direct API connection."""
        if not TOPSTEP_API_AVAILABLE:
            print(f"{RED}TopstepX API not available.{RESET}")
            return

        print(f"{CYAN}Connecting...{RESET}", end=' ', flush=True)

        # Initialize and authenticate
        self._client = TopstepXClient(
            api_key=TOPSTEP_API_KEY,
            username=TOPSTEP_USERNAME,
            environment="LIVE",
        )

        if not await self._client.authenticate():
            print(f"{RED}Auth failed!{RESET}")
            return

        accounts = await self._client.get_accounts()
        if not accounts:
            print(f"{RED}No accounts!{RESET}")
            return

        account_id = accounts[0].id
        contracts = await self._client.get_contracts("MES", account_id)
        if not contracts:
            print(f"{RED}No MES contract!{RESET}")
            return

        self.contract_id = contracts[0].id

        # Connect to market data
        self._market_stream = MarketDataStream(
            token=self._client._token,
            on_quote_callback=self._on_quote,
            on_trade_callback=self._on_trade,
        )

        await self._market_stream.connect()
        await self._market_stream.subscribe(self.contract_id)
        print(f"{GREEN}Connected!{RESET} [{len(self._bars)} bars cached]")
        print(f"{YELLOW}[T]=Take [S]=Skip [X]=Exit | Ctrl+C=Quit{RESET}\n")

        # Log startup (to file only)
        alert_logger.info("="*40)
        alert_logger.info(f"STARTED | {self.contract_id} | {self.min_confidence}% min")
        alert_logger.info("="*40)

        # Main loop - process signals every second for real-time responsiveness
        self._is_running = True
        self._last_signal_check = datetime.now()
        self._last_reconnect_check = datetime.now()
        self._connection_start = datetime.now()
        reconnect_hours = 20  # Reconnect before 24hr token expiry

        try:
            while self._is_running:
                # Check keyboard
                key = get_key()
                if key:
                    self._handle_key(key)

                now = datetime.now()

                # Process signals every second
                if (now - self._last_signal_check).total_seconds() >= 1.0:
                    self._process_signals()
                    self._last_signal_check = now

                # Check if we need to reconnect (every hour, reconnect after 20hrs)
                if (now - self._last_reconnect_check).total_seconds() >= 3600:
                    self._last_reconnect_check = now
                    hours_connected = (now - self._connection_start).total_seconds() / 3600

                    if hours_connected >= reconnect_hours:
                        print(f"\n{YELLOW}[Auto-reconnect] Token refresh after {hours_connected:.1f} hours...{RESET}")
                        alert_logger.info(f"AUTO-RECONNECT | Token refresh after {hours_connected:.1f} hours")

                        # Disconnect and reconnect
                        await self._reconnect()
                        self._connection_start = datetime.now()

                    # Also check if stream disconnected unexpectedly
                    elif self._market_stream and not self._market_stream.is_connected:
                        print(f"\n{YELLOW}[Auto-reconnect] Stream disconnected, reconnecting...{RESET}")
                        alert_logger.info("AUTO-RECONNECT | Stream disconnected")
                        await self._reconnect()

                await asyncio.sleep(0.05)  # 50ms for responsive keyboard

        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}Stopped. Alerts: {self._alert_count}{RESET}")
            alert_logger.info(f"WATCHER STOPPED | Alerts: {self._alert_count}")
        finally:
            await self.stop()

    async def stop(self):
        """Stop and disconnect."""
        self._is_running = False
        if self._market_stream:
            await self._market_stream.disconnect()
        if self._client and self._client._http_client:
            await self._client._http_client.aclose()

    async def _reconnect(self):
        """Reconnect to API and market stream."""
        try:
            # Disconnect existing
            if self._market_stream:
                await self._market_stream.disconnect()

            # Re-authenticate
            if not await self._client.authenticate():
                print(f"{RED}[Reconnect] Authentication failed!{RESET}")
                return False

            # Reconnect market stream with new token
            self._market_stream = MarketDataStream(
                token=self._client._token,
                on_quote_callback=self._on_quote,
                on_trade_callback=self._on_trade,
            )
            await self._market_stream.connect()
            await self._market_stream.subscribe(self.contract_id)

            print(f"{GREEN}[Reconnect] Success - back online{RESET}")
            return True

        except Exception as e:
            print(f"{RED}[Reconnect] Error: {e}{RESET}")
            alert_logger.error(f"RECONNECT FAILED | {e}")
            return False

    def _on_quote(self, data):
        """Handle quote update from WebSocket.

        SignalR sends: [contract_id, {quote_data}]
        Quote format: {'lastPrice': 7002.5, 'bestBid': 7002.5, 'bestAsk': 7002.75, ...}
        """
        try:
            # Handle list format from SignalR
            if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], dict):
                quote = data[1]
            elif isinstance(data, dict):
                quote = data
            else:
                return

            # Extract price - lastPrice is the most recent trade price
            price = quote.get('lastPrice') or quote.get('bestBid') or quote.get('bestAsk')
            if price:
                self._current_price = float(price)
                self._update_bar(self._current_price)
                # Log tick to file (forever)
                self._log_tick(self._current_price, 0, 'Q')
        except Exception as e:
            pass  # Ignore malformed quotes

    def _on_trade(self, data):
        """Handle trade print from WebSocket.

        SignalR sends: [contract_id, [{trade1}, {trade2}, ...]]
        Trade format: {'price': 7002.75, 'volume': 2, ...}
        """
        try:
            # Handle list format from SignalR
            if isinstance(data, list) and len(data) >= 2:
                trades = data[1]
                # Trades come as a list of trade objects
                if isinstance(trades, list) and trades:
                    # Log ALL trades in the batch (not just last)
                    for trade in trades:
                        price = trade.get('price')
                        volume = trade.get('volume', 0)
                        if price:
                            self._current_price = float(price)
                            self._update_bar(self._current_price)
                            # Log tick to file (forever)
                            self._log_tick(self._current_price, volume, 'T')
        except Exception as e:
            pass

    def _update_bar(self, price: float):
        """Update current bar with new price tick."""
        now = datetime.now()

        # Start new bar if needed
        if self._bar_start_time is None:
            self._start_new_bar(price, now)
            return

        # Check if bar is complete
        elapsed = (now - self._bar_start_time).total_seconds()
        if elapsed >= self.bar_seconds:
            # Close current bar and add to history
            if self._current_bar:
                self._bars.append(self._current_bar)
                # Save to cache every bar (so restarts resume instantly)
                self._save_bar_cache()
            # Start new bar
            self._start_new_bar(price, now)
            return

        # Update current bar with new tick
        if self._current_bar:
            self._current_bar.H = max(self._current_bar.H, price)
            self._current_bar.L = min(self._current_bar.L, price)
            self._current_bar.C = price

    def _start_new_bar(self, price: float, timestamp: datetime):
        """Start a new bar."""
        # idx is the next index after all existing bars
        next_idx = len(self._bars)
        self._current_bar = APIBar(
            timestamp=timestamp,
            O=price,
            H=price,
            L=price,
            C=price,
            volume=0,
            idx=next_idx,
        )
        self._bar_start_time = timestamp

    def _process_signals(self):
        """Process signals when a bar completes."""
        pst_time = get_pst_time().strftime('%H:%M:%S')
        session = get_session_for_ny_time()

        # Show status even during warmup - different thresholds for different strategies
        bar_count = len(self._bars)
        if bar_count < 10:
            bars_needed = 10 - bar_count
            print(f"\r{CYAN}[{pst_time}]{RESET} {session['current']} | {WHITE}{self._current_price:.2f}{RESET} | Warming up... {bar_count} bars ({bars_needed} more for SMC)    ", end='', flush=True)
            return

        # Show which strategies are active based on bar count
        active_strats = ['SMC']
        if bar_count >= 20:
            active_strats.append('London')
        if bar_count >= 150:
            active_strats.append('Williams')

        # Convert bars to list for strategy functions
        bars_list = list(self._bars)
        if self._current_bar:
            bars_list.append(self._current_bar)

        # Check active trade status
        if self._active_trade:
            self._check_trade_status(bars_list[-1].C if bars_list else self._current_price)
            return  # Don't generate new signals while in trade

        # Check pending signal
        if self._pending_signal:
            self._show_pending_status()
            return

        # Get SMC signals
        signals = get_high_confidence_signals(
            bars_list,
            min_confidence=self.min_confidence,
            live_price=self._current_price
        )

        # Add discovered strategies
        williams = detect_williams_fractals(bars_list)
        if williams:
            signals.append(williams)

        gap = detect_gap_signal(bars_list)
        if gap:
            signals.append(gap)

        london = detect_london_mean_reversion(bars_list)
        if london:
            signals.append(london)

        # Sort by confidence
        signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        signal_hash = str([(s['strategy'], s['direction'], s['confidence']) for s in signals])

        if signal_hash != self._last_signal_hash and signals:
            if signal_hash == self._skipped_signal_hash:
                return  # Already skipped this signal

            self._skipped_signal_hash = None
            self._alert_count += 1
            top = signals[0]

            # Print alert
            sig_color = GREEN if top['direction'] == 'LONG' else RED
            risk_pts = top.get('risk', abs(top['entry'] - top['stop_loss']))
            target_pts = abs(top['take_profit'] - top['entry'])

            print(f"\n{BRIGHT}{YELLOW}{'!'*60}{RESET}")
            print(f"  {BRIGHT}NEW ALERT #{self._alert_count}{RESET} @ {pst_time}")
            print(f"{BRIGHT}{YELLOW}{'!'*60}{RESET}")
            print(f"  {sig_color}{BRIGHT}{top['direction']}{RESET} {top['confidence']}% [{top['strategy']}]")
            print(f"  Entry:  {WHITE}{top['entry']:.2f}{RESET}")
            print(f"  Stop:   {RED}{top['stop_loss']:.2f}{RESET} ({risk_pts:.1f} pts = ${risk_pts*50:.0f} ES / ${risk_pts*5:.0f} MES)")
            print(f"  Target: {GREEN}{top['take_profit']:.2f}{RESET} ({target_pts:.1f} pts = ${target_pts*50:.0f} ES / ${target_pts*5:.0f} MES)")
            print(f"  R:R:    {top.get('rr', target_pts/risk_pts if risk_pts > 0 else 0):.1f}")
            print(f"  Reason: {top['reasoning']}")
            print(f"\n  {YELLOW}{BRIGHT}>>> Press [T] to TAKE or [S] to SKIP <<<{RESET}\n", flush=True)

            # Audio
            if not self.silent:
                try:
                    speak_signal(top, voice=self.voice, chime=True,
                                chime_style='airport', overlap_seconds=3.0,
                                speech_rate='+25%')
                except Exception as e:
                    print('\a', end='', flush=True)

            # Log
            rr = top.get('rr', target_pts/risk_pts if risk_pts > 0 else 0)
            alert_logger.info(f"NEW ALERT | {top['direction']} {top['confidence']}% | {top['strategy']}")
            alert_logger.info(f"  Entry: {top['entry']:.2f} | Stop: {top['stop_loss']:.2f} | Target: {top['take_profit']:.2f} | R:R {rr:.1f}")
            for handler in alert_logger.handlers:
                handler.flush()

            # Set pending
            self._pending_signal = {
                'direction': top['direction'],
                'entry': top['entry'],
                'stop': top['stop_loss'],
                'target': top['take_profit'],
                'time': pst_time
            }

            self._alert_history.append({
                'time': pst_time,
                'direction': top['direction'],
                'confidence': top['confidence'],
                'entry': top['entry'],
                'stop': top['stop_loss'],
                'target': top['take_profit'],
                'rr': rr
            })

        elif signals:
            # Show current signal status
            top = signals[0]
            sig_color = GREEN if top['direction'] == 'LONG' else RED
            print(f"\r{CYAN}[{pst_time}]{RESET} {session['current']} | {WHITE}{self._current_price:.2f}{RESET} | {sig_color}{top['direction']}{RESET} {top['confidence']}%    ", end='', flush=True)
        else:
            strats_str = '+'.join(active_strats)
            bars_to_williams = 150 - bar_count if bar_count < 150 else 0
            extra = f" ({bars_to_williams} bars to Williams)" if bars_to_williams > 0 else ""
            print(f"\r{CYAN}[{pst_time}]{RESET} {session['current']} | {WHITE}{self._current_price:.2f}{RESET} | [{strats_str}] No signal{extra}    ", end='', flush=True)

        self._last_signal_hash = signal_hash

    def _handle_key(self, key: str):
        """Handle keyboard input."""
        if key == 'T' and self._pending_signal and not self._active_trade:
            self._active_trade = self._pending_signal
            self._pending_signal = None
            print(f"\n{GREEN}{BRIGHT}>>> TRADE TAKEN{RESET} - Tracking {self._active_trade['direction']} @ {self._active_trade['entry']:.2f}")
            alert_logger.info(f"USER TOOK TRADE | {self._active_trade['direction']} @ {self._active_trade['entry']:.2f}")

        elif key == 'S' and self._pending_signal:
            print(f"\n{YELLOW}>>> SKIPPED{RESET} - Waiting for next signal")
            alert_logger.info(f"USER SKIPPED | {self._pending_signal['direction']} @ {self._pending_signal['entry']:.2f}")
            self._skipped_signal_hash = self._last_signal_hash
            self._pending_signal = None

        elif key == 'X' and self._active_trade:
            exit_price = self._current_price
            if self._active_trade['direction'] == 'LONG':
                pnl = exit_price - self._active_trade['entry']
            else:
                pnl = self._active_trade['entry'] - exit_price
            pnl_color = GREEN if pnl >= 0 else RED
            print(f"\n{MAGENTA}{BRIGHT}>>> MANUAL EXIT{RESET} @ {exit_price:.2f} | P&L: {pnl_color}{pnl:+.2f}{RESET} pts")
            alert_logger.info(f"USER EXITED | {self._active_trade['direction']} @ {exit_price:.2f} | P&L: {pnl:+.2f} pts")
            self._active_trade = None

    def _check_trade_status(self, current_price: float):
        """Check if active trade hit stop or target."""
        if not self._active_trade:
            return

        pst_time = get_pst_time().strftime('%H:%M:%S')
        session = get_session_for_ny_time()

        pnl = 0
        status = 'ACTIVE'

        if self._active_trade['direction'] == 'LONG':
            pnl = current_price - self._active_trade['entry']
            if current_price <= self._active_trade['stop']:
                status = 'STOPPED'
                pnl = self._active_trade['stop'] - self._active_trade['entry']
            elif current_price >= self._active_trade['target']:
                status = 'TARGET'
                pnl = self._active_trade['target'] - self._active_trade['entry']
        else:
            pnl = self._active_trade['entry'] - current_price
            if current_price >= self._active_trade['stop']:
                status = 'STOPPED'
                pnl = self._active_trade['entry'] - self._active_trade['stop']
            elif current_price <= self._active_trade['target']:
                status = 'TARGET'
                pnl = self._active_trade['entry'] - self._active_trade['target']

        if status in ['STOPPED', 'TARGET']:
            status_color = RED if status == 'STOPPED' else GREEN
            print(f"\n{BRIGHT}{status_color}{'='*60}{RESET}")
            print(f"  TRADE {status}! P&L: {pnl:+.2f} pts (${pnl*50:+.0f} ES)")
            print(f"  Entry: {self._active_trade['entry']:.2f} -> Exit: {current_price:.2f}")
            print(f"{BRIGHT}{status_color}{'='*60}{RESET}\n")
            alert_logger.info(f"TRADE {status} | {self._active_trade['direction']} | P&L: {pnl:+.2f} pts")
            self._active_trade = None
            self._last_signal_hash = None
        else:
            pnl_color = GREEN if pnl >= 0 else RED
            sig_color = GREEN if self._active_trade['direction'] == 'LONG' else RED
            print(f"\r{CYAN}[{pst_time}]{RESET} {session['current']} | {WHITE}{current_price:.2f}{RESET} | {sig_color}{self._active_trade['direction']}{RESET} @ {self._active_trade['entry']:.2f} | P&L: {pnl_color}{pnl:+.2f}{RESET} pts {YELLOW}[X=exit]{RESET}    ", end='', flush=True)

    def _show_pending_status(self):
        """Show status while waiting for user input on pending signal."""
        if not self._pending_signal:
            return

        pst_time = get_pst_time().strftime('%H:%M:%S')
        sig_color = GREEN if self._pending_signal['direction'] == 'LONG' else RED

        if self._pending_signal['direction'] == 'LONG':
            pnl = self._current_price - self._pending_signal['entry']
        else:
            pnl = self._pending_signal['entry'] - self._current_price

        pnl_color = GREEN if pnl >= 0 else RED
        print(f"\r{CYAN}[{pst_time}]{RESET} {WHITE}{self._current_price:.2f}{RESET} | {sig_color}{BRIGHT}{self._pending_signal['direction']}{RESET} @ {self._pending_signal['entry']:.2f} | {pnl_color}{pnl:+.2f}{RESET} pts | {YELLOW}{BRIGHT}[T]ake [S]kip?{RESET}    ", end='', flush=True)


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

ALERT_LOG_FILE = os.path.join(ALERT_LOG_DIR, 'signal_alerts.log')

def setup_logging():
    """Setup logging to file"""

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

    # Calculate dollar risk
    risk_pts = sig.get('risk', abs(sig['entry'] - sig['stop_loss']))
    target_pts = abs(sig['take_profit'] - sig['entry'])
    rr = sig.get('rr', target_pts/risk_pts if risk_pts > 0 else 0)

    return f"""
{'='*60}
  {arrow} [{sig['strategy']}] {sig['direction']} - {sig['confidence']}% confidence
{'='*60}
  Entry:  {sig['entry']:.2f}
  Stop:   {sig['stop_loss']:.2f} ({risk_pts:.2f} pts = ${risk_pts*50:.0f} ES / ${risk_pts*5:.0f} MES)
  Target: {sig['take_profit']:.2f} ({target_pts:.2f} pts = ${target_pts*50:.0f} ES / ${target_pts*5:.0f} MES)
  R:R:    1:{rr:.1f}
  Reason: {sig['reasoning']}
"""


def watch_signals(log_path: str, interval: int = 60, min_confidence: int = 70,
                  voice: str = 'aria', silent: bool = False):
    """
    Continuously watch for trading signals.

    Args:
        log_path: Path to fractal_bot log
        interval: Seconds between checks
        min_confidence: Minimum confidence to alert
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

            # Get SMC signals (Order Blocks, Breakers, Inducement)
            signals = get_high_confidence_signals(
                bars,
                min_confidence=min_confidence,
                live_price=current_price
            )

            # Add discovered strategies
            williams = detect_williams_fractals(bars)
            if williams:
                signals.append(williams)

            gap = detect_gap_signal(bars)
            if gap:
                signals.append(gap)

            london = detect_london_mean_reversion(bars)
            if london:
                signals.append(london)

            # Sort all signals by confidence
            signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)

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
                risk_pts = top.get('risk', abs(top['entry'] - top['stop_loss']))
                target_pts = abs(top['take_profit'] - top['entry'])
                rr = top.get('rr', target_pts/risk_pts if risk_pts > 0 else 0)

                print(f"\n{BRIGHT}CURRENT SIGNAL:{RESET}")
                print(f"  {sig_color}{BRIGHT}{top['direction']}{RESET} {top['confidence']}% [{top['strategy']}]")
                print(f"  Entry:  {WHITE}{top['entry']:.2f}{RESET}")
                print(f"  Stop:   {RED}{top['stop_loss']:.2f}{RESET} ({risk_pts:.1f} pts = ${risk_pts*50:.0f} ES / ${risk_pts*5:.0f} MES)")
                print(f"  Target: {GREEN}{top['take_profit']:.2f}{RESET} ({target_pts:.1f} pts = ${target_pts*50:.0f} ES / ${target_pts*5:.0f} MES)")
                print(f"  R:R:    {rr:.1f}")
            else:
                print(f"\n{WHITE}No active signal - waiting for setup{RESET}")

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
                   voice: str = 'aria', silent: bool = False):
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

            # Get SMC signals
            signals = get_high_confidence_signals(
                bars, min_confidence=min_confidence,
                live_price=current_price
            )

            # Add discovered strategies
            williams = detect_williams_fractals(bars)
            if williams:
                signals.append(williams)

            gap = detect_gap_signal(bars)
            if gap:
                signals.append(gap)

            london = detect_london_mean_reversion(bars)
            if london:
                signals.append(london)

            # Sort all signals by confidence
            signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)

            signal_hash = str([(s['strategy'], s['direction'], s['confidence']) for s in signals])

            if signal_hash != last_signal_hash and signals:
                # NEW SIGNAL - clear any previous skipped signal
                skipped_signal_hash = None
                alert_count += 1
                top = signals[0]
                sig_color = GREEN if top['direction'] == 'LONG' else RED
                risk_pts = top.get('risk', abs(top['entry'] - top['stop_loss']))
                target_pts = abs(top['take_profit'] - top['entry'])

                # Print FIRST (immediate visual feedback)
                print(f"\n{BRIGHT}{YELLOW}{'!'*60}{RESET}")
                print(f"  {BRIGHT}NEW ALERT #{alert_count}{RESET} @ {pst_time}")
                print(f"{BRIGHT}{YELLOW}{'!'*60}{RESET}")
                print(f"  {sig_color}{BRIGHT}{top['direction']}{RESET} {top['confidence']}% [{top['strategy']}]")
                print(f"  Entry:  {WHITE}{top['entry']:.2f}{RESET}")
                print(f"  Stop:   {RED}{top['stop_loss']:.2f}{RESET} ({risk_pts:.1f} pts = ${risk_pts*50:.0f} ES / ${risk_pts*5:.0f} MES)")
                print(f"  Target: {GREEN}{top['take_profit']:.2f}{RESET} ({target_pts:.1f} pts = ${target_pts*50:.0f} ES / ${target_pts*5:.0f} MES)")
                print(f"  R:R:    {top.get('rr', target_pts/risk_pts if risk_pts > 0 else 0):.1f}")
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
                rr = top.get('rr', target_pts/risk_pts if risk_pts > 0 else 0)
                alert_logger.info(f"NEW ALERT | {top['direction']} {top['confidence']}% | {top['strategy']}")
                alert_logger.info(f"  Entry: {top['entry']:.2f} | Stop: {top['stop_loss']:.2f} ({risk_pts:.1f}pts/${risk_pts*50:.0f}) | Target: {top['take_profit']:.2f} | R:R {rr:.1f}")
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
                    'rr': rr
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
    parser.add_argument('--api', action='store_true',
                        help='API mode: connect directly to TopstepX API (no log file needed)')
    parser.add_argument('--4h-high', type=float, dest='four_h_high',
                        help='Manual 4H range high')
    parser.add_argument('--4h-low', type=float, dest='four_h_low',
                        help='Manual 4H range low')

    args = parser.parse_args()

    if args.api:
        # Direct API connection mode
        if not TOPSTEP_API_AVAILABLE:
            print(f"{RED}ERROR: TopstepX API not available.{RESET}")
            print(f"Make sure topstep-trading-bot is at: {TOPSTEP_BOT_PATH}")
            print(f"And has the required dependencies installed.")
            sys.exit(1)

        watcher = TopstepXSignalWatcher(
            min_confidence=args.min_confidence,
            voice=args.voice,
            silent=args.silent,
        )
        asyncio.run(watcher.start())
    elif args.realtime:
        watch_realtime(
            args.log,
            min_confidence=args.min_confidence,
            voice=args.voice,
            silent=args.silent
        )
    else:
        watch_signals(
            args.log,
            interval=args.interval,
            min_confidence=args.min_confidence,
            voice=args.voice,
            silent=args.silent
        )
