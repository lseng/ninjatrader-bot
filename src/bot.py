#!/usr/bin/env python3
"""
NinjaTrader Trading Bot

Live trading bot using Williams Fractals strategy.
Validated on 2-year backtest: 60.6% win rate, 1.94 profit factor.

For $1,000 account trading MES futures.
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Optional
from collections import deque
import pandas as pd
import pytz

from api.client import NinjaTraderClient
from api.models import OrderRequest, Bar, Position
from strategies.williams_fractals_strategy import WilliamsFractalsStrategy, TradeSignal
from config import TradingConfig, APIConfig, get_position_size

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TradingBot:
    """
    Live trading bot for NinjaTrader/Tradovate.

    Uses Williams Fractals strategy with proper risk management.
    """

    def __init__(
        self,
        config: TradingConfig = None,
        api_config: APIConfig = None
    ):
        self.config = config or TradingConfig()
        self.api_config = api_config or APIConfig()

        # Initialize strategy
        self.strategy = WilliamsFractalsStrategy(
            fractal_period=self.config.fractal_period,
            ma_fast=self.config.ma_fast,
            ma_medium=self.config.ma_medium,
            ma_slow=self.config.ma_slow,
            atr_period=self.config.atr_period,
            atr_multiplier=self.config.atr_multiplier,
            target_rr=self.config.target_rr,
            contract_value=self.config.contract_value,
            flatten_hour=self.config.flatten_hour
        )

        # API client
        self.client: Optional[NinjaTraderClient] = None
        self.account_id: Optional[str] = None

        # State
        self.capital = self.config.initial_capital
        self.daily_pnl = 0.0
        self.position: Optional[Position] = None
        self.pending_orders = []

        # Bar history (need ~200 bars for indicators)
        self.bars: deque = deque(maxlen=300)
        self.last_bar_time: Optional[datetime] = None

        # Session tracking
        self.trades_today = []
        self.session_start_capital = self.config.initial_capital

        # Timezone
        self.tz = pytz.timezone('America/Chicago')  # Central Time

        # Running state
        self.running = False

    async def connect(self):
        """Connect to the API."""
        logger.info("Connecting to NinjaTrader API...")

        self.client = NinjaTraderClient(
            username=self.api_config.username,
            password=self.api_config.password,
            api_key=self.api_config.api_key,
            demo=self.api_config.demo
        )

        await self.client.connect()

        # Get account
        accounts = await self.client.get_accounts()
        if accounts:
            self.account_id = accounts[0].account_id
            self.capital = accounts[0].cash_balance
            logger.info(f"Connected to account: {self.account_id}")
            logger.info(f"Account balance: ${self.capital:,.2f}")
        else:
            raise RuntimeError("No accounts found")

        # Check existing positions
        await self.sync_position()

    async def disconnect(self):
        """Disconnect from API."""
        if self.client:
            await self.client.disconnect()
            logger.info("Disconnected from API")

    async def sync_position(self):
        """Sync position state with broker."""
        if not self.client:
            return

        positions = await self.client.get_positions(self.account_id)

        for pos in positions:
            if self.config.symbol in str(pos.symbol):
                self.position = pos
                self.strategy.position = pos.quantity
                self.strategy.entry_price = pos.average_price
                logger.info(f"Existing position: {pos.quantity} @ {pos.average_price}")
                return

        self.position = None
        self.strategy.position = 0

    def is_trading_hours(self) -> bool:
        """Check if within trading hours."""
        now = datetime.now(self.tz)

        # Check day of week (Monday=0, Sunday=6)
        if now.weekday() >= 5:  # Saturday/Sunday
            return False

        hour = now.hour
        minute = now.minute

        # Flatten check - stop trading before flatten time
        if hour >= self.config.flatten_hour and minute >= self.config.flatten_minute:
            return False

        # Within trading window
        if hour < self.config.trading_start_hour:
            return False

        return True

    def should_flatten(self) -> bool:
        """Check if should flatten all positions."""
        now = datetime.now(self.tz)
        hour = now.hour
        minute = now.minute

        if hour > self.config.flatten_hour:
            return True
        if hour == self.config.flatten_hour and minute >= self.config.flatten_minute:
            return True

        return False

    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit hit."""
        if self.daily_pnl <= -self.config.daily_loss_limit:
            logger.warning(f"Daily loss limit reached: ${self.daily_pnl:,.2f}")
            return True
        return False

    async def on_bar(self, bar: Bar):
        """Process new bar."""
        # Add to history
        self.bars.append({
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })

        self.last_bar_time = bar.timestamp

        # Need minimum history
        if len(self.bars) < 200:
            logger.debug(f"Building history: {len(self.bars)}/200 bars")
            return

        # Check trading conditions
        if not self.is_trading_hours():
            if self.should_flatten() and self.position:
                await self.flatten_position("End of day")
            return

        if self.check_daily_loss_limit():
            if self.position:
                await self.flatten_position("Daily loss limit")
            return

        # Convert to DataFrame for strategy
        df = pd.DataFrame(list(self.bars))

        # Check for exit first
        if self.position:
            await self.check_exit(bar)

        # Check for entry
        if not self.position:
            await self.check_entry(df, bar)

    async def check_entry(self, df: pd.DataFrame, current_bar: Bar):
        """Check for entry signal."""
        signal = self.strategy.generate_signal(df)

        if signal.direction == 0:
            return

        # Calculate position size
        size = get_position_size(
            capital=self.capital,
            entry=signal.entry_price,
            stop=signal.stop_loss,
            risk_pct=self.config.risk_per_trade,
            contract_value=self.config.contract_value,
            max_contracts=self.config.max_contracts
        )

        # Log signal
        direction = "LONG" if signal.direction == 1 else "SHORT"
        logger.info(f"SIGNAL: {direction} @ {signal.entry_price:.2f}")
        logger.info(f"  Stop: {signal.stop_loss:.2f} | Target: {signal.take_profit:.2f}")
        logger.info(f"  Size: {size} contracts | Reason: {signal.reason}")

        # Place orders
        await self.enter_position(signal, size)

    async def enter_position(self, signal: TradeSignal, size: int):
        """Enter a position."""
        if not self.client:
            logger.error("Not connected to API")
            return

        try:
            # Market entry order
            side = "buy" if signal.direction == 1 else "sell"

            entry_order = OrderRequest(
                symbol=self.config.symbol,
                side=side,
                order_type="market",
                quantity=size
            )

            order = await self.client.place_order(entry_order, self.account_id)
            logger.info(f"Entry order placed: {order.order_id}")

            # Place stop loss
            stop_side = "sell" if signal.direction == 1 else "buy"
            stop_order = OrderRequest(
                symbol=self.config.symbol,
                side=stop_side,
                order_type="stop",
                quantity=size,
                stop_price=signal.stop_loss
            )

            stop = await self.client.place_order(stop_order, self.account_id)
            logger.info(f"Stop order placed: {stop.order_id} @ {signal.stop_loss:.2f}")

            # Place take profit
            tp_order = OrderRequest(
                symbol=self.config.symbol,
                side=stop_side,
                order_type="limit",
                quantity=size,
                price=signal.take_profit
            )

            tp = await self.client.place_order(tp_order, self.account_id)
            logger.info(f"Target order placed: {tp.order_id} @ {signal.take_profit:.2f}")

            # Update state
            self.strategy.position = size if signal.direction == 1 else -size
            self.strategy.entry_price = signal.entry_price
            self.strategy.stop_loss = signal.stop_loss
            self.strategy.take_profit = signal.take_profit

            self.pending_orders = [stop.order_id, tp.order_id]

        except Exception as e:
            logger.error(f"Failed to enter position: {e}")

    async def check_exit(self, bar: Bar):
        """Check for exit conditions."""
        if not self.position:
            return

        # Check if stop or target hit (broker should handle, but double-check)
        entry = self.strategy.entry_price
        stop = self.strategy.stop_loss
        target = self.strategy.take_profit

        direction = 1 if self.position.quantity > 0 else -1

        if direction == 1:  # Long
            if bar.low <= stop:
                await self.exit_position(stop, "Stop Loss")
            elif bar.high >= target:
                await self.exit_position(target, "Take Profit")
        else:  # Short
            if bar.high >= stop:
                await self.exit_position(stop, "Stop Loss")
            elif bar.low <= target:
                await self.exit_position(target, "Take Profit")

    async def exit_position(self, exit_price: float, reason: str):
        """Exit current position."""
        if not self.client or not self.position:
            return

        try:
            # Cancel pending orders
            for order_id in self.pending_orders:
                await self.client.cancel_order(order_id)

            # Market exit
            size = abs(self.position.quantity)
            side = "sell" if self.position.quantity > 0 else "buy"

            exit_order = OrderRequest(
                symbol=self.config.symbol,
                side=side,
                order_type="market",
                quantity=size
            )

            await self.client.place_order(exit_order, self.account_id)

            # Calculate P&L
            direction = 1 if self.position.quantity > 0 else -1
            entry = self.strategy.entry_price
            gross_pnl = (exit_price - entry) * direction * self.config.contract_value * size
            commission = self.config.commission_per_contract * size
            net_pnl = gross_pnl - commission

            # Update state
            self.capital += net_pnl
            self.daily_pnl += net_pnl

            self.trades_today.append({
                'entry': entry,
                'exit': exit_price,
                'direction': 'LONG' if direction == 1 else 'SHORT',
                'size': size,
                'pnl': net_pnl,
                'reason': reason
            })

            result = "WIN" if net_pnl > 0 else "LOSS"
            logger.info(f"EXIT: {reason} | {result} | P&L: ${net_pnl:,.2f} | Balance: ${self.capital:,.2f}")

            # Reset state
            self.position = None
            self.strategy.position = 0
            self.strategy.entry_price = 0
            self.pending_orders = []

        except Exception as e:
            logger.error(f"Failed to exit position: {e}")

    async def flatten_position(self, reason: str):
        """Flatten all positions (EOD or emergency)."""
        if not self.position:
            return

        logger.warning(f"FLATTEN: {reason}")
        # Get current price for exit
        quote = await self.client.get_quote(self.config.symbol)
        exit_price = quote.last
        await self.exit_position(exit_price, f"Flatten: {reason}")

    async def run(self):
        """Main bot loop."""
        logger.info("=" * 60)
        logger.info("NINJATRADER TRADING BOT")
        logger.info(f"Strategy: Williams Fractals")
        logger.info(f"Symbol: {self.config.symbol}")
        logger.info(f"Capital: ${self.config.initial_capital:,.2f}")
        logger.info(f"Risk per trade: {self.config.risk_per_trade * 100:.1f}%")
        logger.info(f"Demo mode: {self.api_config.demo}")
        logger.info("=" * 60)

        await self.connect()

        self.running = True
        self.session_start_capital = self.capital

        try:
            # Stream real-time bars
            logger.info("Streaming market data...")

            async for bar in self.stream_bars():
                if not self.running:
                    break

                await self.on_bar(bar)

        except KeyboardInterrupt:
            logger.info("Shutdown requested...")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            if self.position:
                await self.flatten_position("Bot shutdown")
            await self.disconnect()
            self.print_session_summary()

    async def stream_bars(self):
        """Stream 5-minute bars."""
        # First load historical bars
        from datetime import datetime, timedelta

        logger.info("Loading historical bars...")

        request = type('HistoricalDataRequest', (), {
            'symbol': self.config.symbol,
            'timeframe': self.config.timeframe,
            'start_date': datetime.now() - timedelta(days=5),
            'end_date': None
        })()

        try:
            bars = await self.client.get_historical_bars(request)
            for bar in bars:
                self.bars.append({
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })
            logger.info(f"Loaded {len(bars)} historical bars")
        except Exception as e:
            logger.warning(f"Could not load historical bars: {e}")

        # Now stream real-time quotes and aggregate to bars
        current_bar = None
        bar_start = None

        async for quote in self.client.stream_quotes([self.config.symbol]):
            now = quote.timestamp

            # Round to bar boundary
            if self.config.timeframe == "5min":
                bar_boundary = now.replace(
                    minute=(now.minute // 5) * 5,
                    second=0,
                    microsecond=0
                )
            else:
                bar_boundary = now.replace(second=0, microsecond=0)

            # New bar?
            if bar_start != bar_boundary:
                # Emit previous bar
                if current_bar:
                    yield Bar(
                        symbol=self.config.symbol,
                        timestamp=bar_start,
                        open=current_bar['open'],
                        high=current_bar['high'],
                        low=current_bar['low'],
                        close=current_bar['close'],
                        volume=current_bar['volume'],
                        timeframe=self.config.timeframe
                    )

                # Start new bar
                bar_start = bar_boundary
                current_bar = {
                    'open': quote.last,
                    'high': quote.last,
                    'low': quote.last,
                    'close': quote.last,
                    'volume': quote.volume
                }
            else:
                # Update current bar
                if current_bar:
                    current_bar['high'] = max(current_bar['high'], quote.last)
                    current_bar['low'] = min(current_bar['low'], quote.last)
                    current_bar['close'] = quote.last
                    current_bar['volume'] += quote.volume

    def print_session_summary(self):
        """Print end-of-session summary."""
        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info("=" * 60)

        if not self.trades_today:
            logger.info("No trades executed")
            return

        total_trades = len(self.trades_today)
        winners = [t for t in self.trades_today if t['pnl'] > 0]
        losers = [t for t in self.trades_today if t['pnl'] <= 0]

        total_pnl = sum(t['pnl'] for t in self.trades_today)
        win_pnl = sum(t['pnl'] for t in winners) if winners else 0
        loss_pnl = sum(t['pnl'] for t in losers) if losers else 0

        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Winners: {len(winners)} ({len(winners)/total_trades*100:.1f}%)")
        logger.info(f"Losers: {len(losers)} ({len(losers)/total_trades*100:.1f}%)")
        logger.info(f"Gross Win: ${win_pnl:,.2f}")
        logger.info(f"Gross Loss: ${loss_pnl:,.2f}")
        logger.info(f"Net P&L: ${total_pnl:,.2f}")
        logger.info(f"Starting Balance: ${self.session_start_capital:,.2f}")
        logger.info(f"Ending Balance: ${self.capital:,.2f}")
        logger.info(f"Return: {(self.capital - self.session_start_capital) / self.session_start_capital * 100:.2f}%")

    def stop(self):
        """Stop the bot."""
        self.running = False


async def main():
    """Run the trading bot."""
    bot = TradingBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
